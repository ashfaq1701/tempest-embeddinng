"""Centroid-to-centroid head on the Poincaré ball — the Gromov product with LEARNABLE term weights:

    s(u,v) = w . [-d(P_u, P_v), rho_v],   w init [1, 1],   rho_x = dist0(P_x)

At w = [1, 1] this is exactly the Gromov product 2*(u|v)_o - rho_u (LCA depth on a tree, the defining
quantity of Gromov hyperbolicity). rho_u is dropped because it is constant within a query and cancels
from both the loss and the gradient under the per-query softmax CE (sum_c dL/ds_c = 0); the 0.5 is only
a logit scale. The weights are added to READ where the ranking optimum sits relative to the theoretical
[1,1]: w[1] << 1 (or < 0) means the ranking prefers a popularity prior over deep-LCA agreement.
P_x is the weighted gyro-midpoint of x's walk-token bag.

Pooling weights come from a small MLP (hidden = 8*N_FEAT) over four per-token scalars: -(age/mnia), -pos,
and the geodesic distance / cosine alignment to the bag's unweighted midpoint (the last two are centre
features with the per-bag level removed). mnia (mean node inter-arrival) is a fixed age scale."""
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class PoincareManifold:
    """Poincaré-ball geometry (c=1). `manifold` is kept for E's init + RiemannianAdam."""

    def __init__(self, c: float = 1.0):
        self.manifold = geoopt.PoincareBall(c=c)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Elementwise geodesic distance, broadcasting over leading dims. LOWER = closer."""
        return self.manifold.dist(x, y)

    def dist0(self, x: torch.Tensor) -> torch.Tensor:
        """Hyperbolic radius 2*artanh(||x||)."""
        return self.manifold.dist0(x)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted gyro-midpoint: x [Q,T,d], w [Q,T] -> [Q,d]."""
        return self.manifold.weighted_midpoint(x, weights=w, reducedim=[-2], dim=-1, keepdim=False)


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([-(age/mnia), -pos, spr, ang])); hidden = 8*N_FEAT, randomly
    initialised, with mnia as a fixed age scale. The softmax over tokens dampens the random init."""

    N_FEAT = 4

    def __init__(self, mnia: float):
        super().__init__()
        self.mnia = float(mnia)                 # fixed age scale
        hidden = 8 * self.N_FEAT                # 8x expansion, random init
        self.net = nn.Sequential(nn.Linear(self.N_FEAT, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @staticmethod
    def centre_feats(geom: PoincareManifold, x: torch.Tensor,
                     valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token geometry relative to the bag's unweighted midpoint C, per-bag level removed from both:
        spr = d(x_p, C) / mean_q d(x_q, C),  ang = cos(x_p, C) - mean_q cos(x_q, C).  Each -> [Q, T]."""
        vf = valid.to(x.dtype)
        n = vf.sum(dim=-1, keepdim=True).clamp_min(1.0)
        c = geom.midpoint(x, vf / n).unsqueeze(-2)                              # [Q, 1, d]
        dc = geom.dist(c, x) * vf
        spr = dc / (dc.sum(dim=-1, keepdim=True) / n).clamp_min(1e-6)
        cs = F.cosine_similarity(c, x, dim=-1, eps=1e-6) * vf
        return spr, (cs - cs.sum(dim=-1, keepdim=True) / n) * vf

    def forward(self, geom: PoincareManifold, tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        rec = -(tokens.ages.clamp_min(0).float() / self.mnia)                   # -(age/mnia)
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)                    # -pos
        spr, ang = self.centre_feats(geom, x.detach(), valid)
        feat = torch.stack([rec, pos, spr, ang], dim=-1).to(x.dtype)            # [Q, T, 4]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; BagWeights is the only other trained module — the score itself
    has no parameters."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float,
                 init_irange: float = 1e-3):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(mean_node_inter_arrival)

        # Near-origin init: uniform(-irange, irange) per coord -> r ~ 2*irange*sqrt(d/3).
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.projx(
                (torch.rand(self.num_nodes, self.d_emb) * 2 - 1) * float(init_irange))
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        # Learnable weights on the Gromov terms [-d, rho_v], init [1, 1] = the exact Gromov product
        # (2*(u|v)_o with rho_u dropped). Read whether they stay near [1,1] or move (e.g. w[1] < 0
        # would mean the ranking prefers a popularity prior over the deep-LCA Gromov quantity).
        self.score_w = nn.Parameter(torch.tensor([1.0, 1.0]))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> P [Q, d], the pooling-weighted gyro-midpoint."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)                                               # all-padding walk -> use seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(self.geom, tokens, x, valid)                       # [Q, T] sums to 1, 0 on padding
        return self.geom.midpoint(x, w)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        s(u,v) = w . [-d(P_u,P_v), rho_v]; the per-query-constant rho_u is dropped (cancels under the
        per-query softmax CE). w init [1,1] = the exact Gromov product 2*(u|v)_o; read where w goes."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]

        rho_v = self.geom.dist0(p_v)                                            # [B, C]
        dist = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C]
        feats = torch.stack([-dist, rho_v], dim=-1)                             # [B, C, 2]
        return (feats * self.score_w).sum(dim=-1)
