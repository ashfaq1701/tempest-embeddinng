"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = w . f + b_v, f = [geo, cos, geo_spread_v,
cos_spread_v, coh_v] (geo=-d(P_u,P_v), cos=cos(P_u,P_v); the spreads are the candidate cloud's
pooling-weighted mean geo/cos to its centroid; coh_v is the candidate cloud's (w*r)-weighted mean
off-diagonal cosine coherence), learnable w init [1,1,0,0,0]; P_x is the weighted gyro-midpoint of x's
walk-token bag. Pooling weights come from
four per-token scalars (recency, position, and geodesic distance and cosine alignment to the bag's
unweighted midpoint, both centre features with the per-bag level removed) via w·[rec, pos, spr, ang] plus a
zero-init MLP correction, w init (1, 1, 0, 0)."""
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
        """Hyperbolic radius (geodesic distance from the origin)."""
        return self.manifold.dist0(x)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted gyro-midpoint: x [Q,T,d], w [Q,T] -> [Q,d]."""
        return self.manifold.weighted_midpoint(x, weights=w, reducedim=[-2], dim=-1, keepdim=False)


class BagWeights(nn.Module):
    """logit = w·[rec, pos, spr, ang] + MLP([rec, pos, spr, ang]); w init (1, 1, 0, 0), MLP zero-init."""

    def __init__(self, mnia: float, hidden: int = 32):
        super().__init__()
        self.mnia = float(mnia)
        self.w = nn.Parameter(torch.tensor([1.0, 1.0, 0.0, 0.0]))
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

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
        rec = -torch.log1p(tokens.ages.clamp_min(0).float() / self.mnia)
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)
        spr, ang = self.centre_feats(geom, x.detach(), valid)
        feat = torch.stack([rec, pos, spr, ang], dim=-1).to(x.dtype)            # [Q, T, 4]
        logits = (feat * self.w).sum(dim=-1) + self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; BagWeights is the only other trained module."""

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

        self.pop_bias = nn.Embedding(self.num_nodes, 1)
        nn.init.zeros_(self.pop_bias.weight)
        # Learnable channel weights [geo, cos, geo_spread_v, cos_spread_v], no calibration; init 1,1,0,0
        # (the candidate cloud-spread channels are off at step 0).
        self.score_w = nn.Parameter(torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor):
        """Bag -> (P [Q,d], geo_spread [Q], cos_spread [Q], coherence [Q]): spreads are the pooling-weighted
        mean geodesic distance / cosine of the cloud tokens to their centroid P; coherence is the
        (w*r)-weighted mean off-diagonal cosine within the cloud (r = token hyperbolic radius)."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)                                               # all-padding walk -> use seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(self.geom, tokens, x, valid)                       # [Q, T] sums to 1, 0 on padding
        p = self.geom.midpoint(x, w)                                            # [Q, d]
        pe = p.unsqueeze(-2)                                                    # [Q, 1, d]
        geo_sp = (w * self.geom.dist(pe, x)).sum(dim=-1)                        # [Q] weighted mean geo to P
        cos_sp = (w * F.cosine_similarity(pe, x, dim=-1, eps=1e-6)).sum(dim=-1) # [Q] weighted mean cos to P
        r = self.geom.dist0(x) * valid.to(x.dtype)                             # [Q, T] token radii
        wr = w * r
        lo2 = (wr ** 2).sum(dim=-1)
        hi2 = wr.sum(dim=-1) ** 2
        rp = self.geom.dist0(p)
        coh = (rp ** 2 - lo2) / (hi2 - lo2).clamp_min(1e-9)                     # [Q]
        return p, geo_sp, cos_sp, coh

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        score = w . [geo, cos, geo_spread_v, cos_spread_v] + b_v."""
        emb = self.E.weight
        p_u, _, _, _ = self.pool(src_tokens, emb)                               # [B, d]
        p_v, sg_v, sc_v, coh_v = self.pool(cand_tokens, emb)                    # [B*C,d] + 3x [B*C]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        v_nodes = cand_tokens.seeds.view(b, c)                                  # [B, C] candidate node ids
        geo = -self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic
        cos = F.cosine_similarity(p_u.unsqueeze(1), p_v, dim=-1, eps=1e-6)      # [B, C] direction agreement
        w = self.score_w
        return (w[0] * geo + w[1] * cos
                + w[2] * sg_v.view(b, c) + w[3] * sc_v.view(b, c)               # candidate cloud spreads
                + w[4] * coh_v.view(b, c)                                       # candidate cloud coherence
                + self.pop_bias(v_nodes).squeeze(-1))
