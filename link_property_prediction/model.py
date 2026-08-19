"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = w . f, f = [geo, cos, geo_spread_v,
cos_spread_v, log1p(deg_v)] (geo=-d(P_u,P_v), cos=cos(P_u,P_v); the spreads are the candidate cloud's
pooling-weighted mean geo/cos to its centroid; deg_v is the candidate's interaction count strictly before
the query time), learnable w init [1,1,0,0,0]; P_x is the weighted gyro-midpoint of x's
walk-token bag. Pooling weights come from a small MLP (hidden = 8*N_FEAT, random init) over four per-token
scalars: -(age/mnia), -pos, and the geodesic distance / cosine alignment to the bag's unweighted midpoint
(the last two are centre features with the per-bag level removed). mnia (mean node inter-arrival) is a fixed
age scale. On Yelp this matched the previous linear-prior + zero-init-MLP-correction pooling."""
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

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted gyro-midpoint: x [Q,T,d], w [Q,T] -> [Q,d]."""
        return self.manifold.weighted_midpoint(x, weights=w, reducedim=[-2], dim=-1, keepdim=False)


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([-(age/mnia), -pos, spr, ang])); hidden = 8*N_FEAT, random init,
    fixed age scale mnia (no learnable tau, no linear residual). The softmax over tokens dampens the
    random init, so unlike a score-level MLP the prior/residual are not needed here."""

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

        # Channel weights [geo, cos, geo_spread_v, cos_spread_v, log1p(deg_v)]; init 1,1,0,0,0
        # (the spread and degree channels start off and learn their weights).
        self.score_w = nn.Parameter(torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bag -> (P [Q,d], geo_spread [Q], cos_spread [Q]): the spreads are the pooling-weighted mean
        geodesic distance / cosine of the cloud tokens to their centroid P."""
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
        return p, geo_sp, cos_sp

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens,
                cand_deg: torch.Tensor) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        score = w . [geo, cos, geo_spread_v, cos_spread_v, log1p(deg_v)]. cand_deg [B, C] = candidate's
        interaction count strictly before its query time."""
        emb = self.E.weight
        p_u, _, _ = self.pool(src_tokens, emb)                                  # [B, d]
        p_v, sg_v, sc_v = self.pool(cand_tokens, emb)                           # [B*C,d] + 2x [B*C]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = -self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic
        cos = F.cosine_similarity(p_u.unsqueeze(1), p_v, dim=-1, eps=1e-6)      # [B, C] direction agreement
        w = self.score_w
        return (w[0] * geo + w[1] * cos
                + w[2] * sg_v.view(b, c) + w[3] * sc_v.view(b, c)               # candidate cloud spreads
                + w[4] * torch.log1p(cand_deg))                                # online candidate degree (pop)
