"""Centroid-to-centroid head on the Poincaré ball:  s(u,v) = -d(P_u, P_v) + scale*cos(P_u, P_v) + b_v,
where P_x is the weighted gyro-midpoint of x's walk-token bag (seeds included). The geodesic term is
radius-confounded (radius encodes a u-independent popularity prior); the explicit angular channel
cos(P_u, P_v) carries the relation and the per-node bias b_v carries popularity, so radius is freed and
E stops expanding. (Diagnosis + A/B: see the diag/radial-confound and experiment/angular-scoring branches.)

Pooling weights are learned from three per-token scalars: recency, position, and the token's geodesic
distance from its bag's unweighted midpoint relative to the bag's mean such distance. The relative form is
deliberate — raw distance-to-centre carries bag spread, which tracks node degree and would leak a popularity
signal. The logit is a linear base w·[rec, pos, spr] plus a zero-init MLP correction, with w init (1, 1, 0),
so step 0 is exactly the recency+position prior; a random-init pooler collapsed at ep2 where this does not."""
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_NORM_EPS = 1e-5
_ACOSH_EPS = 1e-7


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

    @staticmethod
    def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x [...,n,d], y [...,m,d] -> [...,n,m]. Not used by forward; kept for external probes."""
        x2 = (x * x).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)
        y2 = (y * y).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)
        xy = torch.matmul(x, y.transpose(-1, -2))
        sq = (x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2.0 * xy).clamp_min(0.0)
        denom = (1.0 - x2).unsqueeze(-1) * (1.0 - y2).unsqueeze(-2)
        return torch.acosh((1.0 + 2.0 * sq / denom).clamp_min(1.0 + _ACOSH_EPS))


class BagWeights(nn.Module):
    """logit = w·[rec, pos, spr] + MLP([rec, pos, spr]); w init (1, 1, 0) and the MLP zero-init, so step 0
    reproduces the recency+position prior exactly."""

    def __init__(self, mnia: float, hidden: int = 32):
        super().__init__()
        self.mnia = float(mnia)
        self.w = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def relative_spread(geom: PoincareManifold, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """d(x_p, C) / mean_q d(x_q, C) over valid slots, C = unweighted midpoint -> [Q, T]."""
        vf = valid.to(x.dtype)
        n = vf.sum(dim=-1, keepdim=True).clamp_min(1.0)
        c = geom.midpoint(x, vf / n)
        dc = geom.dist(c.unsqueeze(-2), x) * vf
        return dc / (dc.sum(dim=-1, keepdim=True) / n).clamp_min(1e-6)

    def forward(self, geom: PoincareManifold, tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        rec = -torch.log1p(tokens.ages.clamp_min(0).float() / self.mnia)
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)
        spr = self.relative_spread(geom, x.detach(), valid)
        feat = torch.stack([rec, pos, spr], dim=-1).to(x.dtype)                 # [Q, T, 3]
        logits = (feat * self.w).sum(dim=-1) + self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; BagWeights is the only other trained module."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(mean_node_inter_arrival)

        # geo_cos scoring extras: an explicit per-node popularity bias b_v (init 0) and a learned cosine
        # scale (init ~10; ball-point directions in high-d start with cos std ~1/sqrt(d)). Decouple the
        # popularity prior (which the geodesic smuggles through radius) into b_v, freeing radius, and add an
        # explicit angular channel cos(P_u, P_v) on top of the geodesic.
        self.pop_bias = nn.Embedding(self.num_nodes, 1)
        nn.init.zeros_(self.pop_bias.weight)
        self.log_cos_scale = nn.Parameter(torch.log(torch.tensor(10.0)))

        # spread init (geoopt random, std=1), not the near-origin wrapped normal
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> one manifold point [Q, d]."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)                                               # all-padding walk -> use seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(self.geom, tokens, x, valid)                       # [Q, T]
        return self.geom.midpoint(x, w)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        geo_cos score: -d_H(P_u, P_v) + scale*cos(P_u, P_v) + b_v."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        v_nodes = cand_tokens.seeds.view(b, c)                                  # [B, C] candidate node ids
        geo = -self.geom.dist(p_u.unsqueeze(1), p_v)                           # [B, C] geodesic
        cos = F.cosine_similarity(p_u.unsqueeze(1), p_v, dim=-1, eps=1e-6)      # [B, C] direction agreement
        return geo + torch.exp(self.log_cos_scale) * cos + self.pop_bias(v_nodes).squeeze(-1)
