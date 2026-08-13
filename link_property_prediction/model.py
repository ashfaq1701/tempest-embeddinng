"""Centroid-to-centroid head on the Poincaré ball. Each side's walk-token bag (seeds included) is pooled to a
single point and the score is the geodesic between them:
    s(u,v) = -d(P_u, P_v)
Pooling is the TANGENT MEAN at the origin, expmap0(sum_p w_p logmap0(x_p)) — not the gyro-midpoint, whose
conformal reweighting 2/(1-||x||^2) would fight the learned weights as E moves.

The weights are LEARNED from three raw per-token scalars — recency -log1p(age/mnia), position -(pos-1), and
the token's distance from its own bag's unweighted centre RELATIVE to the bag's mean such distance — with NO
frequency encoding. The logit is a linear base w·[rec, pos, spr] plus a zero-init MLP correction, and w is
init (1, 1, 0) so at step 0 the head is exactly the recency+position prior; training starts there and learns
a smooth correction. Three smooth low-dim features plus prior-init avoid the random-init pooler instability
that collapsed the encoded (cos/sin + hop-embedding) head. The relative form of the spread is deliberate: raw
distance-to-centre carries the bag's spread, which tracks node degree and would leak a popularity signal,
whereas dividing by the bag mean removes the per-bag level exactly and leaves only within-bag outlier
structure. The spread is detached (no second gradient path into E); a detach-vs-not A/B was a dead tie."""
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_NORM_EPS = 1e-5      # ||x||^2 clamp: stay strictly inside the ball
_ACOSH_EPS = 1e-7     # arcosh arg clamp: finite gradient at coincidence


class PoincareManifold:
    """Poincaré-ball geometry (c=1). `manifold` (geoopt ball) is kept for E's init + RiemannianAdam."""

    def __init__(self, c: float = 1.0):
        self.manifold = geoopt.PoincareBall(c=c)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Elementwise geodesic distance (geoopt), broadcasting over leading dims. LOWER = closer."""
        return self.manifold.dist(x, y)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Manifold -> tangent at the origin. One global chart, so all queries stay comparable."""
        return self.manifold.logmap0(x)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Tangent at the origin -> manifold. geoopt clamps ||out|| to 1-4e-3 (fp32); never hits the boundary."""
        return self.manifold.expmap0(v)

    @staticmethod
    def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2))) for x [...,n,d], y [...,m,d] -> [...,n,m]."""
        x2 = (x * x).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., n]
        y2 = (y * y).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., m]
        xy = torch.matmul(x, y.transpose(-1, -2))                               # [..., n, m]
        sq = (x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2.0 * xy).clamp_min(0.0)     # [..., n, m]
        denom = (1.0 - x2).unsqueeze(-1) * (1.0 - y2).unsqueeze(-2)             # [..., n, m]
        arg = (1.0 + 2.0 * sq / denom).clamp_min(1.0 + _ACOSH_EPS)
        return torch.acosh(arg)


class BagWeights(nn.Module):
    """Per-token softmax weights from three raw scalars — recency, position, distance-to-centre — with no
    frequency encoding. logit = w·[rec, pos, spr] + MLP([rec, pos, spr]); w is init (1, 1, 0) so at step 0
    the head is exactly the recency+position prior and the MLP contributes nothing. Training starts at that
    known-stable point and learns a smooth correction, avoiding the random-init pooler instability of the
    encoded head."""

    def __init__(self, mnia: float, hidden: int = 32):
        super().__init__()
        self.mnia = float(mnia)
        # (recency, position, spread) base coefficients; init reproduces the recency+position prior, spread off.
        self.w = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)    # nonlinear correction starts at 0 -> logit == prior at step 0
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def relative_spread(z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """||z_p - c|| / mean_q ||z_q - c||, c = unweighted centre over VALID slots only [Q, T]. Dividing by
        the bag mean removes the per-bag level (the degree-correlated spread) and keeps outlier structure."""
        vf = valid.unsqueeze(-1).to(z.dtype)                                    # [Q, T, 1]
        n = vf.sum(dim=-2).clamp_min(1.0)                                       # [Q, 1]
        c = (z * vf).sum(dim=-2, keepdim=True) / n.unsqueeze(-2)                # [Q, 1, d]
        dc = (z - c).norm(dim=-1) * valid.to(z.dtype)                           # [Q, T]
        return dc / (dc.sum(dim=-1, keepdim=True) / n).clamp_min(1e-6)          # [Q, T]

    def forward(self, tokens: WalkTokens, z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """tokens, z [Q,T,d] tangent coords, valid [Q,T] -> w [Q,T] summing to 1 (0 on padding)."""
        rec = -torch.log1p(tokens.ages.clamp_min(0).float() / self.mnia)        # [Q, T]  0 at seed (age 0)
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)                    # [Q, T]  0 at seed, linear
        spr = self.relative_spread(z.detach(), valid)                          # [Q, T]  detached
        feat = torch.stack([rec, pos, spr], dim=-1).to(z.dtype)                # [Q, T, 3]
        logits = (feat * self.w).sum(dim=-1) + self.net(feat).squeeze(-1)      # [Q, T]  base + correction
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)  # [Q, T]


class LinkPredHead(nn.Module):
    """Centroid-to-centroid head with tangent-mean pooling. E is a ManifoldParameter; BagWeights is the only
    other trained module."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float, scorer: str = "geodesic"):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.scorer = str(scorer)                                              # geodesic | cosine | geo_cos
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(mean_node_inter_arrival)

        # Angular-scoring extras (scorer in {cosine, geo_cos}). Decouple the relational signal (direction of
        # the pooled tangent) from the popularity prior (radius, which the geodesic confounds): score the
        # DIRECTION via cosine, and give popularity its own explicit per-node scalar b_v (init 0). cos is
        # bounded [-1,1]; the learned scale lifts it to a usable logit range (init ~10; random directions in
        # high-d start with cos std ~1/sqrt(d), so a scale is needed for early signal).
        self.pop_bias = nn.Embedding(self.num_nodes, 1)
        nn.init.zeros_(self.pop_bias.weight)
        self.log_cos_scale = nn.Parameter(torch.log(torch.tensor(10.0)))

        # Spread init: geoopt random (std=1), not the near-origin wrapped normal. ManifoldParameter so
        # RiemannianAdam keeps E in the ball.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> pooled TANGENT vector t [Q, d] (the origin-tangent weighted mean; expmap0(t) is the manifold
        point, ||t|| its hyperbolic radius, t/||t|| its direction). Cold-bag guard falls back to the seed so an
        all -inf softmax row can't go NaN."""
        nodes = tokens.nodes.clamp_min(0).clone()                               # [Q, T] padding(-1) -> 0
        valid = tokens.mask.clone()                                             # [Q, T] real slots (seed incl.)
        cold = ~valid.any(dim=-1)
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        z = self.geom.logmap0(F.embedding(nodes, emb))                          # [Q, T, d]
        w = self.bag_weights(tokens, z, valid)                                  # [Q, T]
        return (w.unsqueeze(-1) * z).sum(dim=-2)                                # [Q, d] pooled tangent

    def score(self, t_u: torch.Tensor, t_v: torch.Tensor, v_nodes: torch.Tensor) -> torch.Tensor:
        """t_u [B, d], t_v [B, C, d] pooled tangents, v_nodes [B, C] candidate node ids -> logits [B, C].
        geodesic: -d_H(P_u, P_v)  (baseline, radius-confounded).
        cosine:    scale*cos(t_u, t_v) + b_v   (direction only + explicit popularity; radius removed).
        geo_cos:   -d_H + scale*cos + b_v       (keep geodesic, add explicit cosine; b_v frees radius)."""
        s = t_u.new_zeros(t_v.shape[:-1])                                       # [B, C]
        if self.scorer in ("geodesic", "geo_cos"):
            p_u = self.geom.expmap0(t_u)
            p_v = self.geom.expmap0(t_v)
            s = s - self.geom.dist(p_u.unsqueeze(1), p_v)
        if self.scorer in ("cosine", "geo_cos"):
            cos = F.cosine_similarity(t_u.unsqueeze(1), t_v, dim=-1, eps=1e-6)  # [B, C]
            s = s + torch.exp(self.log_cos_scale) * cos + self.pop_bias(v_nodes).squeeze(-1)
        return s

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries (seeds u); cand = B*C candidate queries (seeds v), query-major. -> [B, C]."""
        emb = self.E.weight
        t_u = self.pool(src_tokens, emb)                                        # [B, d] tangent
        t_v = self.pool(cand_tokens, emb)                                       # [B*C, d] tangent
        b, d = t_u.shape
        c = t_v.shape[0] // b
        t_v = t_v.view(b, c, d)                                                 # [B, C, d]
        v_nodes = cand_tokens.seeds.view(b, c)                                  # [B, C] candidate node ids
        return self.score(t_u, t_v, v_nodes)                                    # [B, C] higher = closer
