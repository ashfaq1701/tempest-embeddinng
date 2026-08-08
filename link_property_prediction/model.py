"""Two-sided centroid-vs-token head on the Poincaré ball. E is the only trained tensor. The seed is handled
by an explicit identity term; the centroid + cross terms operate on the NON-seed neighbourhood:
    s(u,v) = -[ d(E_u,E_v) + sum_q w_q^v * d(P_u, x_q^v) + sum_p w_p^u * d(P_v, x_p^u) ]
P_x = weighted gyro-midpoint of x's non-seed bag (seed excluded via seed_mask); x_p/x_q = raw non-seed token
embeddings; w = softmax of the -(log1p(age)+log1p(hop-1)) prior over non-seed slots. No centroid-centroid term."""
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

    @staticmethod
    def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2))) for x [...,n,d], y [...,m,d] -> [...,n,m].
        ||x-y||^2 expanded as ||x||^2+||y||^2-2<x,y> so the cross term is one matmul (no [...,n,m,d] diff)."""
        x2 = (x * x).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., n]
        y2 = (y * y).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., m]
        xy = torch.matmul(x, y.transpose(-1, -2))                               # [..., n, m]
        sq = (x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2.0 * xy).clamp_min(0.0)     # [..., n, m]
        denom = (1.0 - x2).unsqueeze(-1) * (1.0 - y2).unsqueeze(-2)             # [..., n, m]
        arg = (1.0 + 2.0 * sq / denom).clamp_min(1.0 + _ACOSH_EPS)
        return torch.acosh(arg)


class LinkPredHead(nn.Module):
    """Two-sided centroid-vs-token head. Owns E (ManifoldParameter, trained by the link CE); no other
    parameter."""

    def __init__(self, num_nodes: int, d_emb: int):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()

        # Spread init: geoopt random (std=1), not the near-origin wrapped normal. ManifoldParameter so
        # RiemannianAdam keeps E in the ball.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    @staticmethod
    def bag_weight_logits(tokens: WalkTokens) -> torch.Tensor:
        """Recency/hop prior LOGITS [Q, T] = -(log1p(age) + log1p(hop-1)); 0 (max) for the seed (age 0,
        hop 1), decaying with age and hop."""
        age = tokens.ages.clamp_min(0).to(torch.float32)                        # [Q, T]  seed=0, ctx>=1
        hop = tokens.positions.clamp_min(1).to(torch.float32)                   # [Q, T]  seed=1, ctx>=2
        return -(torch.log1p(age) + torch.log1p(hop - 1.0))                     # [Q, T]  <= 0

    @staticmethod
    def bag_weights(tokens: WalkTokens, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """(nodes [Q,T], w [Q,T]): softmax the recency/hop prior over NON-seed context slots
        (mask & ~seed_mask), 0 on the seed and padding, sums to 1 per row. Cold-bag guard (no context, e.g.
        a seed-only walk) falls back to the seed; without it that row's all -inf softmax would be NaN."""
        nodes = tokens.nodes.clamp_min(0).clone()                               # [Q, T] padding(-1) -> 0
        valid = (tokens.mask & ~tokens.seed_mask).clone()                       # [Q, T] non-seed context slots

        cold = ~valid.any(dim=-1)                                               # [Q]  fully-empty walk guard
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        logits = LinkPredHead.bag_weight_logits(tokens).to(dtype)               # [Q, T] <= 0
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)    # [Q, T] sums to 1
        return nodes, w

    def bag_centroid(self, nodes: torch.Tensor, w: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """P_x = weighted gyro-midpoint of the bag's token embeddings (weights w sum to 1)."""
        x = F.embedding(nodes, emb)                                            # [Q, T, d]
        return self.geom.manifold.weighted_midpoint(
            x, weights=w, reducedim=[-2], dim=-1, keepdim=False)               # [Q, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries (seeds u); cand = B*C candidate queries (seeds v), query-major. -> [B, C]."""
        emb = self.E.weight

        nodes_u, w_u = self.bag_weights(src_tokens, emb.dtype)                 # [B, T]
        nodes_v, w_v = self.bag_weights(cand_tokens, emb.dtype)               # [B*C, T]

        x_u = F.embedding(nodes_u, emb)                                        # [B, T, d]
        x_v = F.embedding(nodes_v, emb)                                        # [B*C, T, d]
        p_u = self.bag_centroid(nodes_u, w_u, emb)                            # [B, d]  non-seed centroid
        p_v = self.bag_centroid(nodes_v, w_v, emb)                            # [B*C, d]
        e_u = F.embedding(src_tokens.seeds, emb)                               # [B, d]   E[u]
        e_v = F.embedding(cand_tokens.seeds, emb)                              # [B*C, d] E[v]

        b, d = p_u.shape
        c = p_v.shape[0] // b

        p_v = p_v.view(b, c, d)                                                # [B, C, d]
        x_v = x_v.view(b, c, x_u.shape[1], d)                                  # [B, C, T, d]
        w_v = w_v.view(b, c, x_u.shape[1])                                     # [B, C, T]
        e_v = e_v.view(b, c, d)                                                # [B, C, d]
        e_u = e_u.unsqueeze(1).expand(b, c, d)                                 # [B, C, d]

        # Identity (seed vs seed) + P_u vs v's non-seed tokens + P_v vs u's non-seed tokens.
        identity = self.geom.dist(e_u, e_v)                                    # [B, C]  d(E_u, E_v)
        d_pu_xv = self.geom.pairwise_dist(p_u[:, None, None, :], x_v).squeeze(-2)  # [B, C, T]
        term_v = (w_v * d_pu_xv).sum(-1)                                       # [B, C]
        d_pv_xu = self.geom.pairwise_dist(p_v, x_u)                            # [B, C, T]
        term_u = (w_u.unsqueeze(1) * d_pv_xu).sum(-1)                          # [B, C]

        raw = identity + term_v + term_u                                       # [B, C]
        return -raw                                                           # [B, C] higher = closer
