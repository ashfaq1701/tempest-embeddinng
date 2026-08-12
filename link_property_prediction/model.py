"""Centroid-to-centroid head on the Poincaré ball. E is the only trained tensor. Each side's walk-token bag
is pooled to a single gyro-midpoint and the score is the geodesic between them:
    s(u,v) = -d(P_u, P_v)
P_x = weighted gyro-midpoint of x's full bag (seeds included); w = softmax of the -(log1p(age)+log1p(hop-1))
prior. No per-token terms."""
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

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float = 1.0):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        # data_stats mean-field per-node inter-event time (T_train*N/2E) — the characteristic AGE scale.
        # The pooling recency weight normalises age by it: log1p(age / mean_node_inter_arrival), which makes
        # the softmax scale-invariant across datasets (review ~1e7 vs wiki ~1e4) and stops the huge age range
        # from collapsing the gyro-midpoint onto E[seed]. A single dataset constant (not per-node, not
        # per-bag), so between-bag differences are preserved.
        self.mean_node_inter_arrival = float(mean_node_inter_arrival)
        self.geom = PoincareManifold()

        # Spread init: geoopt random (std=1), not the near-origin wrapped normal. ManifoldParameter so
        # RiemannianAdam keeps E in the ball.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    def bag_weight_logits(self, tokens: WalkTokens) -> torch.Tensor:
        """Recency/hop prior LOGITS [Q, T] = -(log1p(age / mean_node_inter_arrival) + log1p(hop-1)); 0 (max)
        for the seed (age 0, hop 1). Age is normalised by the dataset's mean-field per-node inter-event time
        (the characteristic age scale), so the softmax is scale-invariant: measured in node-timescale units
        the seed-vs-context gap is bounded (~log of a small ratio) on any dataset, which stops the huge raw
        age range from collapsing the gyro-midpoint onto E[seed]. One dataset constant → between-bag structure
        is preserved (a staler bag keeps larger age/scale)."""
        age = tokens.ages.clamp_min(0).to(torch.float32)                        # [Q, T]  seed=0, ctx>=1
        hop = tokens.positions.clamp_min(1).to(torch.float32)                   # [Q, T]  seed=1, ctx>=2
        return -(torch.log1p(age / self.mean_node_inter_arrival) + torch.log1p(hop - 1.0))  # [Q, T]  <= 0

    def bag_weights(self, tokens: WalkTokens, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """(nodes [Q,T], w [Q,T]): softmax the recency/hop prior over ALL real slots (seed included), 0 on
        padding, sums to 1 per row. Cold-bag guard handles a fully-empty walk (all padding) -> falls back to
        the seed; without it that row's all -inf softmax would be NaN."""
        nodes = tokens.nodes.clamp_min(0).clone()                               # [Q, T] padding(-1) -> 0
        valid = tokens.mask.clone()                                             # [Q, T] real slots (seed incl.)

        cold = ~valid.any(dim=-1)                                               # [Q]  fully-empty walk guard
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        logits = self.bag_weight_logits(tokens).to(dtype)                       # [Q, T] <= 0
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

        p_u = self.bag_centroid(nodes_u, w_u, emb)                            # [B, d]
        p_v = self.bag_centroid(nodes_v, w_v, emb)                            # [B*C, d]

        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                # [B, C, d]

        # Centroid-to-centroid geodesic: s(u,v) = -d(P_u, P_v).
        raw = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C]
        return -raw                                                           # [B, C] higher = closer
