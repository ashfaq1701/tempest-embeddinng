"""Monotone weighted-mean metric head on the Poincaré ball. E is the only trained tensor; the score is a
parameter-free distance aggregate  s(u,v) = -[ d(E_u,E_v) + mean(E_v,B_u) + mean(E_u,B_v) ]  where B_x is
x's walk-token bag. The weighted mean is a convex combination, so ds/dd_p <= 0 and the link CE trains E
end-to-end with no detach."""
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
    """Two-sided monotone weighted-mean head. Owns E (ManifoldParameter, trained by the link CE); no other
    parameter. Symmetric across the two directions (v vs B_u, u vs B_v) since the task is undirected."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float = 1.0):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        # data_stats mean-field per-node inter-event time (characteristic age scale). Stored for a
        # future recency-temperature init; NOT used in the score yet.
        self.mean_node_inter_arrival = float(mean_node_inter_arrival)
        self.geom = PoincareManifold()

        # Spread init: geoopt random (std=1), not the near-origin wrapped normal. ManifoldParameter so
        # RiemannianAdam keeps E in the ball.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    @staticmethod
    def bag_weight_logits(tokens: WalkTokens) -> torch.Tensor:
        """Recency/hop prior LOGITS [Q, T] = -(log1p(age) + log1p(hop-1)); 0 (max) for a just-happened
        adjacent token, decaying with age and hop."""
        age = tokens.ages.clamp_min(0).to(torch.float32)                        # [Q, T]  seed=0, ctx>=1
        hop = tokens.positions.clamp_min(1).to(torch.float32)                   # [Q, T]  seed=1, ctx>=2
        return -(torch.log1p(age) + torch.log1p(hop - 1.0))                     # [Q, T]  <= 0

    @staticmethod
    def bag_weights(tokens: WalkTokens, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """(nodes [Q,T], log_w [Q,T]): softmax the recency/hop prior over context slots (mask & ~seed_node),
        -inf elsewhere. A cold bag (no context) falls back to {(seed, w=1)} -> mean = identity distance."""
        nodes = tokens.nodes.clamp_min(0).clone()                               # [Q, T] padding(-1) -> 0
        valid = (tokens.mask & ~tokens.seed_node_mask).clone()                  # [Q, T] context slots

        cold = ~valid.any(dim=-1)                                               # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        logits = LinkPredHead.bag_weight_logits(tokens).to(dtype)               # [Q, T] <= 0
        log_w = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
        return nodes, log_w

    @staticmethod
    def bag_mean(d: torch.Tensor, log_w: torch.Tensor) -> torch.Tensor:
        """Weighted mean sum_p exp(log_w_p) * d_p over the last axis (excluded slots have weight 0)."""
        return (log_w.exp() * d).sum(dim=-1)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries (seeds u); cand = B*C candidate queries (seeds v), query-major. -> [B, C]."""
        emb = self.E.weight

        nodes_u, logw_u = self.bag_weights(src_tokens, emb.dtype)               # [B, T]
        nodes_v, logw_v = self.bag_weights(cand_tokens, emb.dtype)              # [B*C, T]

        e_u = F.embedding(src_tokens.seeds, emb)                                # [B, d]
        x_u = F.embedding(nodes_u, emb)                                         # [B, T, d]
        e_v = F.embedding(cand_tokens.seeds, emb)                               # [B*C, d]
        x_v = F.embedding(nodes_v, emb)                                         # [B*C, T, d]

        b, d = e_u.shape
        c = e_v.shape[0] // b
        t = nodes_u.shape[1]
        e_v = e_v.view(b, c, d)                                                 # [B, C, d]
        x_v = x_v.view(b, c, t, d)                                              # [B, C, T, d]
        logw_v = logw_v.view(b, c, t)                                           # [B, C, T]

        d_id = self.geom.pairwise_dist(e_u.unsqueeze(-2), e_v).squeeze(-2)      # [B, C]  identity: d(E_u,E_v)

        d_v_bu = self.geom.pairwise_dist(e_v, x_u)                              # [B, C, T]  v vs u's bag
        mean_v_bu = self.bag_mean(d_v_bu, logw_u.unsqueeze(-2))                 # [B, C]

        d_u_bv = self.geom.pairwise_dist(e_u[:, None, None, :], x_v).squeeze(-2)  # [B, C, T]  u vs v's bag
        mean_u_bv = self.bag_mean(d_u_bv, logw_v)                               # [B, C]

        raw = d_id + mean_v_bu + mean_u_bv                                      # [B, C]
        return -raw                                                             # higher = closer = better
