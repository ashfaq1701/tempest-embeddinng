"""Cross-bag head on the Poincaré ball. E is the only trained tensor. The cross term is the weighted
all-pairs geodesic between the two walk-token bags:
    X(u,v) = sum_p sum_q w_p^u * w_q^v * d(x_p^u, x_q^v)
The product weights form a joint distribution over the TxT grid (both marginals sum to 1), so X is a convex
combination of geodesics: dX/dd_pq <= 0, and the (seed,seed) cell carries d(E_u,E_v) at weight
w_seed^u * w_seed^v. `cross_only=False` adds back the two centroid-probe terms A and B for the ablation:
    A = sum_q w_q^v * d(P_u, x_q^v),  B = sum_p w_p^u * d(P_v, x_p^u)
With cross_only=True no gyro-midpoint is computed at all."""
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
    """Cross-bag head. Owns E (ManifoldParameter, trained by the link CE); no other parameter.
    cross_only=True  -> s = -X                 (no centroid anywhere)
    cross_only=False -> s = -(X + A + B)       (centroid-probe terms added back)"""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float = 1.0,
                 cross_only: bool = True):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        # data_stats mean-field per-node inter-event time (T_train*N/2E) — the characteristic AGE scale.
        # The pooling recency weight normalises age by it: log1p(age / mean_node_inter_arrival), which makes
        # the softmax scale-invariant across datasets (review ~1e7 vs wiki ~1e4).
        self.mean_node_inter_arrival = float(mean_node_inter_arrival)
        self.cross_only = bool(cross_only)
        self.geom = PoincareManifold()

        # Spread init: geoopt random (std=1), not the near-origin wrapped normal. ManifoldParameter so
        # RiemannianAdam keeps E in the ball.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    def bag_weight_logits(self, tokens: WalkTokens) -> torch.Tensor:
        """Recency/hop prior LOGITS [Q, T] = -(log1p(age / mean_node_inter_arrival) + log1p(hop-1)); 0 (max)
        for the seed (age 0, hop 1)."""
        age = tokens.ages.clamp_min(0).to(torch.float32)                        # [Q, T]  seed=0, ctx>=1
        hop = tokens.positions.clamp_min(1).to(torch.float32)                   # [Q, T]  seed=1, ctx>=2
        return -(torch.log1p(age / self.mean_node_inter_arrival) + torch.log1p(hop - 1.0))  # [Q, T]  <= 0

    def bag_weights(self, tokens: WalkTokens, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """(nodes [Q,T], w [Q,T]): softmax the recency/hop prior over ALL real slots (seed included), 0 on
        padding, sums to 1 per row. Cold-bag guard handles a fully-empty walk -> falls back to the seed."""
        nodes = tokens.nodes.clamp_min(0).clone()                               # [Q, T] padding(-1) -> 0
        valid = tokens.mask.clone()                                             # [Q, T] real slots (seed incl.)

        cold = ~valid.any(dim=-1)                                               # [Q]  fully-empty walk guard
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        logits = self.bag_weight_logits(tokens).to(dtype)                       # [Q, T] <= 0
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)    # [Q, T] sums to 1
        return nodes, w

    def bag_centroid(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """P_x = weighted gyro-midpoint of the bag's token embeddings (weights w sum to 1). Only called when
        cross_only=False."""
        return self.geom.manifold.weighted_midpoint(
            x, weights=w, reducedim=[-2], dim=-1, keepdim=False)               # [Q, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries (seeds u); cand = B*C candidate queries (seeds v), query-major. -> [B, C]."""
        emb = self.E.weight

        nodes_u, w_u = self.bag_weights(src_tokens, emb.dtype)                 # [B, T]
        nodes_v, w_v = self.bag_weights(cand_tokens, emb.dtype)                # [B*C, T]

        x_u = F.embedding(nodes_u, emb)                                        # [B, T, d]
        x_v = F.embedding(nodes_v, emb)                                        # [B*C, T, d]

        b, t = nodes_u.shape
        d = emb.shape[1]
        c = nodes_v.shape[0] // b

        x_v = x_v.view(b, c, t, d)                                             # [B, C, T, d]
        w_v = w_v.view(b, c, t)                                                # [B, C, T]

        # X: all token pairs between the two bags, weighted by w_p^u * w_q^v.
        d_cross = self.geom.pairwise_dist(x_u.unsqueeze(1), x_v)               # [B, C, T, T]
        raw = torch.einsum('bp,bcq,bcpq->bc', w_u, w_v, d_cross)               # [B, C]

        if not self.cross_only:
            p_u = self.bag_centroid(x_u, w_u)                                  # [B, d]
            p_v = self.bag_centroid(x_v.view(b * c, t, d), w_v.view(b * c, t)).view(b, c, d)
            d_pu_xv = self.geom.pairwise_dist(p_u[:, None, None, :], x_v).squeeze(-2)  # [B, C, T]
            raw = raw + (w_v * d_pu_xv).sum(-1)                                # + A
            d_pv_xu = self.geom.pairwise_dist(p_v, x_u)                        # [B, C, T]
            raw = raw + (w_u.unsqueeze(1) * d_pv_xu).sum(-1)                   # + B

        return -raw                                                            # [B, C] higher = closer
