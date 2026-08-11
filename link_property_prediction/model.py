"""Monotone weighted-mean metric head on the Poincaré ball. E is the only trained tensor; the score is a
parameter-free distance aggregate  s(u,v) = -[ mean(E_v, B_u) + mean(E_u, B_v) ]  where B_x is x's walk-token
bag INCLUDING the seed. Because the seed is in the bag (and carries the top recency weight), each mean already
contains the identity distance d(E_u,E_v) via the seed slot — so there is no separate identity term. The
weighted mean is a convex combination, so ds/dd_p <= 0 and the link CE trains E end-to-end with no detach.

Contrast with the centroid head (feature/centroid-token-cross): that head aggregates the bag to a single
gyro-midpoint P_x and scores d(E_v, P_u) — pulling the other side toward the neighbourhood CENTROID. This head
keeps mean-OF-distances (not distance-to-mean); with the seed dominating the weights it pulls the other side
toward the SEED node. Same bag + same recency weights; only the aggregation differs."""
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_NORM_EPS = 1e-5      # ||x||^2 clamp: stay strictly inside the ball
_ACOSH_EPS = 1e-7     # arcosh arg clamp: finite gradient at coincidence


class BagWeights(nn.Module):
    """Learned pooling weights over a walk-token bag. Two scalars: tau, the characteristic age scale inside
    the log (init at the dataset's mean-field per-node inter-event time), and the hop decay rate, forced
    negative via -exp(.). At init this reproduces the fixed -(log1p(age/mnia) + log1p(hop-1)) prior exactly."""

    def __init__(self, mnia: float):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(mnia))))   # tau = mnia at init
        self.log_c_hop = nn.Parameter(torch.zeros(()))                      # c_hop = -exp(.) = -1 at init

    def logits(self, tokens: WalkTokens) -> torch.Tensor:
        """[Q, T] pooling logit; softmax'd in forward. Higher = more weight. Seed (age 0, hop 1) is always 0,
        the max, so tau sets how far the context falls below it."""
        age = tokens.ages.clamp_min(0).float()                              # [Q, T]  seed=0
        hop = tokens.positions.clamp_min(1).float()                         # [Q, T]  seed=1
        a = torch.log1p(age / self.log_tau.exp())                           # [Q, T]
        h = torch.log1p(hop - 1.0)                                          # [Q, T]
        return -(a + self.log_c_hop.exp() * h)                              # [Q, T]

    def forward(self, tokens: WalkTokens, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """(nodes [Q,T], w [Q,T]): softmax over real slots (seed included), 0 on padding, sums to 1 per row.
        Cold-bag guard handles a fully-empty walk -> falls back to the seed (all -inf would be NaN)."""
        nodes = tokens.nodes.clamp_min(0).clone()                           # [Q, T] padding(-1) -> 0
        valid = tokens.mask.clone()                                         # [Q, T] real slots (seed incl.)

        cold = ~valid.any(dim=-1)                                           # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        z = self.logits(tokens).to(dtype)                                   # [Q, T]
        w = torch.softmax(z.masked_fill(~valid, float("-inf")), dim=-1)     # [Q, T] sums to 1
        return nodes, w

    @property
    def tau(self) -> float:
        return float(self.log_tau.exp())

    @property
    def c_hop(self) -> float:
        return -float(self.log_c_hop.exp())


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
    """Two-sided monotone weighted-mean head. Owns E (ManifoldParameter) plus the small BagWeights pooling
    parameters (log_tau, log_c_hop); all trained by the link CE. Symmetric across the two directions
    (v vs B_u, u vs B_v) since the task is undirected. The bag INCLUDES the seed, so the identity distance
    is folded into each mean (no separate term)."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        # Scale-normalise the age term of the pooling recency weight by the dataset's mean-field per-node
        # inter-event time (T_train*N/2E, the characteristic age scale). Makes the pooling softmax
        # timestamp-scale-neutral across datasets (log(c*age)=log(age)+log(c) is a softmax-invariant shift).
        # Since the seed (age 0) is now IN the bag, mnia also stops the seed from collapsing the weights onto
        # itself (seed_w ~0.5 rather than ~1), keeping the neighbour tokens alive in the mean.
        self.mean_node_inter_arrival = float(mean_node_inter_arrival)
        self.geom = PoincareManifold()
        # Learned recency/hop pooling (init = the fixed -(log1p(age/mnia)+log1p(hop-1)) prior).
        self.bag_weights = BagWeights(mnia=self.mean_node_inter_arrival)

        # Spread init: geoopt random (std=1), not the near-origin wrapped normal. ManifoldParameter so
        # RiemannianAdam keeps E in the ball.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    @staticmethod
    def bag_mean(d: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted mean sum_p w_p * d_p over the last axis (w sums to 1; padding weight 0)."""
        return (w * d).sum(dim=-1)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries (seeds u); cand = B*C candidate queries (seeds v), query-major. -> [B, C]."""
        emb = self.E.weight

        nodes_u, w_u = self.bag_weights(src_tokens, emb.dtype)                  # [B, T]  (seed incl.)
        nodes_v, w_v = self.bag_weights(cand_tokens, emb.dtype)                 # [B*C, T]

        e_u = F.embedding(src_tokens.seeds, emb)                                # [B, d]
        x_u = F.embedding(nodes_u, emb)                                         # [B, T, d]
        e_v = F.embedding(cand_tokens.seeds, emb)                               # [B*C, d]
        x_v = F.embedding(nodes_v, emb)                                         # [B*C, T, d]

        b, d = e_u.shape
        c = e_v.shape[0] // b
        t = nodes_u.shape[1]
        e_v = e_v.view(b, c, d)                                                 # [B, C, d]
        x_v = x_v.view(b, c, t, d)                                              # [B, C, T, d]
        w_v = w_v.view(b, c, t)                                                 # [B, C, T]

        # TWO terms only. The seed is in each bag, so d(E_u,E_v) is already inside both means (via the seed
        # slot, weight ~0.5) — no separate identity term. mean-of-distances (not distance-to-centroid) pulls
        # the other side toward the seed node.
        d_v_bu = self.geom.pairwise_dist(e_v, x_u)                              # [B, C, T]  v vs u's bag
        mean_v_bu = self.bag_mean(d_v_bu, w_u.unsqueeze(-2))                    # [B, C]

        d_u_bv = self.geom.pairwise_dist(e_u[:, None, None, :], x_v).squeeze(-2)  # [B, C, T]  u vs v's bag
        mean_u_bv = self.bag_mean(d_u_bv, w_v)                                  # [B, C]

        raw = mean_v_bu + mean_u_bv                                            # [B, C]  two terms
        return -raw                                                             # higher = closer = better
