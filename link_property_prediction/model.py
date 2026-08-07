"""Link head — Poincaré-ball node embeddings + a MONOTONE weighted-mean metric score over walk-token bags.

There is no scorer MLP, no centroid / P[·] projection, no exp map, and no learnable head parameter at all —
E is the only trained tensor. Per (u, candidate v) the logit is the NEGATIVE of a sum of geodesic-distance
channels:

    s(u, v) = -[ d(E_u, E_v)                        identity (direct pair distance)
               + mean_p d(E_v, node_p in B_u)       v vs u's neighbour bag
               + mean_p d(E_u, node_p in B_v) ]     u vs v's neighbour bag

where B_x is x's walk-token bag as a weighted set {(node_p, w_p)} with the fixed recency/hop prior
w = softmax(-(log1p(age) + log1p(hop-1))), and mean is the WEIGHTED average sum_p w_p d_p.

MONOTONICITY is the load-bearing property. The weighted mean is a convex combination of the distances with
weights >= 0 that do NOT depend on the distances, so ds/dd_p <= 0 EVERYWHERE — moving any neighbour closer
can only raise the score. The pull direction into E is therefore fixed by construction rather than learned,
which is what lets the link loss train E end-to-end with no detach and no free-riding readout (there is
nothing the head can reshape instead of moving E).

BAG MASKING: every slot whose node == the bag's own seed is dropped before the mean (the seed carries age
0 / hop 1 => maximal prior weight, and d(E_v, E_u) for u in B_u merely duplicates the identity channel).
A cold bag (no context slot survives) is the single atom {(seed, w=1)}, so the mean = d(anchor, E_seed)
exactly and the whole score degrades to the identity distance — branch-free, never log(0).
"""
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

# Numerical floors for the closed-form Poincaré distance.
_NORM_EPS = 1e-5      # ||x||^2 clamped to <= 1 - _NORM_EPS   (stay strictly inside the ball)
_ACOSH_EPS = 1e-7     # arcosh argument clamped to >= 1 + _ACOSH_EPS (finite gradient at coincidence)


class PoincareManifold:
    """Poincaré-ball geometry (c = 1). Exposes `pairwise_dist(X, Y)` — the ONLY geometric primitive the
    head needs (no centroid / exp-map projection). `manifold` is still the geoopt ball, used for E's init
    and for RiemannianAdam's retraction (E is a ManifoldParameter)."""

    def __init__(self, c: float = 1.0):
        self.manifold = geoopt.PoincareBall(c=c)

    @staticmethod
    def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance between every row of `x` and every row of `y`.

        x: [..., n, d], y: [..., m, d] with BROADCASTABLE leading dims -> [..., n, m].

            d(x, y) = arcosh( 1 + 2 ||x-y||^2 / ((1-||x||^2)(1-||y||^2)) )

        ||x-y||^2 is expanded as ||x||^2 + ||y||^2 - 2<x,y> so the cross term is ONE batched matmul —
        the [..., n, m, d] difference tensor is never materialised. LOWER = closer."""
        x2 = (x * x).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., n]
        y2 = (y * y).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., m]
        xy = torch.matmul(x, y.transpose(-1, -2))                               # [..., n, m]
        sq = (x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2.0 * xy).clamp_min(0.0)     # [..., n, m]
        denom = (1.0 - x2).unsqueeze(-1) * (1.0 - y2).unsqueeze(-2)             # [..., n, m]
        arg = (1.0 + 2.0 * sq / denom).clamp_min(1.0 + _ACOSH_EPS)
        return torch.acosh(arg)


def bag_weights(tokens: WalkTokens, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce a WalkTokens bag to (node ids, LOG pooling weights).

    Returns (nodes [Q, T] int64, log_w [Q, T] `dtype`) with log_w = -inf on excluded slots and
    logsumexp(log_w, dim=-1) == 0 on every row.

    The prior itself is NOT recomputed here — it is `WalkTokens.weight_logits`, the single definition of
    -(log1p(age) + log1p(hop-1)) shared by every consumer. This function only (a) decides which slots are
    live and (b) normalises over them.

    Context = mask & ~seed_node_mask: real slots that are NOT the bag's own seed node (the origin slot AND
    any mid-walk revisit). Masking happens BEFORE the softmax so the mean's mass redistributes over the
    real context.

    COLD bag (no context slot survives): slot 0 is overwritten with (seed, log_w = 0) and every other slot
    excluded, i.e. B = {(seed, 1)}. Downstream this makes the mean = the identity distance d(E_anchor,
    E_seed) exactly."""
    nodes = tokens.nodes.clamp_min(0).clone()                                   # [Q, T] padding(-1) -> 0
    valid = (tokens.mask & ~tokens.seed_node_mask).clone()                      # [Q, T] context slots

    cold = ~valid.any(dim=-1)                                                   # [Q]
    if bool(cold.any()):
        nodes[cold, 0] = tokens.seeds[cold]
        valid[cold, 0] = True                                                   # exactly one live slot

    logits = tokens.weight_logits.to(dtype)                                     # [Q, T] <= 0
    log_w = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
    return nodes, log_w


def bag_mean(d: torch.Tensor, log_w: torch.Tensor) -> torch.Tensor:
    """Weighted mean of anchor-to-bag distances along the last axis.

    d: [..., T] distances from one anchor to a bag's slots. log_w: [..., T] log pooling weights
    (broadcastable against d; -inf on excluded slots). Returns [...] = sum_p exp(log_w_p) * d_p, the
    weighted average over the LIVE slots — excluded slots have weight exp(-inf) = 0, so they drop out.

    The mean is a convex combination of the distances with weights that do NOT depend on d, so it is
    monotone non-decreasing in each d_p — which is what keeps the score monotone in the distances.
    bag_weights guarantees >= 1 live slot per row (a cold row's single live slot makes the mean equal the
    identity distance)."""
    return (log_w.exp() * d).sum(dim=-1)                                        # [...]  weighted average


class LinkPredHead(nn.Module):
    """Two-sided monotone weighted-mean metric head. Owns E (a ManifoldParameter on the Poincaré ball,
    trained end-to-end by the link loss). E is the head's ONLY parameter — the score is a parameter-free
    geometric aggregate (identity distance + the two bag means), symmetrised across the two directions
    (v vs B_u and u vs B_v) because the task is undirected."""

    def __init__(self, num_nodes: int, d_emb: int):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()

        # E lives in the ball: pure geoopt random init (default std=1, spread across the ball) rather than
        # the near-origin std=1e-2 wrapped normal; wrapped as a ManifoldParameter so RiemannianAdam keeps
        # it in the ball. E is the head's ONLY trained tensor.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in QUERY-MAJOR order, each walked with its query's cutoff. Returns logits
        [B, C] (higher = more likely link; the score is the negated distance aggregate)."""
        emb = self.E.weight                                                     # E trained end-to-end

        nodes_u, logw_u = bag_weights(src_tokens, emb.dtype)                    # [B, T]
        nodes_v, logw_v = bag_weights(cand_tokens, emb.dtype)                   # [B*C, T]

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

        # Identity channel: d(E_u, E_v) per (query, candidate).
        d_id = self.geom.pairwise_dist(e_u.unsqueeze(-2), e_v).squeeze(-2)      # [B, C]

        # Candidate seed vs the SOURCE's bag — "is v close to what u recently touched".
        d_v_bu = self.geom.pairwise_dist(e_v, x_u)                              # [B, C, T]
        mean_v_bu = bag_mean(d_v_bu, logw_u.unsqueeze(-2))                      # [B, C]

        # Source seed vs each CANDIDATE's bag — "is u close to what v recently touched". u stays in B_v
        # when the pair has interacted, so its (near-)zero distance IS the recurrence signal, for free.
        d_u_bv = self.geom.pairwise_dist(e_u[:, None, None, :], x_v).squeeze(-2)  # [B, C, T]
        mean_u_bv = bag_mean(d_u_bv, logw_v)                                    # [B, C]

        # Sum the identity distance and both sides' weighted-mean bag distances. No learnable mixing.
        raw = d_id + mean_v_bu + mean_u_bv                                      # [B, C]
        return -raw                                                             # higher = closer = better
