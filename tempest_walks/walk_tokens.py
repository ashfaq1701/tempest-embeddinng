"""Per-query RAW walk tensors (nodes + node-aligned timestamps), straight from Tempest.

A "query" is a (seed node, cutoff time t) pair. `build_query_walk_tokens` runs K backward walks
per query, each bounded by cutoff = t (every edge has t_edge < t — strict causal past), and
returns them in their RAW per-walk layout — no packing, no dedup, no seed/origin exclusions:

    seeds       [Q]         int64  the query/source node u of each query (the walk origin). Kept
                                   explicitly so the consumer has u even for cold/empty walks,
                                   where the seed is not placed in `nodes`.
    nodes       [Q, K, L]   int64  walk node ids; backward — oldest predecessor at position 0,
                                   the seed at the LAST real position (lens-1); padding = -1.
    nodes_mask  [Q, K, L]   bool   True on real walk positions (nodes != -1), False on padding.
    ages        [Q, K, L]   int64  NODE-ALIGNED age of each node = cutoff − t_edge (query-relative):
                                   a non-seed node's age is (cutoff − time of the edge that reached
                                   it), ≥ 1 since the cutoff is exclusive; the SEED (last node) sits
                                   at "now" → age 0. Padding = -1. Shares `nodes_mask`.
    cutoffs     [Q]         int64  each query's exclusive cutoff t (kept so ages can be re-derived).

This RAW layout is the SHARED walk contract for every head (point / velocity / …): the trainer
samples a query's backward walks into it (`build_query_walk_tokens`) and each head turns it into
the flat token bag its μ needs via `flatten_tokens` — so the sampling pipeline is
identical across heads and only the scoring model differs.

Requires shuffle_walk_order=False at the Tempest constructor so a query's K walk rows are
contiguous (rows [q*K, (q+1)*K)) and reshape cleanly to [Q, K, L].
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# INT64_MAX — Tempest's raw sentinel at the seed slot (lens-1) of every walk (no outgoing edge there).
_TS_SENTINEL = torch.iinfo(torch.int64).max


@dataclass
class WalkTokens:
    """Raw per-query backward walks (see module docstring).

        seeds         [Q]           int64  query/source node u (walk origin; present even when cold)
        nodes         [Q, K, L]     int64  node ids; seed at the last real position, padding -1
        nodes_mask    [Q, K, L]     bool   True on real walk positions
        ages          [Q, K, L]     int64  node-aligned age = cutoff − t_edge; seed 0, edges ≥ 1, pad -1
        cutoffs       [Q]           int64  per-query exclusive cutoff t
        edge_features [Q, K, L*d_ef] float per-position edge features flattened over the L axis; the
                                          seed position and padding carry [0]*d_ef. None if the dataset
                                          has no edge features.
    """

    seeds: torch.Tensor                        # [Q]
    nodes: torch.Tensor                        # [Q, K, L]
    nodes_mask: torch.Tensor                   # [Q, K, L]
    ages: torch.Tensor                         # [Q, K, L]
    cutoffs: torch.Tensor                      # [Q]
    edge_features: Optional[torch.Tensor] = None   # [Q, K, L*d_ef]


def build_query_walk_tokens(
    walk_gen,
    device: torch.device,
    query_seeds: torch.Tensor,
    query_cutoffs: torch.Tensor,
    *,
    max_walk_len: int,
    num_walks_per_node: int,
    start_bias: Optional[str] = None,
    walk_bias: Optional[str] = None,
) -> WalkTokens:
    """Per-query backward walks → raw [Q, K, L] nodes + mask + node-aligned timestamps (see WalkTokens)."""
    seeds_t = query_seeds.detach().to(device=device, dtype=torch.long)        # [Q]
    cutoffs_t = query_cutoffs.detach().to(device=device, dtype=torch.long)    # [Q]
    q = int(seeds_t.shape[0])

    if q == 0:
        shape = (0, num_walks_per_node, max_walk_len)
        return WalkTokens(
            seeds_t,
            torch.empty(shape, dtype=torch.int64, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty(shape, dtype=torch.int64, device=device),
            cutoffs_t)

    # ── Walk: K backward walks per query, each bounded by its own cutoff t. ──
    wd = walk_gen.walks_for_nodes(
        np.ascontiguousarray(seeds_t.cpu().numpy(), dtype=np.int32),
        max_walk_len=max_walk_len,
        num_walks_per_node=num_walks_per_node,
        start_bias=start_bias,
        walk_bias=walk_bias,
        cutoff_times=np.ascontiguousarray(cutoffs_t.cpu().numpy(), dtype=np.int64),
    )
    k, length = int(wd.K), int(wd.nodes.shape[1])

    # Rows [q*K, (q+1)*K) are query q's K walks ⇒ reshape [Q*K, L] -> [Q, K, L].
    nodes = wd.nodes.to(device=device, dtype=torch.int64).reshape(q, k, length)        # [Q, K, L]
    nodes_mask = nodes != -1                                                            # [Q, K, L]

    # Node-aligned AGE = cutoff − t_edge (query-relative). Each non-seed node's edge time is < cutoff
    # (exclusive) → age ≥ 1; the seed slot (Tempest sentinel) is "now" → age 0; padding → -1.
    ts = wd.timestamps.to(device=device, dtype=torch.int64).reshape(q, k, length)      # [Q, K, L] raw edge time
    edge_ts = torch.where(ts == _TS_SENTINEL, cutoffs_t.view(q, 1, 1), ts)             # seed sentinel → cutoff
    ages = cutoffs_t.view(q, 1, 1) - edge_ts                                           # seed → 0, edges ≥ 1
    ages = torch.where(nodes_mask, ages, torch.full_like(ages, -1))                    # padding → -1

    # Per-position edge features → [Q, K, L*d_ef]. wd.edge_feats is [N*K, L, d_ef] node-aligned; the
    # seed position (age 0) and padding are forced to [0]*d_ef so they carry no edge. None if absent.
    edge_features = None
    if wd.edge_feats is not None:
        d_ef = int(wd.edge_feats.shape[-1])
        ef = wd.edge_feats.to(device=device, dtype=torch.float32).reshape(q, k, length, d_ef)
        real = (nodes_mask & (ages != 0)).unsqueeze(-1)                                # [Q, K, L, 1] non-seed, non-pad
        edge_features = (ef * real).reshape(q, k, length * d_ef)                       # zero seed + padding

    return WalkTokens(seeds_t, nodes, nodes_mask, ages, cutoffs_t, edge_features)


def flatten_tokens(
    tokens: WalkTokens,
    exclude_seed_positions: bool = True,
    exclude_seed_tokens: bool = True,
):
    """Collapse the raw [Q, K, L] walks into one flat [Q, K*L] token bag for the μ pooling.

    Padding (nodes == -1) is ALWAYS masked. Two flags control whether the seed node u is also
    masked out of the bag. Empirically, excluding the seed lifts wiki MRR substantially: E[u] as a
    token self-anchors P[u] toward u itself (a recurrence shortcut that overfits), so removing it
    frees μ to describe u's neighbourhood rather than u.

      exclude_seed_tokens (default True) — TAKES PRECEDENCE. When True, mask EVERY occurrence of
          the seed node u: its walk-origin slot AND any mid-walk recurrence (ids == seed).
      exclude_seed_positions (default True) — checked ONLY when exclude_seed_tokens is False. When
          True, mask ONLY the seed's walk-origin slot — the position where the seed sits at the walk
          end, identified by timestamp == cutoff (age 0). Mid-walk recurrences of u are KEPT.
      both False — no seed filtering; the seed is kept everywhere (padding-only mask).

    Returns (ages are NOT returned — read them from the instance as tokens.ages, [Q, K, L]):
        ids   [Q, T]  int64  token node ids (−1 in padding/masked slots; clamp before embedding)
        mask  [Q, T]  bool   True on kept tokens
        pos   [Q, T]  int64  within-walk HOP position from the seed: 1 = seed (walk end), 2 = its
                             immediate predecessor, …, lens = oldest node; 0 on padding. Backward
                             walks are stored oldest→seed, so pos = lens − array_index.
    with T = K*L. The per-walk (K) structure is intentionally flattened away — every head consumes
    one flat token bag; the raw [Q, K, L] shape exists only to share the walk-sampling pipeline."""
    q = tokens.nodes.shape[0]
    L = tokens.nodes.shape[2]
    ids = tokens.nodes.reshape(q, -1)                                       # [Q, T]
    mask = tokens.nodes_mask.reshape(q, -1)                                 # [Q, T]  padding
    # Hop position from the seed: seed (last real slot, index lens-1) = 1, oldest (index 0) = lens.
    lens = tokens.nodes_mask.sum(dim=-1, keepdim=True)                      # [Q, K, 1] real length
    arange = torch.arange(L, device=tokens.nodes.device).view(1, 1, L)
    pos = (lens - arange).clamp_min(0).reshape(q, -1)                       # [Q, T]  pad → 0
    if exclude_seed_tokens:                                                 # every occurrence of node u
        mask = mask & (ids != tokens.seeds.view(q, 1))
    elif exclude_seed_positions:                                            # only the walk-origin slot (age 0)
        mask = mask & (tokens.ages.reshape(q, -1) != 0)
    return ids, mask, pos
