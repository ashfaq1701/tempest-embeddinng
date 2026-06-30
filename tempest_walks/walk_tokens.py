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
    timestamps  [Q, K, L]   int64  NODE-ALIGNED time of each node: for a non-seed node it is the
                                   time of the edge that reached it (== Tempest's timestamps[p],
                                   the edge (nodes[p], nodes[p+1])); the SEED (last node) uses the
                                   query cutoff t (it sits at "now", age 0). Padding = -1. Shares
                                   `nodes_mask` — no separate timestamp mask.
    cutoffs     [Q]         int64  each query's exclusive cutoff t; every non-seed time is < t,
                                   the seed time == t.

Requires shuffle_walk_order=False at the Tempest constructor so a query's K walk rows are
contiguous (rows [q*K, (q+1)*K)) and reshape cleanly to [Q, K, L].
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# INT64_MAX — Tempest's sentinel at the seed slot (lens-1) of every walk (no outgoing edge there).
_TS_SENTINEL = torch.iinfo(torch.int64).max


@dataclass
class WalkTokens:
    """Raw per-query backward walks (see module docstring).

        seeds       [Q]         int64  query/source node u (walk origin; present even when cold)
        nodes       [Q, K, L]   int64  node ids; seed at the last real position, padding -1
        nodes_mask  [Q, K, L]   bool   True on real walk positions
        timestamps  [Q, K, L]   int64  node-aligned times; seed = cutoff, edges < cutoff, pad -1
        cutoffs     [Q]         int64  per-query exclusive cutoff t
    """

    seeds: torch.Tensor        # [Q]
    nodes: torch.Tensor        # [Q, K, L]
    nodes_mask: torch.Tensor   # [Q, K, L]
    timestamps: torch.Tensor   # [Q, K, L]
    cutoffs: torch.Tensor      # [Q]


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

    # Node-aligned times: each non-seed node keeps its arrival-edge time; the seed slot (the
    # sentinel) takes the query cutoff t — the seed is "now", age 0. Padding stays -1.
    ts = wd.timestamps.to(device=device, dtype=torch.int64).reshape(q, k, length)      # [Q, K, L]
    timestamps = torch.where(ts == _TS_SENTINEL, cutoffs_t.view(q, 1, 1), ts)

    return WalkTokens(seeds_t, nodes, nodes_mask, timestamps, cutoffs_t)


def build_query_walk_tokens_multi(
    walk_gen,
    device: torch.device,
    query_seeds: torch.Tensor,
    query_cutoffs: torch.Tensor,
    *,
    max_walk_len: int,
    configs: "list[tuple[int, str, str]]",
) -> WalkTokens:
    """Sample several (num_walks, start_bias, walk_bias) walk configs for the SAME queries and
    concatenate them into one token bag.

    Each config samples its own per-query backward walks (sharing seeds, cutoffs, and `max_walk_len`
    so all bags are [Q, K_i, L] with a common L), and the bags are concatenated along the K axis →
    [Q, ΣK_i, L]. The head consumes the union, so e.g. (ExponentialWeight) + (ExponentialWeight­-
    InverseDegree) gives a μ-fit over both a degree-favouring and a degree-discounting sample.
    `seeds`/`cutoffs` are identical across configs and taken from the first."""
    if len(configs) == 1:
        nw, sb, wb = configs[0]
        return build_query_walk_tokens(
            walk_gen, device, query_seeds, query_cutoffs,
            max_walk_len=max_walk_len, num_walks_per_node=nw,
            start_bias=sb, walk_bias=wb)

    parts = [
        build_query_walk_tokens(
            walk_gen, device, query_seeds, query_cutoffs,
            max_walk_len=max_walk_len, num_walks_per_node=nw,
            start_bias=sb, walk_bias=wb)
        for (nw, sb, wb) in configs
    ]
    return WalkTokens(
        parts[0].seeds,
        torch.cat([p.nodes for p in parts], dim=1),
        torch.cat([p.nodes_mask for p in parts], dim=1),
        torch.cat([p.timestamps for p in parts], dim=1),
        parts[0].cutoffs,
    )
