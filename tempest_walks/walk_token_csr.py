"""Per-query walk tokens in CSR layout, plus a per-query deduplicated-neighbour CSR.

A "query" is a (seed node, cutoff time t) pair. `build_query_walk_tokens` runs K backward
walks per query, each bounded by cutoff=t (every token has t_edge < t, strict causal past),
and returns two parallel CSR views over the same Q queries — see `WalkTokenCSR`. No node-level
dedup ACROSS queries (the same node at two query times needs two cutoffs); the neighbour CSR
deduplicates WITHIN a query.

Excluded before packing (same contract as the dense layout it replaces):
  - the seed slot (position lens-1, the INT64_MAX sentinel) and padding (pos >= lens),
  - the walk's own origin node-id wherever it recurs at a context position
    (Log_{E[seed]}(E[seed]) = 0 — it would pull μ toward zero and let a node witness itself).

Requires shuffle_walk_order=False at the Tempest constructor so a query's K walk rows are
contiguous (rows [q*K, (q+1)*K)).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class WalkTokenCSR:
    """Per-query walk tokens in CSR layout, plus a per-query deduplicated-neighbour CSR.

    A "query" is a (seed node, cutoff time t) pair. Two parallel CSR views over the same
    Q queries:

    TOKEN STREAM (raw, count-free) — every surviving walk context token, kept in
    occurrence order; multiplicity implicit in repetition (a node reached k times = k
    entries). Drives μ.

        seeds         [Q]      int64   origin node of each query (NOT deduped across queries)
        cutoffs       [Q]      int64   each query's exclusive cutoff t (== its prediction
                                       time); every token has pos_ts < cutoffs[q], so token
                                       ages = cutoffs[q] − pos_ts are all > 0. Self-contained:
                                       a consumer needs no external t_query.
        node_ids      [T]      int64   flat token node ids, all queries concatenated
        pos_ts        [T]      int64   raw t_edge per token, index-aligned with node_ids
        node_ids_ptr  [Q + 1]  int64   row pointer: query q owns node_ids[node_ids_ptr[q] : node_ids_ptr[q+1]]

    NEIGHBOUR CSR (deduplicated, explicit count) — the same tokens collapsed to unique
    nodes per query with their occurrence counts. The id-level multiplicity the token
    stream leaves implicit, made explicit for set-overlap / common-neighbour /
    path-density signals.

        neighbors        [Tn]     int64   unique reached node ids per query, concatenated
        neighbors_count  [Tn]     int64   occurrence count per neighbour, aligned with `neighbors`
        neighbors_ptr    [Q + 1]  int64   row pointer: query q owns neighbors[neighbors_ptr[q] : neighbors_ptr[q+1]]

    T  = total tokens over all queries; Tn = total unique neighbours over all queries (Tn ≤ T,
    and Σ_q over a query's neighbors_count == that query's token length). Empty segment
    (ptr[q] == ptr[q+1]) = a cold query with no tokens. Both pointers share the leading-0
    CSR convention so query q slices with no q==0 special case.
    """

    seeds: torch.Tensor            # [Q]
    cutoffs: torch.Tensor          # [Q]  per-query cutoff t (== prediction time)
    node_ids: torch.Tensor         # [T]
    pos_ts: torch.Tensor           # [T]
    node_ids_ptr: torch.Tensor     # [Q + 1]

    neighbors: torch.Tensor        # [Tn]
    neighbors_count: torch.Tensor  # [Tn]
    neighbors_ptr: torch.Tensor    # [Q + 1]


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
) -> WalkTokenCSR:
    """Per-query backward walks → token-stream CSR + neighbour CSR (see WalkTokenCSR).

       walk_gen        a WalkGenerator (cutoff-aware walks_for_nodes)
       device          torch device for the packed tensors
       query_seeds     [Q] long   one seed node per query
       query_cutoffs   [Q] long   each query's exclusive cutoff time t
       -> WalkTokenCSR
    """
    seeds_t = query_seeds.detach().to(device=device, dtype=torch.long)        # [Q]
    cutoffs_t = query_cutoffs.detach().to(device=device, dtype=torch.long)    # [Q]
    seeds_np = np.ascontiguousarray(seeds_t.cpu().numpy(), dtype=np.int32)
    cutoffs_np = np.ascontiguousarray(cutoffs_t.cpu().numpy(), dtype=np.int64)
    q = int(seeds_t.shape[0])

    empty_ptr = torch.zeros(q + 1, device=device, dtype=torch.int64)
    if q == 0:
        e = torch.empty(0, device=device, dtype=torch.int64)
        return WalkTokenCSR(seeds_t, cutoffs_t, e, e, empty_ptr, e, e.clone(), empty_ptr.clone())

    wd = walk_gen.walks_for_nodes(
        seeds_np,
        max_walk_len=max_walk_len,
        num_walks_per_node=num_walks_per_node,
        start_bias=start_bias,
        walk_bias=walk_bias,
        cutoff_times=cutoffs_np,
    )
    k, length = int(wd.K), int(wd.nodes.shape[1])
    w = int(wd.nodes.shape[0])                                  # Q*K walk rows

    nodes = wd.nodes.to(device)                                # [W, L] int64 (-1 pad)
    ts = wd.timestamps.to(device)                              # [W, L] int64 t_edge
    lens = wd.lens.to(device)                                  # [W]

    pos = torch.arange(length, device=device).view(1, length)
    is_ctx = pos < (lens.view(w, 1) - 1)                       # excl seed slot + padding
    # rows [q·K, (q+1)·K) belong to query q (shuffle_walk_order=False) → origin per row.
    origin = wd.seeds.to(device).repeat_interleave(k)          # [W]
    valid = is_ctx & (nodes != origin.view(w, 1))              # [W, L] real, non-origin

    # ── TOKEN STREAM CSR ── collapse the K walk rows of each query: [W, L] = [Q·K, L] ->
    # [Q, K·L] (rows are query-major). Boolean-masking a [Q, K·L] tensor returns elements
    # row-major, i.e. query-major in OCCURRENCE order — exactly the flat token stream.
    nodes_q = nodes.reshape(q, k * length)                     # [Q, K·L]
    ts_q = ts.reshape(q, k * length)
    valid_q = valid.reshape(q, k * length)

    counts = valid_q.sum(dim=1)                                # [Q] tokens per query
    node_ids = nodes_q[valid_q].to(torch.int64)               # [T]
    pos_ts = ts_q[valid_q].to(torch.int64)                    # [T]
    node_ids_ptr = torch.zeros(q + 1, device=device, dtype=torch.int64)
    node_ids_ptr[1:] = torch.cumsum(counts, dim=0)
    t = int(node_ids.shape[0])

    # ── NEIGHBOUR CSR ── per-query unique nodes + occurrence counts. Build the per-token
    # query index, then a single global unique over (query, node) pairs (sorted by query,
    # then node) gives the deduped neighbours grouped by query in one shot.
    if t == 0:
        e = torch.empty(0, device=device, dtype=torch.int64)
        return WalkTokenCSR(seeds_t, cutoffs_t, node_ids, pos_ts, node_ids_ptr,
                            e, e.clone(), torch.zeros(q + 1, device=device, dtype=torch.int64))

    seg = torch.repeat_interleave(
        torch.arange(q, device=device, dtype=torch.int64), counts)   # [T] query per token
    pairs = torch.stack([seg, node_ids], dim=1)                       # [T, 2]
    uniq, nbr_count = torch.unique(pairs, dim=0, return_counts=True)  # [Tn, 2], [Tn]
    nbr_seg = uniq[:, 0]
    neighbors = uniq[:, 1].contiguous()
    neighbors_count = nbr_count.to(torch.int64)
    per_q = torch.bincount(nbr_seg, minlength=q)                      # [Q] uniques per query
    neighbors_ptr = torch.zeros(q + 1, device=device, dtype=torch.int64)
    neighbors_ptr[1:] = torch.cumsum(per_q, dim=0)

    return WalkTokenCSR(
        seeds=seeds_t,
        cutoffs=cutoffs_t,
        node_ids=node_ids,
        pos_ts=pos_ts,
        node_ids_ptr=node_ids_ptr,
        neighbors=neighbors,
        neighbors_count=neighbors_count,
        neighbors_ptr=neighbors_ptr,
    )
