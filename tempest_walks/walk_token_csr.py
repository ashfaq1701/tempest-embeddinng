"""Per-query walk tokens in a DENSE layout, plus a per-query deduplicated-neighbour bag.

A "query" is a (seed node, cutoff time t) pair. `build_query_walk_tokens` runs K backward
walks per query, each bounded by cutoff=t (every token has t_edge < t, strict causal past),
and packs the result into two parallel DENSE [Q, ·] views (see `WalkTokens`). Dense — rather
than a ragged CSR — so the per-query reduction is a plain `torch.softmax(..., dim=-1)` over a
padded axis (deterministic, no scatter-add jitter), and the neighbour set is a padded matrix.

Excluded before packing:
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
class WalkTokens:
    """Per-query walk tokens in a dense layout, plus a per-query deduplicated-neighbour bag.

    A "query" is a (seed node, cutoff time t) pair. Two parallel DENSE views over the same
    Q queries, both left-packed with masks (padding never contributes):

        seeds            [Q]       int64   origin node of each query (NOT deduped across queries)
        cutoffs          [Q]       int64   each query's exclusive cutoff t (== its prediction
                                           time); every token has pos_ts < cutoffs[q], so token
                                           ages = cutoffs[q] − pos_ts are all > 0. Self-contained.

    TOKEN BAG (raw, count-free) — every surviving walk context token; multiplicity implicit in
    repetition (a node reached k times = k entries). Drives μ via a per-row softmax over U.

        node_ids         [Q, U]    int64   token node ids, -1 in padding slots
        node_mask        [Q, U]    bool     True on real tokens
        pos_ts           [Q, U]    int64   raw t_edge per token, index-aligned with node_ids

    NEIGHBOUR BAG (deduplicated, explicit count) — the token bag collapsed to unique nodes per
    query with their occurrence counts. The id-level multiplicity the token bag leaves implicit,
    made explicit for set-overlap / common-neighbour / path-density (co-reach) signals.

        neighbors        [Q, Un]   int64   unique reached node ids per query, -1 padding
        neighbors_count  [Q, Un]   int64   occurrence count per neighbour (0 in padding)
        neighbors_mask   [Q, Un]   bool     True on real neighbours

    U = max tokens over queries; Un = max unique neighbours over queries (Un ≤ U, and per row
    Σ neighbors_count == node_mask.sum()). A cold query (no tokens) is an all-False row.
    """

    seeds: torch.Tensor            # [Q]
    cutoffs: torch.Tensor          # [Q]

    node_ids: torch.Tensor         # [Q, U]
    node_mask: torch.Tensor        # [Q, U]
    pos_ts: torch.Tensor           # [Q, U]

    neighbors: torch.Tensor        # [Q, Un]
    neighbors_count: torch.Tensor  # [Q, Un]
    neighbors_mask: torch.Tensor   # [Q, Un]


def _left_pack(values_q: torch.Tensor, valid_q: torch.Tensor, fill):
    """Left-compact the True entries of each row of valid_q [Q, M] into [Q, W], pulling the
    aligned `values_q` along. Returns (packed [Q, W], mask [Q, W]); W = max per-row count."""
    device = valid_q.device
    q = valid_q.shape[0]
    counts = valid_q.sum(dim=1)                                 # [Q]
    width = max(int(counts.max().item()) if q > 0 else 1, 1)
    out = torch.full((q, width), fill, device=device, dtype=values_q.dtype)
    mask = torch.zeros((q, width), device=device, dtype=torch.bool)
    if bool(valid_q.any()):
        slot = (valid_q.cumsum(dim=1) - 1).clamp_min(0)        # [Q, M] dest column
        qidx = torch.arange(q, device=device).view(q, 1).expand_as(valid_q)
        sel = valid_q
        out[qidx[sel], slot[sel]] = values_q[sel]
        mask[qidx[sel], slot[sel]] = True
    return out, mask


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
    """Per-query backward walks → dense token bag + dense neighbour bag (see WalkTokens)."""
    seeds_t = query_seeds.detach().to(device=device, dtype=torch.long)        # [Q]
    cutoffs_t = query_cutoffs.detach().to(device=device, dtype=torch.long)    # [Q]
    seeds_np = np.ascontiguousarray(seeds_t.cpu().numpy(), dtype=np.int32)
    cutoffs_np = np.ascontiguousarray(cutoffs_t.cpu().numpy(), dtype=np.int64)
    q = int(seeds_t.shape[0])

    def _empty(width_tok=1, width_nbr=1):
        zt = torch.full((q, width_tok), -1, device=device, dtype=torch.int64)
        zn = torch.full((q, width_nbr), -1, device=device, dtype=torch.int64)
        fb = torch.zeros((q, width_tok), device=device, dtype=torch.bool)
        nb = torch.zeros((q, width_nbr), device=device, dtype=torch.bool)
        return WalkTokens(seeds_t, cutoffs_t, zt, fb, torch.zeros_like(zt),
                          zn, torch.zeros_like(zn), nb)

    if q == 0:
        return _empty()

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
    origin = wd.seeds.to(device).repeat_interleave(k)          # [W] each query's seed
    valid = is_ctx & (nodes != origin.view(w, 1))              # [W, L] real, non-origin

    # collapse the K walk rows of each query: [W, L] = [Q·K, L] -> [Q, K·L] (query-major).
    nodes_q = nodes.reshape(q, k * length)
    ts_q = ts.reshape(q, k * length)
    valid_q = valid.reshape(q, k * length)

    # --- TOKEN BAG (dense, left-packed) ---
    node_ids, node_mask = _left_pack(nodes_q.to(torch.int64), valid_q, fill=-1)
    pos_ts, _ = _left_pack(ts_q.to(torch.int64), valid_q, fill=0)

    # --- NEIGHBOUR BAG (dense, per-query unique + counts) ---
    if not bool(valid_q.any()):
        nz = torch.full((q, 1), -1, device=device, dtype=torch.int64)
        return WalkTokens(seeds_t, cutoffs_t, node_ids, node_mask, pos_ts,
                          nz, torch.zeros_like(nz), torch.zeros((q, 1), device=device, dtype=torch.bool))

    counts = valid_q.sum(dim=1)                                # [Q] tokens per query
    seg = torch.repeat_interleave(
        torch.arange(q, device=device, dtype=torch.int64), counts)   # [T] query per token
    flat_nodes = nodes_q[valid_q].to(torch.int64)              # [T] query-major token nodes
    pairs = torch.stack([seg, flat_nodes], dim=1)              # [T, 2]
    uniq, nbr_count = torch.unique(pairs, dim=0, return_counts=True)  # [Tn, 2], [Tn] (sorted)
    uq = uniq[:, 0]                                            # query of each unique neighbour
    un = uniq[:, 1]                                            # the neighbour node id
    per_q = torch.bincount(uq, minlength=q)                   # [Q] uniques per query
    width_n = max(int(per_q.max().item()), 1)
    nptr = torch.zeros(q + 1, device=device, dtype=torch.int64)
    nptr[1:] = torch.cumsum(per_q, dim=0)
    nslot = torch.arange(uq.shape[0], device=device) - nptr.index_select(0, uq)  # within-query col

    neighbors = torch.full((q, width_n), -1, device=device, dtype=torch.int64)
    neighbors_count = torch.zeros((q, width_n), device=device, dtype=torch.int64)
    neighbors_mask = torch.zeros((q, width_n), device=device, dtype=torch.bool)
    neighbors[uq, nslot] = un
    neighbors_count[uq, nslot] = nbr_count.to(torch.int64)
    neighbors_mask[uq, nslot] = True

    return WalkTokens(
        seeds=seeds_t,
        cutoffs=cutoffs_t,
        node_ids=node_ids,
        node_mask=node_mask,
        pos_ts=pos_ts,
        neighbors=neighbors,
        neighbors_count=neighbors_count,
        neighbors_mask=neighbors_mask,
    )
