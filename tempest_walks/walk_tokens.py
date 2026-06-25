"""Per-query walk tokens in a DENSE layout, plus a per-query deduplicated-neighbour bag.

A "query" is a (seed node, cutoff time t) pair. `build_query_walk_tokens` runs K backward
walks per query, each bounded by cutoff=t (every token has t_edge < t — strict causal past),
and packs the result into two parallel DENSE [Q, ·] views (see `WalkTokens`):

  * TOKEN BAG     — every surviving walk token, left-packed; drives μ via a softmax over the
                    padded axis (deterministic, no scatter-add jitter).
  * NEIGHBOUR BAG — the same tokens collapsed to unique nodes per query, with occurrence
                    counts, for set-overlap / common-neighbour / path-density (co-reach) signals.

Excluded before packing:
  * the seed slot (position lens-1, the INT64_MAX sentinel) and padding (pos >= lens),
  * the origin node-id wherever it recurs at a context position
    (Log_{E[seed]}(E[seed]) = 0 — it would pull μ toward zero and let a node witness itself).

Requires shuffle_walk_order=False at the Tempest constructor so a query's K walk rows are
contiguous (rows [q*K, (q+1)*K)).

Both bags are LEFT-PACKED with masks: real entries pushed to the front of each row, padding
after. The token bag also carries `pos_ts`; the neighbour bag carries integer counts.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class WalkTokens:
    """Two parallel DENSE [Q, ·] views over the same Q queries (see module docstring).

        seeds            [Q]       int64   origin node of each query (NOT deduped across queries)
        cutoffs          [Q]       int64   each query's exclusive cutoff t (== its prediction
                                           time); every token has pos_ts < cutoffs[q], so
                                           ages = cutoffs[q] − pos_ts are all > 0. Self-contained.

      TOKEN BAG (raw, count-free) — a node reached k times is k entries.
        node_ids         [Q, U]    int64   token node ids, -1 in padding slots
        node_mask        [Q, U]    bool    True on real tokens
        pos_ts           [Q, U]    int64   raw t_edge per token, aligned with node_ids

      NEIGHBOUR BAG (deduplicated, explicit count)
        neighbors        [Q, Un]   int64   unique reached node ids per query, -1 padding
        neighbors_count  [Q, Un]   int64   occurrence count per neighbour (0 in padding)
        neighbors_mask   [Q, Un]   bool    True on real neighbours

    U = max tokens over queries; Un = max unique neighbours (Un ≤ U, and per row
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


# ──────────────────────────────────────────────────────────────────────────
# Packing helpers
# ──────────────────────────────────────────────────────────────────────────

def _pack_plan(valid_q: torch.Tensor):
    """Compute the left-pack destination of every real entry, ONCE, for reuse by all payloads.

    valid_q [Q, M] bool  ->  rows [T], cols [T], width.
    Taking the True entries in token-major (row-major) order, entry i lands at out[rows[i],
    cols[i]]; `width` is the widest row (>= 1). `cols` is the within-row running count − 1
    (how many reals precede it in its row). `rows` doubles as the per-token query id used by
    the neighbour dedup, so the segment index never has to be rebuilt separately.
    """
    q = valid_q.shape[0]
    counts = valid_q.sum(dim=1)                                     # [Q] reals per row
    width = max(int(counts.amax().item()), 1)
    cols = (valid_q.cumsum(dim=1) - 1).clamp_min(0)[valid_q]        # [T] dest column
    rows = (torch.arange(q, device=valid_q.device)
            .view(q, 1).expand_as(valid_q)[valid_q])               # [T] dest row == query id
    return rows, cols, width


def _scatter_rows(flat_vals: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
                  q: int, width: int, fill) -> torch.Tensor:
    """Scatter a token-major payload [T] into a left-packed [Q, width] matrix (rest = fill)."""
    out = torch.full((q, width), fill, device=flat_vals.device, dtype=flat_vals.dtype)
    out[rows, cols] = flat_vals
    return out


def _pack_neighbours(seg: torch.Tensor, flat_nodes: torch.Tensor, q: int,
                     device: torch.device):
    """Dedup (query, node) → per-query unique neighbours + occurrence counts, left-packed.

    seg [T] query id per token, flat_nodes [T] token node ids (token-major).
    -> neighbors [Q, Un], neighbors_count [Q, Un], neighbors_mask [Q, Un].
    """
    # One global unique over (query, node) pairs; results come out sorted by (query, node),
    # so all of a query's neighbours are contiguous — grouped by query in a single shot.
    pairs = torch.stack([seg, flat_nodes], dim=1)                    # [T, 2]
    uniq, count = torch.unique(pairs, dim=0, return_counts=True)     # [Tn, 2], [Tn]
    uq, un = uniq[:, 0], uniq[:, 1]                                  # query id, neighbour id

    per_q = torch.bincount(uq, minlength=q)                         # [Q] uniques per query
    width = max(int(per_q.amax().item()), 1)
    # uq is sorted ascending ⇒ within-query column = global position − query's start offset.
    start = torch.zeros(q + 1, dtype=torch.int64, device=device)
    start[1:] = per_q.cumsum(0)
    col = torch.arange(uq.shape[0], device=device) - start[uq]      # [Tn] within-query column

    neighbors = torch.full((q, width), -1, dtype=torch.int64, device=device)
    neighbors_count = torch.zeros((q, width), dtype=torch.int64, device=device)
    neighbors_mask = torch.zeros((q, width), dtype=torch.bool, device=device)
    neighbors[uq, col] = un
    neighbors_count[uq, col] = count.to(torch.int64)
    neighbors_mask[uq, col] = True
    return neighbors, neighbors_count, neighbors_mask


def _empty_bag(q: int, device: torch.device):
    """A width-1, all-padding dense bag — used for cold queries / empty batches."""
    ids = torch.full((q, 1), -1, dtype=torch.int64, device=device)
    counts = torch.zeros((q, 1), dtype=torch.int64, device=device)
    mask = torch.zeros((q, 1), dtype=torch.bool, device=device)
    return ids, counts, mask


# ──────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────

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
    q = int(seeds_t.shape[0])

    # Empty batch / all-cold short-circuits share one builder.
    def _result(node_ids, node_mask, pos_ts, nbr):
        return WalkTokens(seeds_t, cutoffs_t, node_ids, node_mask, pos_ts, *nbr)

    if q == 0:
        ids, counts, mask = _empty_bag(q, device)
        return _result(ids, mask, counts, _empty_bag(q, device))

    # ── 1. Walk: K backward walks per query, each bounded by its own cutoff t. ──
    wd = walk_gen.walks_for_nodes(
        np.ascontiguousarray(seeds_t.cpu().numpy(), dtype=np.int32),
        max_walk_len=max_walk_len,
        num_walks_per_node=num_walks_per_node,
        start_bias=start_bias,
        walk_bias=walk_bias,
        cutoff_times=np.ascontiguousarray(cutoffs_t.cpu().numpy(), dtype=np.int64),
    )
    k, length = int(wd.K), int(wd.nodes.shape[1])
    w = int(wd.nodes.shape[0])                                     # W = Q*K walk rows

    nodes = wd.nodes.to(device)                                   # [W, L] int64 (-1 pad)
    ts = wd.timestamps.to(device)                                 # [W, L] int64 t_edge
    lens = wd.lens.to(device)                                     # [W]

    # ── 2. Keep only real, non-seed-slot, non-origin context positions. ──
    pos = torch.arange(length, device=device).view(1, length)
    is_ctx = pos < (lens.view(w, 1) - 1)                          # drop seed slot + padding
    origin = wd.seeds.to(device).repeat_interleave(k)             # [W] each row's seed id
    valid = is_ctx & (nodes != origin.view(w, 1))                # [W, L]

    # ── 3. Collapse each query's K walk rows: [Q*K, L] -> [Q, K*L], query-major. ──
    nodes_q = nodes.reshape(q, k * length)
    ts_q = ts.reshape(q, k * length)
    valid_q = valid.reshape(q, k * length)

    # ── 4. One packing plan; reuse it for ids, timestamps, mask, and the neighbour seg. ──
    rows, cols, width = _pack_plan(valid_q)            # rows[i] == query id of token i
    flat_nodes = nodes_q[valid_q].to(torch.int64)      # [T] token-major node ids (reused in 5)
    flat_ts = ts_q[valid_q].to(torch.int64)            # [T] token-major timestamps

    node_ids = _scatter_rows(flat_nodes, rows, cols, q, width, fill=-1)
    pos_ts = _scatter_rows(flat_ts, rows, cols, q, width, fill=0)
    node_mask = torch.zeros((q, width), dtype=torch.bool, device=device)
    node_mask[rows, cols] = True

    # ── 5. Neighbour bag (reuses `rows` as the segment id and `flat_nodes` as the stream). ──
    if flat_nodes.numel() == 0:                        # every query cold → empty neighbour bag
        nbr = _empty_bag(q, device)
    else:
        nbr = _pack_neighbours(rows, flat_nodes, q, device)

    return _result(node_ids, node_mask, pos_ts, nbr)
