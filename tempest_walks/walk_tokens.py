"""Per-query walk tokens in a DENSE layout.

A "query" is a (seed node, cutoff time t) pair. `build_query_walk_tokens` runs K backward
walks per query, each bounded by cutoff=t (every token has t_edge < t — strict causal past),
and packs the surviving tokens into a DENSE [Q, U] token bag (see `WalkTokens`): every surviving
walk token, left-packed; drives μ via a softmax over the padded axis (deterministic, no
scatter-add jitter).

Excluded before packing:
  * the seed slot (position lens-1, the INT64_MAX sentinel) and padding (pos >= lens),
  * the origin node-id wherever it recurs at a context position
    (Log_{E[seed]}(E[seed]) = 0 — it would pull μ toward zero and let a node witness itself).

Requires shuffle_walk_order=False at the Tempest constructor so a query's K walk rows are
contiguous (rows [q*K, (q+1)*K)).

The bag is LEFT-PACKED with a mask: real entries pushed to the front of each row, padding after.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class WalkTokens:
    """A DENSE [Q, U] token-bag view over Q queries (see module docstring).

        seeds      [Q]       int64   origin node of each query (NOT deduped across queries)
        cutoffs    [Q]       int64   each query's exclusive cutoff t (== its prediction time);
                                     every token has pos_ts < cutoffs[q], so ages = cutoffs[q] −
                                     pos_ts are all > 0. Self-contained.

      TOKEN BAG (raw, count-free) — a node reached k times is k entries.
        node_ids   [Q, U]    int64   token node ids, -1 in padding slots
        node_mask  [Q, U]    bool    True on real tokens
        pos_ts     [Q, U]    int64   raw t_edge per token, aligned with node_ids

    U = max tokens over queries. A cold query (no tokens) is an all-False row.
    """

    seeds: torch.Tensor       # [Q]
    cutoffs: torch.Tensor     # [Q]

    node_ids: torch.Tensor    # [Q, U]
    node_mask: torch.Tensor   # [Q, U]
    pos_ts: torch.Tensor      # [Q, U]


# ──────────────────────────────────────────────────────────────────────────
# Packing helpers
# ──────────────────────────────────────────────────────────────────────────

def _pack_plan(valid_q: torch.Tensor):
    """Compute the left-pack destination of every real entry, ONCE, for reuse by all payloads.

    valid_q [Q, M] bool  ->  rows [T], cols [T], width.
    Taking the True entries in token-major (row-major) order, entry i lands at out[rows[i],
    cols[i]]; `width` is the widest row (>= 1). `cols` is the within-row running count − 1
    (how many reals precede it in its row); `rows` is the destination row == query id.
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
    """Per-query backward walks → a dense token bag (see WalkTokens)."""
    seeds_t = query_seeds.detach().to(device=device, dtype=torch.long)        # [Q]
    cutoffs_t = query_cutoffs.detach().to(device=device, dtype=torch.long)    # [Q]
    q = int(seeds_t.shape[0])

    if q == 0:
        return WalkTokens(
            seeds_t, cutoffs_t,
            torch.full((q, 1), -1, dtype=torch.int64, device=device),
            torch.zeros((q, 1), dtype=torch.bool, device=device),
            torch.zeros((q, 1), dtype=torch.int64, device=device))

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

    # ── 4. One packing plan; reuse it for ids, timestamps and the mask. ──
    rows, cols, width = _pack_plan(valid_q)
    node_ids = _scatter_rows(nodes_q[valid_q].to(torch.int64), rows, cols, q, width, fill=-1)
    pos_ts = _scatter_rows(ts_q[valid_q].to(torch.int64), rows, cols, q, width, fill=0)
    node_mask = torch.zeros((q, width), dtype=torch.bool, device=device)
    node_mask[rows, cols] = True

    return WalkTokens(seeds_t, cutoffs_t, node_ids, node_mask, pos_ts)
