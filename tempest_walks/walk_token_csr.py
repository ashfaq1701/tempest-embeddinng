"""Per-query walk tokens — one packed token bag per (node, t) query, COUNT-FREE.

Walks are generated PER QUERY: each query is a (seed node, cutoff time t) pair, and
Tempest produces K backward walks for it whose edges are all STRICTLY before t (cutoff=t,
exclusive). Because every query carries its own cutoff, seeds can no longer be deduplicated
across queries — the same node at two different query times needs two different walks — so
there is no unique/inverse/gather phase and no two-level CSR. This is what lets the trainer
ingest the current batch into Tempest BEFORE scoring (TPNet-style): a query (u, t) walked
with cutoff=t never sees the edge at t (incl. the target), only its strict causal past.

A query's K walk rows are contiguous (rows [q*K, (q+1)*K), guaranteed by
shuffle_walk_order=False at the Tempest constructor), so we collapse them straight into one
LEFT-PACKED dense token bag per query:

    node_ids [Q, U]  int64, -1 pad
    node_mask [Q, U] bool
    pos_ts   [Q, U]  int64, raw t_edge   (ages = t_query - t_edge are formed by the caller)

U = max tokens over queries. Excluded before packing:
  - the seed slot (position lens-1, the INT64_MAX sentinel) and padding (pos >= lens),
  - the walk's own origin node-id wherever it recurs at a context position
    (Log_{E[seed]}(E[seed]) = 0 — it would pull μ toward zero and let a node witness itself).
COUNT-FREE: a node reached k times across a query's walks is k tokens; multiplicity is
implicit in token repetition, no explicit count.

Strict-causal: with cutoff=t every emitted token has t_edge < t, so the caller's
ages = t_query - t_edge are all > 0.
"""
from typing import Optional, Tuple

import numpy as np
import torch

DenseTokens = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # node_ids, node_mask, pos_ts


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
) -> DenseTokens:
    """Per-query backward walks → per-query packed token bags.

       walk_gen        a WalkGenerator (cutoff-aware walks_for_nodes)
       device          torch device for the packed tensors
       query_seeds     [Q] long   one seed node per query (NOT deduped)
       query_cutoffs   [Q] long   each query's exclusive cutoff time t
       -> node_ids [Q, U], node_mask [Q, U], pos_ts [Q, U]

    Generates num_walks_per_node walks for EACH query, each bounded by its own cutoff,
    and packs every query's real context tokens (seed slot / padding / origin excluded)
    left-aligned into a [Q, U] bag. NO dedup, NO counts."""
    seeds_np = np.ascontiguousarray(query_seeds.detach().cpu().numpy(), dtype=np.int32)
    cutoffs_np = np.ascontiguousarray(query_cutoffs.detach().cpu().numpy(), dtype=np.int64)
    q = int(seeds_np.shape[0])

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

    # Collapse each query's K walk rows: [W, L] = [Q·K, L] -> [Q, K·L] (rows are
    # query-major, so row q of [Q, K·L] is exactly query q's K walks flattened).
    nodes_q = nodes.reshape(q, k * length)
    ts_q = ts.reshape(q, k * length)
    valid_q = valid.reshape(q, k * length)

    counts = valid_q.sum(dim=1)                                # [Q] tokens per query
    u = max(int(counts.max().item()) if q > 0 else 1, 1)

    node_ids = torch.full((q, u), -1, device=device, dtype=torch.int64)
    node_mask = torch.zeros((q, u), device=device, dtype=torch.bool)
    pos_ts = torch.zeros((q, u), device=device, dtype=torch.int64)

    if bool(valid_q.any()):
        # left-compact each query's valid tokens: dest slot = within-query prefix count - 1.
        slot = (valid_q.cumsum(dim=1) - 1).clamp_min(0)        # [Q, K·L]
        qidx = torch.arange(q, device=device).view(q, 1).expand(q, k * length)
        sel = valid_q
        node_ids[qidx[sel], slot[sel]] = nodes_q[sel]
        node_mask[qidx[sel], slot[sel]] = True
        pos_ts[qidx[sel], slot[sel]] = ts_q[sel]

    return node_ids, node_mask, pos_ts
