"""Per-query FLATTENED backward-walk tensors from Tempest. A query is a (seed,
cutoff t) pair; K walks per query (each bounded by cutoff = t) are flattened to
one [Q, T] bag (T = K·L). Requires shuffle_walk_order=False."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

_TS_SENTINEL = torch.iinfo(torch.int64).max     # Tempest's seed-slot timestamp (no outgoing edge there)


@dataclass
class WalkTokens:
    """Flattened [Q, T] fields (T = K·L), or [Q, T, d_ef] for edge_features."""

    seeds: torch.Tensor                             # [Q]      query/source node u (kept even when cold)
    cutoffs: torch.Tensor                           # [Q]      exclusive cutoff t per query
    nodes: torch.Tensor                             # [Q, T]   walk node ids; padding = -1
    ages: torch.Tensor                              # [Q, T]   cutoff - t_edge; seed = 0, ctx >= 1, pad = -1
    positions: torch.Tensor                         # [Q, T]   hop from seed: 1 = seed, ..., lens = oldest; pad 0
    mask: torch.Tensor                              # [Q, T]   bool, real slots (nodes != -1), incl. seed
    seed_mask: torch.Tensor                         # [Q, T]   bool, ONLY the seed's walk-origin slot
    edge_features: Optional[torch.Tensor] = None    # [Q, T, d_ef]  seed slot + padding zeroed; None if no EF


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
    """Per-query backward walks -> flattened [Q, T] WalkTokens."""
    seeds_t = query_seeds.detach().to(device=device, dtype=torch.long)        # [Q]
    cutoffs_t = query_cutoffs.detach().to(device=device, dtype=torch.long)    # [Q]
    q = int(seeds_t.shape[0])

    if q == 0:
        t = num_walks_per_node * max_walk_len
        empty_i = torch.empty((0, t), dtype=torch.int64, device=device)
        empty_b = torch.empty((0, t), dtype=torch.bool, device=device)
        return WalkTokens(seeds_t, cutoffs_t, empty_i, empty_i, empty_i, empty_b, empty_b)

    wd = walk_gen.walks_for_nodes(
        np.ascontiguousarray(seeds_t.cpu().numpy(), dtype=np.int32),
        max_walk_len=max_walk_len,
        num_walks_per_node=num_walks_per_node,
        start_bias=start_bias,
        walk_bias=walk_bias,
        cutoff_times=np.ascontiguousarray(cutoffs_t.cpu().numpy(), dtype=np.int64),
    )
    k, length = int(wd.K), int(wd.nodes.shape[1])

    # Rows [q*K, (q+1)*K) are query q's K walks -> reshape [Q*K, L] -> [Q, K, L].
    nodes = wd.nodes.to(device=device, dtype=torch.int64).reshape(q, k, length)        # [Q, K, L]
    node_mask = nodes != -1

    # AGE = cutoff - t_edge: seed sentinel -> cutoff (age 0), real edges >= 1, padding -> -1.
    ts = wd.timestamps.to(device=device, dtype=torch.int64).reshape(q, k, length)
    edge_ts = torch.where(ts == _TS_SENTINEL, cutoffs_t.view(q, 1, 1), ts)
    ages = cutoffs_t.view(q, 1, 1) - edge_ts
    ages = torch.where(node_mask, ages, torch.full_like(ages, -1))

    # HOP from seed: pos = lens - array_index (walks stored oldest->seed); padding -> 0.
    lens = node_mask.sum(dim=-1, keepdim=True)                                          # [Q, K, 1]
    arange = torch.arange(length, device=device).view(1, 1, length)
    positions = (lens - arange).clamp_min(0)

    seed_mask = node_mask & (ages == 0)

    # Per-token edge features; seed slot (age 0) and padding forced to [0]*d_ef.
    edge_features = None
    if wd.edge_feats is not None:
        d_ef = int(wd.edge_feats.shape[-1])
        ef = wd.edge_feats.to(device=device, dtype=torch.float32).reshape(q, k, length, d_ef)
        real = (node_mask & (ages != 0)).unsqueeze(-1)
        edge_features = (ef * real).reshape(q, k * length, d_ef)                        # [Q, T, d_ef]

    return WalkTokens(
        seeds_t,
        cutoffs_t,
        nodes.reshape(q, -1),               # [Q, T]
        ages.reshape(q, -1),
        positions.reshape(q, -1),
        node_mask.reshape(q, -1),
        seed_mask.reshape(q, -1),
        edge_features,
    )


def median_context_age(
    walk_gen,
    device: torch.device,
    query_seeds: torch.Tensor,
    query_cutoffs: torch.Tensor,
    *,
    max_walk_len: int,
    num_walks_per_node: int,
    start_bias: Optional[str] = None,
    walk_bias: Optional[str] = None,
) -> Optional[float]:
    """Median age of the CONTEXT walk tokens produced by (seeds, cutoffs).

    The recency feature is -(age / scale); this measures the age distribution the pooler will
    actually see, so the scale is derived from the same walks the model consumes rather than
    from a mean-field estimate over the edge list.

    Lives here because it depends on this module's token invariants: `ages` is 0 on the seed
    slot, >= 1 on context, -1 on padding, and `mask` includes the seed while `seed_mask` marks
    only the walk origin. Seed slots are dropped -- they are structurally 0 (about a quarter of
    valid slots) and would drag the median toward zero -- as are zero ages from timestamp ties.

    Returns None when no context token survives -- on a graph whose queries are all cold there is
    no age distribution to measure. The caller decides what to use instead.

    The result depends on the walk configuration -- num_walks_per_node, max_walk_len and the
    biases all move the age distribution -- so a sweep over those also moves the scale. Record
    it alongside them.
    """
    toks = build_query_walk_tokens(
        walk_gen, device, query_seeds, query_cutoffs,
        max_walk_len=max_walk_len, num_walks_per_node=num_walks_per_node,
        start_bias=start_bias, walk_bias=walk_bias)
    ages = toks.ages[toks.mask & ~toks.seed_mask]        # context slots only
    ages = ages[ages > 0]                                # drop zero-gap ties
    if int(ages.numel()) == 0:
        return None
    # int64 median: ages reach ~1e9 and float32 holds only ~1.7e7 integers exactly. torch.median
    # is the lower order statistic on even counts (not numpy's midpoint average), so the scale is
    # always a real observed age.
    return float(torch.median(ages).item())
