"""Per-query FLATTENED walk tensors (nodes + node-aligned ages + hop positions), from Tempest.

A "query" is a (seed node, cutoff time t) pair. `build_query_walk_tokens` runs K backward walks
per query, each bounded by cutoff = t (every edge has t_edge < t — strict causal past), and returns
them ALREADY FLATTENED into one [Q, T] token bag per query (T = K·L). Nothing consumes the raw
per-walk [Q, K, L] layout, so the K/L structure is collapsed here once; heads read the flat fields
directly (no separate flatten step).

    seeds          [Q]          int64  query/source node u (walk origin; present even when cold)
    cutoffs        [Q]          int64  each query's exclusive cutoff t (kept so ages can be re-derived)
    nodes          [Q, T]       int64  flattened walk node ids; padding = -1 (clamp before embedding)
    ages           [Q, T]       int64  NODE-ALIGNED age = cutoff − t_edge (query-relative): a non-seed
                                       node's age is (cutoff − time of the edge that reached it), ≥ 1
                                       since the cutoff is exclusive; the SEED sits at "now" → age 0;
                                       padding = -1.
    positions      [Q, T]       int64  within-walk HOP from the seed: 1 = seed (walk end), 2 = its
                                       immediate predecessor, …, lens = oldest node; 0 on padding.
    mask [Q, T]       bool   True on real walk positions (nodes != -1) — INCLUDING the seed.
    seed_mask      [Q, T]       bool   True ONLY on the seed's own walk-origin slot (age 0 & real).
    seed_node_mask [Q, T]       bool   True on EVERY real slot whose node == the seed node, i.e. the origin
                                       slot AND any mid-walk recurrence of the seed. Use `mask & ~seed_node_mask`
                                       to drop ALL self-node slots (each contributes Log_{E[u]}(E[u])=0 to μ but
                                       still consumes softmax weight, diluting μ toward 0).
    edge_features  [Q, T, d_ef] float  flattened per-token edge features; the seed slot and padding
                                       carry [0]*d_ef. None if the dataset has no edge features.

Requires shuffle_walk_order=False at the Tempest constructor so a query's K walk rows are contiguous
(rows [q*K, (q+1)*K)) and reshape cleanly before flattening.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# INT64_MAX — Tempest's raw sentinel at the seed slot (lens-1) of every walk (no outgoing edge there).
_TS_SENTINEL = torch.iinfo(torch.int64).max


@dataclass
class WalkTokens:
    """Per-query FLATTENED backward walks (see module docstring). Every tensor except `seeds` /
    `cutoffs` is [Q, T] (T = K·L), or [Q, T, d_ef] for `edge_features`."""

    seeds: torch.Tensor                             # [Q]
    cutoffs: torch.Tensor                           # [Q]
    nodes: torch.Tensor                             # [Q, T]
    ages: torch.Tensor                              # [Q, T]
    positions: torch.Tensor                         # [Q, T]
    mask: torch.Tensor                    # [Q, T]  bool
    seed_mask: torch.Tensor                         # [Q, T]  bool  True ONLY on the seed's origin slot
    seed_node_mask: torch.Tensor                    # [Q, T]  bool  True on EVERY slot whose node == seed node
    edge_features: Optional[torch.Tensor] = None    # [Q, T, d_ef]

    @property
    def weight_logits(self) -> torch.Tensor:
        """Fixed (non-learned) shared per-token pooling weight LOGITS [Q, T] — the recency/hop prior

            log_weight = -( log1p(age) + log1p(hop - 1) )

        maximal (= 0) for a just-happened, adjacent token, decaying with both age and hop. These are
        LOG-weights: a pooling channel masked_fills its non-context slots to -inf and softmaxes, so the
        most RECENT & CLOSEST token gets the highest weight. Convention: age = 0 at the seed / ≥1 for
        context; hop = 1 at the seed / ≥2 for the closest context (so hop-1 = 0 at the seed, ≥1 for
        context). Returned in float32 — cast to the head's dtype at the call site if it differs. The
        SAME prior weights both pooling channels (geometry and features) and the alignment loss."""
        age = self.ages.clamp_min(0).to(torch.float32)                                 # [Q, T]  seed=0, ctx≥1
        hop = self.positions.clamp_min(1).to(torch.float32)                            # [Q, T]  seed=1, ctx≥2
        return -(torch.log1p(age) + torch.log1p(hop - 1.0))                            # [Q, T]  ≤ 0


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
    """Per-query backward walks → FLATTENED [Q, T] nodes + ages + positions + masks (see WalkTokens)."""
    seeds_t = query_seeds.detach().to(device=device, dtype=torch.long)        # [Q]
    cutoffs_t = query_cutoffs.detach().to(device=device, dtype=torch.long)    # [Q]
    q = int(seeds_t.shape[0])

    if q == 0:
        t = num_walks_per_node * max_walk_len
        empty_i = torch.empty((0, t), dtype=torch.int64, device=device)
        empty_b = torch.empty((0, t), dtype=torch.bool, device=device)
        return WalkTokens(seeds_t, cutoffs_t, empty_i, empty_i, empty_i, empty_b, empty_b, empty_b)

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

    # Rows [q*K, (q+1)*K) are query q's K walks ⇒ reshape [Q*K, L] -> [Q, K, L] to build per-walk fields.
    nodes = wd.nodes.to(device=device, dtype=torch.int64).reshape(q, k, length)        # [Q, K, L]
    node_mask = nodes != -1                                                             # [Q, K, L]

    # Node-aligned AGE = cutoff − t_edge (query-relative). Each non-seed node's edge time is < cutoff
    # (exclusive) → age ≥ 1; the seed slot (Tempest sentinel) is "now" → age 0; padding → -1.
    ts = wd.timestamps.to(device=device, dtype=torch.int64).reshape(q, k, length)      # [Q, K, L] raw edge time
    edge_ts = torch.where(ts == _TS_SENTINEL, cutoffs_t.view(q, 1, 1), ts)             # seed sentinel → cutoff
    ages = cutoffs_t.view(q, 1, 1) - edge_ts                                           # seed → 0, edges ≥ 1
    ages = torch.where(node_mask, ages, torch.full_like(ages, -1))                     # padding → -1

    # HOP position from the seed: seed (last real slot, index lens-1) = 1, oldest (index 0) = lens;
    # padding → 0. Backward walks are stored oldest→seed, so pos = lens − array_index.
    lens = node_mask.sum(dim=-1, keepdim=True)                                          # [Q, K, 1] real length
    arange = torch.arange(length, device=device).view(1, 1, length)
    positions = (lens - arange).clamp_min(0)                                            # [Q, K, L]

    seed_mask = node_mask & (ages == 0)                                                 # [Q, K, L] seed origin slot
    seed_node_mask = node_mask & (nodes == seeds_t.view(q, 1, 1))                        # [Q, K, L] ANY seed-node slot

    # Per-token edge features → [Q, T, d_ef]. wd.edge_feats is [N*K, L, d_ef] node-aligned; the seed
    # slot (age 0) and padding are forced to [0]*d_ef so they carry no edge. None if the dataset has none.
    edge_features = None
    if wd.edge_feats is not None:
        d_ef = int(wd.edge_feats.shape[-1])
        ef = wd.edge_feats.to(device=device, dtype=torch.float32).reshape(q, k, length, d_ef)
        real = (node_mask & (ages != 0)).unsqueeze(-1)                                  # [Q, K, L, 1] non-seed, non-pad
        edge_features = (ef * real).reshape(q, k * length, d_ef)                        # [Q, T, d_ef]

    # ── Flatten K·L → T for every field. ──
    return WalkTokens(
        seeds_t,
        cutoffs_t,
        nodes.reshape(q, -1),               # [Q, T]
        ages.reshape(q, -1),                # [Q, T]
        positions.reshape(q, -1),           # [Q, T]
        node_mask.reshape(q, -1),           # [Q, T]  mask
        seed_mask.reshape(q, -1),           # [Q, T]  seed_mask
        seed_node_mask.reshape(q, -1),      # [Q, T]  seed_node_mask
        edge_features,                      # [Q, T, d_ef] or None
    )
