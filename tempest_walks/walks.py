"""Tempest walk sampler wrapper.

Responsibility:
  - Construct a TemporalRandomWalk instance with the right config.
  - Provide walks_for_nodes(seed_nodes) returning the standard
    (nodes, timestamps, lens, edge_feats) tuple with seed at
    position lens-1.
  - Provide add_edges(...) for post-batch ingest (strict-causal).
  - Provide reset() for epoch-boundary state clear.

Walk layout (Tempest convention; verified empirically against the
TemporalRandomWalk source):
  nodes:      [n_0, n_1, ..., n_{lens-1}, padding(-1)...]
              Chronological — n_0 is the oldest predecessor, n_{lens-1}
              is the seed.
  timestamps: [t_0, t_1, ..., t_{lens-2}, INT64_MAX, padding(-1)...]
              t_p = time of the edge between nodes[p] and nodes[p+1]
              (the OUTGOING edge from nodes[p] in chronological time,
              equivalently the edge toward the seed).
              t_{lens-1} = INT64_MAX sentinel (seed has no outgoing edge).
  edge_feats: shape [NK, max_walk_len - 1, d_ef] or None.
              ef[p] is the FEATURE of the same edge whose time is
              timestamps[p]: the edge between nodes[p] and nodes[p+1].
              ef and timestamps are aligned at the same index p.
              Slots p >= lens-1 are zeros (Tempest fills empty slots).
  seed:       nodes[lens-1]

Convention β (edge-toward-seed) attaches walks.edge_feats[p] to
context at walk position p as the edge OUT of that context toward
the seed. Loss-side code in alignment_loss right-pads ef by one row
so the projection sees shape [NK, L, d_ef] aligned with e_ctx.

Grouping contract (load-bearing):
  Walks for seeds[i] occupy rows [i*K, (i+1)*K) of nodes/timestamps/
  edge_feats. Enforced via shuffle_walk_order=False at construction;
  every downstream caller (alignment loss reshape, link-head batching)
  assumes this layout.

CLI-exposed knobs (passed through from train.py):
  num_walks_per_node, max_walk_len, walk_bias, start_bias.
"""

from typing import NamedTuple, Optional

import numpy as np
import torch
from temporal_random_walk import TemporalRandomWalk


class WalkData(NamedTuple):
    """Per-walk arrays. All tensors are returned on CPU; callers move
    them to GPU as needed."""
    nodes: torch.Tensor              # [N*K, L_max] int32, padding=-1
    timestamps: torch.Tensor         # [N*K, L_max] int64, sentinel INT64_MAX at lens-1, padding=-1
    lens: torch.Tensor               # [N*K] int64
    edge_feats: Optional[torch.Tensor]  # [N*K, L_max-1, d_edge] float32 or None
    seeds: torch.Tensor              # [N] int64
    K: int                           # walks per seed


class WalkGenerator:
    def __init__(
        self,
        is_directed: bool,
        use_gpu: bool = False,
        walk_bias: str = "ExponentialWeight",
        start_bias: str = "Uniform",
        max_walk_len: int = 20,
        num_walks_per_node: int = 5,
        timescale_bound: int = 300,
        max_time_capacity: int = -1,
    ):
        # shuffle_walk_order=False is non-negotiable: the K-contiguous
        # row grouping is what every downstream caller assumes. If a
        # future Tempest version drops this kwarg, the call below will
        # raise and the architecture must be rebuilt around the new
        # layout; do NOT silently proceed.
        #
        # max_time_capacity: sliding-window eviction in raw timestamp
        # units. Tempest tracks the max ingested timestamp and removes
        # any edge with ts < (latest - max_time_capacity) on every
        # add_multiple_edges call. -1 = unbounded (keep every ingested
        # edge until walk_gen.reset() at epoch boundary).
        self.trw = TemporalRandomWalk(
            is_directed=is_directed,
            use_gpu=use_gpu,
            enable_weight_computation=True,
            timescale_bound=timescale_bound,
            max_time_capacity=max_time_capacity,
            shuffle_walk_order=False,
        )
        self.walk_bias = walk_bias
        self.start_bias = start_bias
        self.max_walk_len = max_walk_len
        self.num_walks_per_node = num_walks_per_node

    def reset(self) -> None:
        """Drop all ingested edges. Call at start of each training epoch."""
        self.trw.clear()

    def add_edges(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        ts: np.ndarray,
        edge_feat: Optional[np.ndarray] = None,
    ) -> None:
        """Ingest a batch of edges. STRICT-CAUSAL: call AFTER scoring."""
        self.trw.add_multiple_edges(src, tgt, ts, edge_features=edge_feat)

    def walks_for_nodes(
        self,
        seeds: np.ndarray,
        walk_direction: str = "Backward_In_Time",
    ) -> WalkData:
        """Sample walks for the given seed nodes from the CURRENT state.

        walk_direction:
          "Backward_In_Time" — seed at position lens-1 (default).
          "Forward_In_Time"  — seed at position 0 (Tempest mirror;
                                contract verified, see CLAUDE.md).

        Returns a WalkData with walks grouped K-per-seed in input order:
        rows [i*K, (i+1)*K) contain seeds[i]'s K walks.
        """
        seed_arr = np.ascontiguousarray(seeds, dtype=np.int32)
        nodes, ts, lens, ef = self.trw.get_random_walks_and_times_for_nodes(
            seed_nodes=seed_arr,
            max_walk_len=self.max_walk_len,
            walk_bias=self.walk_bias,
            initial_edge_bias=self.start_bias,
            num_walks_per_node=self.num_walks_per_node,
            walk_direction=walk_direction,
        )

        # Fail loud if Tempest's edge_feats shape no longer matches the
        # [NK, L-1, d_ef] convention the loss code assumes. A future
        # Tempest version-skew that returns [NK, L, d_ef] would silently
        # mis-align edge features with positions by one step under
        # convention β.
        if ef is not None:
            N = seed_arr.shape[0]
            expected_2d = (N * self.num_walks_per_node, self.max_walk_len - 1)
            assert ef.shape[:2] == expected_2d, (
                f"Tempest edge_feats shape {ef.shape[:2]} != expected "
                f"{expected_2d} (NK, L-1). The convention-β attachment in "
                f"alignment_loss assumes ef[p] is the edge between nodes[p] "
                f"and nodes[p+1]; revisit if Tempest output changed."
            )

        return WalkData(
            nodes=torch.from_numpy(nodes),
            timestamps=torch.from_numpy(ts),
            lens=torch.from_numpy(lens).to(torch.int64),
            edge_feats=torch.from_numpy(ef) if ef is not None else None,
            seeds=torch.from_numpy(seed_arr.astype(np.int64)),
            K=self.num_walks_per_node,
        )
