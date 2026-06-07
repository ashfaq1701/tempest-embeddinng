"""Tempest walk sampler wrapper.

Responsibility:
  - Construct ONE TemporalRandomWalk instance (Tempest holds the
    ingested-edge state; both directions query the same state).
  - Provide `walks_for_nodes_embedding_backward(seeds)` — the single
    direction consumed by the alignment loss to shape the embedding
    table E. (Forward embedding alignment was ablated and dropped:
    the 2026-06-07 sweep on tgbl-wiki showed backward-only beats
    forward+backward by +0.009 test, outside the noise band.)
  - Provide `walks_for_nodes_link_pred_forward(seeds)` and
    `walks_for_nodes_link_pred_backward(seeds)` — the two directions
    consumed by the walk-mediated link-prediction head.
  - Provide add_edges(...) for post-batch ingest (strict-causal).
  - Provide reset() for epoch-boundary state clear.

Walk layout — BACKWARD (embedding side, seeds = batch targets):
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

Walk layout — FORWARD (link-pred side, seeds = batch sources):
  nodes:      [n_0, n_1, ..., n_{lens-1}, padding(-1)...]
              Chronological — n_0 is the seed, n_{lens-1} is the
              chronologically latest successor.
  timestamps: [INT64_MIN, t_1, ..., t_{lens-1}, padding(-1)...]
              t_p for p in [1, lens-1] = time of the edge
              (nodes[p-1], nodes[p]) — edge INTO nodes[p].
              t_0 = INT64_MIN sentinel (seed has no incoming edge
              in the walk).
  edge_feats: shape [NK, max_walk_len - 1, d_ef] or None.
              ef[q] = feature of edge (nodes[q], nodes[q+1]) for
              q in [0, lens-2] — SAME rule as backward. Note this
              differs from t's attachment: forward ef[q] aligns
              with t[q+1] (off-by-one), backward ef[q] aligns
              with t[q].
  seed:       nodes[0]

Convention β (edge-toward-seed) attaches walks.edge_feats[p] to
context at walk position p as the edge OUT of that context toward
the seed (backward direction). Loss-side code in alignment_loss
right-pads ef by one row so the projection sees shape [NK, L, d_ef]
aligned with e_ctx.

Grouping contract (load-bearing for BOTH directions):
  Walks for seeds[i] occupy rows [i*K, (i+1)*K) of nodes/timestamps/
  edge_feats. Enforced via shuffle_walk_order=False at construction;
  every downstream caller (alignment loss reshape, link-head batching)
  assumes this layout.

CLI-exposed knobs (passed through from train.py):
  embedding side (backward only):
    embedding_backward_walk_bias, embedding_backward_start_bias,
    embedding_num_walks_per_node, embedding_max_walk_len.
  link-pred side (BOTH directions, per-direction biases):
    link_pred_forward_walk_bias,  link_pred_forward_start_bias,
    link_pred_backward_walk_bias, link_pred_backward_start_bias,
    link_pred_num_walks_per_node, link_pred_max_walk_len.

The num_walks_per_node stored on the WalkGenerator is the per-call
K (one direction). Every walks_for_nodes_*_{forward,backward} call
returns exactly that many rows per seed.
"""

from typing import NamedTuple, Optional

import numpy as np
import torch
from temporal_random_walk import TemporalRandomWalk


class WalkData(NamedTuple):
    """Per-walk arrays. All tensors are returned on CPU; callers move
    them to GPU as needed."""
    nodes: torch.Tensor              # [N*K, L_max] int32, padding=-1
    timestamps: torch.Tensor         # [N*K, L_max] int64; sentinel INT64_MAX
                                     # at lens-1 for backward, INT64_MIN at
                                     # 0 for forward; padding=-1.
    lens: torch.Tensor               # [N*K] int64
    edge_feats: Optional[torch.Tensor]  # [N*K, L_max-1, d_edge] float32 or None
    seeds: torch.Tensor              # [N] int64
    K: int                           # walks per seed


class WalkGenerator:
    def __init__(
        self,
        is_directed: bool,
        use_gpu: bool = False,
        # Embedding side — backward walks only (forward was ablated).
        embedding_backward_walk_bias:  str = "ExponentialWeight",
        embedding_backward_start_bias: str = "ExponentialWeight",
        embedding_num_walks_per_node: int = 10,
        embedding_max_walk_len: int = 20,
        # Link-pred side. Both directions, per-direction biases.
        # Defaults reflect the temporal structure of useful walks:
        #   forward  start=Uniform, walk=ExpW
        #     ExpW+ExpW from the source would shoot the walk toward
        #     the chronological HEAD (oldest end) of the seed's
        #     successor set, which is least predictive. Uniform start
        #     spreads coverage across the successor set; ExpW walk
        #     then biases continuations toward recency relative to
        #     the previous hop.
        #   backward start=ExpW,    walk=ExpW
        #     ExpW+ExpW from the target traces the TAIL (most recent
        #     incoming chain) — the most predictive predecessors.
        link_pred_forward_walk_bias:  str = "ExponentialWeight",
        link_pred_forward_start_bias: str = "Uniform",
        link_pred_backward_walk_bias:  str = "ExponentialWeight",
        link_pred_backward_start_bias: str = "ExponentialWeight",
        link_pred_num_walks_per_node: int = 5,
        link_pred_max_walk_len: int = 20,
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
        self.embedding_backward_walk_bias  = embedding_backward_walk_bias
        self.embedding_backward_start_bias = embedding_backward_start_bias
        self.embedding_num_walks_per_node = embedding_num_walks_per_node
        self.embedding_max_walk_len = embedding_max_walk_len
        self.link_pred_forward_walk_bias  = link_pred_forward_walk_bias
        self.link_pred_forward_start_bias = link_pred_forward_start_bias
        self.link_pred_backward_walk_bias  = link_pred_backward_walk_bias
        self.link_pred_backward_start_bias = link_pred_backward_start_bias
        self.link_pred_num_walks_per_node = link_pred_num_walks_per_node
        self.link_pred_max_walk_len = link_pred_max_walk_len

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

    def _walks(
        self,
        seeds: np.ndarray,
        walk_bias: str,
        start_bias: str,
        direction: str,
        num_walks_per_node: int,
        max_walk_len: int,
    ) -> WalkData:
        seed_arr = np.ascontiguousarray(seeds, dtype=np.int32)
        nodes, ts, lens, ef = self.trw.get_random_walks_and_times_for_nodes(
            seed_nodes=seed_arr,
            max_walk_len=max_walk_len,
            walk_bias=walk_bias,
            initial_edge_bias=start_bias,
            num_walks_per_node=num_walks_per_node,
            walk_direction=direction,
        )

        # Fail loud if Tempest's edge_feats shape no longer matches the
        # [NK, L-1, d_ef] convention every downstream consumer assumes.
        if ef is not None:
            N = seed_arr.shape[0]
            expected_2d = (N * num_walks_per_node, max_walk_len - 1)
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
            K=num_walks_per_node,
        )

    def walks_for_nodes_embedding_backward(self, seeds: np.ndarray) -> WalkData:
        """Sample BACKWARD walks for the embedding-side alignment loss.

        Seed at row position lens-1; chronologically oldest predecessor
        at position 0. timestamps[i, lens-1] = INT64_MAX sentinel.

        Returns a WalkData with walks grouped K-per-seed in input order:
        rows [i*K, (i+1)*K) contain seeds[i]'s K walks.
        """
        return self._walks(
            seeds,
            walk_bias=self.embedding_backward_walk_bias,
            start_bias=self.embedding_backward_start_bias,
            direction="Backward_In_Time",
            num_walks_per_node=self.embedding_num_walks_per_node,
            max_walk_len=self.embedding_max_walk_len,
        )

    def walks_for_nodes_link_pred_forward(self, seeds: np.ndarray) -> WalkData:
        """Sample FORWARD walks for the link-prediction side.

        Seed at row position 0; chronologically latest successor at
        position lens-1. timestamps[i, 0] = INT64_MIN sentinel.

        Used by the iter-6 forward-alignment loss term; also reserved
        for a future link-prediction-side scoring head.
        """
        return self._walks(
            seeds,
            walk_bias=self.link_pred_forward_walk_bias,
            start_bias=self.link_pred_forward_start_bias,
            direction="Forward_In_Time",
            num_walks_per_node=self.link_pred_num_walks_per_node,
            max_walk_len=self.link_pred_max_walk_len,
        )

    def walks_for_nodes_link_pred_backward(self, seeds: np.ndarray) -> WalkData:
        """Sample BACKWARD walks for the link-prediction side.

        Seed at row position lens-1; chronologically oldest predecessor
        at position 0. timestamps[i, lens-1] = INT64_MAX sentinel.

        Reserved for a future link-prediction-side scoring path; no
        caller wired into the trainer yet.
        """
        return self._walks(
            seeds,
            walk_bias=self.link_pred_backward_walk_bias,
            start_bias=self.link_pred_backward_start_bias,
            direction="Backward_In_Time",
            num_walks_per_node=self.link_pred_num_walks_per_node,
            max_walk_len=self.link_pred_max_walk_len,
        )
