"""Tempest walk sampler wrapper.

Responsibility:
  - Construct a TemporalRandomWalk instance with the right config.
  - Provide walks_for_nodes(seed_nodes) returning the standard
    (nodes, timestamps, lens, edge_feats) tuple with seed at
    position lens-1.
  - Provide add_edges(...) for post-batch ingest (strict-causal).
  - Provide reset() for epoch-boundary state clear.

Walk layout (Tempest convention):
  nodes:      [n_0, n_1, ..., n_{lens-1}, padding...]
  timestamps: [t_1, t_2, ..., sentinel, padding...]
              where t_k = time of edge between nodes[k-1] and nodes[k]
  seed:       nodes[lens-1]
  Edge feature attached to context at position p (under convention β,
  edge-toward-seed): timestamps and edge_feats index p+1, i.e. the
  edge LEAVING that context toward the seed.

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
    ):
        # shuffle_walk_order=False is non-negotiable: the K-contiguous
        # row grouping is what every downstream caller assumes. If a
        # future Tempest version drops this kwarg, the call below will
        # raise and the architecture must be rebuilt around the new
        # layout; do NOT silently proceed.
        self.trw = TemporalRandomWalk(
            is_directed=is_directed,
            use_gpu=use_gpu,
            enable_weight_computation=True,
            timescale_bound=timescale_bound,
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

    def walks_for_nodes(self, seeds: np.ndarray) -> WalkData:
        """Sample backward walks for the given seed nodes from the CURRENT state.

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
            walk_direction="Backward_In_Time",
        )
        return WalkData(
            nodes=torch.from_numpy(nodes),
            timestamps=torch.from_numpy(ts),
            lens=torch.from_numpy(lens).to(torch.int64),
            edge_feats=torch.from_numpy(ef) if ef is not None else None,
            seeds=torch.from_numpy(seed_arr.astype(np.int64)),
            K=self.num_walks_per_node,
        )
