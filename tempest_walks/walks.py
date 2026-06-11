"""Tempest walk sampler wrapper.

One ``TemporalRandomWalk`` instance holds the streaming temporal graph. The
link head samples K BACKWARD walks per node (graphs treated as undirected);
the seed sits at row position ``lens-1``, the chronologically oldest
predecessor at position 0, padding = -1. Rows ``[i*K, (i+1)*K)`` are seed i's
K walks (``shuffle_walk_order=False`` pins this grouping).

Ordering contract: ``reset()`` per epoch, ``walks_for_nodes()`` at scoring
(pre-ingest state), ``add_edges()`` AFTER scoring (strict-causal).
"""
from typing import NamedTuple, Optional

import numpy as np
import torch
from temporal_random_walk import TemporalRandomWalk


class WalkData(NamedTuple):
    nodes: torch.Tensor   # [N*K, L] int64, padding = -1
    lens: torch.Tensor    # [N*K] int64
    seeds: torch.Tensor   # [N] int64
    K: int                # walks per seed


class WalkGenerator:
    def __init__(
        self,
        use_gpu: bool = False,
        walk_bias: str = "ExponentialWeight",
        start_bias: str = "ExponentialWeight",
        num_walks_per_node: int = 5,
        max_walk_len: int = 20,
        timescale_bound: int = 300,
        max_time_capacity: int = -1,
    ):
        self.trw = TemporalRandomWalk(
            is_directed=False,
            use_gpu=use_gpu,
            enable_weight_computation=True,
            timescale_bound=timescale_bound,
            max_time_capacity=max_time_capacity,
            shuffle_walk_order=False,
        )
        self.walk_bias = walk_bias
        self.start_bias = start_bias
        self.num_walks_per_node = int(num_walks_per_node)
        self.max_walk_len = int(max_walk_len)

    def reset(self) -> None:
        """Drop all ingested edges. Call at the start of each epoch."""
        self.trw.clear()

    def add_edges(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray,
                  edge_feat: Optional[np.ndarray] = None) -> None:
        """Ingest a batch of edges. STRICT-CAUSAL: call AFTER scoring."""
        self.trw.add_multiple_edges(src, tgt, ts, edge_features=edge_feat)

    def walks_for_nodes(self, seeds: np.ndarray) -> WalkData:
        """K BACKWARD walks per seed. ``nodes`` is [N*K, L] with rows
        [i*K, (i+1)*K) = seed i's walks; seed at lens-1, padding = -1."""
        seed_arr = np.ascontiguousarray(seeds, dtype=np.int32)
        nodes, _ts, lens, _ef = self.trw.get_random_walks_and_times_for_nodes(
            seed_nodes=seed_arr,
            max_walk_len=self.max_walk_len,
            walk_bias=self.walk_bias,
            initial_edge_bias=self.start_bias,
            num_walks_per_node=self.num_walks_per_node,
            walk_direction="Backward_In_Time",
        )
        return WalkData(
            nodes=torch.from_numpy(np.asarray(nodes).astype(np.int64)),
            lens=torch.from_numpy(np.asarray(lens).astype(np.int64)),
            seeds=torch.from_numpy(seed_arr.astype(np.int64)),
            K=self.num_walks_per_node,
        )
