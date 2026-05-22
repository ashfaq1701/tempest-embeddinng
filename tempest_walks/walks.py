"""Tempest walk generator (strict-causal API).

Walks are sampled for EXPLICIT seed nodes via
`get_random_walks_and_times_for_nodes`. That way the caller controls
ingest timing — the trainer can sample walks from the PRE-ingest
state (current batch's edges not yet in Tempest) and ingest the batch
afterward. No `for_last_batch` path is exposed: it tempts the
order-leak that destroyed the v1/v2 numbers.

Walk layout (chronological, per Tempest convention):
  nodes:      [n_0, n_1, ..., n_{lens-2}, n_{lens-1}, -1, -1, ...]
  timestamps: [t_0, t_1, ..., t_{lens-2}, sentinel,   -1, -1, ...]
  - nodes[lens-1]    = the SEED (Tempest reverses backward walks
                       in place before returning).
  - timestamps[k]    for k ∈ [0, lens-2] = time of edge between
                       nodes[k] and nodes[k+1].
  - timestamps[lens-1] = INT64_MAX sentinel.
  - positions ≥ lens are padding (-1).
"""

from typing import NamedTuple, Optional

import numpy as np
import torch
from temporal_random_walk import TemporalRandomWalk


class WalkData(NamedTuple):
    """Per-walk arrays, all torch tensors ready for the model."""
    nodes: torch.Tensor          # [W, L] int32, padding=-1
    timestamps: torch.Tensor     # [W, L] int64, sentinel at lens-1
    lens: torch.Tensor           # [W] int64
    edge_feats: Optional[torch.Tensor]  # [W, L-1, d_edge] float32 or None
    seeds: torch.Tensor          # [N] int64, the unique seed node IDs passed in
    K: int                       # walks per seed (so reshape works: W == N*K)


class WalkGenerator:
    def __init__(
        self,
        is_directed: bool,
        use_gpu: bool,
        walk_bias: str = "ExponentialWeight",
        max_walk_len: int = 20,
        num_walks_per_node: int = 5,
        timescale_bound: int = 300,
        seed: Optional[int] = None,
    ):
        # CRITICAL: shuffle_walk_order defaults to True in Tempest's
        # constructor (see temporal_random_walk/src/common/const.cuh
        # DEFAULT_SHUFFLE_WALK_ORDER). When True, Tempest randomly
        # interleaves the [N*K, L] output across all seeds, so the row
        # at index i*K does NOT correspond to seed i. The downstream
        # code (alignment_loss's repeat_interleave at losses.py, the
        # walk-encoder's view(N, K, d) reshape, and the seed→row map
        # in _compute_walk_repr_for) all assume grouped order. We
        # disable the shuffle so the output is laid out as
        # [seed_0×K, seed_1×K, ..., seed_{N-1}×K]. See Lesson 28.
        # `global_seed` (Lesson 33) makes Tempest's internal RNG
        # deterministic across runs with the same Python/torch seed;
        # without it, walks differ run-to-run even when the rest of the
        # pipeline is seeded.
        trw_kwargs = dict(
            is_directed=is_directed,
            use_gpu=use_gpu,
            enable_weight_computation=True,
            timescale_bound=timescale_bound,
            shuffle_walk_order=False,
        )
        if seed is not None:
            trw_kwargs["global_seed"] = int(seed)
        self.trw = TemporalRandomWalk(**trw_kwargs)
        self.walk_bias = walk_bias
        self.max_walk_len = max_walk_len
        self.num_walks_per_node = num_walks_per_node

    def reset(self) -> None:
        """Drop all ingested edges. Call once at the start of each training epoch."""
        self.trw.clear()

    def add_edges(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        ts: np.ndarray,
        edge_feat: Optional[np.ndarray] = None,
    ) -> None:
        """Ingest a batch of edges. Must be called AFTER scoring (strict-causal)."""
        self.trw.add_multiple_edges(src, tgt, ts, edge_features=edge_feat)

    def walks_for_nodes(self, seeds: np.ndarray) -> WalkData:
        """Sample backward walks for the given seed nodes from the CURRENT state."""
        seed_arr = np.ascontiguousarray(seeds, dtype=np.int32)
        nodes, ts, lens, ef = self.trw.get_random_walks_and_times_for_nodes(
            seed_nodes=seed_arr,
            max_walk_len=self.max_walk_len,
            walk_bias=self.walk_bias,
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
