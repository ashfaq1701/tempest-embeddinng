"""Tempest walk sampler wrapper. Samples K BACKWARD walks per node.

Seed sits at row position ``lens-1``, oldest predecessor at 0, padding = -1.
Rows ``[i*K, (i+1)*K)`` are seed i's K walks (``shuffle_walk_order=False``).
Causality is per-query via ``cutoff_times``: a walk for (u, t) uses only edges
with t_edge < t (EXCLUSIVE).
"""
from typing import NamedTuple, Optional

import numpy as np
import torch
from tempest import Tempest


class WalkData(NamedTuple):
    nodes: torch.Tensor        # [N*K, L] int64, padding = -1
    timestamps: torch.Tensor   # [N*K, L] int64; timestamps[p] = time of edge
                               # (nodes[p], nodes[p+1]); INT64_MAX sentinel at
                               # the seed slot (lens-1); padding = -1
    lens: torch.Tensor         # [N*K] int64
    seeds: torch.Tensor        # [N] int64
    K: int                     # walks per seed
    edge_feats: Optional[torch.Tensor] = None
                               # [N*K, L, d_ef] float32, or None if no edge feats.
                               # edge_feats[p] pairs edge (nodes[p], nodes[p+1]) for
                               # p in [0, lens-2]; seed slot and padding are ZERO.


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
        temporal_node2vec_p: float = 4.0,
        temporal_node2vec_q: float = 0.25,
    ):
        # Build node2vec adjacency only when a node2vec bias is requested; p/q
        # are the return / in-out params, inert for other biases.
        enable_n2v = "TemporalNode2Vec" in (walk_bias, start_bias)
        self.tempest = Tempest(
            is_directed=False,
            use_gpu=use_gpu,
            enable_weight_computation=True,
            enable_temporal_node2vec=enable_n2v,
            temporal_node2vec_p=temporal_node2vec_p,
            temporal_node2vec_q=temporal_node2vec_q,
            timescale_bound=timescale_bound,
            max_time_capacity=max_time_capacity,
            shuffle_walk_order=False,
        )
        self.walk_bias = walk_bias
        self.start_bias = start_bias
        self.num_walks_per_node = int(num_walks_per_node)
        self.max_walk_len = int(max_walk_len)

    def add_edges(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray,
                  edge_feat: Optional[np.ndarray] = None) -> None:
        """Ingest edges into Tempest (indexed by time; ingestion order is irrelevant)."""
        self.tempest.add_multiple_edges(src, tgt, ts, edge_features=edge_feat)

    def participation_counts(self, nodes: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
        """Interactions of each node strictly before its cutoff t (int64), one per node. Undirected graph,
        so this is the total degree as-of-t; the exclusive cutoff excludes the query edge itself."""
        node_arr = np.ascontiguousarray(nodes, dtype=np.int32)
        cutoff_arr = np.ascontiguousarray(cutoffs, dtype=np.int64)
        if cutoff_arr.shape[0] != node_arr.shape[0]:
            raise ValueError("cutoffs must have the same length as nodes "
                             f"({cutoff_arr.shape[0]} vs {node_arr.shape[0]})")
        return self.tempest.get_node_participation_counts(
            node_arr, cutoff_times=cutoff_arr, direction="Backward_In_Time")

    def walks_for_nodes(self, seeds: np.ndarray, max_walk_len: Optional[int] = None,
                        num_walks_per_node: Optional[int] = None,
                        start_bias: Optional[str] = None,
                        walk_bias: Optional[str] = None,
                        cutoff_times: Optional[np.ndarray] = None) -> WalkData:
        """K BACKWARD walks per seed. ``nodes`` is [N*K, L] with rows
        [i*K, (i+1)*K) = seed i's walks; seed at lens-1, padding = -1.

        ``cutoff_times`` (int64, one per seed): walk uses only edges with
        t_edge < cutoff (EXCLUSIVE). None = unbounded. Length / count / start-bias /
        walk-bias default to instance values, override per-call."""
        mwl = self.max_walk_len if max_walk_len is None else int(max_walk_len)
        nw = self.num_walks_per_node if num_walks_per_node is None else int(num_walks_per_node)
        sb = self.start_bias if start_bias is None else start_bias
        wb = self.walk_bias if walk_bias is None else walk_bias
        seed_arr = np.ascontiguousarray(seeds, dtype=np.int32)
        cutoff_arr = None
        if cutoff_times is not None:
            cutoff_arr = np.ascontiguousarray(cutoff_times, dtype=np.int64)
            if cutoff_arr.shape[0] != seed_arr.shape[0]:
                raise ValueError(
                    "cutoff_times must have the same length as seeds "
                    f"({cutoff_arr.shape[0]} vs {seed_arr.shape[0]})")
        nodes, ts, lens, ef = self.tempest.get_random_walks_and_times_for_nodes(
            seed_nodes=seed_arr,
            max_walk_len=mwl,
            walk_bias=wb,
            initial_edge_bias=sb,
            num_walks_per_node=nw,
            walk_direction="Backward_In_Time",
            cutoff_times=cutoff_arr,
        )
        nodes_t = torch.from_numpy(np.asarray(nodes).astype(np.int64))

        # Tempest returns edge feats [N*K, L-1, d_ef] (one col shorter than nodes);
        # right-pad one zero column to L so it indexes 1:1 with nodes/timestamps.
        # No edge features -> empty array (ndim 0) -> edge_feats stays None.
        ef_arr = np.asarray(ef)
        edge_feats = None
        if ef_arr.ndim == 3 and ef_arr.size > 0:
            ef_t = torch.from_numpy(np.ascontiguousarray(ef_arr, dtype=np.float32))
            pad_cols = nodes_t.shape[1] - ef_t.shape[1]
            if pad_cols > 0:
                z = torch.zeros(ef_t.shape[0], pad_cols, ef_t.shape[2], dtype=ef_t.dtype)
                ef_t = torch.cat([ef_t, z], dim=1)            # [N*K, L, d_ef]
            edge_feats = ef_t

        return WalkData(
            nodes=nodes_t,
            timestamps=torch.from_numpy(np.asarray(ts).astype(np.int64)),
            lens=torch.from_numpy(np.asarray(lens).astype(np.int64)),
            seeds=torch.from_numpy(seed_arr.astype(np.int64)),
            K=nw,
            edge_feats=edge_feats,
        )
