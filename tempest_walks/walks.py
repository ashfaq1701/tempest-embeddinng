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
    nodes: torch.Tensor        # [N*K, L] int64, padding = -1
    timestamps: torch.Tensor   # [N*K, L] int64; timestamps[p] = time of edge
                               # (nodes[p], nodes[p+1]); INT64_MAX sentinel at
                               # the seed slot (lens-1); padding = -1
    lens: torch.Tensor         # [N*K] int64
    seeds: torch.Tensor        # [N] int64
    K: int                     # walks per seed
    edge_feats: Optional[torch.Tensor] = None
                               # [N*K, L, d_ef] float32, or None when the dataset
                               # carries no edge features. INDEX-ALIGNED with nodes
                               # / timestamps: edge_feats[p] is the feature of the
                               # SAME edge (nodes[p], nodes[p+1]) whose time is
                               # timestamps[p], for p in [0, lens-2]. The seed slot
                               # (p = lens-1) and padding (p >= lens) are ZERO —
                               # Tempest returns [N*K, L-1, d_ef] (no seed-slot
                               # row); we right-pad one zero column so the context
                               # mask (positions < lens-1) selects exactly the real
                               # edge-feature rows. (Pairing verified against the
                               # walk contract in tests/test_walk_edge_feats.py.)


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

    def node_degrees(self, nodes: np.ndarray) -> np.ndarray:
        """Total degree (undirected ⇒ u's edit count) of each node in the CURRENT
        pre-ingest graph. Strict-causal: query before add_edges, same as the walks."""
        return self.trw.get_node_degrees(
            np.ascontiguousarray(nodes, dtype=np.int32), "Backward_In_Time")

    def add_edges(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray,
                  edge_feat: Optional[np.ndarray] = None) -> None:
        """Ingest a batch of edges. STRICT-CAUSAL: call AFTER scoring."""
        self.trw.add_multiple_edges(src, tgt, ts, edge_features=edge_feat)

    def walks_for_nodes(self, seeds: np.ndarray, max_walk_len: Optional[int] = None,
                        num_walks_per_node: Optional[int] = None,
                        start_bias: Optional[str] = None,
                        walk_bias: Optional[str] = None) -> WalkData:
        """K BACKWARD walks per seed. ``nodes`` is [N*K, L] with rows
        [i*K, (i+1)*K) = seed i's walks; seed at lens-1, padding = -1.

        Walk length / count / start-bias / walk-bias default to the instance values
        but accept per-call overrides — the query is read-only over the SAME ingested
        graph, so query-side and candidate-side walks (different lengths/biases) reuse
        this one generator instead of a second Tempest instance."""
        mwl = self.max_walk_len if max_walk_len is None else int(max_walk_len)
        nw = self.num_walks_per_node if num_walks_per_node is None else int(num_walks_per_node)
        sb = self.start_bias if start_bias is None else start_bias
        wb = self.walk_bias if walk_bias is None else walk_bias
        seed_arr = np.ascontiguousarray(seeds, dtype=np.int32)
        nodes, ts, lens, ef = self.trw.get_random_walks_and_times_for_nodes(
            seed_nodes=seed_arr,
            max_walk_len=mwl,
            walk_bias=wb,
            initial_edge_bias=sb,
            num_walks_per_node=nw,
            walk_direction="Backward_In_Time",
        )
        nodes_t = torch.from_numpy(np.asarray(nodes).astype(np.int64))

        # Edge features (when the dataset has them). Tempest returns
        # [N*K, L-1, d_ef], one column SHORTER than nodes, aligned so ef[p] is the
        # feature of the edge (nodes[p], nodes[p+1]) — the same edge as
        # timestamps[p] — for p in [0, lens-2]; it has no row for the seed slot and
        # its tail/padding rows are zero. We right-pad the L-1 axis back up to L
        # (one zero column) so edge_feats indexes 1:1 with nodes / timestamps and
        # the existing context mask (positions < lens-1) selects exactly the real
        # edges. When no edge features were ingested, Tempest hands back an empty
        # object array (ndim 0) -> edge_feats stays None.
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
