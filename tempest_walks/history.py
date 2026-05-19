"""Per-node interaction-history buffers for the DyGFormer-style node encoder.

Each node maintains a sliding window of its last K_history interactions:

    history[u] = [ (neighbor_i, timestamp_i, edge_feat_i, role_i)  for i = 1..K ]

where role indicates whether u was the source (0) or destination (1) of that
interaction.

Storage convention (RIGHT-PADDED, most-recent-LAST among valid):
  Valid entries live at slots [0, valid_cnt[u] − 1].
  Slot valid_cnt[u] − 1 is the MOST RECENT event.
  Slot 0 is the OLDEST valid event.
  Slots [valid_cnt[u], K − 1] are padding.

  When buffer is full (valid_cnt[u] == K) and a new event arrives, we shift
  the contents left by 1 (drop slot 0, the oldest) and write the new event
  into slot K − 1.

Caller mask for "valid position p in row u": `p < valid_cnt[u]`.

Strict-causal contract:

  Per training/eval batch (BOTH phases, exact order):

      1. read_windows_for_nodes(unique_batch_nodes)  ← state ≤ batch B−1
      2. score the batch, compute losses
      3. write_events(batch.src, batch.tgt, batch.ts, batch.edge_feat)  ← LAST

  read_windows_for_nodes(...) MUST NOT be called between write_events(B-1) and
  write_events(B): if it is, batch B's reads would see B-1's events but not
  B's events, which is exactly what we want — but only if write is the
  LAST line of the per-batch loop and read is the FIRST. Caller responsibility.

  read_windows_for_nodes returns a *copy* of the relevant slice (so any
  in-place mutation downstream doesn't corrupt the buffer state).
"""

from typing import Optional, Tuple

import numpy as np


class NodeHistory:
    """Per-node sliding-window interaction history.

    Memory layout (fixed-size dense ring buffer):
      neighbors:  np.int64   [N, K]   neighbor IDs; -1 in padding slots
      timestamps: np.int64   [N, K]   times of events; -1 in padding
      edge_feats: np.float32 [N, K, d_e]   or None if dataset lacks edge feats
      roles:      np.int8    [N, K]   0 = u was src, 1 = u was dst; -1 padding
      valid_cnt:  np.int32   [N]      number of real entries in [0, K]

    All arrays live on CPU. Reads convert to torch tensors and move to GPU
    inside the model code, not here.
    """

    def __init__(
        self,
        n_nodes: int,
        K: int,
        edge_feat_dim: int = 0,
    ):
        if K <= 0:
            raise ValueError(f"K must be > 0, got {K}")
        self.n_nodes = int(n_nodes)
        self.K = int(K)
        self.edge_feat_dim = int(edge_feat_dim)

        # Initialised empty — no events have happened.
        self.neighbors = np.full((n_nodes, K), -1, dtype=np.int64)
        self.timestamps = np.full((n_nodes, K), -1, dtype=np.int64)
        if edge_feat_dim > 0:
            self.edge_feats: Optional[np.ndarray] = np.zeros(
                (n_nodes, K, edge_feat_dim), dtype=np.float32,
            )
        else:
            self.edge_feats = None
        self.roles = np.full((n_nodes, K), -1, dtype=np.int8)
        self.valid_cnt = np.zeros(n_nodes, dtype=np.int32)

    def reset(self) -> None:
        """Drop all events. Call at the start of each training epoch (mirrors
        walk_gen.reset() + reservoir.reset())."""
        self.neighbors.fill(-1)
        self.timestamps.fill(-1)
        if self.edge_feats is not None:
            self.edge_feats.fill(0.0)
        self.roles.fill(-1)
        self.valid_cnt.fill(0)

    # ------------------------------------------------------------------ #
    # Writes (strict-causal: only call from the POST-scoring block)
    # ------------------------------------------------------------------ #

    def write_events(
        self,
        src: np.ndarray,                # [E] int64
        tgt: np.ndarray,                # [E] int64
        ts: np.ndarray,                 # [E] int64
        edge_feat: Optional[np.ndarray],  # [E, d_e] float32 or None
    ) -> None:
        """Append a batch of (src → tgt @ ts) events.

        Each event is recorded TWICE — once for src (role=0) with neighbor=tgt,
        once for tgt (role=1) with neighbor=src — so reading either endpoint's
        history surfaces this event.

        Ordering inside a batch: events are written in the order they appear
        in the input arrays. The most-recent event for any node u ends up in
        the highest-occupied slot of history[u]. If u participates in multiple
        events in the same batch, the LAST one is the most recent in u's
        history. Caller guarantees the input arrays are time-sorted (which
        create_batches already does).
        """
        assert src.shape == tgt.shape == ts.shape
        if edge_feat is not None:
            assert edge_feat.shape[0] == src.shape[0]
            if self.edge_feats is None:
                raise ValueError(
                    "edge_feat was passed but NodeHistory was constructed "
                    "with edge_feat_dim=0",
                )
            assert edge_feat.shape[1] == self.edge_feat_dim

        n_events = int(src.shape[0])
        if n_events == 0:
            return

        # Loop the events. Vectorising this cleanly is tricky because the
        # same node may appear in multiple events of one batch and the K-th
        # write must overwrite the oldest entry — sequential append is the
        # natural model. n_events is at most target_batch_size (default 200),
        # so the Python loop overhead is negligible compared to GPU work.
        for i in range(n_events):
            self._append_one(int(src[i]), int(tgt[i]), int(ts[i]),
                             edge_feat[i] if edge_feat is not None else None,
                             role=0)
            self._append_one(int(tgt[i]), int(src[i]), int(ts[i]),
                             edge_feat[i] if edge_feat is not None else None,
                             role=1)

    def _append_one(
        self,
        u: int,
        neighbor: int,
        timestamp: int,
        edge_feat_row: Optional[np.ndarray],
        role: int,
    ) -> None:
        """Append one event to u's buffer. If buffer is full, evict the
        oldest (slot 0) and slide everything left by 1."""
        vc = int(self.valid_cnt[u])
        K = self.K
        if vc < K:
            # There's empty space: write at slot vc (next free).
            slot = vc
            self.valid_cnt[u] = vc + 1
        else:
            # Buffer is full: shift left, drop the oldest entry.
            self.neighbors[u, :K - 1] = self.neighbors[u, 1:K]
            self.timestamps[u, :K - 1] = self.timestamps[u, 1:K]
            self.roles[u, :K - 1] = self.roles[u, 1:K]
            if self.edge_feats is not None:
                self.edge_feats[u, :K - 1, :] = self.edge_feats[u, 1:K, :]
            slot = K - 1

        self.neighbors[u, slot] = neighbor
        self.timestamps[u, slot] = timestamp
        self.roles[u, slot] = role
        if self.edge_feats is not None and edge_feat_row is not None:
            self.edge_feats[u, slot, :] = edge_feat_row

    # ------------------------------------------------------------------ #
    # Reads (strict-causal: only call BEFORE the post-scoring write block)
    # ------------------------------------------------------------------ #

    def read_windows_for_nodes(
        self,
        node_ids: np.ndarray,  # [N_query] int64
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """Return per-node history slices for the queried nodes.

        Outputs (always shape [N_query, K, *] regardless of valid_cnt):

          neighbors:  [N_query, K] int64    — -1 in invalid slots
          timestamps: [N_query, K] int64    — -1 in invalid slots
          edge_feats: [N_query, K, d_e] float32 or None — zeros in invalid slots
          roles:      [N_query, K] int8     — -1 in invalid slots
          valid_cnt:  [N_query]    int32    — number of valid entries

        Valid entries live at slots [0, valid_cnt[i] − 1] (right-padded).
        Most-recent event is at slot valid_cnt[i] − 1. Caller is
        responsible for masking on `valid_cnt`.

        Returns copies (np advanced indexing always copies anyway, but stated
        here so reviewers don't worry about aliasing).
        """
        if node_ids.dtype != np.int64:
            node_ids = node_ids.astype(np.int64)
        neighbors = self.neighbors[node_ids]
        timestamps = self.timestamps[node_ids]
        edge_feats = self.edge_feats[node_ids] if self.edge_feats is not None else None
        roles = self.roles[node_ids]
        valid_cnt = self.valid_cnt[node_ids].copy()
        return neighbors, timestamps, edge_feats, roles, valid_cnt
