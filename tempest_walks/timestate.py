"""Per-node and per-pair "time-of-last-event" state for the link MLP's
time-encoding inputs (Component 0 of the walk-distribution-matched design).

Two tensors / mappings maintained per training epoch:

  last_event_time : np.ndarray [n_nodes] int64   — time of u's most recent event
  last_edge_time  : dict[(u, v) → int64]         — time of the most recent
                                                    (u, v) edge in either direction
                                                    (treated symmetrically; (v, u)
                                                    aliases (u, v))

Strict-causal contract (mirrors NodeHistory's discipline):

  Per training batch B:
    1. read state for batch B's (u, v) pairs               (state ≤ B-1)
    2. compute Δt features for the link MLP
    3. score
    4. POST-SCORING (the LAST line of the per-batch block):
       update(src, tgt, ts)                                (now state ≤ B)

  Per training epoch: reset() — drops all event times.

Queries are O(P) per call where P = number of pairs scored. Reads return
0 (sentinel) for unseen nodes / unseen pairs — the link MLP receives
explicit cold-start binary flags so it can branch on this case.

Sentinel:
  - last_event_time[*] initialised to 0
  - last_edge_time.get((u, v)) returns 0 for unseen pairs
  - All real timestamps in TGB are ≥ 1 (the datasets use seconds since some
    epoch); 0 is therefore a safe sentinel.
"""

from typing import Tuple

import numpy as np


class NodeTimeState:
    """Holds last_event_time per node + last_edge_time per pair.

    `last_edge_time` uses a Python dict keyed by `(min(u, v), max(u, v))` so
    that (u, v) and (v, u) alias to the same entry. The cost is one Python
    op per (u, v) lookup at query time, which is fine at our pair counts
    (~2000 per training batch).
    """

    def __init__(self, n_nodes: int):
        self.n_nodes = int(n_nodes)
        self.last_event_time: np.ndarray = np.zeros(n_nodes, dtype=np.int64)
        self.last_edge_time: dict = {}  # (min_id, max_id) → int64

    def reset(self) -> None:
        """Drop all event times. Call at the start of each training epoch
        alongside walk_gen.reset() and reservoir.reset()."""
        self.last_event_time.fill(0)
        self.last_edge_time = {}

    # ------------------------------------------------------------------ #
    # Update (POST-SCORING block only)
    # ------------------------------------------------------------------ #

    def update(
        self,
        src: np.ndarray,                # [E] int64
        tgt: np.ndarray,                # [E] int64
        ts: np.ndarray,                 # [E] int64
    ) -> None:
        """Record events for batch B. Must be called AFTER B is scored.
        Per-node and per-pair last-event times are pulled forward to the
        max event time in this batch involving them.

        Symmetric pair keying: (u, v) and (v, u) share the same entry, keyed
        by `(min, max)`. This matches the link MLP's symmetric reading of
        `Δt_uv` for both (u, v) and (v, u) pair orderings.
        """
        assert src.shape == tgt.shape == ts.shape
        n = int(src.shape[0])
        if n == 0:
            return

        src_i = src.astype(np.int64, copy=False)
        tgt_i = tgt.astype(np.int64, copy=False)
        ts_i = ts.astype(np.int64, copy=False)

        # Per-node: max-reduce in case the same node appears multiple times.
        # Vectorised via np.maximum.at (handles duplicate indices correctly).
        np.maximum.at(self.last_event_time, src_i, ts_i)
        np.maximum.at(self.last_event_time, tgt_i, ts_i)

        # Per-pair: Python loop. Symmetric key.
        for i in range(n):
            u, v, t = int(src_i[i]), int(tgt_i[i]), int(ts_i[i])
            key = (u, v) if u <= v else (v, u)
            prev = self.last_edge_time.get(key, 0)
            if t > prev:
                self.last_edge_time[key] = t

    # ------------------------------------------------------------------ #
    # Query (PRE-scoring block only)
    # ------------------------------------------------------------------ #

    def query(
        self,
        u_ids: np.ndarray,              # [P] int64
        v_ids: np.ndarray,              # [P] int64
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (last_event_u, last_event_v, last_edge_uv) for each pair.

        Outputs:
          last_event_u  : [P] int64, last_event_time[u] (0 if cold-start)
          last_event_v  : [P] int64, last_event_time[v]
          last_edge_uv  : [P] int64, last_edge_time.get((min(u,v), max(u,v)), 0)

        The caller is responsible for:
          - Computing Δt = t_query - last_*_time (clamping non-positive to 0,
            though strict-causal means they're always ≥ 0 by construction)
          - Computing is_cold_start_* binary flags from last_*_time == 0
        """
        u_i = u_ids.astype(np.int64, copy=False)
        v_i = v_ids.astype(np.int64, copy=False)
        last_u = self.last_event_time[u_i].copy()
        last_v = self.last_event_time[v_i].copy()
        n = int(u_i.shape[0])
        last_uv = np.zeros(n, dtype=np.int64)
        for i in range(n):
            u, v = int(u_i[i]), int(v_i[i])
            key = (u, v) if u <= v else (v, u)
            last_uv[i] = self.last_edge_time.get(key, 0)
        return last_u, last_v, last_uv
