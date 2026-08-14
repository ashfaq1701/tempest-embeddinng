"""Reusable sparse streaming int64 key->value store over a pandas ``Index`` hash table.
Maps keys to named integer columns, each with its own batch-reducer (``max`` / ``min`` /
``add`` / ``last``). Memory is O(#distinct keys); all ops vectorized over numpy arrays.

Lifecycle: ``reset()`` per epoch, ``upsert()`` AFTER scoring a batch, ``get()`` at
scoring time.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# reducer name -> pandas groupby agg used to collapse duplicate keys WITHIN a batch.
_BATCH_AGG = {"max": "max", "min": "min", "add": "sum", "last": "last"}


class SparseStreamStore:
    """int64 key -> named int64 value columns; vectorized batch upsert / get."""

    def __init__(self, columns: Dict[str, Tuple[str, int]]):
        # columns: name -> (reducer, default). reducer in {max, min, add, last}.
        for name, (red, _) in columns.items():
            if red not in _BATCH_AGG:
                raise ValueError(f"{name}: unknown reducer {red!r}")
        self._columns = dict(columns)
        self.reset()

    def reset(self) -> None:
        self._keys = np.empty(0, dtype=np.int64)
        self._index = pd.Index(self._keys)
        self._vals = {n: np.empty(0, dtype=np.int64) for n in self._columns}

    def _dedup(self, keys: np.ndarray, values: Dict[str, np.ndarray]):
        """Collapse duplicate keys within the incoming batch per-column reducer."""
        df = pd.DataFrame({"__k": keys, **values})
        agg = {n: _BATCH_AGG[r] for n, (r, _) in self._columns.items()}
        g = df.groupby("__k", sort=False).agg(agg)
        return (g.index.to_numpy(dtype=np.int64),
                {n: g[n].to_numpy(dtype=np.int64) for n in self._columns})

    def upsert(self, keys: np.ndarray, values: Dict[str, np.ndarray]) -> None:
        """Insert/update rows. Existing keys fold in via each column's reducer."""
        keys, values = self._dedup(np.asarray(keys, np.int64), values)
        pos = self._index.get_indexer(keys)            # -1 where missing (C hashtable)
        found = pos >= 0
        fp = pos[found]
        for n, (red, _) in self._columns.items():
            v = values[n][found]
            cur = self._vals[n]
            if red == "max":
                cur[fp] = np.maximum(cur[fp], v)
            elif red == "min":
                cur[fp] = np.minimum(cur[fp], v)
            elif red == "add":
                cur[fp] = cur[fp] + v
            else:  # last
                cur[fp] = v
        new = ~found
        if new.any():
            nk = keys[new]
            self._keys = np.concatenate([self._keys, nk])
            self._index = pd.Index(self._keys)         # unique by construction
            for n in self._columns:
                self._vals[n] = np.concatenate([self._vals[n], values[n][new]])

    def get(self, keys: np.ndarray):
        """keys [Q] -> ({name: [Q] int64 (default-filled)}, found_mask [Q] bool)."""
        keys = np.asarray(keys, np.int64)
        pos = self._index.get_indexer(keys)
        found = pos >= 0
        fp = pos[found]
        out = {}
        for n, (_, default) in self._columns.items():
            arr = np.full(keys.shape[0], default, dtype=np.int64)
            arr[found] = self._vals[n][fp]
            out[n] = arr
        return out, found
