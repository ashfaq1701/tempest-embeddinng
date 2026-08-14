"""Streaming exact per-undirected-pair last-interaction time and count, over a sparse
:class:`SparseStreamStore` keyed on the canonical pair ``min(u,v)*N + max(u,v)``.
Memory is O(#distinct pairs).

Lifecycle: ``reset()`` per epoch, ``update()`` AFTER scoring a batch, ``query()`` at
scoring time (pre-ingest state).
"""
import numpy as np
import torch

from .sparse_store import SparseStreamStore


class PairRecencyStore:
    """Streaming exact last-interaction time + count per undirected node pair. Used only
    by the stratification analysis (`stratify.py`); the `pair_dt` return is vestigial."""

    def __init__(self, num_nodes: int):
        self.N = int(num_nodes)
        self._store = SparseStreamStore(
            {"last_ts": ("max", 0), "count": ("add", 0)})

    def reset(self) -> None:
        """Drop all interactions. Call at the start of each epoch (with walk reset)."""
        self._store.reset()

    def _canon(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.minimum(u, v) * self.N + np.maximum(u, v)

    @torch.no_grad()
    def update(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray) -> None:
        """Ingest a batch of edges (undirected). STRICT-CAUSAL: call AFTER scoring."""
        s = np.asarray(src, dtype=np.int64)
        t = np.asarray(tgt, dtype=np.int64)
        ti = np.asarray(ts, dtype=np.int64)
        self._store.upsert(
            self._canon(s, t),
            {"last_ts": ti, "count": np.ones_like(ti)})

    @torch.no_grad()
    def query(self, src: torch.Tensor, cand: torch.Tensor, t_query: torch.Tensor):
        """src [B], cand [B, C], t_query [B] (long) -> (pair_dt [B, C], pair_count_log [B, C])
        on cand.device.
          pair_dt        : Δt_uv = t_query − t_last[(u,v)] (clamped ≥0); never-seen ⇒ 1e18.
          pair_count_log : log1p(#(u,v) interactions) (0 for never-seen)."""
        device = cand.device
        B, C = cand.shape
        s = src.detach().to("cpu", torch.int64).numpy()
        c = cand.detach().to("cpu", torch.int64).numpy()
        tq = t_query.detach().to("cpu", torch.int64).numpy()
        keys = self._canon(s[:, None], c).reshape(-1)              # [B*C]

        out, _ = self._store.get(keys)
        last = out["last_ts"].reshape(B, C)
        cnt = out["count"].reshape(B, C)
        rec = np.clip(tq[:, None] - last, 0, None).astype(np.float32)
        rec[cnt == 0] = 1e18                                       # never-seen ⇒ Δt=∞ ⇒ φ=0
        count_log = np.log1p(cnt.astype(np.float32))               # 0 for never-seen
        return (torch.from_numpy(rec).to(device),
                torch.from_numpy(count_log).to(device))
