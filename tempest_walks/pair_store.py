"""Streaming pairwise-interaction store (exact, strict-causal, sparse).

Holds, for every UNDIRECTED node pair seen so far, the most recent interaction time
and an interaction count — the exact ``A^(1)_{u,v}`` recurrence signal TPNet
approximates with its random-feature Gram (``pair-feature-integration.md`` #1/#2).

A thin feature-specific view over the reusable :class:`SparseStreamStore`: the key is
the canonical pair ``min(u,v)*N + max(u,v)``; columns are ``last_ts`` (reduce=max) and
``count`` (reduce=add). Memory is O(#distinct pairs), so it scales from tgbl-wiki
(13 k pairs) to tgbl-comment (tens of millions) without the dense ``[N*N]`` blow-up.

Lifecycle mirrors the Tempest walk graph: ``reset()`` per epoch, ``update()`` AFTER
scoring a batch, ``query()`` at scoring time (pre-ingest state). Timestamps are
monotone non-decreasing across chronological batches, so ``last_ts`` = amax is the
last interaction time.
"""
import numpy as np
import torch

from .sparse_store import SparseStreamStore


class PairRecencyStore:
    """Exact last-interaction-time + count per undirected node pair, streamed."""

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
        """src [B] long, cand [B, C] long, t_query [B] long ->
        (pair_rec_log [B, C], ever_bit [B, C], count_log [B, C]) on cand.device.

        Cold pairs get last_ts = 0 (recency = t_query, large); the ever-bit flags
        real history."""
        device = cand.device
        B, C = cand.shape
        s = src.detach().to("cpu", torch.int64).numpy()
        c = cand.detach().to("cpu", torch.int64).numpy()
        tq = t_query.detach().to("cpu", torch.int64).numpy()
        keys = self._canon(s[:, None], c).reshape(-1)              # [B*C]

        out, _ = self._store.get(keys)
        last = out["last_ts"].reshape(B, C)
        cnt = out["count"].reshape(B, C)
        rec = np.clip(tq[:, None] - last, 0, None)

        pair_rec_log = torch.log1p(torch.from_numpy(rec.astype(np.float32)))
        ever = torch.from_numpy((cnt > 0).astype(np.float32))
        count_log = torch.log1p(torch.from_numpy(cnt.astype(np.float32)))
        return (pair_rec_log.to(device), ever.to(device), count_log.to(device))


class NodeLastSeenStore:
    """Streaming per-node last-activity time (undirected). Supplies the candidate
    recency term `t_query - t_last[v]` without sampling candidate-side walks — used
    by the source-side-only head, where only the source's walks are sampled."""

    def __init__(self):
        self._store = SparseStreamStore({"last_ts": ("max", 0)})

    def reset(self) -> None:
        self._store.reset()

    @torch.no_grad()
    def update(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray) -> None:
        """Both endpoints of every edge get their last-seen time bumped. AFTER scoring."""
        s = np.asarray(src, dtype=np.int64)
        t = np.asarray(tgt, dtype=np.int64)
        ti = np.asarray(ts, dtype=np.int64)
        self._store.upsert(
            np.concatenate([s, t]), {"last_ts": np.concatenate([ti, ti])})

    @torch.no_grad()
    def query(self, cand: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
        """cand [B, C] long, t_query [B] long -> rec_v_log [B, C] on cand.device.
        Cold nodes get last_ts = 0 (recency = t_query, large)."""
        device = cand.device
        B, C = cand.shape
        c = cand.detach().to("cpu", torch.int64).numpy().reshape(-1)
        tq = t_query.detach().to("cpu", torch.int64).numpy()
        out, _ = self._store.get(c)
        last = out["last_ts"].reshape(B, C)
        rec = np.clip(tq[:, None] - last, 0, None)
        return torch.log1p(torch.from_numpy(rec.astype(np.float32))).to(device)
