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
    """Streaming exact last-interaction time + count per undirected node pair.

    Used ONLY by the stratification analysis (`stratify.py`) — there is no model pair
    channel. stratify queries `count` to split test edges into repeat-pair (count>0) vs
    new-pair (count==0) and localize where MRR is lost; the `pair_dt` return is vestigial."""

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
        (pair_dt [B, C], pair_count_log [B, C]) on cand.device.
          pair_dt        : RAW Δt_uv = t_query − t_last[(u,v)] (clamped ≥0; → ExpDecayBasis).
                           NEVER-seen (count==0) ⇒ Δt = +inf (1e18) ⇒ φ → 0 (clean baseline).
          pair_count_log : log1p(#(u,v) interactions) (0 for never-seen → no count term)."""
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
