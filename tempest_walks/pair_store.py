"""Streaming pairwise-interaction store (exact, strict-causal, scalable).

Mirrors the Tempest walk graph's lifecycle: ``reset()`` each epoch, ``update()``
AFTER scoring a batch, ``query()`` at scoring time (pre-ingest state). Holds, for
every UNDIRECTED node pair seen so far, the most recent interaction time and an
interaction count — the exact ``A^(1)_{u,v}`` recurrence signal TPNet approximates
with its random-feature Gram (``pair-feature-integration.md`` feature #1/#2).

Backed by a torch **open-addressing hash table** (linear probing) keyed on the
canonical pair ``min(u,v)*N + max(u,v)``. Memory is O(#unique pairs), NOT O(N²), so
it scales from tgbl-wiki (9 k nodes, 13 k pairs) to tgbl-comment (995 k nodes, tens
of millions of pairs) — the dense ``[N*N]`` form would have been ~8 TB there.

Everything is vectorized: a batch insert resolves ≤ batch_size unique keys in a few
probe rounds (claim-by-scatter, verify, advance losers); a query probes all ``B*C``
keys in parallel. Load factor is kept < 0.5 (rehash-doubling) so probe chains stay
short. CPU int64 throughout; query returns tensors on the caller's device.

Timestamps are monotone non-decreasing across chronological batches, so "amax"
gives last-interaction-time and "sum" the count.
"""
import numpy as np
import torch

_GOLDEN = 0x9E3779B97F4A7C15  # 64-bit Fibonacci hashing multiplier
_EMPTY = -1


def _next_pow2(x: int) -> int:
    return 1 << max(1, (int(x) - 1).bit_length())


class PairRecencyStore:
    """Exact last-interaction-time + count per undirected node pair, streamed."""

    def __init__(self, num_nodes: int, initial_capacity: int = 1 << 14):
        self.N = int(num_nodes)
        self._cap = _next_pow2(initial_capacity)
        self._alloc(self._cap)

    def _alloc(self, cap: int) -> None:
        self._cap = int(cap)
        self._keys = torch.full((cap,), _EMPTY, dtype=torch.int64)
        self._last_ts = torch.zeros(cap, dtype=torch.int64)
        self._count = torch.zeros(cap, dtype=torch.int64)
        self._size = 0

    def reset(self) -> None:
        """Drop all interactions. Call at the start of each epoch (with walk reset)."""
        self._keys.fill_(_EMPTY)
        self._last_ts.zero_()
        self._count.zero_()
        self._size = 0

    # ── hashing ────────────────────────────────────────────────────────
    def _hash(self, keys: torch.Tensor) -> torch.Tensor:
        # int64 multiply wraps (two's complement) — consistent; mask to capacity.
        return (keys * _GOLDEN) & (self._cap - 1)

    def _canon(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        lo = torch.minimum(u, v)
        hi = torch.maximum(u, v)
        return lo * self.N + hi

    # ── growth ─────────────────────────────────────────────────────────
    def _maybe_grow(self, incoming_unique: int) -> None:
        if (self._size + incoming_unique) * 2 <= self._cap:
            return
        target = _next_pow2((self._size + incoming_unique) * 4)
        old_keys, old_ts, old_cnt = self._keys, self._last_ts, self._count
        occ = old_keys != _EMPTY
        self._alloc(target)
        if occ.any():
            self._insert_unique(old_keys[occ], old_ts[occ], old_cnt[occ])

    # ── insert (vectorized open addressing) ─────────────────────────────
    def _insert_unique(self, keys: torch.Tensor, ts: torch.Tensor,
                       cnt: torch.Tensor) -> None:
        mask = self._cap - 1
        pos = self._hash(keys)
        todo = torch.arange(keys.numel())
        while todo.numel() > 0:
            p = pos[todo]
            k = keys[todo]
            cur = self._keys[p]
            found = cur == k
            empty = cur == _EMPTY
            # update existing (slots distinct: keys are unique) -> last_ts max, count +=
            if found.any():
                fp = p[found]
                self._last_ts[fp] = torch.maximum(self._last_ts[fp], ts[todo[found]])
                self._count[fp] = self._count[fp] + cnt[todo[found]]
            # claim empties by scatter (duplicate slots -> last write wins), verify
            if empty.any():
                ep = p[empty]
                self._keys[ep] = k[empty]
            won = self._keys[p] == k
            claimed = empty & won
            if claimed.any():
                cp = p[claimed]
                self._last_ts[cp] = ts[todo[claimed]]
                self._count[cp] = cnt[todo[claimed]]
                self._size += int(claimed.sum())
            resolved = found | claimed
            unresolved = ~resolved
            pos[todo[unresolved]] = (p[unresolved] + 1) & mask
            todo = todo[unresolved]

    @torch.no_grad()
    def update(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray) -> None:
        """Ingest a batch of edges (undirected). STRICT-CAUSAL: call AFTER scoring."""
        s = torch.as_tensor(np.asarray(src, dtype=np.int64))
        t = torch.as_tensor(np.asarray(tgt, dtype=np.int64))
        ti = torch.as_tensor(np.asarray(ts, dtype=np.int64))
        keys = self._canon(s, t)
        # reduce duplicates within the batch: last_ts = amax, count = sum
        uniq, inv = torch.unique(keys, return_inverse=True)
        u_ts = torch.zeros_like(uniq).scatter_reduce_(
            0, inv, ti, reduce="amax", include_self=False)
        u_cnt = torch.zeros_like(uniq).scatter_reduce_(
            0, inv, torch.ones_like(ti), reduce="sum", include_self=False)
        self._maybe_grow(uniq.numel())
        self._insert_unique(uniq, u_ts, u_cnt)

    @torch.no_grad()
    def query(self, src: torch.Tensor, cand: torch.Tensor, t_query: torch.Tensor):
        """src [B] long, cand [B, C] long, t_query [B] long ->
        (pair_rec_log [B, C], ever_bit [B, C], count_log [B, C]) on cand.device.

        Cold pairs get last_ts = 0 (recency = t_query, large); the ever-bit flags
        real history. Probes all B*C keys in parallel."""
        device = cand.device
        B, C = cand.shape
        s = src.detach().to("cpu", torch.int64).unsqueeze(1).expand(B, C).reshape(-1)
        c = cand.detach().to("cpu", torch.int64).reshape(-1)
        tq = t_query.detach().to("cpu", torch.int64).unsqueeze(1).expand(B, C).reshape(-1)
        keys = self._canon(s, c)                                   # [B*C]

        last = torch.zeros_like(keys)
        cnt = torch.zeros_like(keys)
        mask = self._cap - 1
        pos = self._hash(keys)
        todo = torch.arange(keys.numel())
        while todo.numel() > 0:
            p = pos[todo]
            cur = self._keys[p]
            k = keys[todo]
            found = cur == k
            miss = cur == _EMPTY
            if found.any():
                fi = todo[found]
                last[fi] = self._last_ts[p[found]]
                cnt[fi] = self._count[p[found]]
            done = found | miss
            unresolved = ~done
            pos[todo[unresolved]] = (p[unresolved] + 1) & mask
            todo = todo[unresolved]

        rec = (tq - last).clamp_min(0)
        ever = (cnt > 0).to(torch.float32).reshape(B, C)
        pair_rec_log = torch.log1p(rec.to(torch.float32)).reshape(B, C)
        count_log = torch.log1p(cnt.to(torch.float32)).reshape(B, C)
        return (pair_rec_log.to(device), ever.to(device), count_log.to(device))
