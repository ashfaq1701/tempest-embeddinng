"""Negative samplers.

  - NegativeSampler         : ABC (`sample(batch) → neg_tgt`).
  - UniformNegativeSampler  : random over a destination pool; used for training and
                              to build TGB-Seq's and DyGLib's fixed eval negatives.
  - CollisionFreeNegativeSampler : CRAFT's DyGLib rule (see the class).
  - HistoricalNegativeSampler : TGB's hist_rnd rule, made causal for training (see the class).

Eval-time negatives are suite-native and live behind `Evaluator.sample_negatives`
(`tgb_seq_eval.py`, `tgb_eval.py`), not here: that interface allows a variable K
per positive, which `sample() -> [B, K]` below cannot express.
"""

import abc
from typing import Optional

import numpy as np

from .data import Batch, SplitData


class NegativeSampler(abc.ABC):
    """All samplers expose `sample(batch) -> neg_tgt`, the [B, K] negative destinations."""

    @abc.abstractmethod
    def sample(self, batch: Batch) -> np.ndarray:
        ...


class UniformNegativeSampler(NegativeSampler):
    """Random destinations from `dst_pool`."""

    rule = "uniform"

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        seed: Optional[int] = None,
    ):
        self.num_neg_per_pos = num_neg_per_pos
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(k={self.num_neg_per_pos}, pool={len(self.dst_pool):,})"

    def sample(self, batch: Batch) -> np.ndarray:
        """[B, K] negative destinations. The positive's source is implicit -- every caller
        pairs the negatives with batch.src itself, so no source array is materialised."""
        B = len(batch.src)
        idx = self.rng.integers(0, len(self.dst_pool), (B, self.num_neg_per_pos))
        return self.dst_pool[idx]


class SameTimeDestinations:
    """The destinations each source links to at exactly each timestamp of `edges`."""

    def __init__(self, edges: SplitData, num_nodes: int):
        self._times = np.unique(edges.timestamps)
        self._num_nodes = int(num_nodes)
        if self._num_nodes ** 2 * len(self._times) >= np.iinfo(np.int64).max:
            raise ValueError("(source, time, destination) keys overflow int64")
        pairs, _ = self._pairs(edges.sources, edges.timestamps)
        self._keys = np.unique(pairs * self._num_nodes + edges.destinations)
        self._pair_keys, self._pair_counts = np.unique(self._keys // self._num_nodes,
                                                       return_counts=True)

    def _pairs(self, src: np.ndarray, ts: np.ndarray):
        rank = np.minimum(np.searchsorted(self._times, ts), len(self._times) - 1)
        found = self._times[rank] == ts
        return src.astype(np.int64) * len(self._times) + rank, found

    def contains(self, src: np.ndarray, ts: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """[B, K] bool: dst[i, j] is a destination of src[i] at ts[i]."""
        pairs, found = self._pairs(src, ts)
        keys = pairs[:, None] * self._num_nodes + dst
        i = np.minimum(np.searchsorted(self._keys, keys), len(self._keys) - 1)
        return found[:, None] & (self._keys[i] == keys)

    def count(self, src: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """[B] int: how many destinations src[i] links to at ts[i]."""
        pairs, found = self._pairs(src, ts)
        i = np.minimum(np.searchsorted(self._pair_keys, pairs), len(self._pair_keys) - 1)
        hit = found & (self._pair_keys[i] == pairs)
        return np.where(hit, self._pair_counts[i], 0)


def _repeats(neg: np.ndarray) -> np.ndarray:
    """[B, K] bool: True where an entry equals an earlier entry of its row."""
    order = np.argsort(neg, axis=1, kind="stable")
    ordered = np.take_along_axis(neg, order, axis=1)
    repeat_ordered = np.zeros(neg.shape, dtype=bool)
    repeat_ordered[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
    repeat = np.empty_like(repeat_ordered)
    np.put_along_axis(repeat, order, repeat_ordered, axis=1)
    return repeat


class CollisionFreeNegativeSampler(UniformNegativeSampler):
    """CRAFT's DyGLib collision check (`sample_with_time_collision_check`): never a destination
    the source links to at the query's own timestamp (so never the positive), and no repeats
    within a row unless fewer than K candidates remain, as with CRAFT's
    `replace=len(candidates) < num_negs`. Collisions are redrawn until none remain."""

    rule = "uniform collision-free"

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        edges: SplitData,
        seed: Optional[int] = None,
    ):
        super().__init__(num_neg_per_pos, dst_pool, seed)
        num_nodes = max(self.dst_pool.max(), edges.sources.max(), edges.destinations.max()) + 1
        self.same_time = SameTimeDestinations(edges, num_nodes)

    def sample(self, batch: Batch) -> np.ndarray:
        neg = super().sample(batch)
        src, ts = batch.src, batch.ts
        distinct = len(self.dst_pool) - self.same_time.count(src, ts) >= self.num_neg_per_pos
        rows = np.arange(len(neg))
        while len(rows):
            bad = self.same_time.contains(src[rows], ts[rows], neg[rows])
            bad |= _repeats(neg[rows]) & distinct[rows, None]
            hit = bad.any(axis=1)
            rows, bad = rows[hit], bad[hit]
            redrawn = neg[rows]
            redrawn[bad] = self.dst_pool[self.rng.integers(0, len(self.dst_pool), int(bad.sum()))]
            neg[rows] = redrawn
        return neg


class HistoricalNegativeSampler(UniformNegativeSampler):
    """TGB's hist_rnd rule for training negatives, made causal.

    For each query (s, t) the first floor(hist_ratio * K) slots are filled with s's distinct past
    destinations -- nodes s first linked strictly before t in `edges` (the train split) -- drawn
    without replacement and never a destination s links to at t itself. The remaining slots keep
    the uniform draw. hist_ratio = 0 is UniformNegativeSampler exactly. A query with too few past
    destinations keeps uniform draws in the slots it cannot fill."""

    MAX_REDRAWS = 16

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        edges: SplitData,
        hist_ratio: float,
        seed: Optional[int] = None,
    ):
        super().__init__(num_neg_per_pos, dst_pool, seed)
        if not 0.0 <= hist_ratio <= 1.0:
            raise ValueError(f"hist_ratio must be in [0, 1], got {hist_ratio}")
        self.hist_ratio = float(hist_ratio)
        self.n_hist = int(hist_ratio * num_neg_per_pos)                      # floor: explicit, not round()
        self.rule = f"{hist_ratio:g} historical + uniform"
        if self.n_hist:
            num_nodes = max(self.dst_pool.max(), edges.sources.max(), edges.destinations.max()) + 1
            self.same_time = SameTimeDestinations(edges, num_nodes)
            self._index_first_links(edges, num_nodes)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(k={self.num_neg_per_pos}, hist={self.n_hist}, "
                f"pool={len(self.dst_pool):,})")

    def _index_first_links(self, edges: SplitData, num_nodes: int) -> None:
        """Each distinct (src, dst) pair once, at the rank of the time src first linked dst, sorted by
        (src, rank) so a source's past before any t is a prefix of its block."""
        self._times = np.unique(edges.timestamps)
        rank = np.searchsorted(self._times, edges.timestamps)                # time -> dense rank
        pair = edges.sources.astype(np.int64) * num_nodes + edges.destinations
        order = np.lexsort((rank, pair))                                     # by pair, then time
        first = np.r_[True, pair[order][1:] != pair[order][:-1]]             # each pair's earliest link
        pair, rank = pair[order][first], rank[order][first]
        keys = (pair // num_nodes) * len(self._times) + rank                 # (src, rank) as one int64
        by_key = np.argsort(keys, kind="stable")
        self._keys = keys[by_key]
        self._past_dst = (pair % num_nodes)[by_key].astype(self.dst_pool.dtype)

    def _past_block(self, src: np.ndarray, ts: np.ndarray):
        """For each query: where s's block starts and how many destinations s first linked before t."""
        base = src.astype(np.int64) * len(self._times)
        start = np.searchsorted(self._keys, base)
        end = np.searchsorted(self._keys, base + np.searchsorted(self._times, ts))  # ranks strictly < t
        return start, end - start

    def _draw_past(self, start: np.ndarray, n_past: np.ndarray, cols: int) -> np.ndarray:
        """[B, cols] uniform draws from each query's past block (garbage where n_past == 0, masked by callers)."""
        offsets = (self.rng.random((len(start), cols)) * np.maximum(n_past, 1)[:, None]).astype(np.int64)
        return self._past_dst[np.minimum(start[:, None] + offsets, len(self._past_dst) - 1)]

    def sample(self, batch: Batch) -> np.ndarray:
        neg = super().sample(batch)                                          # [B, K] the uniform draw
        if self.n_hist == 0:
            return neg

        src, ts = batch.src, batch.ts
        start, n_past = self._past_block(src, ts)
        slots = np.arange(self.num_neg_per_pos) < np.minimum(self.n_hist, n_past)[:, None]  # [B, K]
        hist = self._draw_past(start, n_past, self.num_neg_per_pos)          # [B, K]

        # Redraw historical slots that hit a destination of s at t or repeat within the row.
        rows = np.flatnonzero(slots.any(axis=1))
        bad = np.zeros_like(slots)
        for _ in range(self.MAX_REDRAWS):
            in_slots = np.where(slots[rows], hist[rows], -1 - np.arange(self.num_neg_per_pos))  # unique fillers
            bad_rows = slots[rows] & (self.same_time.contains(src[rows], ts[rows], hist[rows])
                                      | _repeats(in_slots))
            bad[rows] = bad_rows
            hit = bad_rows.any(axis=1)
            if not hit.any():
                break
            rows, bad_rows = rows[hit], bad_rows[hit]
            redrawn = self._draw_past(start[rows], n_past[rows], self.num_neg_per_pos)
            hist[rows] = np.where(bad_rows, redrawn, hist[rows])

        # Slots still clashing after MAX_REDRAWS (too few clean past destinations) keep the uniform draw.
        return np.where(slots & ~bad, hist, neg)
