"""Negative samplers.

Three flavours:
  - UniformNegativeSampler  : random over a destination pool (TRAINING default).
  - HistoricalNegativeSampler: per-source reservoir of past destinations,
                                mixed with random fallback. Vectorised O(B)
                                observe / O(B*K) sample, CPU/numpy state.
                                **DO NOT use on tgbl-wiki training** — most
                                eval positives are historical, training the
                                model to score them LOW collapses MRR.
                                Kept here for ablations / other datasets.
  - TGBNegativeSampler      : eval-time. Routes through
                                dataset.negative_sampler.query_batch — the
                                TGB-prescribed protocol.
"""

import abc
from typing import List, Optional, Tuple

import numpy as np

from .data import Batch


class NegativeSampler(abc.ABC):
    """All samplers expose `sample(batch) → (neg_src, neg_tgt)`.

    `reset()` is called by the trainer at the start of every epoch.
    Stateless samplers can rely on the no-op default; samplers that
    accumulate per-source state (e.g., the historical reservoir)
    MUST override to drop carry-over from the prior epoch — without
    it, epoch-1's full chronological pass contaminates epoch-2's
    historical negatives with future-batch positives (strict-causal
    violation, see Lesson 28).
    """

    @abc.abstractmethod
    def sample(self, batch: Batch):
        ...

    def reset(self) -> None:
        return None


class UniformNegativeSampler(NegativeSampler):
    """Random destinations from a pool, keeping the positive's source.

    `dst_pool` is required for bipartite datasets (tgbl-wiki / -review etc.)
    so training negatives stay on the destination side of the bipartite —
    sampling over the full node set would create the trivially easy task
    of "is this node ever a destination?" and won't transfer to eval.
    """

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        seed: Optional[int] = None,
    ):
        self.num_neg_per_pos = num_neg_per_pos
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: Batch) -> Tuple[np.ndarray, np.ndarray]:
        B = len(batch.src)
        neg_src = np.broadcast_to(
            batch.src[:, None], (B, self.num_neg_per_pos),
        ).astype(np.int32, copy=True)
        idx = self.rng.integers(0, len(self.dst_pool), (B, self.num_neg_per_pos))
        neg_tgt = self.dst_pool[idx]
        return neg_src, neg_tgt


class HistoricalNegativeSampler(NegativeSampler):
    """Per-source reservoir of past destinations (Vitter's algorithm R)
    mixed with random fallback. Designed to match TGB eval's 50/50 mix
    on datasets where historical-recurrence is NOT the dominant positive
    signal. NOT recommended for tgbl-wiki.

    Vectorised hot path:
      observe(src, dst):
        - True Vitter R: each item accepted with probability M/(count+1)
          when the reservoir is full, replacing a uniform-random slot.
          Fill phase (count < M) accepts unconditionally into the next
          empty slot. Guarantees every reservoir slot is a uniform draw
          from the source's history at all times after fill.
        - O(B) vectorized; no Python loops.
        - For repeated sources within a batch the LAST accepted write
          to a given (src, position) wins — this is the only remaining
          deviation from strict sequential Vitter R, and it's negligible
          because B (~200) is much smaller than the per-source rate of
          accepted writes.
      sample(batch):
        - O(B*K_hist) reservoir gather + a where() for the invalid /
          false-negative guard, then O(B*K_rand) random draws.

    State (reservoir matrix + count vector) lives on CPU/numpy so the
    sampler scales to multi-million-node datasets without GPU memory
    cost; the only per-call data on GPU is the negatives themselves.
    """

    def __init__(
        self,
        num_nodes: int,
        num_neg_per_pos: int,
        hist_ratio: float,
        reservoir_size: int,
        dst_pool: np.ndarray,
        seed: Optional[int] = None,
    ):
        self.num_nodes = num_nodes
        self.num_neg_per_pos = num_neg_per_pos
        self.M = reservoir_size
        self.K_hist = max(0, min(num_neg_per_pos, int(round(num_neg_per_pos * hist_ratio))))
        self.K_rand = num_neg_per_pos - self.K_hist
        self.reservoir = np.full((num_nodes, self.M), -1, dtype=np.int32)
        self.count = np.zeros(num_nodes, dtype=np.int64)
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.reservoir.fill(-1)
        self.count.fill(0)

    def observe(self, src: np.ndarray, dst: np.ndarray) -> None:
        """Vitter-R update. MUST be called AFTER scoring (strict-causal).

        Per-source reservoir maintains a UNIFORM RANDOM SAMPLE of size M
        from the source's history. When the reservoir is full and the
        (count+1)-th item arrives, accept it with probability M/(count+1)
        and, if accepted, replace a uniformly-chosen slot. Every slot is
        a uniform draw from the source's history at all times after fill.
        """
        B = src.shape[0]
        if B == 0:
            return
        src_i = src.astype(np.int64, copy=False)
        dst_i = dst.astype(np.int32, copy=False)
        pre_count = self.count[src_i]

        fill_mask = pre_count < self.M

        # Full phase: accept with probability M / (count + 1).
        t = (pre_count + 1).astype(np.float64)
        accept_threshold = self.M / t
        accept_draw = self.rng.random(size=B)
        accept_when_full = accept_draw < accept_threshold

        # Combined insert mask.
        do_insert = fill_mask | (~fill_mask & accept_when_full)

        # Slot to insert into. Fill phase: next empty slot. Full phase:
        # uniform random slot.
        rand_pos = self.rng.integers(0, self.M, size=B)
        insert_pos = np.where(fill_mask, pre_count, rand_pos)

        # Apply insertion only where do_insert is True.
        insert_idx = np.where(do_insert)[0]
        if len(insert_idx) > 0:
            self.reservoir[src_i[insert_idx], insert_pos[insert_idx]] = dst_i[insert_idx]

        np.add.at(self.count, src_i, 1)

    def sample(self, batch: Batch) -> Tuple[np.ndarray, np.ndarray]:
        B = len(batch.src)
        pos_src = np.asarray(batch.src, dtype=np.int64)
        pos_tgt = np.asarray(batch.tgt, dtype=np.int32)

        rand_pool_idx = self.rng.integers(
            0, self.dst_pool.shape[0], size=(B, self.K_hist + self.K_rand),
        )
        rand_all = self.dst_pool[rand_pool_idx]

        if self.K_hist > 0:
            rand_slot = self.rng.integers(0, self.M, size=(B, self.K_hist))
            hist_neg = np.take_along_axis(
                self.reservoir[pos_src], rand_slot, axis=1,
            ).astype(np.int32, copy=False)
            invalid = (hist_neg < 0) | (hist_neg == pos_tgt[:, None])
            if invalid.any():
                hist_neg = np.where(invalid, rand_all[:, : self.K_hist], hist_neg)
        else:
            hist_neg = np.empty((B, 0), dtype=np.int32)

        rand_neg = rand_all[:, self.K_hist : self.K_hist + self.K_rand]
        neg_tgt = np.concatenate([hist_neg, rand_neg], axis=1)
        neg_src = np.broadcast_to(
            batch.src[:, None], (B, self.num_neg_per_pos),
        ).astype(np.int32, copy=True)
        return neg_src, neg_tgt


class TGBNegativeSampler(NegativeSampler):
    """Eval-time sampler. Wraps `dataset.negative_sampler.query_batch`,
    which serves TGB's pre-generated per-positive negatives. Variable-K
    per positive — returns list-of-arrays."""

    def __init__(self, dataset: object, split_mode: str):
        if split_mode not in ("val", "test"):
            raise ValueError(f"split_mode must be 'val' or 'test', got {split_mode!r}")
        self.dataset = dataset
        self.split_mode = split_mode
        self._sampler = dataset.negative_sampler

    def sample(self, batch: Batch) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        neg_dst_list = self._sampler.query_batch(
            batch.src, batch.tgt, batch.ts, split_mode=self.split_mode,
        )
        neg_src_list: List[np.ndarray] = []
        neg_tgt_list: List[np.ndarray] = []
        for s, neg_dsts in zip(batch.src, neg_dst_list):
            arr = np.asarray(neg_dsts, dtype=np.int32)
            neg_src_list.append(np.full(len(arr), s, dtype=np.int32))
            neg_tgt_list.append(arr)
        return neg_src_list, neg_tgt_list
