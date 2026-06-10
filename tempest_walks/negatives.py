"""Negative samplers.

Three flavours:
  - UniformNegativeSampler    : random over a destination pool. The
                                default training-side sampler.
  - HistoricalNegativeSampler : per-source reservoir (Vitter R, fixed pool of
                                128) returning a source's PAST destinations as
                                hard negatives. Historical-only; accepts some
                                false negatives by design. observe() after
                                scoring, reset() per epoch. Re-added for
                                cold-start / low-recurrence datasets (e.g.
                                tgbl-review) where eval negatives are partly
                                historical; NOT for recurrence-dominated
                                tgbl-wiki (historical ≈ future positives there).
  - TGBNegativeSampler        : eval-time. Routes through
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
    Stateless samplers (everything we have now) rely on the no-op
    default.
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


class HistoricalNegativeSampler:
    """Per-source reservoir of past destinations → HISTORICAL negatives only.

    Each source keeps a FIXED pool of `reservoir_size` (default 128) of its
    past destinations, maintained as a uniform random sample of that source's
    history via Vitter's Algorithm R (accept the (count+1)-th item with
    probability M/(count+1), replacing a uniform-random slot; the fill phase
    accepts unconditionally). All numpy/CPU, fully vectorised — O(B) observe,
    O(Q·k) query.

    Contract:
      * observe(src, dst) — call AFTER scoring (strict causal). Groups the
        batch's destinations per source and inserts them via Vitter R.
      * reset()           — call at each epoch start (the stream is replayed,
                            so history must clear or future edges leak).
      * sample(nodes, k)  — returns [len(nodes), k] historical negatives.

    Returns historical negatives ONLY (no random mix, no positive-target
    exclusion — some false negatives are accepted by design, which on
    recurrence-heavy graphs is a known cost). Empty reservoir slots (cold or
    under-filled sources with fewer than k distinct partners) fall back to
    random draws from `dst_pool` — the only way to return a full [.,k] row for
    a node with insufficient history.
    """

    def __init__(
        self,
        num_nodes: int,
        dst_pool: np.ndarray,
        reservoir_size: int = 128,
        seed: Optional[int] = None,
    ):
        self.num_nodes = int(num_nodes)
        self.M = int(reservoir_size)
        self.reservoir = np.full((self.num_nodes, self.M), -1, dtype=np.int32)
        self.count = np.zeros(self.num_nodes, dtype=np.int64)
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.reservoir.fill(-1)
        self.count.fill(0)

    def observe(self, src: np.ndarray, dst: np.ndarray) -> None:
        """Vitter-R reservoir update over a batch. MUST run AFTER scoring.

        Vectorised: each (src, dst) is accepted into src's reservoir in the
        fill phase (count < M) into the next empty slot, or in the full phase
        with probability M/(count+1) replacing a uniform-random slot. For a
        source repeated within the batch the last accepted write to a given
        (src, slot) wins — a negligible deviation from sequential Vitter R at
        batch sizes ≪ per-source accepted-write rate.
        """
        B = src.shape[0]
        if B == 0:
            return
        src_i = src.astype(np.int64, copy=False)
        dst_i = dst.astype(np.int32, copy=False)
        pre_count = self.count[src_i]

        fill_mask = pre_count < self.M
        # full phase: accept with probability M / (count + 1)
        accept = self.rng.random(size=B) < (self.M / (pre_count + 1).astype(np.float64))
        do_insert = fill_mask | accept
        # slot: next empty in fill phase, uniform-random in full phase
        insert_pos = np.where(fill_mask, pre_count, self.rng.integers(0, self.M, size=B))

        idx = np.where(do_insert)[0]
        if idx.size:
            self.reservoir[src_i[idx], insert_pos[idx]] = dst_i[idx]
        np.add.at(self.count, src_i, 1)

    def sample(self, nodes: np.ndarray, num_neg: int) -> np.ndarray:
        """Return [len(nodes), num_neg] historical negatives for `nodes`.

        Draws num_neg uniform-random slots per node from its reservoir; empty
        slots (-1) are filled with random destinations from `dst_pool`.
        """
        nodes = np.asarray(nodes, dtype=np.int64)
        Q = nodes.shape[0]
        slot = self.rng.integers(0, self.M, size=(Q, num_neg))
        neg = np.take_along_axis(
            self.reservoir[nodes], slot, axis=1,
        ).astype(np.int32, copy=False)
        empty = neg < 0
        if empty.any():
            rand = self.dst_pool[
                self.rng.integers(0, self.dst_pool.shape[0], size=(Q, num_neg))
            ]
            neg = np.where(empty, rand, neg)
        return neg


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
