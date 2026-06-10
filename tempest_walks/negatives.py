"""Negative samplers.

Two flavours:
  - UniformNegativeSampler  : random over a destination pool. The
                              training-side sampler (the only one).
  - TGBNegativeSampler      : eval-time. Routes through
                              dataset.negative_sampler.query_batch — the
                              TGB-prescribed protocol.

The Historical (per-source reservoir + Vitter R) sampler was dropped:
on recurrence-dominated datasets like tgbl-wiki it actively trained
the model AGAINST the eval signal — most eval positives are nodes
the source has previously interacted with, and historical negatives
push E[u] away from exactly those. Removed wholesale (class +
TrainerConfig fields + CLI args + reservoir-uniformity test).
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
