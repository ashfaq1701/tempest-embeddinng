"""Negative samplers.

  - NegativeSampler         : ABC (`sample(batch) → neg_tgt`).
  - UniformNegativeSampler  : random over a destination pool; used for training and
                              to build TGB-Seq's fixed eval negatives.

The eval-time suite-native sampler lives in `tgb_seq_eval.py`.
"""

import abc
from typing import Optional

import numpy as np

from .data import Batch


class NegativeSampler(abc.ABC):
    """All samplers expose `sample(batch) -> neg_tgt`, the [B, K] negative destinations."""

    @abc.abstractmethod
    def sample(self, batch: Batch) -> np.ndarray:
        ...


class UniformNegativeSampler(NegativeSampler):
    """Random destinations from `dst_pool`."""

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        seed: Optional[int] = None,
    ):
        self.num_neg_per_pos = num_neg_per_pos
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: Batch) -> np.ndarray:
        """[B, K] negative destinations. The positive's source is implicit -- every caller
        pairs the negatives with batch.src itself, so no source array is materialised."""
        B = len(batch.src)
        idx = self.rng.integers(0, len(self.dst_pool), (B, self.num_neg_per_pos))
        return self.dst_pool[idx]
