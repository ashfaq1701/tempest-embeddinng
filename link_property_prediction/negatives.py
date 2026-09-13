"""Negative samplers.

  - NegativeSampler         : ABC (`sample(batch) → neg_tgt`).
  - UniformNegativeSampler  : random over a destination pool; used for training and
                              to build TGB-Seq's fixed eval negatives.
  - PopularityNegativeSampler : draws ~ count**alpha over the same pool (word2vec's 0.75).

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


class PopularityNegativeSampler(NegativeSampler):
    """Draws negatives with probability ~ count**alpha over `dst_pool`, alpha=0.75 as in
    word2vec. Uniform sampling gives every pool node the same number of negative draws
    regardless of how often it is a positive: on ML-20M that is 7,257 draws/epoch against
    positive counts of 5 to 18,500, a push/pull ratio from 0.4:1 to 1,451:1. Radius
    correlates -0.804 with log10(positive count) as a result -- items with <50 positives
    reach r=11.4, items with >5000 max out at r=1.42.

    NOT A DROP-IN IMPROVEMENT -- see the class-level warning in the module docstring of the
    branch commit. TGB-Seq scores against *uniform* negatives, so this trains for a harder
    problem than it is measured on.
    """

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        counts: np.ndarray,
        alpha: float = 0.75,
        seed: Optional[int] = None,
    ):
        self.num_neg_per_pos = num_neg_per_pos
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        w = np.asarray(counts, dtype=np.float64)[self.dst_pool] ** float(alpha)
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("popularity weights must be finite and positive")
        # Inverse-CDF sampling: one searchsorted over a precomputed cumulative, so the
        # per-batch cost matches the uniform sampler's rather than rebuilding a
        # distribution each call.
        self.cdf = np.cumsum(w / total)
        self.cdf[-1] = 1.0
        self.rng = np.random.default_rng(seed)

    def sample(self, batch: Batch) -> np.ndarray:
        B = len(batch.src)
        u = self.rng.random((B, self.num_neg_per_pos))
        return self.dst_pool[np.searchsorted(self.cdf, u)]
