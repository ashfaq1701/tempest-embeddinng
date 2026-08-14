"""Negative samplers.

  - NegativeSampler         : ABC (`sample(batch) → (neg_src, neg_tgt)`).
  - UniformNegativeSampler  : random over a destination pool; used for training on
                              every suite and to build TGB-Seq's fixed eval negatives.

Eval-time suite-native samplers live in their suite modules (`tgb_eval.py`,
`tgb_seq_eval.py`).
"""

import abc
from typing import Optional, Tuple

import numpy as np

from .data import Batch


class NegativeSampler(abc.ABC):
    """All samplers expose `sample(batch) → (neg_src, neg_tgt)`. `reset()` is called
    at the start of every epoch; stateless samplers use the no-op default."""

    @abc.abstractmethod
    def sample(self, batch: Batch):
        ...

    def reset(self) -> None:
        return None


class UniformNegativeSampler(NegativeSampler):
    """Random destinations from `dst_pool`, keeping the positive's source."""

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
