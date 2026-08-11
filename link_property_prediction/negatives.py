"""Negative samplers.

  - NegativeSampler         : the ABC (`sample(batch) → (neg_src, neg_tgt)`).
  - UniformNegativeSampler  : random over a destination pool. Used for TRAINING
                              on every suite, and reused to build TGB-Seq's fixed
                              eval negatives one-shot (see tgb_seq_eval.py).

Eval-time, suite-native samplers live with their suite: TGB's per-positive
pre-generated negatives are `TGBNegativeSampler` in `tgb_eval.py`; TGB-Seq's
fixed negatives are served by `TGBSeqEvaluator` in `tgb_seq_eval.py`.

The Historical (per-source reservoir + Vitter R) sampler was dropped:
on recurrence-dominated datasets like tgbl-wiki it actively trained
the model AGAINST the eval signal — most eval positives are nodes
the source has previously interacted with, and historical negatives
push E[u] away from exactly those.
"""

import abc
from typing import Optional, Tuple

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
