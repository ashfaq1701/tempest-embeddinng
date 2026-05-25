"""Pure helper utilities shared across trainer and CLI.

Functions in this module have no module-level state and don't depend
on any class instance. They are pulled here so trainer.py and
scripts/train.py stay focused on orchestration, not boilerplate.

Contents:
  Determinism:
    - seed_all(seed)              — seed Python/numpy/torch RNGs.
  Dataset derivation:
    - derive_t_train(train_ts)    — span of training timestamps.
    - detect_bipartite(train_split) — src/dst disjointness check.
  LR schedule:
    - make_lr_lambda(warmup_steps, decay_steps, lr_min_ratio)
                                  — closure for LambdaLR that does
                                    linear warmup then cosine decay
                                    to lr_min_ratio.
"""

import math
import random
from typing import Callable

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────


def seed_all(seed: int) -> None:
    """Seed every standard RNG from one root seed.

    Sampler-internal RNGs (negative samplers) are seeded via
    TrainerConfig.seed downstream. Tempest's walk RNG is NOT
    controlled here — Tempest CPU mode uses its own internal RNG
    and may exhibit small run-to-run drift even with the same Python
    seed. Multi-seed anchoring is the correct way to measure this.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────
# Dataset derivation
# ──────────────────────────────────────────────────────────────────────


def derive_t_train(train_ts: np.ndarray) -> float:
    """T_train: training-span (max - min). Required > 0 — used as a
    denominator in alignment_loss's time weighting."""
    if train_ts.size == 0:
        raise ValueError("Empty training timestamps; cannot derive T_train.")
    span = float(train_ts.max() - train_ts.min())
    if span <= 0:
        raise ValueError(f"Non-positive T_train: {span}")
    return span


def detect_bipartite(train_split) -> bool:
    """A graph is bipartite (under the link-pred convention) iff the
    set of source IDs and the set of destination IDs are disjoint.
    Holds for tgbl-wiki (users→pages), tgbl-review (users→items),
    tgbl-subreddit (users→subreddits). Fails for tgbl-coin / tgbl-flight
    / tgbl-comment where any node can be either endpoint."""
    src_set = set(np.unique(train_split.sources).tolist())
    dst_set = set(np.unique(train_split.destinations).tolist())
    return src_set.isdisjoint(dst_set)


# ──────────────────────────────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────────────────────────────


def make_lr_lambda(
    warmup_steps: int,
    decay_steps: int,
    lr_min_ratio: float,
) -> Callable[[int], float]:
    """Build a LambdaLR lambda for linear warmup + cosine decay.

    Shape:
      step 0..warmup_steps    linear ramp from 0 (at step 0) to 1.0
      step warmup..decay      cosine from 1.0 to lr_min_ratio
      step > decay_steps      stay at lr_min_ratio

    lr_min_ratio is lr_min / peak_lr. The lambda returns a scale
    factor that LambdaLR multiplies by the optimizer's initial_lr.
    """

    def lr_lambda(step: int) -> float:
        # step is 0-indexed (PyTorch LambdaLR convention).
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_progress = step - warmup_steps
        decay_total = decay_steps - warmup_steps
        if decay_total <= 0:
            return lr_min_ratio

        progress = float(decay_progress) / float(decay_total)
        if progress >= 1.0:
            return lr_min_ratio

        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cos_factor

    return lr_lambda
