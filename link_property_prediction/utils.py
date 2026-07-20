"""Pure helper utilities shared across trainer and CLI.

Functions in this module have no module-level state and don't depend
on any class instance. They are pulled here so trainer.py and
scripts/train_link_property_prediction.py stay focused on orchestration, not boilerplate.

Contents:
  Determinism:
    - seed_all(seed)              — seed Python/numpy/torch RNGs.
  LR schedule:
    - make_lr_lambda(decay_steps, lr_min_ratio)
                                  — closure for LambdaLR that does
                                    cosine decay to lr_min_ratio.

Dataset-derived constants now live in `link_property_prediction/data_stats.py`
(TrainStats bundle).
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
# LR schedule
# ──────────────────────────────────────────────────────────────────────


def make_lr_lambda(
    decay_steps: int,
    lr_min_ratio: float,
) -> Callable[[int], float]:
    """Build a LambdaLR lambda for cosine decay from 1.0 (step 0) to lr_min_ratio (step decay_steps),
    then flat at lr_min_ratio. lr_min_ratio = lr_min / peak_lr; the lambda scales the optimizer's
    initial_lr. No warmup — a value test on the winner found warmup added nothing (marginally hurt)."""

    def lr_lambda(step: int) -> float:
        # step is 0-indexed (PyTorch LambdaLR convention).
        if decay_steps <= 0:
            return lr_min_ratio
        progress = min(1.0, float(step) / float(decay_steps))
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cos_factor

    return lr_lambda
