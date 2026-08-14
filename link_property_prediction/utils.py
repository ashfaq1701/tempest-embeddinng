"""Stateless helpers shared across trainer and CLI: RNG seeding and a cosine-decay
LambdaLR closure."""

import math
import random
from typing import Callable

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python/numpy/torch RNGs. Tempest's walk RNG is NOT controlled here and may
    drift run-to-run even at the same seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lr_lambda(
    decay_steps: int,
    lr_min_ratio: float,
) -> Callable[[int], float]:
    """LambdaLR lambda: cosine decay from 1.0 (step 0) to lr_min_ratio (step decay_steps),
    then flat. lr_min_ratio = lr_min / peak_lr; scales the optimizer's initial_lr."""

    def lr_lambda(step: int) -> float:
        # step is 0-indexed (PyTorch LambdaLR convention).
        if decay_steps <= 0:
            return lr_min_ratio
        progress = min(1.0, float(step) / float(decay_steps))
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cos_factor

    return lr_lambda
