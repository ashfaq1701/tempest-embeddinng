"""Stateless helpers shared across trainer and CLI: RNG seeding."""

import random

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
