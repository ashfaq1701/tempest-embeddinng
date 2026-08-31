"""Immutable bundle of data-driven training-set constants, computed once at data load.

Fields:
    t_min, t_max                — min/max training timestamp
    T_train                     — span (t_max - t_min), > 0
    median_inter_arrival        — median Δt between consecutive events
    mean_inter_arrival          — mean Δt between consecutive events
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainStats:
    """Immutable bundle of data-driven training-set constants."""

    t_min: int
    t_max: int
    T_train: float
    median_inter_arrival: float
    mean_inter_arrival: float


def compute_train_stats(timestamps: np.ndarray) -> TrainStats:
    """Compute every derived constant from the training-split timestamps.

    Inter-arrival stats use Δt between sorted consecutive events, excluding zero gaps
    (same-timestamp events are common and would skew the central tendency).
    """
    ts = np.asarray(timestamps).astype(np.int64)
    if ts.size == 0:
        raise ValueError("Empty training timestamps; cannot derive TrainStats.")

    t_min = int(ts.min())
    t_max = int(ts.max())
    T_train = float(t_max - t_min)
    if T_train <= 0:
        raise ValueError(f"Non-positive T_train: {T_train}")

    gaps = np.diff(np.sort(ts))
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        # All events share one timestamp: fall back to a tiny scale.
        median_ia = 1.0
        mean_ia = 1.0
    else:
        median_ia = float(np.median(gaps))
        mean_ia = float(np.mean(gaps))

    return TrainStats(
        t_min=t_min,
        t_max=t_max,
        T_train=T_train,
        median_inter_arrival=median_ia,
        mean_inter_arrival=mean_ia,
    )
