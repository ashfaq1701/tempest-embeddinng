"""Immutable bundle of data-driven training-set constants, computed once at data load.

Fields:
    t_min, t_max                — min/max training timestamp
    T_train                     — span (t_max - t_min), > 0
    median_inter_arrival        — median Δt between consecutive events
    mean_inter_arrival          — mean Δt between consecutive events
    ts_quantum                  — p10 Δt between adjacent DISTINCT timestamps: the resolution
                                  of the time axis (86,400 on daily data, 1 on second-grained)
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
    ts_quantum: float


def compute_train_stats(timestamps: np.ndarray) -> TrainStats:
    """Compute every derived constant from the training-split timestamps.

    Inter-arrival stats use Δt between sorted consecutive events, excluding zero gaps
    (same-timestamp events are common and would skew the central tendency). That same
    non-zero gap array is exactly the set of gaps between adjacent DISTINCT timestamps, so
    `ts_quantum` -- the resolution of the time axis -- is a percentile of it.

    ts_quantum uses p10 rather than the minimum: the minimum is a single order statistic that
    one stray off-grid timestamp destroys, and an under-estimated quantum silently disables the
    resolution floor it feeds. Over-estimating it is the graceful direction -- it would take a
    ~100x over-estimate to compress the shortest ages, and p10/min is at most 4x on this suite.
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
        ts_quantum = 1.0
    else:
        median_ia = float(np.median(gaps))
        mean_ia = float(np.mean(gaps))
        ts_quantum = float(np.percentile(gaps, 10))

    return TrainStats(
        t_min=t_min,
        t_max=t_max,
        T_train=T_train,
        median_inter_arrival=median_ia,
        mean_inter_arrival=mean_ia,
        ts_quantum=ts_quantum,
    )
