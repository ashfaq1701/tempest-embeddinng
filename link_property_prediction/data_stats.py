"""Immutable bundle of data-driven training-set constants, computed once at data load.

Fields:
    t_min, t_max                — min/max training timestamp
    T_train                     — span (t_max - t_min) of the TRAIN split, > 0
    T_full                      — span over ALL splits. Ages are measured back from a query
                                  cutoff, so val/test ages exceed T_train; normalising by the
                                  full span keeps them in range
    median_inter_arrival        — median Δt between consecutive events
    mean_inter_arrival          — mean Δt between consecutive events
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrainStats:
    """Immutable bundle of data-driven training-set constants."""

    t_min: int
    t_max: int
    T_train: float
    T_full: float
    median_inter_arrival: float
    mean_inter_arrival: float


def compute_train_stats(timestamps: np.ndarray,
                        all_timestamps: Optional[np.ndarray] = None) -> TrainStats:
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

    # Span over every split. Ages are cutoff - t_edge and the cutoffs run to the end of test, so
    # a val/test age can exceed T_train; anything normalising by the train span alone would push
    # those past 1. Falls back to T_train when the caller passes train timestamps only.
    if all_timestamps is None:
        T_full = T_train
    else:
        a = np.asarray(all_timestamps).astype(np.int64)
        T_full = float(a.max() - a.min()) if a.size else T_train
        if T_full < T_train:
            raise ValueError(f"T_full ({T_full}) < T_train ({T_train}); wrong array passed?")

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
        T_full=T_full,
        median_inter_arrival=median_ia,
        mean_inter_arrival=mean_ia,
    )
