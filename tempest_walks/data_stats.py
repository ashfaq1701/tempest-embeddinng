"""Data-driven training-set statistics, computed once at data load.

Bundles every derived constant the loss / sampler / scheduling code
needs into a single immutable object so the same numbers are computed
in one place and consumed consistently downstream. Add new fields here
when a new data-driven constant is needed; downstream code reads from
the bundle by name and doesn't recompute from raw timestamps.

Today's fields:
    t_min                       — min training timestamp
    t_max                       — max training timestamp
    T_train                     — span (t_max - t_min), > 0
    t_max_full                  — max timestamp across train + val + test
    T_full                      — span (t_max_full - t_min), > 0
    median_inter_arrival        — median Δt between consecutive events
    mean_inter_arrival          — mean   Δt between consecutive events

Recipe to add a new field:
    1. Extend `TrainStats` dataclass with the new field.
    2. Compute it inside `compute_train_stats()` from the timestamp
       array (or, if it needs more than timestamps, change the
       signature to take a `SplitData` and update callers).
    3. Reference it by name from train.py / trainer.py.

Why a dataclass instead of more loose function args:
    - Adding a new constant only touches this file and its callers,
      not every config layer in between.
    - Read-only (`frozen=True`) prevents drift between "what was
      derived once" and "what gets used downstream".
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainStats:
    """Immutable bundle of data-driven training-set constants."""

    t_min: int
    t_max: int
    T_train: float
    t_max_full: int
    T_full: float
    median_inter_arrival: float
    mean_inter_arrival: float


def compute_train_stats(
    timestamps: np.ndarray,
    full_timestamps: np.ndarray = None,
) -> TrainStats:
    """Compute every derived constant from the training-split timestamp
    array (`loaded.train.timestamps`). Called once at data load.

    `full_timestamps`, when supplied (typically `concat(train, val,
    test)`), drives the `t_max_full` / `T_full` fields the v2 link-
    pred head uses to bound its per-position gap normaliser at both
    train and eval. When omitted, `t_max_full = t_max` and
    `T_full = T_train` — the train-only fallback.

    Inter-arrival statistics: Δt between sorted consecutive events,
    excluding zero gaps (multiple events sharing a timestamp are
    common in TGB datasets and would skew the central tendency).
    """
    ts = np.asarray(timestamps).astype(np.int64)
    if ts.size == 0:
        raise ValueError("Empty training timestamps; cannot derive TrainStats.")

    t_min = int(ts.min())
    t_max = int(ts.max())
    T_train = float(t_max - t_min)
    if T_train <= 0:
        raise ValueError(f"Non-positive T_train: {T_train}")

    if full_timestamps is not None and len(full_timestamps) > 0:
        t_max_full = int(np.asarray(full_timestamps).astype(np.int64).max())
    else:
        t_max_full = t_max
    if t_max_full < t_max:
        t_max_full = t_max
    T_full = float(t_max_full - t_min)

    gaps = np.diff(np.sort(ts))
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        # Pathological: all events share one timestamp. Fall back to
        # a tiny scale so downstream exp(-gap/scale) is well-defined.
        median_ia = 1.0
        mean_ia = 1.0
    else:
        median_ia = float(np.median(gaps))
        mean_ia = float(np.mean(gaps))

    return TrainStats(
        t_min=t_min,
        t_max=t_max,
        T_train=T_train,
        t_max_full=t_max_full,
        T_full=T_full,
        median_inter_arrival=median_ia,
        mean_inter_arrival=mean_ia,
    )
