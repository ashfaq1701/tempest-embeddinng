"""Suite-agnostic data containers (`SplitData`, `Batch`, `Loaded`) and the fixed-size
chronological batch iterator. Suite modules do the loading."""

from typing import Iterator, NamedTuple, Optional

import numpy as np


class SplitData(NamedTuple):
    sources: np.ndarray         # [E] int64
    destinations: np.ndarray    # [E] int64
    timestamps: np.ndarray      # [E] int64
    edge_feat: Optional[np.ndarray]   # [E, d_edge] float32 or None


class Batch(NamedTuple):
    src: np.ndarray
    tgt: np.ndarray
    ts: np.ndarray
    edge_feat: Optional[np.ndarray]


class Loaded(NamedTuple):
    train: SplitData
    val: SplitData
    test: SplitData
    dataset: object             # live suite dataset handle (negatives + eval)
    name: str                   # dataset name (for the Evaluator)
    max_node_count: int
    # Train-split time span, max(t) - min(t), in the SAME units the loader emits
    # (already dense-ranked to int64 when the raw stamps are non-integral). It is the
    # scale the pooler divides ages by: ages are cutoff - t_edge on that same axis, so
    # age / T_train is unitless and comparable across datasets, whose raw stamp units
    # differ by orders of magnitude. REQUIRED and no default: it is a divisor, so a
    # silent 0 would put NaNs straight into the pooling weights.
    T_train: int


def concat_splits(*splits: SplitData) -> SplitData:
    """Concatenate splits into ONE SplitData for a one-shot ingest. edge_feat is
    concatenated only when every split has it (else None)."""
    src = np.concatenate([s.sources for s in splits])
    dst = np.concatenate([s.destinations for s in splits])
    ts = np.concatenate([s.timestamps for s in splits])
    efs = [s.edge_feat for s in splits]
    ef = np.concatenate(efs) if all(e is not None for e in efs) else None
    return SplitData(sources=src, destinations=dst, timestamps=ts, edge_feat=ef)


def create_batches(split: SplitData, batch_size: int) -> Iterator[Batch]:
    """Fixed-size chronological batches (train and eval): consecutive
    `batch_size` chunks over the time-sorted stream. The final partial batch is kept
    (drop_last=False). Timestamps split freely across boundaries, so same-timestamp
    edges in different batches see each other's ingested state; within one batch they
    don't (ingest is post-batch)."""
    n = int(split.sources.shape[0])
    for start in range(0, n, batch_size):
        yield _slice(split, start, min(start + batch_size, n))


def _slice(split: SplitData, start: int, end: int) -> Batch:
    ef = split.edge_feat[start:end] if split.edge_feat is not None else None
    return Batch(
        src=split.sources[start:end],
        tgt=split.destinations[start:end],
        ts=split.timestamps[start:end],
        edge_feat=ef,
    )
