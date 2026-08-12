"""Suite-agnostic data structures + fixed-size (TGB-identical) batch iterator.

Benchmark-specific loading lives in the suite modules (`tgb_eval.py`,
`tgb_seq_eval.py`), each of which returns the `Loaded` defined here. This file
only holds the shared containers (`SplitData`, `Batch`, `Loaded`) and the
chronological batcher used by both training and eval on every suite."""

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
    dataset: object             # live LinkPropPredDataset (negative_sampler + eval)
    name: str                   # TGB name (for the Evaluator)
    eval_metric: str            # e.g. "mrr"
    max_node_count: int
    node_feat: Optional[np.ndarray]


def concat_splits(*splits: SplitData) -> SplitData:
    """Concatenate splits into ONE SplitData — the full graph as a single set of arrays, for a
    one-shot Tempest ingest. edge_feat is concatenated only when every split has it (else None)."""
    src = np.concatenate([s.sources for s in splits])
    dst = np.concatenate([s.destinations for s in splits])
    ts = np.concatenate([s.timestamps for s in splits])
    efs = [s.edge_feat for s in splits]
    ef = np.concatenate(efs) if all(e is not None for e in efs) else None
    return SplitData(sources=src, destinations=dst, timestamps=ts, edge_feat=ef)


def create_batches(split: SplitData, batch_size: int) -> Iterator[Batch]:
    """TGB-identical fixed-size chronological batches (train AND eval).

    Consecutive fixed-size chunks of exactly `batch_size` events over the
    already-time-sorted stream; timestamps are split freely across batch
    boundaries — byte-for-byte the partition produced by
    ``torch_geometric.TemporalDataLoader`` and by TPNet's
    ``DataLoader(range(n), batch_size, shuffle=False, drop_last=False)``, which
    every TGB leaderboard baseline uses, so our `batch_size` means exactly what
    theirs does. The final partial batch is kept (drop_last=False) so every eval
    positive is scored.

    Strict-causality consequence (intentional — this is what the stateful TGB
    baselines do): with timestamps now splittable, two edges sharing a timestamp
    can land in different batches, so the later batch sees the earlier edge's
    ingested graph state. Same-timestamp edges that land in
    the SAME batch still don't inform each other (ingest is post-batch). Per-edge
    quantities (t_query, the head's tok_age) are
    computed element-wise from `batch.ts` and are unaffected. Eval MRR is
    identical regardless of batching — TGB negatives are keyed per positive edge,
    not per batch — only causal-state freshness changes.
    """
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
