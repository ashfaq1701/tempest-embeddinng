"""TGB dataset bridge + fixed-size (TGB-identical) batch iterator."""

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


def _strip_version_suffix(name: str) -> str:
    """Drop trailing '-v<digits>' from a TGB dataset name. Used to
    convert user-facing names ('tgbl-review-v2') into the TGB
    registry key ('tgbl-review'); the unstripped form raises 'Dataset
    not supported' inside TGB."""
    # E.g., 'tgbl-wiki-v2' -> 'tgbl-wiki'; 'tgbl-coin' unchanged.
    import re
    return re.sub(r"-v\d+$", "", name)


def load_tgb(name: str, root: str = "datasets") -> Loaded:
    """Load a TGB link-property-prediction dataset using only TGB's APIs.

    `-vN` suffixes (e.g. tgbl-review-v2) are stripped before the call —
    TGB's registry uses bare names ("tgbl-review") and serves the
    current version internally; passing "tgbl-review-v2" raises
    "Dataset not supported" because the suffixed key isn't registered.
    """
    from tgb.linkproppred.dataset import LinkPropPredDataset

    tgb_name = _strip_version_suffix(name)
    dataset = LinkPropPredDataset(name=tgb_name, root=root, preprocess=True)
    full = dataset.full_data
    sources = np.asarray(full["sources"], dtype=np.int64)
    destinations = np.asarray(full["destinations"], dtype=np.int64)
    timestamps = np.asarray(full["timestamps"], dtype=np.int64)
    edge_feat = full.get("edge_feat", None)
    if edge_feat is not None:
        edge_feat = np.asarray(edge_feat, dtype=np.float32)

    train_mask = np.asarray(dataset.train_mask, dtype=bool)
    val_mask = np.asarray(dataset.val_mask, dtype=bool)
    test_mask = np.asarray(dataset.test_mask, dtype=bool)

    def _apply(mask: np.ndarray) -> SplitData:
        ef = edge_feat[mask] if edge_feat is not None else None
        return SplitData(
            sources=sources[mask],
            destinations=destinations[mask],
            timestamps=timestamps[mask],
            edge_feat=ef,
        )

    # NB: explicit None checks — `ndarray or ...` raises "ambiguous truth value" when node_feat is an
    # array (e.g. tgbl-flight), so the short-circuit `or` form would crash exactly when features exist.
    node_feat = getattr(dataset, "node_feat", None)
    if node_feat is None:
        node_feat = full.get("node_feat", None)
    if node_feat is not None:
        node_feat = np.asarray(node_feat, dtype=np.float32)

    return Loaded(
        train=_apply(train_mask),
        val=_apply(val_mask),
        test=_apply(test_mask),
        dataset=dataset,
        # Store the TGB-canonical name (suffix stripped). Downstream
        # consumers (Evaluator) pass this back to TGB and must use
        # the registry key, not the user's input suffix.
        name=tgb_name,
        eval_metric=str(dataset.eval_metric),
        max_node_count=int(max(sources.max(), destinations.max())) + 1,
        node_feat=node_feat,
    )


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
