"""TGB dataset bridge + timestamp-grouping batch iterator."""

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
    t_max: int


class Loaded(NamedTuple):
    train: SplitData
    val: SplitData
    test: SplitData
    dataset: object             # live LinkPropPredDataset (negative_sampler + eval)
    name: str                   # TGB name (for the Evaluator)
    eval_metric: str            # e.g. "mrr"
    max_node_count: int
    is_directed: bool
    node_feat: Optional[np.ndarray]


# Datasets known to be undirected (bipartite or otherwise). The
# check normalises by stripping any "-vN" version suffix, so
# "tgbl-wiki", "tgbl-wiki-v2", "tgbl-wiki-v3" all match — TGB's
# versioning bumps refresh negatives / fix data issues but do
# not change graph orientation.
_UNDIRECTED_BASE_NAMES = {
    "tgbl-wiki",
    "tgbl-review",
    "tgbl-subreddit",
    "tgbl-lastfm",
}


def _strip_version_suffix(name: str) -> str:
    """Drop trailing '-v<digits>' from a TGB dataset name."""
    # E.g., 'tgbl-wiki-v2' -> 'tgbl-wiki'; 'tgbl-coin' unchanged.
    import re
    return re.sub(r"-v\d+$", "", name)


def default_is_directed(name: str) -> bool:
    """Default directedness for known TGB datasets, ignoring version
    suffix. Returns True if the dataset isn't on the known-undirected
    list — but callers should expose an explicit override (the
    knowledge here is a default, not authoritative)."""
    base = _strip_version_suffix(name)
    return base not in _UNDIRECTED_BASE_NAMES


def load_tgb(name: str, root: str = "datasets") -> Loaded:
    """Load a TGB link-property-prediction dataset using only TGB's APIs."""
    from tgb.linkproppred.dataset import LinkPropPredDataset

    dataset = LinkPropPredDataset(name=name, root=root, preprocess=True)
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

    node_feat = getattr(dataset, "node_feat", None) or full.get("node_feat", None)
    if node_feat is not None:
        node_feat = np.asarray(node_feat, dtype=np.float32)

    return Loaded(
        train=_apply(train_mask),
        val=_apply(val_mask),
        test=_apply(test_mask),
        dataset=dataset,
        name=name,
        eval_metric=str(dataset.eval_metric),
        max_node_count=int(max(sources.max(), destinations.max())) + 1,
        is_directed=default_is_directed(name),
        node_feat=node_feat,
    )


def create_batches(split: SplitData, target_batch_size: int) -> Iterator[Batch]:
    """Timestamp-respecting chronological batches.

    All edges sharing a timestamp stay in the same batch (so the strict-
    causal "ingest after scoring" rule still partitions cleanly across
    batches). Batches grow until adding the next timestamp group would
    exceed `target_batch_size`.
    """
    n = int(split.sources.shape[0])
    if n == 0:
        return

    ts_change = np.where(np.diff(split.timestamps) != 0)[0] + 1
    group_starts = np.concatenate([[0], ts_change])
    group_ends = np.concatenate([ts_change, [n]])

    batch_start_group = 0
    for i in range(len(group_starts)):
        edges_so_far = group_ends[i] - group_starts[batch_start_group]
        if edges_so_far > target_batch_size and i > batch_start_group:
            yield _slice(split, group_starts[batch_start_group], group_starts[i])
            batch_start_group = i

    if group_starts[batch_start_group] < n:
        yield _slice(split, group_starts[batch_start_group], n)


def _slice(split: SplitData, start: int, end: int) -> Batch:
    ef = split.edge_feat[start:end] if split.edge_feat is not None else None
    return Batch(
        src=split.sources[start:end],
        tgt=split.destinations[start:end],
        ts=split.timestamps[start:end],
        edge_feat=ef,
        t_max=int(split.timestamps[end - 1]),
    )
