"""TGB dataset bridge + timestamp-grouping batch iterator."""

import contextlib

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


@contextlib.contextmanager
def _tgb_preprocess_compat():
    """Disable pandas Copy-on-Write for the duration of TGB dataset
    construction.

    TGB's first-time preprocessing (`tgb.utils.pre_process.load_edgelist_*`)
    reads edge columns via `df.iloc[:, k].values` and mutates them IN PLACE
    (e.g. `dst += int(src.max()) + 1`). Under pandas Copy-on-Write — the
    default in newer pandas builds — `.values` returns a READ-ONLY array,
    so the in-place op raises `ValueError: output array is read-only`. The
    crash happens inside LinkPropPredDataset construction, before any of
    our code runs. (Confirmed: toggling CoW reproduces it exactly; the dev
    laptop defaults to CoW off, which is why it never failed there.)

    Disabling CoW for this block restores writeable `.values`; pandas
    auto-restores the prior setting on exit. No-op where CoW is already
    off. If a future pandas removes the option, we fall back to running
    without the override rather than breaking dataset loading.
    """
    import pandas as pd

    try:
        cm = pd.option_context("mode.copy_on_write", False)
    except Exception:
        cm = contextlib.nullcontext()
    try:
        cm.__enter__()
    except Exception:
        cm = contextlib.nullcontext()
        cm.__enter__()
    try:
        yield
    finally:
        cm.__exit__(None, None, None)


def load_tgb(name: str, root: str = "datasets") -> Loaded:
    """Load a TGB link-property-prediction dataset using only TGB's APIs.

    `-vN` suffixes (e.g. tgbl-review-v2) are stripped before the call —
    TGB's registry uses bare names ("tgbl-review") and serves the
    current version internally; passing "tgbl-review-v2" raises
    "Dataset not supported" because the suffixed key isn't registered.

    Dataset construction runs inside `_tgb_preprocess_compat()`, which
    disables pandas Copy-on-Write so TGB's in-place-mutation preprocessing
    doesn't hit read-only `.values` arrays (see that helper). No-op when
    CoW is already off.
    """
    from tgb.linkproppred.dataset import LinkPropPredDataset

    tgb_name = _strip_version_suffix(name)
    with _tgb_preprocess_compat():
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

    node_feat = getattr(dataset, "node_feat", None) or full.get("node_feat", None)
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
    )
