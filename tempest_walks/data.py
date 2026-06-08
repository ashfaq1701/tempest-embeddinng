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

    # ===================== TEMP DEBUG (read-only .values crash) =====================
    # Diagnose why TGB preprocessing hits "output array is read-only" on the
    # server. Prints to stderr, then proceeds (may still crash) so the console
    # shows both the diagnostics and the traceback. REMOVE after we have the fix.
    import sys as _sys, pandas as _pd, numpy as _np

    def _p(*a):
        print("[CoW-DEBUG]", *a, file=_sys.stderr, flush=True)

    def _probe(tag):
        try:
            df = _pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            v = df.iloc[:, 1].values
            wr = bool(v.flags.writeable)
            try:
                v += 1
                ip = "OK"
            except Exception as e:
                ip = f"FAIL({type(e).__name__}: {e})"
            _p(f"probe[{tag}] .values.writeable={wr}  in-place+= {ip}")
        except Exception as e:
            _p(f"probe[{tag}] RAISED: {type(e).__name__}: {e}")

    _p("python", _sys.version.split()[0], "| pandas", _pd.__version__,
       "| numpy", _np.__version__)
    try:
        _orig_cow = _pd.get_option("mode.copy_on_write")
        _p("mode.copy_on_write (current):", _orig_cow)
    except Exception as e:
        _orig_cow = None
        _p("get_option(mode.copy_on_write) RAISED:", repr(e))
    _probe("default")
    # Can CoW be turned OFF globally? (pandas 3.0 may forbid it.)
    try:
        _pd.set_option("mode.copy_on_write", False)
        _p("set_option(mode.copy_on_write, False): OK -> now =",
           _pd.get_option("mode.copy_on_write"))
        _probe("after set False")
    except Exception as e:
        _p("set_option(mode.copy_on_write, False) RAISED:", repr(e))
    # option_context path (what the reverted fix used)
    try:
        with _pd.option_context("mode.copy_on_write", False):
            _p("inside option_context(False): mode.copy_on_write =",
               _pd.get_option("mode.copy_on_write"))
            _probe("inside option_context(False)")
    except Exception as e:
        _p("option_context(mode.copy_on_write, False) RAISED:", repr(e))
    # Force read-only regime, then test which copy makes it writeable.
    try:
        _pd.set_option("mode.copy_on_write", True)
        df = _pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        _p("under CoW=True: .values.writeable =",
           bool(df.iloc[:, 1].values.flags.writeable))
        _p("  .values.copy().writeable     =",
           bool(df.iloc[:, 1].values.copy().flags.writeable))
        _p("  np.array(.values).writeable  =",
           bool(_np.array(df.iloc[:, 1].values).flags.writeable))
        _p("  to_numpy(copy=True).writeable=",
           bool(df.iloc[:, 1].to_numpy(copy=True).flags.writeable))
    except Exception as e:
        _p("copy-workaround probe RAISED:", repr(e))
    # Restore CoW to the process's original setting (so construction below
    # reproduces the server's natural state and we leave no side effect).
    try:
        if _orig_cow is not None:
            _pd.set_option("mode.copy_on_write", _orig_cow)
    except Exception:
        pass
    _p("restored mode.copy_on_write ->", _orig_cow,
       "| now calling LinkPropPredDataset(preprocess=True) -- crashes here if unfixed")
    # ===============================================================================

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
