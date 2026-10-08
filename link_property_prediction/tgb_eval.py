"""TGB (`tgb.linkproppred`) — native load, negatives, and metric.

Restores the suite dropped at `ec6e448`, adapted to the `Evaluator` contract as it
stands today. Two deliberate differences from the 2026-08 version:

  - `sample_negatives` returns ONLY the negative destinations, matching the ABC.
    The old version returned `(neg_src_list, neg_tgt_list)` where every element of
    the first was `np.full(len(arr), src)` -- redundant, because
    `UniformNegativeSampler.sample` documents that "the positive's source is
    implicit: every caller pairs the negatives with batch.src itself". Widening the
    ABC back to a pair would force TGB-Seq to return a dummy second element and
    push `if suite == ...` into the trainer.
  - `TGBNegativeSampler` is gone. It existed only to subclass `NegativeSampler`,
    whose `sample` is typed `-> np.ndarray` of shape [B, K]; TGB's negatives are
    variable-K lists and never satisfied that. `Evaluator.sample_negatives` is
    already the variable-K interface, so the query folds directly into the
    evaluator -- the same shape as `TGBSeqEvaluator`.

The two suites differ in exactly the axis MRR is sensitive to, so never pool their
numbers: TGB serves pre-generated per-positive negatives via `query_batch`, keyed on
(src, tgt, ts) and therefore order-independent; TGB-Seq walks a fixed `[N_pos, K]`
array with a cursor and needs contiguous batches. That is why `reset()` is a no-op
here and rewinds the cursor there.
"""
import re
import zipfile
from typing import List

import numpy as np

from .data import Batch, Loaded, SplitData
from .evaluator import DataSuite, Evaluator


def _strip_version_suffix(name: str) -> str:
    """Drop a trailing '-v<digits>' to map to the TGB registry key
    ('tgbl-review-v2' -> 'tgbl-review'); the suffixed form raises inside TGB."""
    return re.sub(r"-v\d+$", "", name)


def _missing_dataset_error(tgb_name: str, data_dir: str, detail: str) -> FileNotFoundError:
    """One message for every way a TGB dataset can end up unusable: where its files
    belong, and what the automatic download fetches."""
    from tgb.utils.info import DATA_URL_DICT

    return FileNotFoundError(
        f"TGB dataset {tgb_name!r} is not usable ({detail}).\n"
        f"Its files (edgelist + val/test negatives) belong in:\n"
        f"  {data_dir}\n"
        f"The automatic download fetches {DATA_URL_DICT.get(tgb_name, '(no URL registered)')}; "
        f"if that host no longer serves the data, upgrade py-tgb for a newer URL or extract "
        f"the dataset there by hand. Pass --data-root to use a different directory.")


def load_tgb(name: str, root: str = "datasets") -> Loaded:
    """Load a TGB link-property-prediction dataset. `-vN` suffixes are stripped
    before the call — TGB's registry uses bare names.

    `root` is translated because TGB does `root = PROJ_DIR + root` (literal string
    concatenation, dataset.py:75), so its `root` is ALWAYS relative to the tgb
    install dir and an absolute path yields e.g.
    `.../site-packages/tgb//mnt/.../datasets/tgbl_review/` -- a real 2.1 GB download
    inside the venv, which is how this was found. Passing the relative path from
    PROJ_DIR makes `PROJ_DIR + root` resolve back to the directory the caller asked
    for, so `--data-root` is honoured rather than silently ignored. A RELATIVE root
    (the default "datasets") therefore lands inside the tgb install dir.

    A missing dataset is downloaded by TGB itself. TGB writes whatever the URL returns
    to `<name>.zip` without checking it: py-tgb 2.2.0's host returned a `NoSuchBucket`
    XML page, TGB crashed on it with `BadZipFile` and left the stub behind. So a bad zip
    is deleted here and turned into an error naming the dataset directory, and the
    val/test negative files are checked too, since TGB only checks the edgelist.
    """
    import os
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.utils.info import PROJ_DIR

    tgb_name = _strip_version_suffix(name)
    tgb_root = os.path.relpath(root, PROJ_DIR) if os.path.isabs(root) else root
    # TGB's own resolution (dataset.py:75-81), reproduced only for the error message.
    data_dir = os.path.normpath(os.path.join(PROJ_DIR + tgb_root, tgb_name.replace("-", "_")))
    try:
        dataset = LinkPropPredDataset(name=tgb_name, root=tgb_root, preprocess=True)
    except zipfile.BadZipFile as err:
        stub = os.path.join(data_dir, tgb_name + ".zip")
        if os.path.exists(stub):
            os.remove(stub)                       # else the next run trips on it too
        raise _missing_dataset_error(tgb_name, data_dir, "download was not a zip") from err
    missing = [dataset.meta_dict[k] for k in ("val_ns", "test_ns")
               if not os.path.exists(dataset.meta_dict[k])]
    if missing:
        raise _missing_dataset_error(tgb_name, data_dir, "missing " + ", ".join(missing))
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

    return Loaded(
        train=_apply(train_mask),
        val=_apply(val_mask),
        test=_apply(test_mask),
        dataset=dataset,
        # TGB-canonical (suffix-stripped) name — the Evaluator passes it back to TGB.
        name=tgb_name,
        max_node_count=int(max(sources.max(), destinations.max())) + 1,
    )


class TGBEvaluator(Evaluator):
    """TGB's native evaluation: pre-generated per-positive negatives from
    `dataset.negative_sampler.query_batch`, plus the official
    `tgb.linkproppred.evaluate.Evaluator` metric.

    Content-addressed by (src, tgt, ts), so it is order-independent and `reset()`
    stays the inherited no-op. K varies per positive.
    """

    def __init__(self, dataset: object, split_mode: str,
                 tgb_dataset_name: str, eval_metric: str):
        from tgb.linkproppred.evaluate import Evaluator as TGBOfficialEvaluator

        if split_mode not in ("val", "test"):
            raise ValueError(f"split_mode must be 'val' or 'test', got {split_mode!r}")
        self._dataset = dataset
        self._split_mode = split_mode
        self._sampler = dataset.negative_sampler
        self.eval_metric = eval_metric
        self._tgb_eval = TGBOfficialEvaluator(name=tgb_dataset_name)

    def sample_negatives(self, batch: Batch) -> List[np.ndarray]:
        """`neg_tgt_list` — one negative-destination array per positive, K may vary."""
        neg_dst_list = self._sampler.query_batch(
            batch.src, batch.tgt, batch.ts, split_mode=self._split_mode,
        )
        return [np.asarray(d, dtype=np.int32) for d in neg_dst_list]

    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        res = self._tgb_eval.eval({
            "y_pred_pos": np.asarray([pos_score], dtype=np.float64),
            "y_pred_neg": np.asarray(neg_scores, dtype=np.float64),
            "eval_metric": [self.eval_metric],
        })
        return float(res[self.eval_metric])


class TGBSuite(DataSuite):
    """TGB adapter. TGB's own pre-generated negatives are used for BOTH val and test,
    so `--k-eval` has no effect on this path: the leaderboard fixes the negatives.
    The notice lives here rather than in the train script so `make_suite` stays the
    only place that knows which suite is running."""

    def _load(self) -> Loaded:
        return load_tgb(self.name, self.root)

    def make_evaluator(self, split_mode: str) -> Evaluator:
        loaded = self.load()
        # TGB requires the split's negative-sampler file loaded before querying;
        # cached on disk after the first call.
        if split_mode == "val":
            loaded.dataset.load_val_ns()
        elif split_mode == "test":
            loaded.dataset.load_test_ns()
        else:
            raise ValueError(f"split_mode must be 'val' or 'test', got {split_mode!r}")
        print(f"  [tgb] {split_mode} negatives: TGB's pre-generated set "
              f"(--k-eval={self.k_eval} ignored)")
        return TGBEvaluator(
            dataset=loaded.dataset,
            split_mode=split_mode,
            tgb_dataset_name=loaded.name,
            # Off the live handle, not off Loaded: `eval_metric` is TGB-specific and
            # Loaded is the vocabulary BOTH suites share. Keeping it out is what lets
            # a suite carry its own metadata without widening the common type.
            eval_metric=str(loaded.dataset.eval_metric),
        )
