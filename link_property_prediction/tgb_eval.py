"""TGB (`tgb.linkproppred`) — native load, negatives, and metric.

The `LinkPropPredDataset` loader, the eval-time per-positive negative sampler
(TGB's pre-generated lists via `negative_sampler.query_batch`), the official
Evaluator wrapper, and the `DataSuite` that wires them together.
"""
import re
from typing import List, Optional, Tuple

import numpy as np

from .data import Batch, Loaded, SplitData
from .evaluator import DataSuite, Evaluator
from .negatives import NegativeSampler


def _strip_version_suffix(name: str) -> str:
    """Drop a trailing '-v<digits>' to map to the TGB registry key
    ('tgbl-review-v2' -> 'tgbl-review'); the suffixed form raises inside TGB."""
    return re.sub(r"-v\d+$", "", name)


def load_tgb(name: str, root: str = "datasets") -> Loaded:
    """Load a TGB link-property-prediction dataset. `-vN` suffixes are stripped
    before the call — TGB's registry uses bare names."""
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
        # TGB-canonical (suffix-stripped) name — the Evaluator passes it back to TGB.
        name=tgb_name,
        eval_metric=str(dataset.eval_metric),
        max_node_count=int(max(sources.max(), destinations.max())) + 1,
        node_feat=node_feat,
    )


class TGBNegativeSampler(NegativeSampler):
    """Eval-time sampler wrapping `dataset.negative_sampler.query_batch`, which
    serves TGB's pre-generated per-positive negatives (variable-K → list-of-arrays).
    Content-addressed by (src, tgt, ts), so it is order-independent."""

    def __init__(self, dataset: object, split_mode: str):
        if split_mode not in ("val", "test"):
            raise ValueError(f"split_mode must be 'val' or 'test', got {split_mode!r}")
        self.dataset = dataset
        self.split_mode = split_mode
        self._sampler = dataset.negative_sampler

    def sample(self, batch: Batch) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        neg_dst_list = self._sampler.query_batch(
            batch.src, batch.tgt, batch.ts, split_mode=self.split_mode,
        )
        neg_src_list: List[np.ndarray] = []
        neg_tgt_list: List[np.ndarray] = []
        for s, neg_dsts in zip(batch.src, neg_dst_list):
            arr = np.asarray(neg_dsts, dtype=np.int32)
            neg_src_list.append(np.full(len(arr), s, dtype=np.int32))
            neg_tgt_list.append(arr)
        return neg_src_list, neg_tgt_list


class TGBEvaluator(Evaluator):
    """TGB's native evaluation: pre-generated per-positive negatives + the
    official `tgb…evaluate.Evaluator` metric. Stateless, so `reset()` is a no-op."""

    def __init__(self, dataset: object, split_mode: str,
                 tgb_dataset_name: str, eval_metric: str):
        from tgb.linkproppred.evaluate import Evaluator as TGBOfficialEvaluator

        self._neg = TGBNegativeSampler(dataset, split_mode)
        self.eval_metric = eval_metric
        self._tgb_eval = TGBOfficialEvaluator(name=tgb_dataset_name)

    def sample_negatives(self, batch: Batch) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return self._neg.sample(batch)

    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        res = self._tgb_eval.eval({
            "y_pred_pos": np.asarray([pos_score], dtype=np.float64),
            "y_pred_neg": np.asarray(neg_scores, dtype=np.float64),
            "eval_metric": [self.eval_metric],
        })
        return float(res[self.eval_metric])


class TGBSuite(DataSuite):
    """TGB adapter. Uses TGB's own pre-generated negatives for BOTH val and test,
    so `--k-eval` is ignored on this path (the leaderboard fixes the negatives)."""

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
        return TGBEvaluator(
            dataset=loaded.dataset,
            split_mode=split_mode,
            tgb_dataset_name=loaded.name,
            eval_metric=loaded.eval_metric,
        )
