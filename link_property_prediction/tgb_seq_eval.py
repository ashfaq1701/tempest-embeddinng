"""TGB-Seq (`tgb_seq.LinkPred`) — native load, negatives, and metric.

Everything TGB-Seq lives here and nothing else: the `TGBSeqLoader` (src/dst/
time/split CSV), TGB-Seq's own batched-MRR `Evaluator`, and its shipped
per-positive TEST negatives (`test_ns.npy`).

Two suite-native decisions are encoded here:
  * TGB-Seq ships TEST negatives but NO validation negatives, so the VAL set is
    built ONCE from our `UniformNegativeSampler` at `k_eval` (fixed & seeded, so
    eval is identical every epoch). TEST uses the shipped `test_ns` verbatim.
  * TGB-Seq timestamps are float64; Tempest's per-query cutoff needs a monotone
    int64, so `_to_int64_time` casts them (integral floats cast directly; the
    rare non-integral case is order-preserving dense-ranked).
"""
from typing import List, Tuple

import numpy as np

from .data import Batch, Loaded, SplitData
from .evaluator import DataSuite, Evaluator
from .negatives import UniformNegativeSampler


def _to_int64_time(time: np.ndarray) -> np.ndarray:
    """Monotone int64 timestamps for Tempest cutoffs.

    Integral floats (the common case — TGB-Seq times are typically unix seconds)
    cast directly, preserving the true gaps the age features read. Non-integral
    inputs fall back to an order-preserving dense rank (equal times → equal
    ranks); this keeps cutoff ordering exact but flattens absolute gaps."""
    t = np.asarray(time)
    if np.isfinite(t).all() and np.array_equal(t, np.floor(t)):
        return t.astype(np.int64)
    order = np.unique(t)                                  # sorted unique values
    return np.searchsorted(order, t).astype(np.int64)     # dense rank, monotone


def load_tgb_seq(name: str, root: str = "datasets") -> Loaded:
    """Native TGB-Seq load → suite-agnostic `Loaded`. Splits come from the CSV's
    `split` column (0=train, 1=val, 2=test); the live `TGBSeqLoader` is kept on
    `Loaded.dataset` so the test evaluator can read its shipped negatives."""
    import os

    from tgb_seq.LinkPred.dataloader import TGBSeqLoader

    ds = TGBSeqLoader(name=name, root=root)
    # Self-heal a half-downloaded dataset. TGBSeqLoader auto-downloads only when the dataset dir or the
    # edgelist CSV is missing; a MISSING test_ns raises a Warning it silently swallows (see its _load_file),
    # then the run dies later at eval-setup. So check BOTH files and, if either is absent, trigger the
    # loader's own download (fetches CSV + test_ns from HF, idempotent for files already present) and reload.
    if not (os.path.exists(ds._edgelist_path) and os.path.exists(ds._test_ns_path)):
        ds._download()
        ds._load_file()

    src = np.asarray(ds.src_node_ids, dtype=np.int64)
    dst = np.asarray(ds.dst_node_ids, dtype=np.int64)
    ts = _to_int64_time(ds.node_interact_times)

    edge_feat = ds.edge_features
    edge_feat = np.asarray(edge_feat, dtype=np.float32) if edge_feat is not None else None
    node_feat = ds.node_features
    node_feat = np.asarray(node_feat, dtype=np.float32) if node_feat is not None else None

    def _split(mask: np.ndarray) -> SplitData:
        m = np.asarray(mask, dtype=bool)
        ef = edge_feat[m] if edge_feat is not None else None
        return SplitData(sources=src[m], destinations=dst[m], timestamps=ts[m], edge_feat=ef)

    return Loaded(
        train=_split(ds.train_mask),
        val=_split(ds.val_mask),
        test=_split(ds.test_mask),
        dataset=ds,
        name=name,
        eval_metric="mrr",                                # TGB-Seq evaluates MRR only
        max_node_count=int(max(src.max(), dst.max())) + 1,
        node_feat=node_feat,
    )


def build_eval_negatives(split: SplitData, dst_pool: np.ndarray,
                         k_eval: int, seed: int) -> np.ndarray:
    """One-shot fixed `[N_pos, k_eval]` negatives for a split, drawn once from
    our `UniformNegativeSampler` over `dst_pool` (seeded → deterministic across
    epochs). The whole split is one sampler call; row i are positive i's negs."""
    sampler = UniformNegativeSampler(num_neg_per_pos=k_eval, dst_pool=dst_pool, seed=seed)
    whole = Batch(src=split.sources, tgt=split.destinations,
                  ts=split.timestamps, edge_feat=None)
    _, neg_tgt = sampler.sample(whole)                    # [N_pos, k_eval]
    return neg_tgt.astype(np.int32)


class TGBSeqEvaluator(Evaluator):
    """Fixed per-positive negatives (a `[N_pos, K]` array, row-aligned to the
    split's chronological order) + TGB-Seq's native batched-MRR `Evaluator`.

    Eval batches are contiguous chronological slices, so a cursor walks the
    negative array in lockstep; `reset()` rewinds it at the start of each eval
    pass so every epoch scores against the same negatives. (Split truncation is
    a prefix, so a negatives array built over the full split stays aligned.)"""

    def __init__(self, neg_dst: np.ndarray):
        from tgb_seq.LinkPred.evaluator import Evaluator as TGBSeqOfficialEvaluator

        self._neg_dst = np.asarray(neg_dst, dtype=np.int32)   # [N_pos, K]
        self._cursor = 0
        self._eval = TGBSeqOfficialEvaluator()

    def reset(self) -> None:
        self._cursor = 0

    def sample_negatives(self, batch: Batch) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        b = len(batch.src)
        rows = self._neg_dst[self._cursor:self._cursor + b]
        self._cursor += b
        neg_tgt_list = [rows[i] for i in range(b)]
        neg_src_list = [np.full(rows.shape[1], batch.src[i], dtype=np.int32) for i in range(b)]
        return neg_src_list, neg_tgt_list

    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        pos = np.asarray([pos_score], dtype=np.float64)
        neg = np.asarray(neg_scores, dtype=np.float64).reshape(1, -1)
        return float(self._eval.eval(pos, neg)[0])            # mrr_list[0]


class TGBSeqSuite(DataSuite):
    """TGB-Seq adapter. VAL negatives are built once from our sampler at
    `k_eval`; TEST uses TGB-Seq's shipped `test_ns` verbatim."""

    def _load(self) -> Loaded:
        return load_tgb_seq(self.name, self.root)

    def make_evaluator(self, split_mode: str) -> Evaluator:
        loaded = self.load()
        if split_mode == "val":
            neg = build_eval_negatives(loaded.val, self.dst_pool(), self.k_eval, self.seed)
            return TGBSeqEvaluator(neg_dst=neg)
        if split_mode == "test":
            neg = loaded.dataset.negative_samples          # shipped test_ns.npy [N_test, K]
            if neg is None:
                raise FileNotFoundError(
                    f"TGB-Seq test negatives (test_ns) not found for {self.name!r}; "
                    "they normally download with the dataset — re-run the loader to fetch them.")
            neg = np.asarray(neg, dtype=np.int32)
            n_test = len(loaded.test.sources)
            if neg.shape[0] != n_test:
                raise ValueError(
                    f"test_ns rows ({neg.shape[0]}) != test positives ({n_test}); "
                    "the shipped negatives are not row-aligned to the test split.")
            return TGBSeqEvaluator(neg_dst=neg)
        raise ValueError(f"split_mode must be 'val' or 'test', got {split_mode!r}")
