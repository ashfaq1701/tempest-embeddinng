"""Benchmark-agnostic evaluation interfaces (Evaluator, DataSuite) and the
`make_suite` factory.

Two suites: TGB-Seq in `tgb_seq_eval.py` and TGB in `tgb_eval.py`. Everything
suite-specific lives behind these two ABCs -- `trainer.py`, `negatives.py` and
`data.py` never branch on the suite, and `make_suite` is the single place a name
maps to a class. A new suite is therefore additive: one module, one branch here,
one `--data-suite` choice.

The suites agree on the vocabulary (`Loaded`, `SplitData`, `Batch`) and on the
`Evaluator` contract; they differ only inside, which is what makes them
interchangeable. They do NOT agree on the negative-sampling protocol, so their
MRRs are not comparable with each other -- see `tgb_eval.py`."""
import abc
from typing import List, Optional

import numpy as np

from .data import Batch, Loaded


class Evaluator(abc.ABC):
    """Supplies per-positive negatives and the suite's native per-positive metric."""

    @abc.abstractmethod
    def sample_negatives(self, batch: Batch) -> List[np.ndarray]:
        """`neg_tgt_list` — one negative-destination array per positive (K may vary)."""

    @abc.abstractmethod
    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        """The suite's native metric for one positive scored against its negatives."""

    def reset(self) -> None:
        """Called at the start of every eval pass; rewinds per-positive cursors for
        fixed-negative evaluators. Stateless evaluators no-op."""
        return None


class DataSuite(abc.ABC):
    """Benchmark adapter: native load + native evaluator construction. One instance
    per run."""

    def __init__(self, name: str, root: str, is_bipartite: bool,
                 k_eval: int, seed: int):
        self.name = name
        self.root = root
        self.is_bipartite = bool(is_bipartite)
        self.k_eval = int(k_eval)
        self.seed = seed
        self._loaded: Optional[Loaded] = None

    def load(self) -> Loaded:
        """Native load, cached so downstream calls reuse one copy."""
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    @abc.abstractmethod
    def _load(self) -> Loaded:
        """Do the suite's native load and return a `Loaded`."""

    @abc.abstractmethod
    def make_evaluator(self, split_mode: str) -> Evaluator:
        """Native evaluator for `split_mode` in {'val', 'test'}."""

    def dst_pool(self) -> np.ndarray:
        """Negative-destination universe (int32) from the TRAIN split. Bipartite ->
        unique train destinations; non-bipartite -> all unique train nodes (src ∪ dst)."""
        train = self.load().train
        if self.is_bipartite:
            pool = np.unique(train.destinations)
        else:
            pool = np.unique(np.concatenate([train.sources, train.destinations]))
        return pool.astype(np.int32)


#: Valid `--data-suite` values. The train script reads this for its `choices`, so the
#: flag and the dispatch below cannot drift apart.
SUITES = ("tgb-seq", "tgb")


def make_suite(data_suite: str, **kwargs) -> DataSuite:
    """Dispatch `--data-suite` to its native suite. The ONLY dispatch site: callers
    pass `args.data_suite` through and never branch on it. Each suite is imported
    lazily to avoid an import cycle (they subclass the ABCs defined here) and so that
    a missing optional dependency only bites the suite that needs it."""
    if data_suite == "tgb-seq":
        from .tgb_seq_eval import TGBSeqSuite
        return TGBSeqSuite(**kwargs)
    if data_suite == "tgb":
        from .tgb_eval import TGBSuite
        return TGBSuite(**kwargs)
    raise ValueError(
        f"unknown --data-suite {data_suite!r} (expected one of {sorted(SUITES)})")
