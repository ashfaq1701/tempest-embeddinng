"""Benchmark-agnostic evaluation interfaces (Evaluator, DataSuite) and the
`make_suite` factory.

Three suites: TGB-Seq in `tgb_seq_eval.py`, TGB in `tgb_eval.py` and DyGLib in
`dyglib_eval.py`. Everything suite-specific lives behind these two ABCs --
`trainer.py`, `negatives.py` and `data.py` never branch on the suite, and
`make_suite` is the single place a name maps to a class. A new suite is therefore
additive: one module, one branch here, one `--data-suite` choice.

The suites agree on the vocabulary (`Loaded`, `SplitData`, `Batch`) and on the
`Evaluator` contract; they differ only inside, which is what makes them
interchangeable. They do NOT agree on the negative-sampling protocol, so their
MRRs are not comparable with each other -- see `tgb_eval.py`."""
import abc
from functools import cached_property
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

    def __init__(self, name: str, root: str, is_bipartite: bool, k_eval: int, seed: int):
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

    @property
    def candidate_kind(self) -> str:
        """What a negative can be: destinations on a bipartite graph, any node otherwise."""
        return "destinations" if self.is_bipartite else "nodes"

    def _candidates(self, *splits) -> np.ndarray:
        """Unique negative candidates (int32) over `splits`: their destinations when the
        graph is bipartite, their sources ∪ destinations when it is not."""
        ids = [s.destinations for s in splits]
        if not self.is_bipartite:
            ids += [s.sources for s in splits]
        return np.unique(np.concatenate(ids)).astype(np.int32)

    @cached_property
    def train_negative_pool(self) -> np.ndarray:
        """Training negatives come from the train split only, so a node that first appears
        in val/test is never trained on purely as a negative."""
        return self._candidates(self.load().train)

    @cached_property
    def eval_negative_pool(self) -> np.ndarray:
        """Our val and test negatives come from the full dataset (train ∪ val ∪ test), as
        CRAFT/DyGLib draw both."""
        loaded = self.load()
        return self._candidates(loaded.train, loaded.val, loaded.test)


#: Valid `--data-suite` values. The train script reads this for its `choices`, so the
#: flag and the dispatch below cannot drift apart.
SUITES = ("tgb-seq", "tgb", "dyglib")


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
    if data_suite == "dyglib":
        from .dyglib_eval import DyGLibSuite
        return DyGLibSuite(**kwargs)
    raise ValueError(
        f"unknown --data-suite {data_suite!r} (expected one of {sorted(SUITES)})")
