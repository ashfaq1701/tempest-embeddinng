"""Evaluation interface + data-suite factory — benchmark-agnostic.

Two abstractions the trainer and the training script build on. The concrete,
NATIVE implementations for each benchmark live in `tgb_eval.py` (TGB) and
`tgb_seq_eval.py` (TGB-Seq); this module holds only the interfaces and the
`make_suite` dispatch, so the two paths never interleave.

  Evaluator  — the trainer's ONLY eval dependency. Given a batch it yields
               per-positive negatives, and given one positive's scores it
               returns that suite's NATIVE metric. Nothing about the model
               architecture lives behind this seam.

  DataSuite  — one cohesive object per benchmark that owns its NATIVE load and
               NATIVE evaluator construction. `--data-suite` picks one; after
               that single dispatch every downstream call routes through the
               chosen suite object, with no per-suite branching anywhere else.
"""
import abc
from typing import List, Optional, Tuple

import numpy as np

from .data import Batch, Loaded


class Evaluator(abc.ABC):
    """Per-positive negatives + per-positive metric — the trainer's eval seam.

    A training loop asks this evaluator for a batch's negatives, scores them
    with its own model, and folds each (positive, negative-array) pair through
    `score_to_metric`. The suite decides where the negatives come from and which
    metric is applied; the trainer stays architecture- and benchmark-agnostic.
    """

    @abc.abstractmethod
    def sample_negatives(self, batch: Batch) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """`(neg_src_list, neg_tgt_list)` — one variable-length negative array
        per positive in `batch` (K may differ per positive across suites)."""

    @abc.abstractmethod
    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        """The suite's NATIVE metric for a single positive scored against its
        negatives (e.g. reciprocal rank → MRR)."""

    def reset(self) -> None:
        """Called by the trainer at the START of every eval pass. Stateless
        evaluators (TGB, whose negatives are content-addressed) no-op. The
        fixed-negative TGB-Seq evaluator rewinds its per-positive cursor so the
        same negatives are served every epoch, in split order."""
        return None


class DataSuite(abc.ABC):
    """A benchmark adapter: NATIVE load + NATIVE evaluator construction.

    One instance per run, selected by `--data-suite`. `load()` returns the
    suite-agnostic `Loaded` the trainer consumes (cached — evaluators reuse it);
    `make_evaluator(split_mode)` returns that split's native `Evaluator`.

    `dst_pool()` is shared: the `--is-bipartite` arg drives it, and the SAME
    pool feeds both the training negatives and any eval negatives a suite builds
    from our sampler, so training and evaluation draw from one destination
    universe.
    """

    def __init__(self, name: str, root: str, is_bipartite: bool,
                 k_eval: int, seed: int):
        self.name = name
        self.root = root
        self.is_bipartite = bool(is_bipartite)
        self.k_eval = int(k_eval)
        self.seed = seed
        self._loaded: Optional[Loaded] = None

    def load(self) -> Loaded:
        """Native load, cached so `make_evaluator`/`dst_pool` reuse one copy."""
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
        """Negative-destination universe (int32), from the TRAIN split.

        Bipartite -> unique train DESTINATIONS (keep negatives on the
        destination side; sampling all nodes would make the trivial "is this
        ever a destination?" task that doesn't transfer to eval). Non-bipartite
        -> all unique train nodes (src ∪ dst), since any node can be a target.
        """
        train = self.load().train
        if self.is_bipartite:
            pool = np.unique(train.destinations)
        else:
            pool = np.unique(np.concatenate([train.sources, train.destinations]))
        return pool.astype(np.int32)


def make_suite(data_suite: str, **kwargs) -> DataSuite:
    """Dispatch `--data-suite` to its native suite. Concrete suites are imported
    lazily HERE (not at module top) so this interface module carries no import
    cycle with the suite modules that import it back."""
    if data_suite == "tgb":
        from .tgb_eval import TGBSuite
        return TGBSuite(**kwargs)
    if data_suite == "tgb-seq":
        from .tgb_seq_eval import TGBSeqSuite
        return TGBSeqSuite(**kwargs)
    raise ValueError(
        f"unknown --data-suite {data_suite!r} (expected 'tgb' or 'tgb-seq')")
