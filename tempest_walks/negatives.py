"""Negative samplers.

Training-side per-node interface (`NegativeSampler`):
    sample(nodes, num_neg) -> [len(nodes), num_neg] int negatives.
Two implementations share it:
  - UniformNegativeSampler    : uniform draws from a destination pool.
  - HistoricalNegativeSampler : per-source reservoir (Vitter R, fixed 128 pool)
                                returning a source's past partners as hard
                                negatives; observe() AFTER scoring, reset() per
                                epoch. Historical-only; accepts some false
                                negatives by design (cold-start / low-recurrence
                                data; NOT recurrence-dominated tgbl-wiki).
`combine_negatives(historical, uniform)` (free function) merges two per-node
negative arrays along the negative axis.

Eval uses a SEPARATE contract (not the per-node interface):
  - TGBNegativeSampler        : batch-based, wraps TGB's pregenerated negatives.
"""

import abc
from typing import List, Optional, Tuple

import numpy as np

from .data import Batch


class NegativeSampler(abc.ABC):
    """Training-side per-node negative sampler.

    `sample(nodes, num_neg) -> [len(nodes), num_neg]`. Stateful samplers update
    in `observe()` (called AFTER scoring — strict-causal) and clear in
    `reset()` (per epoch); stateless samplers no-op both (the defaults).
    """

    @abc.abstractmethod
    def sample(self, nodes: np.ndarray, num_neg: int) -> np.ndarray:
        """Return [len(nodes), num_neg] negatives for the given nodes."""

    def observe(self, src: np.ndarray, dst: np.ndarray) -> None:
        return None

    def reset(self) -> None:
        return None


class UniformNegativeSampler(NegativeSampler):
    """Uniform negatives from a destination pool, independent of node history.

    `dst_pool` keeps negatives on the destination side of a bipartite graph
    (tgbl-wiki / -review etc.): sampling the full node set would make the task
    the trivial "is this ever a destination?" and would not transfer to eval.
    """

    def __init__(self, dst_pool: np.ndarray, seed: Optional[int] = None):
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def sample(self, nodes: np.ndarray, num_neg: int) -> np.ndarray:
        """[len(nodes), num_neg] uniform negatives (the negs don't depend on
        the node — only its count is used)."""
        Q = np.asarray(nodes).shape[0]
        idx = self.rng.integers(0, self.dst_pool.shape[0], size=(Q, num_neg))
        return self.dst_pool[idx]


class HistoricalNegativeSampler(NegativeSampler):
    """Per-source reservoir of past destinations → HISTORICAL negatives.

    Each source keeps a FIXED pool of `reservoir_size` (default 128) of its
    past destinations, maintained as a uniform random sample of that source's
    history via Vitter's Algorithm R (accept the (count+1)-th item with
    probability M/(count+1), replacing a uniform-random slot; the fill phase
    accepts unconditionally). All numpy/CPU, fully vectorised.

    No positive-target exclusion — some false negatives are accepted by design.
    Empty reservoir slots (cold / under-filled sources) fall back to random
    `dst_pool` draws, so a full [., num_neg] row is always returned.
    """

    def __init__(
        self,
        num_nodes: int,
        dst_pool: np.ndarray,
        reservoir_size: int = 128,
        seed: Optional[int] = None,
    ):
        self.num_nodes = int(num_nodes)
        self.M = int(reservoir_size)
        self.reservoir = np.full((self.num_nodes, self.M), -1, dtype=np.int32)
        self.count = np.zeros(self.num_nodes, dtype=np.int64)
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.reservoir.fill(-1)
        self.count.fill(0)

    def observe(self, src: np.ndarray, dst: np.ndarray) -> None:
        """Vitter-R reservoir update over a batch. MUST run AFTER scoring.

        For a source repeated within the batch the last accepted write to a
        given (src, slot) wins — a negligible deviation from sequential Vitter R
        at batch sizes ≪ per-source accepted-write rate.
        """
        B = src.shape[0]
        if B == 0:
            return
        src_i = src.astype(np.int64, copy=False)
        dst_i = dst.astype(np.int32, copy=False)
        pre_count = self.count[src_i]

        fill_mask = pre_count < self.M
        accept = self.rng.random(size=B) < (self.M / (pre_count + 1).astype(np.float64))
        do_insert = fill_mask | accept
        insert_pos = np.where(fill_mask, pre_count, self.rng.integers(0, self.M, size=B))

        idx = np.where(do_insert)[0]
        if idx.size:
            self.reservoir[src_i[idx], insert_pos[idx]] = dst_i[idx]
        np.add.at(self.count, src_i, 1)

    def sample(self, nodes: np.ndarray, num_neg: int) -> np.ndarray:
        """[len(nodes), num_neg] historical negatives; empty slots (-1) filled
        with random `dst_pool` draws."""
        nodes = np.asarray(nodes, dtype=np.int64)
        Q = nodes.shape[0]
        slot = self.rng.integers(0, self.M, size=(Q, num_neg))
        neg = np.take_along_axis(
            self.reservoir[nodes], slot, axis=1,
        ).astype(np.int32, copy=False)
        empty = neg < 0
        if empty.any():
            rand = self.dst_pool[
                self.rng.integers(0, self.dst_pool.shape[0], size=(Q, num_neg))
            ]
            neg = np.where(empty, rand, neg)
        return neg


def combine_negatives(historical: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """Merge per-node historical and uniform negatives along the negative axis:
    [N, n_hist] + [N, n_unif] -> [N, n_hist + n_unif]."""
    historical = np.asarray(historical)
    uniform = np.asarray(uniform)
    if historical.shape[0] != uniform.shape[0]:
        raise ValueError(
            f"node-dim mismatch: historical {historical.shape[0]} vs "
            f"uniform {uniform.shape[0]}"
        )
    return np.concatenate([historical, uniform], axis=1)


class TGBNegativeSampler:
    """Eval-time sampler (NOT the per-node training interface). Wraps
    `dataset.negative_sampler.query_batch`, which serves TGB's pre-generated
    per-positive negatives. Batch-based, variable-K per positive — returns
    list-of-arrays."""

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
