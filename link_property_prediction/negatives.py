"""Negative samplers.

Interface (one seam for training AND eval):
  - NegativeSampler.sample(batch) -> (neg_src, neg_tgt)
  - NegativeSampler.observe(src, dst) -> None   (no-op default; overridden by
                                                 stateful samplers, called by
                                                 the trainer POST-scoring)
  - NegativeSampler.reset() -> None             (no-op default; called at the
                                                 start of every epoch)

Because `observe` / `reset` are no-ops on the base, the trainer calls them
unconditionally on whatever training sampler it holds — the historical/random
mix or a pure uniform sampler — with no `isinstance` branching.

Training samplers (this file):
  - UniformNegativeSampler   : random destinations from a pool (a provider; also
                               usable standalone). Dense [B, K]. Reused to build
                               TGB-Seq's fixed eval negatives one-shot (see
                               tgb_seq_eval.py).
  - HistoricalReservoir      : per-source reservoir (Vitter R) of PAST
                               destinations — a PROVIDER for the mixer, not a
                               standalone sampler. Exposes observe()/reset()/
                               draw(); the random fallback for cold sources is
                               the MIXER's job.
  - MixedNegativeSampler     : THE training entry point. Splits K into
                               K_hist/K_rand at `hist_ratio`, delegates each
                               portion to the historical / uniform providers,
                               backfills invalid historical slots with random,
                               returns the combined [B, K]. hist_ratio=0 ->
                               pure uniform (no reservoir allocated).

Eval-time, suite-native samplers live with their suite: TGB's per-positive
pre-generated negatives are `TGBNegativeSampler` in `tgb_eval.py`; TGB-Seq's
fixed negatives are served by `TGBSeqEvaluator` in `tgb_seq_eval.py`.

Historical negatives are NOT for tgbl-wiki: on recurrence-dominated datasets
most eval positives ARE historical, so training against them pushes E[u] away
from the eval signal and collapses MRR. Intended for low-recurrence datasets
(e.g. tgbl-review, ~8% historical eval negatives — see CLAUDE.md TGB stats).
"""

import abc
from typing import Optional, Tuple

import numpy as np

from .data import Batch


class NegativeSampler(abc.ABC):
    """Shared seam for training AND eval samplers."""

    @abc.abstractmethod
    def sample(self, batch: Batch):
        ...

    def observe(self, src: np.ndarray, dst: np.ndarray) -> None:
        """Feed observed positives to stateful samplers. Called by the trainer
        POST-scoring (strict-causal). No-op for stateless samplers."""
        return None

    def reset(self) -> None:
        """Drop per-epoch carry-over. No-op for stateless samplers."""
        return None


class UniformNegativeSampler(NegativeSampler):
    """Random destinations from a pool, keeping the positive's source.

    `dst_pool` is required for bipartite datasets (tgbl-wiki / -review etc.)
    so training negatives stay on the destination side of the bipartite —
    sampling over the full node set would create the trivially easy task
    of "is this node ever a destination?" and won't transfer to eval.

    Doubles as the random PROVIDER for MixedNegativeSampler via `draw`.
    """

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        seed: Optional[int] = None,
    ):
        self.num_neg_per_pos = num_neg_per_pos
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def draw(self, src: np.ndarray, n: int) -> np.ndarray:
        """[B, n] random destinations (with replacement, matching prior training
        behaviour — training negatives are not de-duplicated)."""
        B = len(src)
        if n == 0:
            return np.empty((B, 0), dtype=np.int32)
        idx = self.rng.integers(0, len(self.dst_pool), (B, n))
        return self.dst_pool[idx]

    def sample(self, batch: Batch) -> Tuple[np.ndarray, np.ndarray]:
        B = len(batch.src)
        neg_tgt = self.draw(batch.src, self.num_neg_per_pos)
        neg_src = np.broadcast_to(
            batch.src[:, None], (B, self.num_neg_per_pos),
        ).astype(np.int32, copy=True)
        return neg_src, neg_tgt


class HistoricalReservoir:
    """Per-source reservoir of past destinations (Vitter's algorithm R) — a
    PROVIDER of historical negatives for MixedNegativeSampler (not a standalone
    NegativeSampler; it never does its own random fallback).

    Vectorised hot path:
      observe(src, dst): true Vitter R — each item accepted with probability
        M/(count+1) when the reservoir is full (replacing a uniform-random
        slot); fill phase (count < M) accepts unconditionally into the next
        empty slot. Every reservoir slot is a uniform draw from the source's
        history at all times after fill. O(B), no Python loops. For a source
        repeated within a batch the LAST accepted write to a (src, position)
        wins — the only deviation from strict sequential Vitter R, negligible
        because per-batch B is small vs the per-source accepted-write rate.
      draw(pos_src, pos_tgt, n): O(B*n) reservoir gather + a validity mask
        (empty slot / equals the positive). The MIXER backfills invalid slots.

    State (reservoir matrix + count vector) is CPU/numpy, so it scales to
    multi-million-node datasets without GPU memory cost. Reservoir_size M
    should be ~ the dataset's typical per-source history depth: an under-filled
    reservoir (count < M) yields more invalid slots, which the mixer then
    backfills with random (i.e. the effective historical fraction dilutes).
    """

    def __init__(
        self,
        num_nodes: int,
        reservoir_size: int,
        seed: Optional[int] = None,
    ):
        self.num_nodes = num_nodes
        self.M = reservoir_size
        self.reservoir = np.full((num_nodes, self.M), -1, dtype=np.int32)
        self.count = np.zeros(num_nodes, dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.reservoir.fill(-1)
        self.count.fill(0)

    def observe(self, src: np.ndarray, dst: np.ndarray) -> None:
        """Vitter-R update. MUST be called AFTER scoring (strict-causal)."""
        B = src.shape[0]
        if B == 0:
            return
        src_i = src.astype(np.int64, copy=False)
        dst_i = dst.astype(np.int32, copy=False)
        pre_count = self.count[src_i]

        fill_mask = pre_count < self.M

        # Full phase: accept with probability M / (count + 1).
        t = (pre_count + 1).astype(np.float64)
        accept_when_full = self.rng.random(size=B) < (self.M / t)
        do_insert = fill_mask | (~fill_mask & accept_when_full)

        # Slot: fill phase -> next empty slot; full phase -> uniform random slot.
        rand_pos = self.rng.integers(0, self.M, size=B)
        insert_pos = np.where(fill_mask, pre_count, rand_pos)

        insert_idx = np.where(do_insert)[0]
        if len(insert_idx) > 0:
            self.reservoir[src_i[insert_idx], insert_pos[insert_idx]] = dst_i[insert_idx]

        np.add.at(self.count, src_i, 1)

    def draw(
        self, pos_src: np.ndarray, pos_tgt: np.ndarray, n: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (negs[B, n], valid[B, n]). `valid` is False where the drawn
        slot is empty (-1) or collides with the positive target — the mixer
        backfills those with random negatives."""
        B = len(pos_src)
        if n == 0:
            return np.empty((B, 0), dtype=np.int32), np.zeros((B, 0), dtype=bool)
        pos_src = np.asarray(pos_src, dtype=np.int64)
        pos_tgt = np.asarray(pos_tgt, dtype=np.int32)
        rand_slot = self.rng.integers(0, self.M, size=(B, n))
        negs = np.take_along_axis(
            self.reservoir[pos_src], rand_slot, axis=1,
        ).astype(np.int32, copy=False)
        valid = (negs >= 0) & (negs != pos_tgt[:, None])
        return negs, valid


class MixedNegativeSampler(NegativeSampler):
    """Training-side entry point: mixes historical + random negatives at a
    fixed `hist_ratio`, delegating each portion to the two providers.

    K_hist = round(K * hist_ratio) historical (reservoir), K_rand = K - K_hist
    random. Invalid historical slots (cold source / collision) are backfilled
    with random draws, so every row always returns exactly K negatives.

    hist_ratio == 0  ->  pure uniform, NO reservoir allocated (behaviour- and
    memory-identical to using UniformNegativeSampler directly). Safe default.
    """

    def __init__(
        self,
        num_neg_per_pos: int,
        hist_ratio: float,
        dst_pool: np.ndarray,
        num_nodes: int,
        reservoir_size: int = 256,
        seed: Optional[int] = None,
    ):
        self.K = num_neg_per_pos
        self.hist_ratio = float(hist_ratio)
        self.K_hist = max(0, min(self.K, int(round(self.K * self.hist_ratio))))
        self.K_rand = self.K - self.K_hist
        self.uniform = UniformNegativeSampler(
            num_neg_per_pos=self.K, dst_pool=dst_pool, seed=seed,
        )
        # Only allocate the reservoir when historical negatives are requested.
        self.historical: Optional[HistoricalReservoir] = None
        if self.K_hist > 0:
            self.historical = HistoricalReservoir(
                num_nodes=num_nodes,
                reservoir_size=reservoir_size,
                seed=(None if seed is None else seed + 1),
            )

    def sample(self, batch: Batch) -> Tuple[np.ndarray, np.ndarray]:
        B = len(batch.src)
        neg_src = np.broadcast_to(
            batch.src[:, None], (B, self.K),
        ).astype(np.int32, copy=True)

        if self.historical is None:  # pure uniform (hist_ratio == 0)
            return neg_src, self.uniform.draw(batch.src, self.K)

        hist, valid = self.historical.draw(batch.src, batch.tgt, self.K_hist)
        # K_rand for the random portion + K_hist spares to backfill invalid hist.
        rand = self.uniform.draw(batch.src, self.K_rand + self.K_hist)
        hist = np.where(valid, hist, rand[:, self.K_rand:])          # backfill
        neg_tgt = np.concatenate([hist, rand[:, : self.K_rand]], axis=1)  # [B, K]
        return neg_src, neg_tgt

    def observe(self, src: np.ndarray, dst: np.ndarray) -> None:
        if self.historical is not None:
            self.historical.observe(src, dst)

    def reset(self) -> None:
        if self.historical is not None:
            self.historical.reset()
