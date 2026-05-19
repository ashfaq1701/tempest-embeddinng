"""Streaming evaluator that delegates the metric to TGB's official scorer.

For each batch:
  1. Get per-positive negatives via the injected TGBNegativeSampler.
  2. Score every (positive, neg_i) row through the link predictor in chunks
     (so 1000-neg eval batches don't overrun 8 GB VRAM).
  3. For each positive, hand (y_pred_pos, y_pred_neg) to
     `tgb.linkproppred.evaluate.Evaluator.eval` — same code path as
     the TGB leaderboard.

Per-row activation grows linearly with d_emb, so the row budget scales
inversely with d_emb.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from .data import Batch
from .history import NodeHistory
from .memory import NodeMemory
from .model import (
    CoOccurrenceEncoder,
    CrossPairAttention,
    EmbeddingStore,
    LinkPredictor,
    NodeEncoder,
    WalkEncoder,
    masked_mean_pool,
)
from .negatives import NegativeSampler
from .walks import WalkGenerator


_ROW_BUDGET_AT_D128 = 500_000

# Cross-pair attention is much heavier per row than the bare link MLP —
# every pair needs h_u_seq [L, d] and h_v_seq [L, d] gathered plus an
# attention matrix [L, L, n_heads] computed in two directions. Empirically
# the safe budget at L=20, d=128, n_heads=4 on 8 GB VRAM is ~10K rows
# per chunk (vs 500K for the plain link MLP).
_ROW_BUDGET_AT_D128_XPAIR = 10_000


def _row_budget_for_d_emb(d_emb: int, with_cross_pair: bool) -> int:
    base = _ROW_BUDGET_AT_D128_XPAIR if with_cross_pair else _ROW_BUDGET_AT_D128
    return max(5_000 if with_cross_pair else 50_000, base * 128 // d_emb)


class Evaluator:
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        link_predictor: LinkPredictor,
        neg_sampler: NegativeSampler,
        device: torch.device,
        tgb_dataset_name: str,
        eval_metric: str,
        # Phase 2 additions (required when link_predictor uses walk-encoded
        # blocks). At eval time the same protocol applies: walks come from
        # the PRE-batch Tempest state.
        walk_gen: Optional[WalkGenerator] = None,
        walk_encoder: Optional[WalkEncoder] = None,
        cross_pair_attn: Optional[CrossPairAttention] = None,
        node_history: Optional[NodeHistory] = None,
        node_encoder: Optional[NodeEncoder] = None,
        co_encoder: Optional[CoOccurrenceEncoder] = None,
        memory: Optional[NodeMemory] = None,
        time_scale: Optional[float] = None,
    ):
        # Lazy import so this module loads without py-tgb installed.
        from tgb.linkproppred.evaluate import Evaluator as TGBEvaluator

        self.embedding_store = embedding_store
        self.link_predictor = link_predictor
        self.neg_sampler = neg_sampler
        self.device = device
        self.eval_metric = eval_metric
        self.tgb_eval = TGBEvaluator(name=tgb_dataset_name)
        # Cross-pair attention is dropped on this branch — row budget can
        # stay at the higher 500K-rows ceiling (the only per-row cost in
        # eval is the link MLP forward at d_emb width).
        self.row_budget = _row_budget_for_d_emb(
            embedding_store.d_emb,
            with_cross_pair=False,
        )
        self.walk_gen = walk_gen
        self.walk_encoder = walk_encoder
        self.cross_pair_attn = cross_pair_attn
        self.node_history = node_history
        self.node_encoder = node_encoder
        self.co_encoder = co_encoder
        self.memory = memory
        self.time_scale = time_scale
        # Consistency checks: each augmentation needs its full support set.
        if walk_encoder is not None and (
            walk_gen is None or cross_pair_attn is None or time_scale is None
        ):
            raise ValueError(
                "Evaluator with walk_encoder requires walk_gen, "
                "cross_pair_attn, and time_scale to all be provided.",
            )
        if node_encoder is not None and (
            node_history is None or time_scale is None
        ):
            raise ValueError(
                "Evaluator with node_encoder requires node_history and "
                "time_scale to be provided.",
            )

    @torch.no_grad()
    def evaluate_batch(self, batch: Batch) -> Tuple[float, int]:
        """Returns (sum_of_metric_over_batch, num_positives)."""
        neg_src, neg_tgt = self.neg_sampler.sample(batch)
        B = len(batch.src)

        all_u, all_v, counts = self._interleave(batch, neg_src, neg_tgt, B)

        # Compute the unique seed set ONCE, share between walk encoder
        # and node encoder.
        all_u_np = all_u.cpu().numpy().astype(np.int64)
        all_v_np = all_v.cpu().numpy().astype(np.int64)
        seeds_np = np.unique(np.concatenate([all_u_np, all_v_np]))

        # Apply memory's pending raw messages from prior batch BEFORE scoring.
        # No-grad here (we're inside @torch.no_grad). State after this reflects
        # events ≤ prior eval/training batch.
        if self.memory is not None:
            seeds_t = torch.from_numpy(seeds_np).long().to(self.device)
            self.memory.apply_pending(seeds_t)

        # Walk-encode all unique nodes; produce per-seed W (seed-pooled).
        seed_w_lookup: Optional[Tuple[np.ndarray, torch.Tensor]] = None
        if self.walk_encoder is not None:
            seed_w_lookup = self._encode_walks_for_batch(seeds_np, batch.t_max)

        # Phase 4: read each unique node's history and run the node encoder,
        # producing a per-seed dynamic node embedding. Also cache nb_t/vc_t
        # for the co-occurrence encoder to reuse without re-reading.
        node_h_lookup: Optional[Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]] = None
        if self.node_encoder is not None:
            node_h_lookup = self._encode_history_for_batch(seeds_np, batch.t_max)

        scores = self._score_chunked(
            all_u, all_v, counts, seed_w_lookup, node_h_lookup,
        )

        scores_np = scores.cpu().numpy()
        total = 0.0
        cursor = 0
        for K in counts:
            pos = np.asarray([scores_np[cursor]], dtype=np.float64)
            neg = scores_np[cursor + 1 : cursor + 1 + K].astype(np.float64)
            res = self.tgb_eval.eval({
                "y_pred_pos": pos,
                "y_pred_neg": neg,
                "eval_metric": [self.eval_metric],
            })
            total += float(res[self.eval_metric])
            cursor += 1 + K
        return total, B

    def _encode_walks_for_batch(
        self,
        seeds_np: np.ndarray,
        t_max: int,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample walks for every unique node in this batch, run the encoder,
        seed-pool (mean of h_walks[w, lens_w-1, :] over the K walks per seed,
        target() fallback for cold-start), return per-seed W.

        Returns:
            seeds_np : np.ndarray  sorted unique seed ids
            seed_w   : [N, d_emb]  seed-pooled walk-summary (same recipe as
                                   the trainer's seed-pool path).

        Strict-causal: walks come from the CURRENT Tempest state (the trainer
        ingests AFTER calling evaluate_batch). Same protocol as training.
        """
        assert self.walk_gen is not None and self.walk_encoder is not None

        walks = self.walk_gen.walks_for_nodes(seeds_np)
        nodes = walks.nodes.to(self.device).long().clamp_min(0)
        timestamps = walks.timestamps.to(self.device).long()
        walk_lens = walks.lens.to(self.device).long()
        edge_feats = (
            walks.edge_feats.to(self.device) if walks.edge_feats is not None else None
        )
        seeds_tensor = walks.seeds.to(self.device).long()
        K = walks.K
        t_query_per_walk = torch.full(
            (seeds_tensor.shape[0] * K,), int(t_max),
            dtype=torch.long, device=self.device,
        )
        h_walks = self.walk_encoder(
            walk_nodes=nodes,
            walk_timestamps=timestamps,
            lens=walk_lens,
            walk_edge_feats=edge_feats,
            t_query=t_query_per_walk,
            time_scale=self.time_scale,
        )                                                                          # [N*K, L, d]

        # Seed-pool (matches trainer's recipe): pick h at the per-walk seed
        # position (lens-1), average over K walks per seed, target() fallback
        # for all-cold-start seeds.
        arange_w = torch.arange(h_walks.shape[0], device=self.device)
        seed_pos = (walk_lens - 1).clamp_min(0)
        h_at_seed = h_walks[arange_w, seed_pos, :]                                 # [N*K, d]
        has_walk = (walk_lens > 0).float().unsqueeze(-1)                           # [N*K, 1]
        N = seeds_tensor.shape[0]
        h_at_seed = h_at_seed.reshape(N, K, -1)
        has_walk = has_walk.reshape(N, K, 1)
        h_sum = (h_at_seed * has_walk).sum(dim=1)                                  # [N, d]
        h_cnt = has_walk.sum(dim=1).clamp_min(1.0)
        seed_w = h_sum / h_cnt
        no_walk = (has_walk.sum(dim=1) == 0).float()
        target_seeds_fb = self.embedding_store.target(seeds_tensor)
        seed_w = seed_w * (1.0 - no_walk) + target_seeds_fb * no_walk              # [N, d]
        return seeds_np, seed_w

    def _encode_history_for_batch(
        self,
        seeds_np: np.ndarray,
        t_max: int,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read per-node interaction history for every unique seed,
        run the DyGFormer-style node encoder, AND cache the neighbor /
        valid-count tensors so the co-occurrence encoder can reuse them.

        Returns:
            seeds_np : np.ndarray  sorted unique seed ids (same as input)
            node_h   : [N, d_emb]  per-seed dynamic embedding; zero rows
                                   are cold-start fallbacks.
            nb_t     : [N, K] long — gathered neighbor IDs (for co-occur)
            vc_t     : [N]    long — valid counts (for co-occur)

        Strict-causal: history reflects events ≤ batch B−1 because the
        trainer's evaluate() writes events AFTER calling evaluate_batch.
        """
        assert self.node_encoder is not None and self.node_history is not None
        assert self.time_scale is not None

        nb, hts, ef_hist, ro, vc = self.node_history.read_windows_for_nodes(seeds_np)
        nb_t = torch.from_numpy(nb).long().to(self.device)
        hts_t = torch.from_numpy(hts).long().to(self.device)
        ef_hist_t = (
            torch.from_numpy(ef_hist).float().to(self.device)
            if ef_hist is not None else None
        )
        ro_t = torch.from_numpy(ro).long().to(self.device)
        vc_t = torch.from_numpy(vc).long().to(self.device)
        tq = torch.full(
            (seeds_np.shape[0],), int(t_max),
            dtype=torch.long, device=self.device,
        )
        node_h, _has_hist = self.node_encoder(
            neighbors=nb_t,
            timestamps=hts_t,
            edge_feats=ef_hist_t,
            roles=ro_t,
            valid_cnt=vc_t,
            t_query=tq,
            time_scale=self.time_scale,
        )
        return seeds_np, node_h, nb_t, vc_t

    def _score_chunked(
        self,
        all_u: torch.Tensor,
        all_v: torch.Tensor,
        counts: List[int],
        seed_w_lookup: Optional[Tuple[np.ndarray, torch.Tensor]] = None,
        node_h_lookup: Optional[Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Score [P] rows through the link predictor in row_budget-sized chunks.

        Groups (1 pos + K negs) stay atomic — we never split a positive
        across chunks. Pure cap on memory; the math is identical to the
        unchunked path.

        Cross-pair attention runs PER CHUNK because every row has its own
        (u, v) pair → its own pair-conditioned W(u), W(v). The per-batch
        walk encoding happens once (in `_encode_walks_for_batch`); per-chunk
        we just gather sequences and run MHA.
        """
        group_sizes = [1 + c for c in counts]
        rows = []
        rows_in_chunk = 0
        cursor = 0
        for i, gs in enumerate(group_sizes):
            if rows_in_chunk > 0 and rows_in_chunk + gs > self.row_budget:
                rows.append((cursor - rows_in_chunk, cursor))
                rows_in_chunk = 0
            rows_in_chunk += gs
            cursor += gs
        if rows_in_chunk > 0:
            rows.append((cursor - rows_in_chunk, cursor))

        out = torch.empty(int(all_u.shape[0]), dtype=torch.float32, device=self.device)

        # Walk-encoded path requires the lookup AND the cross-pair attention.
        if seed_w_lookup is None:
            raise RuntimeError(
                "Evaluator was constructed without walk_encoder; the current "
                "LinkPredictor requires walk-encoded W(u), W(v). Wire those in.",
            )

        seeds_np, seed_w_pooled = seed_w_lookup                          # [N_unique, d]
        all_u_np = all_u.cpu().numpy().astype(np.int64)
        all_v_np = all_v.cpu().numpy().astype(np.int64)
        u_idx_full = torch.from_numpy(
            np.searchsorted(seeds_np, all_u_np),
        ).long().to(self.device)
        v_idx_full = torch.from_numpy(
            np.searchsorted(seeds_np, all_v_np),
        ).long().to(self.device)

        # Optional node-encoder + co-occurrence lookups.
        node_h_tensor: Optional[torch.Tensor] = None
        hist_nb_tensor: Optional[torch.Tensor] = None
        hist_vc_tensor: Optional[torch.Tensor] = None
        if node_h_lookup is not None:
            assert np.array_equal(node_h_lookup[0], seeds_np), (
                "node_h_lookup seed array must match the walk-encoder's "
                "seed array — they're sampled from the same union."
            )
            node_h_tensor = node_h_lookup[1]                  # [N_unique, d]
            hist_nb_tensor = node_h_lookup[2]                 # [N_unique, K]
            hist_vc_tensor = node_h_lookup[3]                 # [N_unique]

        for start, end in rows:
            u = all_u[start:end]
            v = all_v[start:end]
            u_idx = u_idx_full[start:end]
            v_idx = v_idx_full[start:end]

            w_u = seed_w_pooled[u_idx]                                  # [P_chunk, d]
            w_v = seed_w_pooled[v_idx]

            e_t_u = self.embedding_store.target(u)
            e_t_v = self.embedding_store.target(v)
            e_c_u = self.embedding_store.context(u)
            e_c_v = self.embedding_store.context(v)
            if node_h_tensor is not None:
                e_t_u = e_t_u + node_h_tensor[u_idx]
                e_t_v = e_t_v + node_h_tensor[v_idx]
            if self.memory is not None:
                # Memory state is read here (in eval no_grad). State was
                # advanced via apply_pending at the start of evaluate_batch.
                e_t_u = e_t_u + self.memory.read(u)
                e_t_v = e_t_v + self.memory.read(v)

            co_feat: Optional[torch.Tensor] = None
            if (
                self.co_encoder is not None
                and hist_nb_tensor is not None
                and hist_vc_tensor is not None
            ):
                nb_u = hist_nb_tensor[u_idx]
                nb_v = hist_nb_tensor[v_idx]
                vc_u = hist_vc_tensor[u_idx]
                vc_v = hist_vc_tensor[v_idx]
                co_feat = self.co_encoder(nb_u, nb_v, vc_u, vc_v)

            out[start:end] = self.link_predictor(
                e_t_u, e_t_v, e_c_u, e_c_v, w_u, w_v, co_feat,
            )

        return out

    def _interleave(
        self, batch: Batch, neg_src, neg_tgt, B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Layout: [pos_0, neg_0_1..neg_0_K0, pos_1, neg_1_1..neg_1_K1, ...]
        Returns (all_u, all_v, counts). Supports fixed-K (training-time
        UniformNegativeSampler shape) and variable-K (TGBNegativeSampler
        list-of-arrays) uniformly.
        """
        fixed_k = isinstance(neg_src, np.ndarray) and neg_src.ndim == 2
        if fixed_k:
            K = neg_src.shape[1]
            pos_src = torch.from_numpy(batch.src).long().to(self.device)
            pos_tgt = torch.from_numpy(batch.tgt).long().to(self.device)
            neg_src_t = torch.from_numpy(neg_src).long().to(self.device)
            neg_tgt_t = torch.from_numpy(neg_tgt).long().to(self.device)
            all_u = torch.cat([pos_src.unsqueeze(1), neg_src_t], dim=1).flatten()
            all_v = torch.cat([pos_tgt.unsqueeze(1), neg_tgt_t], dim=1).flatten()
            return all_u, all_v, [K] * B

        src_parts: List[np.ndarray] = []
        dst_parts: List[np.ndarray] = []
        counts: List[int] = []
        for i in range(B):
            K_i = len(neg_tgt[i])
            counts.append(K_i)
            src_parts.append(np.array([batch.src[i]], dtype=np.int32))
            src_parts.append(np.asarray(neg_src[i], dtype=np.int32))
            dst_parts.append(np.array([batch.tgt[i]], dtype=np.int32))
            dst_parts.append(np.asarray(neg_tgt[i], dtype=np.int32))
        all_u = torch.from_numpy(np.concatenate(src_parts)).long().to(self.device)
        all_v = torch.from_numpy(np.concatenate(dst_parts)).long().to(self.device)
        return all_u, all_v, counts
