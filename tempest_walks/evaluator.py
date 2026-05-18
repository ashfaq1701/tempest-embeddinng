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
from .model import EmbeddingStore, LinkPredictor, WalkEncoder
from .negatives import NegativeSampler
from .walks import WalkGenerator


_ROW_BUDGET_AT_D128 = 500_000


def _row_budget_for_d_emb(d_emb: int) -> int:
    return max(50_000, _ROW_BUDGET_AT_D128 * 128 // d_emb)


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
        self.row_budget = _row_budget_for_d_emb(embedding_store.d_emb)
        self.walk_gen = walk_gen
        self.walk_encoder = walk_encoder
        self.time_scale = time_scale
        # Consistency check: if the link MLP wants walk blocks, we need
        # both a walk generator and encoder.
        if link_predictor.n_walk_blocks > 0 and (
            walk_gen is None or walk_encoder is None or time_scale is None
        ):
            raise ValueError(
                "LinkPredictor.n_walk_blocks > 0 but Evaluator was constructed "
                "without walk_gen / walk_encoder / time_scale — eval would "
                "have no h_u/h_v to feed into the head.",
            )

    @torch.no_grad()
    def evaluate_batch(self, batch: Batch) -> Tuple[float, int]:
        """Returns (sum_of_metric_over_batch, num_positives)."""
        neg_src, neg_tgt = self.neg_sampler.sample(batch)
        B = len(batch.src)

        all_u, all_v, counts = self._interleave(batch, neg_src, neg_tgt, B)

        # Phase 2: walk-encode all unique nodes in this batch once, then look
        # h up per-pair. Within-batch deduplication is the cache — same role
        # as Phase 3's planned cross-batch cache, but scoped to one batch.
        seed_h_lookup: Optional[Tuple[np.ndarray, torch.Tensor]] = None
        if self.walk_encoder is not None:
            seed_h_lookup = self._encode_walks_for_batch(all_u, all_v, batch.t_max)

        scores = self._score_chunked(all_u, all_v, counts, seed_h_lookup)

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
        all_u: torch.Tensor,
        all_v: torch.Tensor,
        t_max: int,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Sample walks for every unique node in this batch, run the encoder
        once, return (sorted_seeds_np, seed_h: [N_unique, d_emb]).

        Strict-causal: walks come from the CURRENT Tempest state (the trainer
        ingests AFTER calling evaluate_batch). Same protocol as training.
        """
        assert self.walk_gen is not None and self.walk_encoder is not None
        all_u_np = all_u.cpu().numpy().astype(np.int64)
        all_v_np = all_v.cpu().numpy().astype(np.int64)
        seeds_np = np.unique(np.concatenate([all_u_np, all_v_np]))

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

        # Per-seed walk summary: mean of h[w, lens_w-1] over K walks,
        # target() fallback for all-cold-start seeds.
        arange_w = torch.arange(h_walks.shape[0], device=self.device)
        seed_pos = (walk_lens - 1).clamp_min(0)
        h_at_seed_per_walk = h_walks[arange_w, seed_pos, :]                        # [N*K, d]
        has_walk = (walk_lens > 0).float().unsqueeze(-1)                           # [N*K, 1]
        N = seeds_tensor.shape[0]
        h_at_seed_per_walk = h_at_seed_per_walk.reshape(N, K, -1)
        has_walk = has_walk.reshape(N, K, 1)
        h_sum = (h_at_seed_per_walk * has_walk).sum(dim=1)
        h_cnt = has_walk.sum(dim=1).clamp_min(1.0)
        h_avg = h_sum / h_cnt
        no_walk = (has_walk.sum(dim=1) == 0).float()
        target_seeds = self.embedding_store.target(seeds_tensor)
        seed_h = h_avg * (1.0 - no_walk) + target_seeds * no_walk                  # [N, d]
        return seeds_np, seed_h

    def _score_chunked(
        self,
        all_u: torch.Tensor,
        all_v: torch.Tensor,
        counts: List[int],
        seed_h_lookup: Optional[Tuple[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Score [P] rows through the link predictor in row_budget-sized chunks.

        Groups (1 pos + K negs) stay atomic — we never split a positive
        across chunks. Pure cap on memory; the math is identical to the
        unchunked path.

        When `seed_h_lookup` is supplied (Phase 2), the link MLP also reads
        h_u/h_v per row, gathered from the per-batch encoded lookup.
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

        if seed_h_lookup is not None:
            seeds_np, seed_h = seed_h_lookup
            # Pre-compute the (u, v) → seed-index lookups once for the whole
            # batch — keeps per-chunk overhead minimal.
            all_u_np = all_u.cpu().numpy().astype(np.int64)
            all_v_np = all_v.cpu().numpy().astype(np.int64)
            u_idx_full = torch.from_numpy(
                np.searchsorted(seeds_np, all_u_np),
            ).long().to(self.device)
            v_idx_full = torch.from_numpy(
                np.searchsorted(seeds_np, all_v_np),
            ).long().to(self.device)

        for start, end in rows:
            u = all_u[start:end]
            v = all_v[start:end]
            if seed_h_lookup is not None:
                h_u = seed_h[u_idx_full[start:end]]
                h_v = seed_h[v_idx_full[start:end]]
                out[start:end] = self.link_predictor(
                    self.embedding_store.target(u),
                    self.embedding_store.target(v),
                    self.embedding_store.context(u),
                    self.embedding_store.context(v),
                    h_u, h_v,
                )
            else:
                out[start:end] = self.link_predictor(
                    self.embedding_store.target(u),
                    self.embedding_store.target(v),
                    self.embedding_store.context(u),
                    self.embedding_store.context(v),
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
