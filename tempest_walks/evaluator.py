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

from typing import List, Tuple

import numpy as np
import torch

from .data import Batch
from .model import EmbeddingStore, LinkPredictor
from .negatives import NegativeSampler


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

    @torch.no_grad()
    def evaluate_batch(self, batch: Batch) -> Tuple[float, int]:
        """Returns (sum_of_metric_over_batch, num_positives)."""
        neg_src, neg_tgt = self.neg_sampler.sample(batch)
        B = len(batch.src)

        all_u, all_v, counts = self._interleave(batch, neg_src, neg_tgt, B)
        scores = self._score_chunked(all_u, all_v, counts)

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

    def _score_chunked(
        self, all_u: torch.Tensor, all_v: torch.Tensor, counts: List[int],
    ) -> torch.Tensor:
        """Score [P] rows through the link predictor in row_budget-sized chunks.

        Groups (1 pos + K negs) stay atomic — we never split a positive
        across chunks. Pure cap on memory; the math is identical to the
        unchunked path.
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
        for start, end in rows:
            u = all_u[start:end]
            v = all_v[start:end]
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
