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
from .model import EmbeddingStore, LinkPredictor, TimeEncoder
from .negatives import NegativeSampler
from .timestate import NodeTimeState


_ROW_BUDGET_AT_D128 = 100_000


def _row_budget_for_d_emb(d_emb: int) -> int:
    # Lowered from 500k → 100k after a CUDA OOM on tgbl-review's full-precision
    # eval (730k positives × ~999 negs per pass). Review has 352k nodes →
    # embedding tables alone occupy ~1.4 GB with Adam state, leaving little
    # headroom for the link MLP forward at 500k rows × 1123-dim input.
    # 100k rows × 1123 = ~450 MB input, fits comfortably under 8 GB VRAM.
    return max(20_000, _ROW_BUDGET_AT_D128 * 128 // d_emb)


class Evaluator:
    def __init__(
        self,
        embedding_store: EmbeddingStore,
        link_predictor: LinkPredictor,
        neg_sampler: NegativeSampler,
        device: torch.device,
        tgb_dataset_name: str,
        eval_metric: str,
        # Component 0 hooks: time encoder + per-node state. When the
        # link predictor was constructed with use_time_encoding=True,
        # these MUST be provided — otherwise we raise loudly rather than
        # silently feeding zeros and producing wrong scores.
        time_encoder: Optional[TimeEncoder] = None,
        time_state: Optional[NodeTimeState] = None,
        time_scale: Optional[float] = None,
        cold_start_dt_clamp_factor: float = 100.0,
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
        self.time_encoder = time_encoder
        self.time_state = time_state
        self.time_scale = time_scale
        self.cold_start_dt_clamp_factor = cold_start_dt_clamp_factor
        # Wiring consistency: if the link predictor expects time inputs,
        # the encoder + state + scale must all be wired.
        if getattr(link_predictor, "use_time_encoding", False):
            if time_encoder is None or time_state is None or time_scale is None:
                raise ValueError(
                    "LinkPredictor was built with use_time_encoding=True, but "
                    "Evaluator was constructed without time_encoder / time_state "
                    "/ time_scale. Wire those in (typically via trainer.*)."
                )

    @torch.no_grad()
    def evaluate_batch(self, batch: Batch, sample_pct: float = 1.0) -> Tuple[float, int]:
        """Returns (sum_of_metric_over_batch, num_positives_scored).

        When `sample_pct < 1.0`, the positive set in this batch is randomly
        subsampled BEFORE scoring — used for cheap per-epoch monitoring on
        large datasets (e.g. tgbl-review-v2's 730k val positives, where
        full eval costs ~15 min per pass). The full-eval path (sample_pct=1)
        is bit-identical to the original behaviour.

        Important: the caller (Trainer.evaluate) still ingests the FULL
        batch into walk_gen/time_state regardless of sampling — only the
        scoring step is subsampled. State evolution stays exact.
        """
        B = len(batch.src)
        if 0.0 < sample_pct < 1.0:
            n_keep = max(1, int(B * sample_pct))
            keep_idx = np.random.choice(B, size=n_keep, replace=False)
            sub_batch = Batch(
                src=batch.src[keep_idx],
                tgt=batch.tgt[keep_idx],
                ts=batch.ts[keep_idx],
                edge_feat=(
                    batch.edge_feat[keep_idx] if batch.edge_feat is not None else None
                ),
                t_max=int(batch.ts[keep_idx].max()) if len(keep_idx) > 0 else batch.t_max,
            )
            score_batch = sub_batch
            B_scored = n_keep
        else:
            score_batch = batch
            B_scored = B

        neg_src, neg_tgt = self.neg_sampler.sample(score_batch)

        all_u, all_v, counts = self._interleave(score_batch, neg_src, neg_tgt, B_scored)
        scores = self._score_chunked(all_u, all_v, counts, t_query=int(score_batch.t_max))

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
        return total, B_scored

    def _time_features_for_chunk(
        self,
        u_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        t_query: int,
    ) -> tuple:
        """Compute (Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv), is_cold_u, is_cold_v, is_cold_uv)
        for a chunk of (u, v) pairs. Reads PRE-batch NodeTimeState — strict
        causal: evaluator's evaluate_batch is called BEFORE the trainer's
        post-scoring time_state.update for this batch."""
        u_np = u_chunk.cpu().numpy().astype(np.int64)
        v_np = v_chunk.cpu().numpy().astype(np.int64)
        last_u_np, last_v_np, last_uv_np = self.time_state.query(u_np, v_np)
        clamp_to = float(self.time_scale) * float(self.cold_start_dt_clamp_factor)
        dt_u_np = np.clip((float(t_query) - last_u_np).astype(np.float32), 0.0, clamp_to)
        dt_v_np = np.clip((float(t_query) - last_v_np).astype(np.float32), 0.0, clamp_to)
        dt_uv_np = np.clip((float(t_query) - last_uv_np).astype(np.float32), 0.0, clamp_to)
        cold_u_np = (last_u_np == 0).astype(np.float32)
        cold_v_np = (last_v_np == 0).astype(np.float32)
        cold_uv_np = (last_uv_np == 0).astype(np.float32)

        dt_u = torch.from_numpy(dt_u_np).to(self.device)
        dt_v = torch.from_numpy(dt_v_np).to(self.device)
        dt_uv = torch.from_numpy(dt_uv_np).to(self.device)
        phi_u = self.time_encoder(dt_u)
        phi_v = self.time_encoder(dt_v)
        phi_uv = self.time_encoder(dt_uv)
        cold_u = torch.from_numpy(cold_u_np).to(self.device).unsqueeze(-1)
        cold_v = torch.from_numpy(cold_v_np).to(self.device).unsqueeze(-1)
        cold_uv = torch.from_numpy(cold_uv_np).to(self.device).unsqueeze(-1)
        return phi_u, phi_v, phi_uv, cold_u, cold_v, cold_uv

    def _score_chunked(
        self,
        all_u: torch.Tensor,
        all_v: torch.Tensor,
        counts: List[int],
        t_query: int,
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
            if self.time_encoder is not None and self.time_state is not None:
                phi_u, phi_v, phi_uv, cold_u, cold_v, cold_uv = self._time_features_for_chunk(
                    u, v, t_query,
                )
                out[start:end] = self.link_predictor(
                    self.embedding_store.target(u),
                    self.embedding_store.target(v),
                    self.embedding_store.context(u),
                    self.embedding_store.context(v),
                    phi_u, phi_v, phi_uv,
                    cold_u, cold_v, cold_uv,
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
