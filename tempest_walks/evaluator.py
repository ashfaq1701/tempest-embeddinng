"""Streaming evaluator that delegates the metric to TGB's official scorer.

For each batch:
  1. Get per-positive negatives via the injected TGBNegativeSampler.
  2. Score every (positive, neg_i) row via link_predictor in chunks.
  3. Component 0 features (Φ(Δt_*) + cold-start bits) computed PRE-batch
     (strict-causal — caller hasn't ingested batch yet).
  4. Source-side e_t_u comes from `walk_repr_fn(u_chunk, t_query)` —
     currently the static target(u) lookup (the walk encoder is on
     backup/important-walk-embedding, see Lesson 35). The kwarg name
     is retained for API stability when the encoder is restored.
     Destination + context slots use the static embedding tables.
  5. Per-positive scoring goes through TGB Evaluator.eval(...), same as
     leaderboard.

`row_budget` chunks the link forward to fit 8 GB VRAM at d=128.
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
    """100k rows × 1123-dim input ≈ 450 MB — fits comfortably under 8 GB VRAM."""
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
        time_encoder: TimeEncoder,
        time_state: NodeTimeState,
        time_scale: float,
        walk_repr_fn,             # callable(u_np, t_query) → [N, d_emb] tensor
        cold_start_dt_clamp_factor: float = 100.0,
    ):
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
        self.walk_repr_fn = walk_repr_fn
        self.cold_start_dt_clamp_factor = cold_start_dt_clamp_factor

    @torch.no_grad()
    def evaluate_batch(self, batch: Batch, sample_pct: float = 1.0) -> Tuple[float, int]:
        """Returns (sum_metric_over_batch, num_positives_scored).
        If sample_pct < 1.0, positives are sub-sampled BEFORE scoring (for
        cheap per-epoch monitoring on review-scale datasets)."""
        B = len(batch.src)
        if 0.0 < sample_pct < 1.0:
            n_keep = max(1, int(B * sample_pct))
            keep_idx = np.random.choice(B, size=n_keep, replace=False)
            sub_batch = Batch(
                src=batch.src[keep_idx],
                tgt=batch.tgt[keep_idx],
                ts=batch.ts[keep_idx],
                edge_feat=(batch.edge_feat[keep_idx] if batch.edge_feat is not None else None),
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
        self, u_chunk: torch.Tensor, v_chunk: torch.Tensor, t_query: int,
    ) -> tuple:
        u_np = u_chunk.cpu().numpy().astype(np.int64)
        v_np = v_chunk.cpu().numpy().astype(np.int64)
        last_u, last_v, last_uv = self.time_state.query(u_np, v_np)
        clamp_to = float(self.time_scale) * float(self.cold_start_dt_clamp_factor)
        dt_u = np.clip((float(t_query) - last_u).astype(np.float32), 0.0, clamp_to)
        dt_v = np.clip((float(t_query) - last_v).astype(np.float32), 0.0, clamp_to)
        dt_uv = np.clip((float(t_query) - last_uv).astype(np.float32), 0.0, clamp_to)
        cold_u = (last_u == 0).astype(np.float32)
        cold_v = (last_v == 0).astype(np.float32)
        cold_uv = (last_uv == 0).astype(np.float32)
        dev = self.device
        phi_u = self.time_encoder(torch.from_numpy(dt_u).to(dev))
        phi_v = self.time_encoder(torch.from_numpy(dt_v).to(dev))
        phi_uv = self.time_encoder(torch.from_numpy(dt_uv).to(dev))
        cold_u_t = torch.from_numpy(cold_u).to(dev).unsqueeze(-1)
        cold_v_t = torch.from_numpy(cold_v).to(dev).unsqueeze(-1)
        cold_uv_t = torch.from_numpy(cold_uv).to(dev).unsqueeze(-1)
        return phi_u, phi_v, phi_uv, cold_u_t, cold_v_t, cold_uv_t

    def _score_chunked(
        self,
        all_u: torch.Tensor,
        all_v: torch.Tensor,
        counts: List[int],
        t_query: int,
    ) -> torch.Tensor:
        """Score rows in row_budget-sized chunks. Groups (1 pos + K negs)
        stay atomic (never split across chunks). Math identical to unchunked."""
        group_sizes = [1 + c for c in counts]
        rows = []
        rows_in_chunk = 0
        cursor = 0
        for gs in group_sizes:
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
            u_np = u.cpu().numpy()
            e_t_u = self.walk_repr_fn(u_np, t_query)
            phi_u, phi_v, phi_uv, cold_u, cold_v, cold_uv = self._time_features_for_chunk(
                u, v, t_query,
            )
            out[start:end] = self.link_predictor(
                e_t_u,
                self.embedding_store.target(v),
                self.embedding_store.context(u),
                self.embedding_store.context(v),
                phi_u, phi_v, phi_uv,
                cold_u, cold_v, cold_uv,
            )
        return out

    def _interleave(
        self, batch: Batch, neg_src, neg_tgt, B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Layout: [pos_0, neg_0_1..neg_0_K0, pos_1, neg_1_1..neg_1_K1, ...]"""
        # neg_src/tgt are typically padded ndarrays; handle ragged per-positive K.
        u_list: List[int] = []
        v_list: List[int] = []
        counts: List[int] = []
        for i in range(B):
            u_list.append(int(batch.src[i]))
            v_list.append(int(batch.tgt[i]))
            ns = neg_src[i]
            nt = neg_tgt[i]
            # Pad sentinels (-1) get dropped here.
            mask = nt >= 0
            ns_i = ns[mask]
            nt_i = nt[mask]
            u_list.extend(int(x) for x in ns_i)
            v_list.extend(int(x) for x in nt_i)
            counts.append(int(mask.sum()))
        all_u = torch.tensor(u_list, dtype=torch.long, device=self.device)
        all_v = torch.tensor(v_list, dtype=torch.long, device=self.device)
        return all_u, all_v, counts
