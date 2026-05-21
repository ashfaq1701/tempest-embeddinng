"""Phase 0.5 diagnostics — three probes on the trained Phase 0.5 model.

DIAGNOSTIC 1: Cold-start prevalence at val/test eval.
  Fraction of scored pairs where is_cold_start_uv / _u / _v == 1.

DIAGNOSTIC 2: Zero-out ablation (no retrain).
  Re-evaluate with the time-encoding inputs (Φ(Δt_*) and cold-start bits)
  forced to zero at the link MLP input. Measures how much the model
  actually leans on the Component 0 channels.

DIAGNOSTIC 3: First-Linear column-norm analysis.
  L2 norm of each column of the link MLP's first Linear (post-LayerNorm).
  Reports mean column norm for:
    - cross-table positions (slots 0 : 8·d)
    - time-encoding positions (slots 8·d : 8·d + 3·d_time)
    - cold-start bit positions (last 3 slots)
  If bit-column norms are << cross-table norms, the §12.1 LayerNorm-wash
  concern is real.

Trains Phase 0.5 from scratch (15 min) then runs all three diagnostics
inline. No checkpoint saving required.
"""

import json
from pathlib import Path

import numpy as np
import torch

from tempest_walks.config import Config
from tempest_walks.data import create_batches, load_tgb
from tempest_walks.evaluator import Evaluator
from tempest_walks.negatives import TGBNegativeSampler
from tempest_walks.trainer import Trainer


def _row_budget_for_d_emb_local(d_emb: int) -> int:
    return max(50_000, 500_000 * 128 // d_emb)


class _CountingEvaluator(Evaluator):
    """Evaluator that also accumulates cold-start fractions across batches.

    Adds three int counters (n_cold_u, n_cold_v, n_cold_uv) and n_total.
    Reset by calling reset_counts() between val and test phases.
    """

    def reset_counts(self):
        self.n_cold_u = 0
        self.n_cold_v = 0
        self.n_cold_uv = 0
        self.n_total = 0

    def _time_features_for_chunk(self, u_chunk, v_chunk, t_query):
        # Reuse parent's compute but also accumulate cold-start counts.
        phi_u, phi_v, phi_uv, cold_u, cold_v, cold_uv = super()._time_features_for_chunk(
            u_chunk, v_chunk, t_query,
        )
        self.n_cold_u += int(cold_u.sum().item())
        self.n_cold_v += int(cold_v.sum().item())
        self.n_cold_uv += int(cold_uv.sum().item())
        self.n_total += int(u_chunk.shape[0])
        return phi_u, phi_v, phi_uv, cold_u, cold_v, cold_uv


class _ZeroTimeEvaluator(_CountingEvaluator):
    """Like CountingEvaluator but zeros out Φ(Δt_*) and the cold-start bits
    before they reach the link MLP. Used for Diagnostic 2."""

    def _time_features_for_chunk(self, u_chunk, v_chunk, t_query):
        # Still call parent to update counts, then zero everything out.
        phi_u, phi_v, phi_uv, cold_u, cold_v, cold_uv = super()._time_features_for_chunk(
            u_chunk, v_chunk, t_query,
        )
        zero_phi = torch.zeros_like(phi_u)
        zero_cold = torch.zeros_like(cold_u)
        return zero_phi, zero_phi.clone(), zero_phi.clone(), zero_cold, zero_cold.clone(), zero_cold.clone()


def main():
    out = {}
    loaded = load_tgb("tgbl-wiki", root="datasets")
    edge_feat_dim = int(loaded.train.edge_feat.shape[1]) if loaded.train.edge_feat is not None else 0
    train_dst_pool = np.unique(loaded.train.destinations)

    config = Config(
        tgb_name=loaded.name,
        max_node_count=loaded.max_node_count,
        is_directed=loaded.is_directed,
        d_emb=128, d_hidden_link=128,
        max_walk_len=20, num_walks_per_node=5, walk_bias="ExponentialWeight",
        temporal_decay_exp=0.5, alignment_time_scale=-1.0,
        eta_uniform=1.0, uniformity_temperature=2.0,
        align_weighting="A",
        use_time_encoding=True, time_enc_k=16, cold_start_dt_clamp_factor=100.0,
        num_neg_per_pos=10, hist_neg_ratio=0.5, reservoir_size=32,
        emb_lr=1e-3, link_lr=1e-3,
        target_batch_size=200, num_epochs=50,
        seed=42, tgb_root="datasets", use_gpu=True,
    )
    trainer = Trainer(
        config=config, train_dst_pool=train_dst_pool,
        node_feat=loaded.node_feat, edge_feat_dim=edge_feat_dim,
    )
    # Derive time_scale (master baseline formula)
    ts_arr = loaded.train.timestamps
    span = float(ts_arr.max() - ts_arr.min())
    derived = span / 20.0
    trainer.set_time_scale(derived)
    print(f"time_scale = {derived:.1f}")

    print("Loading TGB negatives (val + test)…")
    loaded.dataset.load_val_ns()
    loaded.dataset.load_test_ns()

    # Phase 0.5 training (same as baseline run)
    print("=== Training Phase 0.5 ===")
    trainer.train(create_batches(loaded.train, config.target_batch_size))

    # Mirror trainer.evaluate()'s mode-switch for the diagnostic eval loops.
    trainer.embedding_store.eval()
    trainer.link_predictor.eval()
    if trainer.time_encoder is not None:
        trainer.time_encoder.eval()

    # --- DIAGNOSTIC 3: column norms of link MLP first Linear (after LayerNorm) ---
    lp = trainer.link_predictor
    first_linear: torch.nn.Linear = lp.net[0]
    W = first_linear.weight.detach().cpu()             # [hidden, in_d]
    col_norms = W.norm(dim=0)                          # [in_d]
    d = config.d_emb
    d_time = 2 * config.time_enc_k

    cross_table_slice = slice(0, 8 * d)
    time_slice = slice(8 * d, 8 * d + 3 * d_time)
    cold_slice = slice(8 * d + 3 * d_time, 8 * d + 3 * d_time + 3)

    cross_table_mean = float(col_norms[cross_table_slice].mean().item())
    time_mean = float(col_norms[time_slice].mean().item())
    cold_mean = float(col_norms[cold_slice].mean().item())
    out["diag3_column_norms"] = {
        "cross_table_8d_mean":          cross_table_mean,
        "time_encoding_3d_time_mean":   time_mean,
        "cold_start_3bits_mean":        cold_mean,
        "cold_to_cross_table_ratio":    cold_mean / max(cross_table_mean, 1e-12),
        "cold_to_time_ratio":           cold_mean / max(time_mean, 1e-12),
    }

    # --- Build counting Evaluators for val + test (DIAGNOSTIC 1 + baseline MRR) ---
    eval_kwargs_val = dict(
        embedding_store=trainer.embedding_store,
        link_predictor=trainer.link_predictor,
        device=trainer.device,
        tgb_dataset_name=loaded.name,
        eval_metric=loaded.eval_metric,
        time_encoder=trainer.time_encoder,
        time_state=trainer.time_state,
        time_scale=trainer._time_scale,
        cold_start_dt_clamp_factor=config.cold_start_dt_clamp_factor,
    )
    eval_val = _CountingEvaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="val"),
        **eval_kwargs_val,
    )
    eval_val.reset_counts()

    # Snapshot Tempest + time_state state BEFORE val eval so we can reset
    # for the test-side and zero-out re-runs (TGB convention: state from
    # end of train carries into val; we need to "rewind" once).
    # Trick: the snapshot is *just the post-train state*. We need to
    # re-run val/test eval three times: (val full, val zeroed, test full,
    # test zeroed). But the trainer.evaluate() loop also adds val edges
    # to walk_gen / time_state during eval — so by the time test eval
    # runs, walk_gen has val edges in it (TGB streaming convention).
    # For zero-out re-runs, the state has to be the SAME as the first
    # run for the comparison to be fair.

    # The cleanest fix: re-create walk_gen + time_state from scratch and
    # re-ingest the train edges, then run val+test fresh. This is
    # expensive (re-ingest ~110k edges). Alternative: snapshot the
    # state dictionaries. NodeTimeState is just numpy + dict; cheap to
    # snapshot. walk_gen is Tempest's internal state — harder.
    # We don't actually need Tempest state for the diagnostic runs
    # because Component 0 doesn't use walks; it only reads NodeTimeState
    # and the embedding tables. So we just need to snapshot NodeTimeState.

    # Snapshot NodeTimeState as it stands at the start of val (i.e., the
    # POST-train state).
    state_snapshot = {
        "last_event_time": trainer.time_state.last_event_time.copy(),
        "last_edge_time":  dict(trainer.time_state.last_edge_time),
    }

    # === Pass 1: val + test, full Component 0 (baseline MRR + cold-start counts)
    print("=== Eval (full Component 0): val ===")
    total_val_full = 0.0
    n_val = 0
    for batch in create_batches(loaded.val, config.target_batch_size):
        m, b = eval_val.evaluate_batch(batch)
        total_val_full += m
        n_val += b
        # Mirror trainer.evaluate post-scoring updates so test side sees the
        # right state. Walks don't matter for Component 0 alone.
        trainer.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
        trainer.time_state.update(batch.src, batch.tgt, batch.ts)
    val_full_mrr = total_val_full / max(n_val, 1)
    val_cold_uv = eval_val.n_cold_uv / max(eval_val.n_total, 1)
    val_cold_u  = eval_val.n_cold_u  / max(eval_val.n_total, 1)
    val_cold_v  = eval_val.n_cold_v  / max(eval_val.n_total, 1)

    eval_test = _CountingEvaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="test"),
        **eval_kwargs_val,
    )
    eval_test.reset_counts()
    print("=== Eval (full Component 0): test ===")
    total_test_full = 0.0
    n_test = 0
    for batch in create_batches(loaded.test, config.target_batch_size):
        m, b = eval_test.evaluate_batch(batch)
        total_test_full += m
        n_test += b
        trainer.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
        trainer.time_state.update(batch.src, batch.tgt, batch.ts)
    test_full_mrr = total_test_full / max(n_test, 1)
    test_cold_uv = eval_test.n_cold_uv / max(eval_test.n_total, 1)
    test_cold_u  = eval_test.n_cold_u  / max(eval_test.n_total, 1)
    test_cold_v  = eval_test.n_cold_v  / max(eval_test.n_total, 1)

    out["diag1_cold_start_fractions"] = {
        "val":  {"uv": val_cold_uv,  "u": val_cold_u,  "v": val_cold_v},
        "test": {"uv": test_cold_uv, "u": test_cold_u, "v": test_cold_v},
    }
    out["phase0_5_baseline_mrr"] = {"val": val_full_mrr, "test": test_full_mrr}

    # === Pass 2: zero-out ablation — restore time_state snapshot, re-eval ===
    trainer.time_state.last_event_time[:] = state_snapshot["last_event_time"]
    trainer.time_state.last_edge_time = dict(state_snapshot["last_edge_time"])

    eval_val_zero = _ZeroTimeEvaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="val"),
        **eval_kwargs_val,
    )
    eval_val_zero.reset_counts()
    print("=== Eval (Component 0 ZEROED): val ===")
    total_val_zero = 0.0
    n_val_z = 0
    for batch in create_batches(loaded.val, config.target_batch_size):
        m, b = eval_val_zero.evaluate_batch(batch)
        total_val_zero += m
        n_val_z += b
        trainer.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
        trainer.time_state.update(batch.src, batch.tgt, batch.ts)
    val_zero_mrr = total_val_zero / max(n_val_z, 1)

    eval_test_zero = _ZeroTimeEvaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="test"),
        **eval_kwargs_val,
    )
    eval_test_zero.reset_counts()
    print("=== Eval (Component 0 ZEROED): test ===")
    total_test_zero = 0.0
    n_test_z = 0
    for batch in create_batches(loaded.test, config.target_batch_size):
        m, b = eval_test_zero.evaluate_batch(batch)
        total_test_zero += m
        n_test_z += b
        trainer.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
        trainer.time_state.update(batch.src, batch.tgt, batch.ts)
    test_zero_mrr = total_test_zero / max(n_test_z, 1)

    out["diag2_zero_out"] = {
        "val":  {"full": val_full_mrr,  "zeroed": val_zero_mrr,  "drop": val_full_mrr - val_zero_mrr},
        "test": {"full": test_full_mrr, "zeroed": test_zero_mrr, "drop": test_full_mrr - test_zero_mrr},
    }

    # --- Report ---
    print()
    print("===== DIAGNOSTICS SUMMARY =====")
    print(json.dumps(out, indent=2))
    Path("runs").mkdir(exist_ok=True)
    out_path = Path("runs") / "phase0_5_diag.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
