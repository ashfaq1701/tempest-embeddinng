"""Entry point. One required flag: --tgb-name (e.g. tgbl-wiki).

Loads the dataset via TGB, runs strict-causal training, evaluates with
TGB's official Evaluator on val and test.
"""

import argparse

import numpy as np
import torch

from tempest_walks.config import Config
from tempest_walks.data import create_batches, load_tgb
from tempest_walks.evaluator import Evaluator
from tempest_walks.negatives import TGBNegativeSampler
from tempest_walks.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tempest-walks-v3 trainer")
    p.add_argument("--tgb-name", required=True)
    p.add_argument("--tgb-root", default="datasets")
    p.add_argument("--use-gpu", action="store_true")
    # `--is-directed` / `--no-is-directed` overrides the per-dataset default
    # in data.py::default_is_directed (e.g. tgbl-wiki is undirected by default;
    # pass `--is-directed` to feed Tempest in directed mode instead).
    p.add_argument("--is-directed", default=None, action=argparse.BooleanOptionalAction)

    # Model
    p.add_argument("--d-emb", type=int, default=128)
    p.add_argument("--d-hidden-link", type=int, default=128)

    # Walks
    p.add_argument("--max-walk-len", type=int, default=20)
    p.add_argument("--num-walks-per-node", type=int, default=5)
    p.add_argument("--walk-bias", default="ExponentialWeight")

    # Losses
    p.add_argument("--temporal-decay-exp", type=float, default=0.5)
    p.add_argument("--alignment-time-scale", type=float, default=-1.0)
    p.add_argument("--eta-uniform", type=float, default=1.0)
    p.add_argument("--uniformity-temperature", type=float, default=2.0)
    # Phase 1 ablation: alignment-loss weighting variant
    p.add_argument("--align-weighting", choices=["A", "B", "C"], default="A",
                   help="A=1/K·(1+Δt/τ)^(-β) [control]; B=1/K only; C=uniform α=1")
    # Phase S Group A2 (v2.2 §4.1): scalar on the alignment loss.
    # 1.0 = anchor (alignment on). 0.0 = walks-supervision OFF.
    p.add_argument("--lambda-align", type=float, default=1.0,
                   help="Scalar on alignment loss. 0 turns walks-supervision off "
                        "(Phase S Group A2). Anchor uses 1.0.")
    # Phase S Group C — joint training scalar on link BCE -> embeddings.
    # 0 (default) keeps decoupled training. > 0 lets BCE update embeddings,
    # scaled by this value. Candidate fix for InfoNCE/SGNS rapid breakdown.
    p.add_argument("--lambda-link", type=float, default=0.0,
                   help="Scalar on link BCE's gradient into embedding tables "
                        "(Phase S Group C). 0=decoupled, 1.0=full joint "
                        "training. Only meaningful under head_mode=cross_table.")
    # Phase S Group E (v2.2 §4.1 / §6.4): link MLP head structure.
    p.add_argument("--head-mode", choices=["cross_table", "component_0_only"],
                   default="cross_table",
                   help="E.1 cross_table (anchor) vs E.2 component_0_only "
                        "(drops cross-table reads entirely).")
    p.add_argument("--cross-table-dropout", type=float, default=0.0,
                   help="E.3: dropout on the 8·d cross-table block before "
                        "concatenation with Component 0. Only used when "
                        "--head-mode=cross_table.")
    # §4.8.2 architectural fixes for cliff remediation.
    p.add_argument("--link-mlp-n-layers", type=int, default=3,
                   help="Link MLP depth (number of Linear layers). Default 3 "
                        "matches v2.3 spec. Increase to 5 for §4.8.2 ablation.")
    p.add_argument("--link-mlp-dropout", type=float, default=0.0,
                   help="Dropout between hidden layers of the link MLP.")
    p.add_argument("--embedding-dropout", type=float, default=0.0,
                   help="Dropout applied to embedding lookups AT the link MLP "
                        "read site (training only). Regulariser without "
                        "touching the embedding-side loss.")

    # Phase S §4.7 — loss-family search. Selects the primary embedding-side
    # loss and an optional norm-brake auxiliary. All hyperparameters under
    # each primary are literature-default; no per-dataset tuning (§4.7.0).
    p.add_argument("--primary-loss", choices=["alignment", "triplet", "infonce", "sgns"],
                   default="alignment",
                   help="Embedding-side loss family. 'alignment'=v2.2 default; "
                        "'triplet'/'infonce'/'sgns'=§4.7 candidates.")
    # Norm-brake auxiliary
    p.add_argument("--lambda-normbrake", type=float, default=0.0,
                   help="Weight on §4.4 norm-brake regularizer. 0=off.")
    p.add_argument("--normbrake-threshold", type=float, default=0.54,
                   help="Per-column L2 threshold for norm-brake. Default 0.54 "
                        "= 1.5 × wiki anchor mean col-norm. Recalibrate per "
                        "dataset (see v2.3 §4.7.4).")
    # Debug instrumentation for deep-analysis runs. When set, logs per-epoch
    # gradient norms (E_target, E_context), link MLP first-Linear cross-table
    # col-norm, AND runs full test eval every epoch (not just on val improvement).
    # Adds ~50% eval cost per epoch; only enable for cliff/plateau analysis.
    p.add_argument("--log-debug", action="store_true",
                   help="Verbose per-epoch logging: grad norms, link-MLP "
                        "col-norm, full per-epoch test MRR trajectory.")

    # Optimization
    p.add_argument("--num-neg-per-pos", type=int, default=10)
    p.add_argument("--hist-neg-ratio", type=float, default=0.5,
                   help="Fraction of training negatives drawn from each "
                        "source's reservoir of past destinations (Vitter R). "
                        "0.5 matches TGB's eval-time historical/random mix. "
                        "0 disables — uniform random only.")
    p.add_argument("--reservoir-size", type=int, default=32)
    p.add_argument("--emb-lr", type=float, default=1e-3)
    p.add_argument("--link-lr", type=float, default=1e-3)
    p.add_argument("--target-batch-size", type=int, default=200)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    # Early stopping. When > 0, per-epoch val eval runs after each
    # training epoch; trainer keeps the best-val checkpoint and breaks
    # after this many epochs of no improvement. test_mrr is pinned to
    # the same epoch as best_val_mrr (same model snapshot).
    # When 0 (default), behaves like before: trains num_epochs flat,
    # then val + test eval once.
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Patience (in epochs) for early stopping on val "
                        "MRR. 0 = disabled (legacy behaviour).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading TGB dataset: {args.tgb_name}")
    loaded = load_tgb(args.tgb_name, root=args.tgb_root)
    is_directed = (
        args.is_directed if args.is_directed is not None else loaded.is_directed
    )
    directed_note = "" if args.is_directed is None else " (CLI override)"
    print(
        f"  N={loaded.max_node_count}  "
        f"train={len(loaded.train.sources)}  val={len(loaded.val.sources)}  "
        f"test={len(loaded.test.sources)}  is_directed={is_directed}{directed_note}  "
        f"eval_metric={loaded.eval_metric}",
    )

    config = Config(
        tgb_name=loaded.name,
        max_node_count=loaded.max_node_count,
        is_directed=is_directed,
        d_emb=args.d_emb,
        d_hidden_link=args.d_hidden_link,
        max_walk_len=args.max_walk_len,
        num_walks_per_node=args.num_walks_per_node,
        walk_bias=args.walk_bias,
        temporal_decay_exp=args.temporal_decay_exp,
        alignment_time_scale=args.alignment_time_scale,
        eta_uniform=args.eta_uniform,
        uniformity_temperature=args.uniformity_temperature,
        align_weighting=args.align_weighting,
        lambda_align=args.lambda_align,
        head_mode=args.head_mode,
        cross_table_dropout=args.cross_table_dropout,
        link_mlp_n_layers=args.link_mlp_n_layers,
        link_mlp_dropout=args.link_mlp_dropout,
        embedding_dropout=args.embedding_dropout,
        primary_loss=args.primary_loss,
        lambda_normbrake=args.lambda_normbrake,
        normbrake_threshold=args.normbrake_threshold,
        lambda_link=args.lambda_link,
        num_neg_per_pos=args.num_neg_per_pos,
        hist_neg_ratio=args.hist_neg_ratio,
        reservoir_size=args.reservoir_size,
        emb_lr=args.emb_lr,
        link_lr=args.link_lr,
        target_batch_size=args.target_batch_size,
        num_epochs=args.num_epochs,
        seed=args.seed,
        tgb_root=args.tgb_root,
        use_gpu=args.use_gpu,
    )

    train_dst_pool = np.unique(loaded.train.destinations)
    # Node features are dataset-static (when present); edge features are
    # per-edge — Tempest already carries them in walks. Both flow into
    # EmbeddingStore as learned residual projections when available.
    edge_feat_dim = (
        int(loaded.train.edge_feat.shape[1]) if loaded.train.edge_feat is not None else 0
    )
    print(
        f"  node_feat: {'present, d=' + str(loaded.node_feat.shape[1]) if loaded.node_feat is not None else 'absent'}  "
        f"edge_feat: {'present, d=' + str(edge_feat_dim) if edge_feat_dim > 0 else 'absent'}"
    )
    trainer = Trainer(
        config=config,
        train_dst_pool=train_dst_pool,
        node_feat=loaded.node_feat,
        edge_feat_dim=edge_feat_dim,
        # SGNS unigram^0.75 needs raw (non-unique) destination counts; pass
        # them through unconditionally — they're a numpy view, ~free.
        train_destinations_full=loaded.train.destinations,
    )

    # Derive alignment time-scale: (training time span) / L_REF, where
    # L_REF is a fixed reference constant (NOT --max-walk-len).
    #
    # Empirically on tgbl-wiki, time_scale ≈ span / 20 = 93k sec ≈ 1.08
    # days outperforms the per-node mean inter-event time (155k sec ≈
    # 1.8 days) by ~0.02 test MRR. The alignment recency term wants a
    # sharper scale than the per-node recurrence period — closer to a
    # within-session timescale.
    #
    # Why L_REF is fixed: tying time_scale to --max-walk-len was a real
    # bug (Lesson 11). Bumping L from 20 → 50 collapsed the scale from
    # 93k → 37k and crushed the recency weight. Keeping L_REF=20 fixed
    # means changing the walk length does not perturb the decay rate.
    if config.alignment_time_scale <= 0:
        ts = loaded.train.timestamps
        span = float(ts.max() - ts.min())
        L_REF = 20
        derived = span / L_REF
        trainer.set_time_scale(derived)
        print(
            f"  alignment_time_scale (derived): {derived:.3f}  "
            f"[ span={span:.1f}  L_REF={L_REF} ]"
        )

    print("Loading TGB negatives (val + test)…")
    loaded.dataset.load_val_ns()
    loaded.dataset.load_test_ns()

    _eval_kwargs = dict(
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
    eval_val = Evaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="val"),
        **_eval_kwargs,
    )
    eval_test = Evaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="test"),
        **_eval_kwargs,
    )

    print(f"Model device: {trainer.device}    Tempest device: cpu")
    if args.early_stop_patience and args.early_stop_patience > 0:
        # Early-stopping path: per-epoch val (+ test on new-best) eval
        # happens inside trainer.train(); the test number is pinned to
        # the same epoch as best val, so we report directly from the
        # summary dict — no second outer eval pass.
        print(f"=== Training (early stop patience={args.early_stop_patience}) ===")
        summary = trainer.train(
            create_batches(loaded.train, config.target_batch_size),
            val_evaluator=eval_val,
            val_batches_factory=lambda: create_batches(
                loaded.val, config.target_batch_size,
            ),
            test_evaluator=eval_test,
            test_batches_factory=lambda: create_batches(
                loaded.test, config.target_batch_size,
            ),
            early_stop_patience=args.early_stop_patience,
            log_debug=args.log_debug,
        )
        print(f"\n=== Summary (best epoch {summary['best_epoch']}) ===")
        print(f"  stopped_at_epoch    : {summary['stopped_at_epoch']}")
        print(f"  best_val_{loaded.eval_metric:<14}: {summary['best_val_mrr']:.4f}")
        if summary["best_test_mrr"] is not None:
            print(f"  best_test_{loaded.eval_metric:<13}: {summary['best_test_mrr']:.4f}")
        print(f"  per_epoch_val_{loaded.eval_metric:<8}: "
              + ", ".join(f"{v:.4f}" for v in summary["per_epoch_val_mrr"]))
        if summary.get("per_epoch_col_norm"):
            print("  per_epoch_col_norm    : "
                  + ", ".join(f"{v:.4f}" for v in summary["per_epoch_col_norm"]))
        if summary.get("per_epoch_normbrake"):
            print("  per_epoch_L_normbrake : "
                  + ", ".join(f"{v:.4f}" for v in summary["per_epoch_normbrake"]))
        # Debug-instrumentation outputs (only populated when --log-debug)
        if summary.get("per_epoch_test_mrr_all"):
            print(f"  per_epoch_test_{loaded.eval_metric:<8}: "
                  + ", ".join(f"{v:.4f}" for v in summary["per_epoch_test_mrr_all"]))
        if summary.get("per_epoch_grad_target"):
            print("  per_epoch_grad_E_targ : "
                  + ", ".join(f"{v:.4f}" for v in summary["per_epoch_grad_target"]))
        if summary.get("per_epoch_grad_context"):
            print("  per_epoch_grad_E_ctx  : "
                  + ", ".join(f"{v:.4f}" for v in summary["per_epoch_grad_context"]))
        if summary.get("per_epoch_link_w_norm"):
            print("  per_epoch_link_w_norm : "
                  + ", ".join(f"{v:.4f}" for v in summary["per_epoch_link_w_norm"]))
    else:
        print("=== Training ===")
        trainer.train(create_batches(loaded.train, config.target_batch_size))

        print("=== Validation ===")
        val_metric = trainer.evaluate(
            create_batches(loaded.val, config.target_batch_size), eval_val,
        )
        print(f"Val {loaded.eval_metric}: {val_metric:.4f}")

        print("=== Test ===")
        test_metric = trainer.evaluate(
            create_batches(loaded.test, config.target_batch_size), eval_test,
        )
        print(f"Test {loaded.eval_metric}: {test_metric:.4f}")


if __name__ == "__main__":
    main()
