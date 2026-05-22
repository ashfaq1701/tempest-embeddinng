"""Entry point — tempest-walks-v3 (minimal production).

Required: --tgb-name.

Loads TGB dataset, derives alignment_time_scale from train span,
constructs Trainer (static target(u) source-side + Component 0 +
WD_link), runs strict-causal training, evaluates with TGB Evaluator.
The walk encoder lives on backup/important-walk-embedding pending
re-stabilization on master.
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
    p = argparse.ArgumentParser(description="tempest-walks-v3 trainer (minimal)")
    p.add_argument("--tgb-name", required=True)
    p.add_argument("--tgb-root", default="datasets")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--is-directed", default=None, action=argparse.BooleanOptionalAction,
                   help="Override the per-dataset default in data.py.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="0 disables; >0 = patience on val MRR.")
    p.add_argument("--target-batch-size", type=int, default=200)

    # Review-scale conveniences.
    p.add_argument("--monitor-sample-pct", type=float, default=1.0,
                   help="Fraction of val/test positives scored per epoch.")
    p.add_argument("--skip-final-full-eval", action="store_true",
                   help="Skip the final full-precision eval pass.")

    p.add_argument("--log-debug", action="store_true",
                   help="Verbose per-epoch logging (reserved).")

    # Ablation toggle — freeze identity tables. The --use-walk-encoder
    # and --num-walks-per-node flags were removed alongside the walk
    # encoder itself (Lesson 35); restoring requires checking out
    # backup/important-walk-embedding.
    p.add_argument("--freeze-tables", action="store_true",
                   help="Freeze E_target and E_context (Sanity-style cell).")
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
    print(
        f"  N={loaded.max_node_count}  "
        f"train={len(loaded.train.sources)}  val={len(loaded.val.sources)}  "
        f"test={len(loaded.test.sources)}  is_directed={is_directed}  "
        f"eval_metric={loaded.eval_metric}",
    )

    config_kwargs = dict(
        tgb_name=loaded.name,
        max_node_count=loaded.max_node_count,
        is_directed=is_directed,
        seed=args.seed,
        num_epochs=args.num_epochs,
        early_stop_patience=args.early_stop_patience,
        target_batch_size=args.target_batch_size,
        monitor_sample_pct=args.monitor_sample_pct,
        skip_final_full_eval=args.skip_final_full_eval,
        log_debug=args.log_debug,
        tgb_root=args.tgb_root,
        use_gpu=args.use_gpu,
        freeze_tables=args.freeze_tables,
    )
    config = Config(**config_kwargs)

    train_dst_pool = np.unique(loaded.train.destinations)
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
    )

    # Derive alignment_time_scale (train_span / L_REF=20).
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

    # Source-side e_t_u function for the evaluator — static target(u)
    # lookup. The kwarg name `walk_repr_fn` is retained for evaluator
    # API stability; under the current architecture it doesn't produce
    # a walk representation, just a target embedding.
    walk_repr_fn = trainer._e_t_u_for

    _eval_kwargs = dict(
        embedding_store=trainer.embedding_store,
        link_predictor=trainer.link_predictor,
        device=trainer.device,
        tgb_dataset_name=loaded.name,
        eval_metric=loaded.eval_metric,
        time_encoder=trainer.time_encoder,
        time_state=trainer.time_state,
        time_scale=trainer._time_scale,
        walk_repr_fn=walk_repr_fn,
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
    print(f"=== Training (early stop patience={args.early_stop_patience}) ===")
    summary = trainer.train(
        train_batches_factory=lambda: create_batches(loaded.train, config.target_batch_size),
        val_evaluator=eval_val,
        val_batches_factory=lambda: create_batches(loaded.val, config.target_batch_size),
        test_evaluator=eval_test,
        test_batches_factory=lambda: create_batches(loaded.test, config.target_batch_size),
    )

    print("=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, list):
            if len(v) <= 50:
                print(f"  {k:24}: " + ", ".join(f"{x:.4f}" for x in v))
            else:
                print(f"  {k:24}: [list of {len(v)}]")
        else:
            print(f"  {k:24}: {v}")


if __name__ == "__main__":
    main()
