"""Anchor validation — 3 seeds × 2 epochs on a TGB dataset.

Gate A for the locked-v2 architecture. Verifies the minimal production
codebase reproduces the v2.2 §3 anchor (test MRR 0.7070 ± 0.0016 on
tgbl-wiki-v2). If this passes, the production pipeline is intact.

Usage:
  python -m scripts.anchor_validate --tgb-name tgbl-wiki --use-gpu

Each seed runs in-process with a fresh Trainer + Evaluator. RNG is
fully reset between seeds.
"""

import argparse
import json
import time

import numpy as np
import torch

from tempest_walks.config import Config
from tempest_walks.data import create_batches, load_tgb
from tempest_walks.evaluator import Evaluator
from tempest_walks.negatives import TGBNegativeSampler
from tempest_walks.trainer import Trainer


SEEDS = [42, 7, 13]
NUM_EPOCHS = 2
L_REF = 20


def run_one(loaded, seed: int, use_gpu: bool) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    edge_feat_dim = (
        int(loaded.train.edge_feat.shape[1])
        if loaded.train.edge_feat is not None
        else 0
    )
    train_dst_pool = np.unique(loaded.train.destinations)

    config = Config(
        tgb_name=loaded.name,
        max_node_count=loaded.max_node_count,
        is_directed=loaded.is_directed,
        num_epochs=NUM_EPOCHS,
        seed=seed,
        use_gpu=use_gpu,
    )
    trainer = Trainer(
        config=config,
        train_dst_pool=train_dst_pool,
        node_feat=loaded.node_feat,
        edge_feat_dim=edge_feat_dim,
    )

    ts = loaded.train.timestamps
    span = float(ts.max() - ts.min())
    derived = span / float(L_REF)
    trainer.set_time_scale(derived)

    walk_repr_fn = trainer._compute_walk_repr_for
    eval_kwargs = dict(
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
        **eval_kwargs,
    )
    eval_test = Evaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="test"),
        **eval_kwargs,
    )

    print(f"  Model device: {trainer.device}    Tempest device: cpu")
    print(f"  time_scale = {derived:.1f}")
    t0 = time.perf_counter()
    summary = trainer.train(
        train_batches_factory=lambda: create_batches(loaded.train, config.target_batch_size),
        val_evaluator=eval_val,
        val_batches_factory=lambda: create_batches(loaded.val, config.target_batch_size),
        test_evaluator=eval_test,
        test_batches_factory=lambda: create_batches(loaded.test, config.target_batch_size),
    )
    dt = time.perf_counter() - t0
    return {
        "seed": seed,
        "val_mrr": summary["best_val_mrr"],
        "test_mrr": summary["best_test_mrr"],
        "wall_s": dt,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tgb-name", default="tgbl-wiki")
    p.add_argument("--tgb-root", default="datasets")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--out", default="runs/anchor_validation.json")
    args = p.parse_args()

    print(f"Loading TGB dataset: {args.tgb_name}")
    loaded = load_tgb(args.tgb_name, root=args.tgb_root)
    print(
        f"  N={loaded.max_node_count}  "
        f"train={len(loaded.train.sources)}  val={len(loaded.val.sources)}  "
        f"test={len(loaded.test.sources)}  is_directed={loaded.is_directed}  "
        f"eval_metric={loaded.eval_metric}",
    )
    loaded.dataset.load_val_ns()
    loaded.dataset.load_test_ns()

    results = []
    t_all = time.perf_counter()
    for s in SEEDS:
        print(f"\n========== seed {s} ({NUM_EPOCHS} epochs) ==========")
        r = run_one(loaded, s, args.use_gpu)
        print(f"  seed {s} → val {r['val_mrr']:.4f}  test {r['test_mrr']:.4f}  "
              f"({r['wall_s']:.1f}s)")
        results.append(r)
    total_dt = time.perf_counter() - t_all

    vals = np.array([r["val_mrr"] for r in results])
    tests = np.array([r["test_mrr"] for r in results])
    print(f"\n========== ANCHOR VALIDATION SUMMARY ==========")
    for r in results:
        print(f"  seed {r['seed']:3d}  val {r['val_mrr']:.4f}  test {r['test_mrr']:.4f}")
    print(f"  ---")
    print(f"  val  mean {vals.mean():.4f}  std {vals.std():.4f}")
    print(f"  test mean {tests.mean():.4f}  std {tests.std():.4f}")
    print(f"  total wall: {total_dt:.1f}s")

    # v2.2 §3 anchor: test MRR 0.7070 ± 0.0016 on tgbl-wiki.
    if args.tgb_name == "tgbl-wiki":
        target_mean, target_std = 0.7070, 0.0016
        if abs(tests.mean() - target_mean) <= 2 * target_std:
            print(f"\n  Verdict: CONFIRMED  (target {target_mean} ± {target_std})")
        else:
            print(f"\n  Verdict: DRIFT  (actual {tests.mean():.4f}, target {target_mean} ± {target_std})")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "vals": vals.tolist(), "tests": tests.tolist()}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
