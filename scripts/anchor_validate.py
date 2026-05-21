"""Anchor validation per v2.2 §3.

Runs the Phase 0.5 architecture (Component 0 + dual identity tables +
8-block cross-table link MLP + alignment + uniformity) for 3 seeds
{42, 7, 13} at num_epochs=2, reports mean +/- std of val/test MRR, and
applies v2.2 §3.2 decision gate.

Reference smoke from `phase0_5_diag.py` at seed=42, 2 epochs:
  val 0.7451, test 0.7070  (the number anchor validation is verifying)

The configuration here mirrors phase0_5_diag.py exactly except for
num_epochs (2 instead of 50) and seed (looped over {42, 7, 13}).
That's the only honest way to test whether the 0.71 reproduces.

Each seed runs in-process with a fresh Trainer + Evaluator. The TGB
dataset and val/test negatives are loaded once and reused across seeds
(they are read-only; load_val_ns / load_test_ns are idempotent).

Usage:
    python3 scripts/anchor_validate.py
    python3 scripts/anchor_validate.py --tgb-name tgbl-wiki --use-gpu

Writes results JSON to runs/anchor_validation.json.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from tempest_walks.config import Config
from tempest_walks.data import create_batches, load_tgb
from tempest_walks.evaluator import Evaluator
from tempest_walks.negatives import TGBNegativeSampler
from tempest_walks.trainer import Trainer


SEEDS = (42, 7, 13)
NUM_EPOCHS = 2
L_REF = 20  # fixed reference for time_scale derivation, per Lesson 11


def _build_config(loaded, seed: int, use_gpu: bool) -> Config:
    """Phase 0.5 config — matches phase0_5_diag.py exactly except for
    num_epochs (2) and the looped seed. All Config defaults that match
    are left implicit; deviations from default are explicit."""
    return Config(
        tgb_name=loaded.name,
        max_node_count=loaded.max_node_count,
        is_directed=loaded.is_directed,
        num_epochs=NUM_EPOCHS,
        seed=seed,
        use_gpu=use_gpu,
    )


def run_one(loaded, seed: int, use_gpu: bool) -> dict:
    """Train + eval one seed. Fully resets RNG and re-constructs all
    stateful objects so seeds are independent."""
    # RNG: seed BEFORE constructing anything that draws random numbers
    # (EmbeddingStore init, optimizer momentum buffers, neg_sampler).
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

    config = _build_config(loaded, seed, use_gpu)
    trainer = Trainer(
        config=config,
        train_dst_pool=train_dst_pool,
        node_feat=loaded.node_feat,
        edge_feat_dim=edge_feat_dim,
    )

    # time_scale derivation (Lesson 11: fixed L_REF=20, NOT max_walk_len).
    ts = loaded.train.timestamps
    span = float(ts.max() - ts.min())
    derived = span / float(L_REF)
    trainer.set_time_scale(derived)

    eval_kwargs = dict(
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
        **eval_kwargs,
    )
    eval_test = Evaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="test"),
        **eval_kwargs,
    )

    print(f"  Model device: {trainer.device}    Tempest device: cpu")
    print(f"  time_scale = {derived:.1f}")
    t_train_0 = time.perf_counter()
    trainer.train(create_batches(loaded.train, config.target_batch_size))
    train_dt = time.perf_counter() - t_train_0

    t_eval_0 = time.perf_counter()
    val_mrr = trainer.evaluate(
        create_batches(loaded.val, config.target_batch_size), eval_val,
    )
    test_mrr = trainer.evaluate(
        create_batches(loaded.test, config.target_batch_size), eval_test,
    )
    eval_dt = time.perf_counter() - t_eval_0

    return {
        "seed": seed,
        "val_mrr": float(val_mrr),
        "test_mrr": float(test_mrr),
        "train_time_sec": float(train_dt),
        "eval_time_sec": float(eval_dt),
    }


def decision_gate(mean_test: float, std_test: float) -> dict:
    """v2.2 §3.2, verbatim thresholds."""
    if mean_test >= 0.70 and std_test <= 0.02:
        return {
            "verdict": "CONFIRMED",
            "action": "Phase S anchors at the mean. Proceed.",
        }
    if 0.65 <= mean_test < 0.70:
        return {
            "verdict": "PARTIAL",
            "action": (
                "Phase S anchors at the verified mean (not 0.71 from the smoke). "
                "v2.2 §4.4 success criterion adjusts accordingly."
            ),
        }
    if mean_test < 0.65 or std_test > 0.04:
        return {
            "verdict": "STOP",
            "action": (
                "Investigate before Phase S. Likely causes: diagnostic config "
                "drift, batch-ordering nondeterminism, or walk-gen state "
                "differences."
            ),
        }
    # Defensive fallthrough — shouldn't trigger given the conditions above,
    # but if it does, surface the gap rather than hide it.
    return {
        "verdict": "AMBIGUOUS",
        "action": "Mean falls outside named bands; manual review required.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Anchor validation (v2.2 §3)")
    p.add_argument("--tgb-name", default="tgbl-wiki")
    p.add_argument("--tgb-root", default="datasets")
    p.add_argument("--use-gpu", action="store_true",
                   help="Run the model on CUDA (Tempest always on CPU "
                        "per 8 GB VRAM constraint).")
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
    print("Loading TGB negatives (val + test)…")
    loaded.dataset.load_val_ns()
    loaded.dataset.load_test_ns()

    runs = []
    wall_start = time.perf_counter()
    for seed in SEEDS:
        print(f"\n========== seed {seed} ({NUM_EPOCHS} epochs) ==========")
        r = run_one(loaded, seed, args.use_gpu)
        print(
            f"  seed {seed} → val {r['val_mrr']:.4f}  test {r['test_mrr']:.4f}  "
            f"(train {r['train_time_sec']:.1f}s, eval {r['eval_time_sec']:.1f}s)"
        )
        runs.append(r)
    total_wall = time.perf_counter() - wall_start

    val_mrrs = [r["val_mrr"] for r in runs]
    test_mrrs = [r["test_mrr"] for r in runs]
    val_mean = statistics.mean(val_mrrs)
    val_std = statistics.stdev(val_mrrs) if len(val_mrrs) > 1 else 0.0
    test_mean = statistics.mean(test_mrrs)
    test_std = statistics.stdev(test_mrrs) if len(test_mrrs) > 1 else 0.0

    gate = decision_gate(test_mean, test_std)

    summary = {
        "config": "Phase 0.5 (Component 0 + alignment + uniformity)",
        "num_epochs": NUM_EPOCHS,
        "seeds": list(SEEDS),
        "L_REF": L_REF,
        "runs": runs,
        "val_mean": val_mean,
        "val_std": val_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "decision": gate,
        "total_wall_sec": float(total_wall),
        "reference_smoke": {
            "seed": 42,
            "num_epochs": 2,
            "val_mrr": 0.7451,
            "test_mrr": 0.7070,
            "source": "phase0_5_diag.py 2-epoch run",
        },
    }

    print("\n========== ANCHOR VALIDATION SUMMARY (v2.2 §3) ==========")
    for r in runs:
        print(f"  seed {r['seed']:3d}  val {r['val_mrr']:.4f}  test {r['test_mrr']:.4f}")
    print(f"  ---")
    print(f"  val  mean {val_mean:.4f}  std {val_std:.4f}")
    print(f"  test mean {test_mean:.4f}  std {test_std:.4f}")
    print(f"  total wall: {total_wall:.1f}s")
    print(f"\n  Verdict: {gate['verdict']}")
    print(f"  Action:  {gate['action']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
