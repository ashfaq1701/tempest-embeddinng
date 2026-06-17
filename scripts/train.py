"""CLI entry point for tempest-embedding training.

Single-binary training script. Loads a TGB dataset via data.py,
constructs a Trainer, runs training, prints results. No experiment-
management logic — for parameter sweeps, invoke this script
repeatedly with different CLI args.

Hyperparameters exposed at CLI (and their grouping):
  Dataset:        --dataset, --tgb-root
  Model:          --d-emb
  Link/head:      --tau-link, --k-train
  Walks:          --{num-walks-per-node,max-walk-len,walk-bias,start-bias}-
                  {query,candidate}-side, --tempest-batch-window-multiplier
                  (backward-only, undirected; query=source→μ, candidate=v→connectors)
  Optimisation:   --lr, --lr-min, --warmup-fraction,
                  --warmup-steps-cap, --decay-horizon-epochs, --weight-decay,
                  --batch-size, --eval-batch-size, --num-epochs,
                  --early-stop-patience
  System:         --seed, --use-gpu, --use-gpu-tempest
  Analysis:       --stratify (post-train per-slice test-MRR stratification)

Derived from the dataset (not exposed): num_nodes, dst_pool, and
mean_inter_arrival (TrainStats) for the Tempest sliding-window cap.
"""

import argparse
import pathlib
import sys
import time
from typing import Any, Dict

# Allow direct invocation (`python scripts/train.py ...`) by putting
# the project root on sys.path. `python -m scripts.train ...` works
# without this; the bootstrap is for the spec's first invocation form.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from tempest_walks.data import Loaded, create_batches, load_tgb
from tempest_walks.data_stats import compute_train_stats
from tempest_walks.evaluator import Evaluator
from tempest_walks.negatives import TGBNegativeSampler
from tempest_walks.trainer import Trainer, TrainerConfig
from tempest_walks.utils import compute_max_time_capacity, seed_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tempest walks-supervised temporal embedding training"
    )

    # Dataset.
    p.add_argument("--dataset", required=True, type=str,
                   help="TGB dataset name, e.g. tgbl-wiki, tgbl-review")
    p.add_argument("--tgb-root", default="datasets", type=str)


    # Model.
    p.add_argument("--d-emb", default=128, type=int)

    # Link loss / head.
    p.add_argument(
        "--tau-link", default=1.0, type=float,
        help="Link-prediction softmax-CE temperature (per-query ranking loss).",
    )
    p.add_argument(
        "--k-train", type=int, default=100,
        help="Per-query training negatives. The head sees [B, 1+K_train] "
             "candidates per query; positive at column 0.",
    )

    # Pair features. Off by default => the baseline cross-GRU head is reproduced
    # byte-identically.
    p.add_argument(
        "--use-pair-features", action="store_true",
        help="Add exact pairwise (u,v) recurrence + history from a streaming store "
             "as one logit term: Time2Vec(time-since-last (u,v) interaction) ‖ "
             "ever-interacted bit ‖ decayed log interaction-count. Multi-seed "
             "confirmed +~0.02 test on tgbl-wiki.",
    )

    # Chronological subsample (wiki-sized window on big datasets, e.g. review).
    p.add_argument(
        "--max-train-edges", default=0, type=int,
        help="If >0, train on only the most-recent N train edges (fixed "
             "chronological suffix) — a wiki-sized subsample of big datasets.")
    p.add_argument(
        "--max-eval-edges", default=0, type=int,
        help="If >0, eval on only the first N official val/test edges (prefix; "
             "keeps TGB pre-generated negatives valid).")

    # Walks (BACKWARD only, undirected). Decoupled QUERY-side (source u → μ) and
    # CANDIDATE-side (v → connectors for the cross channel); same Tempest graph.
    p.add_argument("--num-walks-per-node-query-side", default=5, type=int,
                   help="K walks per source u (build μ).")
    p.add_argument("--max-walk-len-query-side", default=20, type=int,
                   help="L, max walk length for the query-side walks.")
    p.add_argument("--walk-bias-query-side", default="ExponentialWeight", type=str,
                   help="Per-hop edge bias for the query-side backward walks.")
    p.add_argument("--start-bias-query-side", default="ExponentialWeight", type=str,
                   help="Initial-edge bias for the query-side backward walks.")

    p.add_argument("--num-walks-per-node-candidate-side", default=10, type=int,
                   help="K walks per candidate v; their context nodes are the cross "
                        "channel's connectors (FREE LENGTH — direct + indirect).")
    p.add_argument("--max-walk-len-candidate-side", default=2, type=int,
                   help="L for the candidate-side walks. 2 = v's direct neighbours "
                        "only; >2 reaches indirect (co-reachability) neighbours.")
    p.add_argument("--walk-bias-candidate-side", default="Linear", type=str,
                   help="Per-hop edge bias for the candidate-side walks.")
    p.add_argument("--start-bias-candidate-side", default="Linear", type=str,
                   help="Initial-edge bias for the candidate-side walks.")

    # The link head (LinkPredHead) has no architecture knobs beyond
    # max_walk_len, which is set from --link-pred-max-walk-len.
    p.add_argument(
        "--tempest-batch-window-multiplier", default=-1.0, type=float,
        help="Tempest sliding-window cap expressed as a multiple of the "
             "mean batch's time-span. The effective max_time_capacity "
             "passed to Tempest is "
             "round(multiplier * batch_size * mean_inter_arrival) — see "
             "tempest_walks/utils.py:compute_max_time_capacity. -1.0 "
             "(default) is the unbounded sentinel: Tempest retains all "
             "ingested edges until walk_gen.reset() at the epoch "
             "boundary. The multiplier interface is dataset-agnostic; "
             "the raw window depends on the dataset's calendar density.",
    )

    # Optimisation. Single RiemannianAdam over {E (sphere), GRU, scale}.
    p.add_argument(
        "--lr", default=1e-3, type=float, help="Peak LR (RiemannianAdam).",
    )
    p.add_argument(
        "--lr-min", default=1e-5, type=float,
        help="Cosine-decay LR floor (~peak/1000).",
    )
    p.add_argument(
        "--warmup-fraction", default=0.05, type=float,
        help="Warmup as fraction of decay horizon steps.",
    )
    p.add_argument(
        "--warmup-steps-cap", default=500, type=int,
        help="Maximum warmup steps regardless of fraction.",
    )
    p.add_argument(
        "--decay-horizon-epochs", default=50, type=int,
        help="Target epoch count for cosine decay to reach lr-min. "
             "SEPARATE from --num-epochs — short runs stay near peak; "
             "full decay is hit only at num_epochs = horizon.",
    )
    p.add_argument("--weight-decay", default=1e-4, type=float)
    p.add_argument(
        "--batch-size", default=500, type=int,
        help="Train batch size. Under the per-query ranking link "
             "loss each batch does B*(1+K_train) link_head forwards. "
             "Default 500 keeps the per-step compute envelope "
             "comparable to historical baselines.",
    )
    p.add_argument(
        "--eval-batch-size", default=200, type=int,
        help="Batch size for val/test eval batches. The link head "
             "materialises tensors of shape [eval_batch_size, 1+K_eval, "
             "d_emb] where K_eval is TGB's per-positive negative count "
             "(wiki=999, review=100, coin=20, comment=20). Comfortable "
             "values at d_emb=128 on 8 GB: wiki ~25-50, review ~200-500, "
             "coin/comment ~2000+. Default 200 fits review/coin/comment; "
             "wiki needs --eval-batch-size 25-50 explicitly.",
    )
    p.add_argument("--num-epochs", default=50, type=int)
    p.add_argument("--early-stop-patience", default=0, type=int)

    # System.
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--use-gpu", action="store_true",
                   help="Move PyTorch tensors (E, projections, link head, "
                        "losses) to CUDA. Does NOT affect Tempest.")
    p.add_argument(
        "--use-gpu-tempest",
        action="store_true",
        help="Run Tempest's walk sampler in GPU mode. Independent from "
             "--use-gpu, which controls PyTorch tensor placement. "
             "Tempest GPU mode allocates a multi-GB arena that may "
             "collide with PyTorch's allocator on small GPUs; default "
             "off, enable only if you have headroom.",
    )
    p.add_argument(
        "--stratify",
        action="store_true",
        help="After training, re-run the strict-causal TEST eval on the best-val "
             "model and stratify per-positive MRR by pair-recurrence, "
             "transductivity (endpoint-seen), and source-degree — localizing where "
             "MRR is lost. Writes logs/stratify/<dataset>_seed<seed>_strata.{md,json}. "
             "Re-seeds the trainer's causal stores over train+val first; training is "
             "untouched. Off by default.",
    )
    p.add_argument(
        "--export-best-embedding-table",
        action="store_true",
        help="After training, dump the best-val-restored embedding-table "
             "weights to logs/embeddings/<dataset>_seed<seed>_demb<d_emb>"
             "_ep<stopped_at_epoch>.npy. Raw float32 [num_nodes, d_emb] "
             "array; node ids follow TGB's contiguous integer ordering. "
             "NOTE: rows are now UNIT-NORM (sphere-constrained E) — analyse "
             "by direction (cosine), not magnitude. "
             "Off by default.",
    )

    return p.parse_args()


def main() -> Dict[str, Any]:
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(
        "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    )

    print("=== tempest-embedding training ===")
    print(f"dataset: {args.dataset}")
    print(f"device:  {device}")
    print(f"seed:    {args.seed}")

    # ─── Load dataset ──────────────────────────────────────────────
    t0 = time.time()
    loaded: Loaded = load_tgb(name=args.dataset, root=args.tgb_root)
    print(f"loaded in {time.time() - t0:.1f}s")

    # TGB requires negative-sampler files to be loaded before val/test
    # negatives can be queried. They're cached on disk after first call.
    loaded.dataset.load_val_ns()
    loaded.dataset.load_test_ns()

    # Derived dataset constants.
    num_nodes = loaded.max_node_count

    # Optional chronological subsample: recent suffix of train (so the graph the
    # walks see stays causal), official prefix of val/test (keeps TGB's
    # pre-generated negatives valid). Used to run a wiki-sized window on big
    # datasets (e.g. review) that otherwise OOM an 8 GB GPU at full size.
    def _trunc(split, n, tail):
        if n <= 0 or n >= int(split.sources.shape[0]):
            return split
        sl = slice(-n, None) if tail else slice(0, n)
        ef = split.edge_feat[sl] if split.edge_feat is not None else None
        return split._replace(
            sources=split.sources[sl], destinations=split.destinations[sl],
            timestamps=split.timestamps[sl], edge_feat=ef)

    train_sp = _trunc(loaded.train, args.max_train_edges, tail=True)
    val_sp = _trunc(loaded.val, args.max_eval_edges, tail=False)
    test_sp = _trunc(loaded.test, args.max_eval_edges, tail=False)

    dst_pool = np.unique(train_sp.destinations).astype(np.int32)
    # Full-dataset timestamps (train + val + test) feed compute_train_stats
    # so it can populate t_max_full / T_full — the v2 link-pred head's
    # time channel normaliser, bounded across train and eval splits.
    full_ts = np.concatenate([
        train_sp.timestamps,
        val_sp.timestamps,
        test_sp.timestamps,
    ])
    stats = compute_train_stats(train_sp.timestamps, full_timestamps=full_ts)

    print(f"  num_nodes:     {num_nodes:,}")
    print(f"  dst_pool:      {len(dst_pool):,} unique destinations")
    print(f"  t_min:         {stats.t_min}")
    print(f"  t_max:         {stats.t_max}")
    print(f"  T_train:       {stats.T_train:.0f}")
    print(f"  t_max_full:    {stats.t_max_full}")
    print(f"  T_full:        {stats.T_full:.0f}")
    print(f"  median_inter_arrival: {stats.median_inter_arrival:.1f}")
    print(f"  mean_inter_arrival:   {stats.mean_inter_arrival:.1f}")
    print(f"  train edges:   {len(train_sp.sources):,}")
    print(f"  val edges:     {len(val_sp.sources):,}")
    print(f"  test edges:    {len(test_sp.sources):,}")

    # ─── Build batch factories ─────────────────────────────────────
    # create_batches consumes a SplitData and yields Batches in
    # chronological order. We wrap it in a lambda so the trainer can
    # re-iterate the split each epoch. Eval uses a separate batch
    # size from train — the eval-side per-batch pair count blows up
    # as eval_batch_size * (1 + K) for TGB's pregenerated negatives,
    # so it's a memory-fitting knob distinct from train.
    train_batches_factory = (
        lambda: create_batches(train_sp, args.batch_size)
    )
    val_batches_factory = (
        lambda: create_batches(val_sp, args.eval_batch_size)
    )
    test_batches_factory = (
        lambda: create_batches(test_sp, args.eval_batch_size)
    )

    # ─── Build evaluators ──────────────────────────────────────────
    val_eval = Evaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="val"),
        tgb_dataset_name=loaded.name,
        eval_metric=loaded.eval_metric,
    )
    test_eval = Evaluator(
        neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="test"),
        tgb_dataset_name=loaded.name,
        eval_metric=loaded.eval_metric,
    )

    # ─── Build TrainerConfig ───────────────────────────────────────
    config = TrainerConfig(
        num_nodes=num_nodes,
        dst_pool=dst_pool,
        t_train=float(stats.T_train),

        d_emb=args.d_emb,

        tau_link=args.tau_link,
        K_train=args.k_train,

        use_pair_features=args.use_pair_features,

        num_walks_per_node_query_side=args.num_walks_per_node_query_side,
        max_walk_len_query_side=args.max_walk_len_query_side,
        walk_bias_query_side=args.walk_bias_query_side,
        start_bias_query_side=args.start_bias_query_side,
        num_walks_per_node_candidate_side=args.num_walks_per_node_candidate_side,
        max_walk_len_candidate_side=args.max_walk_len_candidate_side,
        walk_bias_candidate_side=args.walk_bias_candidate_side,
        start_bias_candidate_side=args.start_bias_candidate_side,
        max_time_capacity=compute_max_time_capacity(
            args.tempest_batch_window_multiplier,
            args.batch_size,
            stats.mean_inter_arrival,
        ),

        lr=args.lr,
        lr_min=args.lr_min,
        warmup_fraction=args.warmup_fraction,
        warmup_steps_cap=args.warmup_steps_cap,
        decay_horizon_epochs=args.decay_horizon_epochs,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        early_stop_patience=args.early_stop_patience,

        seed=args.seed,
        use_gpu=args.use_gpu,
        use_gpu_tempest=args.use_gpu_tempest,
    )

    print("\n=== Config ===")
    for k, v in vars(config).items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: <ndarray shape={v.shape} dtype={v.dtype}>")
        else:
            print(f"  {k}: {v}")

    # ─── Instantiate Trainer ───────────────────────────────────────
    trainer = Trainer(config=config, device=device)

    print("\n=== Parameter counts ===")
    n_E = sum(p.numel() for p in trainer.embedding_table.parameters())
    n_H = sum(p.numel() for p in trainer.link_head.parameters())
    print(f"  embedding_table: {n_E:>12,}")
    print(f"  link_head:       {n_H:>12,}")
    print(f"  TOTAL trainable: {n_E + n_H:>12,}")

    # ─── Train ─────────────────────────────────────────────────────
    print("\n=== Training ===")
    result = trainer.train(
        train_batches_factory=train_batches_factory,
        val_evaluator=val_eval,
        val_batches_factory=val_batches_factory,
        test_evaluator=test_eval,
        test_batches_factory=test_batches_factory,
    )

    # ─── Results ───────────────────────────────────────────────────
    print("\n=== Final results ===")
    print(f"  dataset:           {args.dataset}")
    print(f"  seed:              {args.seed}")
    print(f"  stopped_at_epoch:  {result['stopped_at_epoch']}")
    print(f"  best_val_mrr:      {result['best_val_mrr']:.4f}")
    print(f"  best_test_mrr:     {result['best_test_mrr']:.4f}")

    # Optional: stratify the best-val model's test MRR to localize the gap.
    if args.stratify:
        from tempest_walks.stratify import run_stratification
        meta = {
            "dataset": args.dataset, "seed": args.seed, "d_emb": args.d_emb,
            "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
            "head": type(trainer.link_head).__name__
            + (" + pair features" if args.use_pair_features else ""),
            "best_epoch": result["stopped_at_epoch"],
            "best_val": result["best_val_mrr"], "best_test": result["best_test_mrr"],
        }
        run_stratification(
            trainer, train_batches_factory, val_batches_factory,
            test_eval, test_batches_factory, num_nodes, meta)

    # Optional: dump best-val-restored embedding table for downstream
    # analysis. Raw [num_nodes, d_emb] float32 array; node ids follow
    # TGB's contiguous integer ordering. Gated by --export-best-
    # embedding-table; off by default to keep runs side-effect-free.
    if args.export_best_embedding_table:
        emb_dir = pathlib.Path("logs/embeddings")
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_path = emb_dir / (
            f"{args.dataset}_seed{args.seed}_demb{args.d_emb}"
            f"_ep{result['stopped_at_epoch']}.npy"
        )
        np.save(
            emb_path,
            trainer.embedding_table.E.weight.detach().cpu().numpy(),
        )
        print(f"  embedding_table:   saved to {emb_path}")

    return result


if __name__ == "__main__":
    main()
