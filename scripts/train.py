"""CLI entry point for tempest-embedding training.

Single-binary training script. Loads a TGB dataset via data.py,
constructs a Trainer, runs training, prints results. No experiment-
management logic — for parameter sweeps, invoke this script
repeatedly with different CLI args.

Hyperparameters exposed at CLI (and their grouping):
  Dataset:        --dataset, --tgb-root, --is-directed
  Model:          --d-emb
  Loss:           --tau-align, --tau-link, --gamma-recency,
                  --k-train, --alignment-chunk-size
  Walks:          --embedding-num-walks-per-node, --embedding-max-walk-len,
                  --embedding-backward-walk-bias, --embedding-backward-start-bias,
                  --link-pred-num-walks-per-node, --link-pred-max-walk-len,
                  --link-pred-forward-walk-bias, --link-pred-forward-start-bias,
                  --link-pred-backward-walk-bias, --link-pred-backward-start-bias,
                  --tempest-batch-window-multiplier
                  (Embedding side is backward-only. Link-pred side picks
                   forward or backward by --is-directed.)
  Optimisation:   --lr, --lr-min, --warmup-fraction, --warmup-steps-cap,
                  --decay-horizon-epochs, --weight-decay, --batch-size,
                  --eval-batch-size, --num-epochs, --early-stop-patience
  System:         --seed, --use-gpu, --use-gpu-tempest

Derived from the dataset (not exposed):
  num_nodes, is_directed, dst_pool,
  TrainStats (t_min, t_max, T_train, median_inter_arrival,
              mean_inter_arrival) — see tempest_walks/data_stats.py.
  recency_scale defaults to TrainStats.mean_inter_arrival and is
  frozen (not learnable) — held as a plain Python float on the Trainer.
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


def _compute_train_deg(loaded: Loaded) -> np.ndarray:
    """Per-node incidence count over the train split. Undirected:
    each edge counts once on each endpoint. Used by inverse-degree
    seed weighting in the alignment loss."""
    deg = np.zeros(loaded.max_node_count, dtype=np.int64)
    src = loaded.train.sources.astype(np.int64)
    tgt = loaded.train.destinations.astype(np.int64)
    np.add.at(deg, src, 1)
    np.add.at(deg, tgt, 1)
    return deg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tempest walks-supervised temporal embedding training"
    )

    # Dataset.
    p.add_argument("--dataset", required=True, type=str,
                   help="TGB dataset name, e.g. tgbl-wiki, tgbl-review")
    p.add_argument("--tgb-root", default="datasets", type=str)

    # Directedness is an explicit caller-supplied flag with no
    # internal fallback table. Default OFF (treat the graph as
    # undirected). Pass --is-directed for datasets where the
    # topology is genuinely directed; this only affects the walk
    # sampler (Tempest constructor) — eval scoring is always
    # task-directional regardless.
    p.add_argument(
        "--is-directed", action="store_true",
        help="Treat the graph as directed (default: undirected). "
             "Consumed by the walk sampler (Tempest) only.",
    )

    # Model.
    p.add_argument("--d-emb", default=128, type=int)

    # Loss.
    p.add_argument(
        "--tau-align", default=0.5, type=float,
        help="InfoNCE alignment temperature (walks-side contrastive).",
    )
    p.add_argument(
        "--tau-link", default=1.0, type=float,
        help="Link-prediction softmax-CE temperature (per-query "
             "ranking loss). Default 1.0 — pending a sweep.",
    )
    p.add_argument(
        "--gamma-recency", default=0.4, type=float,
        help="Convex-combination weight between hop and stationary-"
             "recency profiles in the alignment-loss per-position "
             "weight. γ=0 is hop-only; γ=1 is recency-only; default "
             "0.4 mixes both. Recency_scale is data-driven from the "
             "train split's median inter-arrival time and is NOT a CLI "
             "knob — see tempest_walks/data_stats.py.",
    )
    p.add_argument(
        "--k-train", type=int, default=100,
        help="Per-query training negatives for the ranking link "
             "loss. The link head sees [B, 1+K_train] candidates "
             "per query; positive at column 0. Larger K_train means "
             "harder per-query competition and stronger ranking "
             "gradients, at proportional compute cost.",
    )
    p.add_argument(
        "--alignment-chunk-size", default=8192, type=int,
        help="Slices the unique-pool dimension V when computing the "
             "InfoNCE partition log Z. Each chunk's forward is "
             "gradient-checkpointed, so backward peak memory is "
             "bounded by O(NK·chunk_size) rather than O(NK·V). When "
             "V ≤ chunk_size the loop runs once and behaviour reduces "
             "to the dense path. Default 8192 fits wiki/coin in one "
             "chunk and bounds review's pathological pools.",
    )

    # Walks.
    # Both the embedding and link-pred sides sample BOTH directions per
    # seed. *-num-walks-per-node is the TOTAL K, split half/half between
    # forward and backward at the trainer level — defaults to 10
    # (5+5 per direction). Bias knobs are per direction; defaults
    # reflect Tempest's chronological semantics:
    #   - forward + ExpW start shoots toward the head (oldest end) of
    #     the successor set, the least predictive slice. Uniform start
    #     + ExpW walk spreads coverage, then biases continuations
    #     toward recency relative to the previous hop.
    #   - backward + ExpW start lands on the seed's most recent
    #     predecessor (the tail / most predictive end), so ExpW + ExpW.
    p.add_argument(
        "--embedding-num-walks-per-node", default=10, type=int,
        help="K for embedding-side walks per seed (all spent on the "
             "single backward direction).",
    )
    p.add_argument(
        "--embedding-max-walk-len", default=20, type=int,
        help="L for embedding-side walks.",
    )
    p.add_argument(
        "--embedding-backward-walk-bias", default="ExponentialWeight", type=str,
        help="Per-hop edge bias for backward embedding-side walks.",
    )
    p.add_argument(
        "--embedding-backward-start-bias", default="ExponentialWeight", type=str,
        help="Initial-edge bias for backward embedding-side walks.",
    )
    p.add_argument(
        "--link-pred-num-walks-per-node", default=10, type=int,
        help="TOTAL K for link-pred-side walks; split half/half into "
             "forward + backward at construction (so default 10 → 5+5).",
    )
    p.add_argument(
        "--link-pred-max-walk-len", default=20, type=int,
        help="L for link-pred-side walks.",
    )
    p.add_argument(
        "--link-pred-forward-walk-bias", default="ExponentialWeight", type=str,
        help="Per-hop edge bias for FORWARD link-pred-side walks.",
    )
    p.add_argument(
        "--link-pred-forward-start-bias", default="Uniform", type=str,
        help="Initial-edge bias for FORWARD link-pred-side walks.",
    )
    p.add_argument(
        "--link-pred-backward-walk-bias", default="ExponentialWeight", type=str,
        help="Per-hop edge bias for BACKWARD link-pred-side walks.",
    )
    p.add_argument(
        "--link-pred-backward-start-bias", default="ExponentialWeight", type=str,
        help="Initial-edge bias for BACKWARD link-pred-side walks.",
    )
    # Symmetric forward+backward embedding alignment and inverse-degree
    # seed weighting are always on. Iter 6 analysis (analysis/REPORT.md)
    # established both as load-bearing improvements over baseline (test
    # +0.010 combined); there is no scenario where either should be off,
    # so no CLI knob.

    # LinkPredHeadV2 — walk-mediated link-pred head. Flags scope the
    # ablation sweep designed in analysis/REPORT.md §9; once a winning
    # config emerges they will be collapsed into hardcoded defaults on
    # the post-experiment cleanup branch.
    p.add_argument("--link-head-d-K",      default=16, type=int)
    p.add_argument("--link-head-d-pos",    default=96, type=int)
    p.add_argument("--link-head-d-direct", default=64, type=int)
    p.add_argument(
        "--link-head-chunk-c", default=0, type=int,
        help="0 (default) = no chunking; pass a positive N to enable "
             "candidate-dim chunking inside the walk tower with chunk "
             "size N. Pure memory knob; loss/gradient are identical "
             "to non-chunked. Smaller N → less peak memory but more "
             "kernel launches per step.",
    )
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

    # Optimisation.
    p.add_argument(
        "--lr", default=1e-3, type=float,
        help="Peak learning rate (after warmup). Default 1e-3 — "
             "wiki bs=200 seed-42 A/B (sampled-neg K=64): lr=1e-3 "
             "hit val 0.4454 vs lr=1e-2 at val 0.4301. The K=64 "
             "sampled-negative gradients are noisier than the full "
             "in-batch InfoNCE's, so a smaller step size converges "
             "more reliably.",
    )
    p.add_argument(
        "--lr-min", default=1e-5, type=float,
        help="Minimum LR at end of cosine decay. Default 1e-5 follows "
             "contrastive-SSL convention (SimCLR/MoCo/BYOL cosine to ~0; "
             "we use 1e-5 ≈ peak/1000).",
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
        "--export-best-embedding-table",
        action="store_true",
        help="After training, dump the best-val-restored embedding-table "
             "weights to logs/embeddings/<dataset>_seed<seed>_demb<d_emb>"
             "_ep<stopped_at_epoch>.npy. Raw float32 [num_nodes, d_emb] "
             "array; node ids follow TGB's contiguous integer ordering. "
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
    is_directed = args.is_directed
    dst_pool = np.unique(loaded.train.destinations).astype(np.int32)
    # Full-dataset timestamps (train + val + test) feed compute_train_stats
    # so it can populate t_max_full / T_full — the v2 link-pred head's
    # time channel normaliser, bounded across train and eval splits.
    full_ts = np.concatenate([
        loaded.train.timestamps,
        loaded.val.timestamps,
        loaded.test.timestamps,
    ])
    stats = compute_train_stats(loaded.train.timestamps, full_timestamps=full_ts)

    print(f"  num_nodes:     {num_nodes:,}")
    print(f"  directed:      {is_directed}  (--is-directed)")
    print(f"  dst_pool:      {len(dst_pool):,} unique destinations")
    print(f"  t_min:         {stats.t_min}")
    print(f"  t_max:         {stats.t_max}")
    print(f"  T_train:       {stats.T_train:.0f}")
    print(f"  t_max_full:    {stats.t_max_full}")
    print(f"  T_full:        {stats.T_full:.0f}")
    print(f"  median_inter_arrival: {stats.median_inter_arrival:.1f}")
    print(f"  mean_inter_arrival:   {stats.mean_inter_arrival:.1f}")
    print(f"  train edges:   {len(loaded.train.sources):,}")
    print(f"  val edges:     {len(loaded.val.sources):,}")
    print(f"  test edges:    {len(loaded.test.sources):,}")

    # ─── Build batch factories ─────────────────────────────────────
    # create_batches consumes a SplitData and yields Batches in
    # chronological order. We wrap it in a lambda so the trainer can
    # re-iterate the split each epoch. Eval uses a separate batch
    # size from train — the eval-side per-batch pair count blows up
    # as eval_batch_size * (1 + K) for TGB's pregenerated negatives,
    # so it's a memory-fitting knob distinct from train.
    train_batches_factory = (
        lambda: create_batches(loaded.train, args.batch_size)
    )
    val_batches_factory = (
        lambda: create_batches(loaded.val, args.eval_batch_size)
    )
    test_batches_factory = (
        lambda: create_batches(loaded.test, args.eval_batch_size)
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
        is_directed=is_directed,
        dst_pool=dst_pool,

        d_emb=args.d_emb,

        tau_align=args.tau_align,
        tau_link=args.tau_link,
        gamma_recency=args.gamma_recency,
        recency_scale=stats.mean_inter_arrival,
        K_train=args.k_train,
        alignment_chunk_size=args.alignment_chunk_size,

        embedding_num_walks_per_node=args.embedding_num_walks_per_node,
        embedding_max_walk_len=args.embedding_max_walk_len,
        embedding_backward_walk_bias=args.embedding_backward_walk_bias,
        embedding_backward_start_bias=args.embedding_backward_start_bias,
        link_pred_num_walks_per_node=args.link_pred_num_walks_per_node,
        link_pred_max_walk_len=args.link_pred_max_walk_len,
        link_pred_forward_walk_bias=args.link_pred_forward_walk_bias,
        link_pred_forward_start_bias=args.link_pred_forward_start_bias,
        link_pred_backward_walk_bias=args.link_pred_backward_walk_bias,
        link_pred_backward_start_bias=args.link_pred_backward_start_bias,
        train_deg=_compute_train_deg(loaded),
        t_min=stats.t_min,
        T_train=stats.T_train,
        t_max_full=stats.t_max_full,
        T_full=stats.T_full,
        link_head_d_K=args.link_head_d_K,
        link_head_d_pos=args.link_head_d_pos,
        link_head_d_direct=args.link_head_d_direct,
        link_head_chunk_c=args.link_head_chunk_c,
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
