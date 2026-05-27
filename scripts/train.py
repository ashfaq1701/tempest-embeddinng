"""CLI entry point for tempest-embedding training.

Single-binary training script. Loads a TGB dataset via data.py,
constructs a Trainer, runs training, prints results. No experiment-
management logic — for parameter sweeps, invoke this script
repeatedly with different CLI args.

Hyperparameters exposed at CLI (and their grouping):
  Dataset:        --dataset, --tgb-root
  Model:          --d-emb, --d-proj
  Loss:           --tau, --beta-time, --num-align-negatives
  Walks:          --num-walks-per-node, --max-walk-len, --walk-bias,
                  --start-bias
  Negatives:      --num-neg-per-pos, --hist-neg-ratio, --reservoir-size
  Optimisation:   --lr, --weight-decay, --batch-size, --num-epochs,
                  --early-stop-patience
  System:         --seed, --use-gpu, --skip-final-full-eval,
                  --monitor-sample-pct

Derived from the dataset (not exposed):
  num_nodes, is_directed, is_bipartite, dst_pool, d_node_feat,
  T_train (= max(train.ts) - min(train.ts)).
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
from tempest_walks.evaluator import Evaluator
from tempest_walks.negatives import TGBNegativeSampler
from tempest_walks.trainer import Trainer, TrainerConfig
from tempest_walks.utils import derive_t_train, detect_bipartite, seed_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tempest walks-supervised temporal embedding training"
    )

    # Dataset.
    p.add_argument("--dataset", required=True, type=str,
                   help="TGB dataset name, e.g. tgbl-wiki, tgbl-review")
    p.add_argument("--tgb-root", default="datasets", type=str)

    # Directedness override. The dataset-default comes from
    # tempest_walks.data.default_is_directed (a curated list with
    # version-suffix normalisation). These flags let the user force a
    # choice when running on a dataset the default table doesn't know.
    directed_group = p.add_mutually_exclusive_group()
    directed_group.add_argument(
        "--directed",
        dest="directed_override",
        action="store_const",
        const=True,
        default=None,
        help="Force is_directed=True (overrides dataset default).",
    )
    directed_group.add_argument(
        "--undirected",
        dest="directed_override",
        action="store_const",
        const=False,
        default=None,
        help="Force is_directed=False (overrides dataset default).",
    )

    # Model.
    p.add_argument("--d-emb", default=128, type=int)
    p.add_argument("--d-proj", default=128, type=int)

    # Loss.
    p.add_argument("--tau", default=0.5, type=float,
                   help="InfoNCE contrastive temperature")
    p.add_argument("--beta-time", default=1.0, type=float)
    p.add_argument(
        "--num-align-negatives", type=int, default=128,
        help="Number of sampled negatives per seed in InfoNCE "
             "alignment loss. Higher = sharper contrastive signal "
             "but more memory. Sweepable. Default 128 — from the "
             "wiki K sweep (3 seeds × 50 ep), knee of the diminishing-"
             "returns curve; ~98% of K=512's test MRR at ~2.6× less "
             "compute and ~2× lower std. Largest K that fits on 8 GB "
             "at comment-scale NK (K=256+ OOMs there).",
    )

    # Walks.
    p.add_argument("--num-walks-per-node", default=5, type=int)
    p.add_argument("--max-walk-len", default=20, type=int)
    p.add_argument("--walk-bias", default="ExponentialWeight", type=str)
    p.add_argument("--start-bias", default="Uniform", type=str)

    # Walk encoder (feature-flagged; default OFF). Produces h_seed
    # consumed by the LINK HEAD only. InfoNCE alignment_loss is
    # unchanged. Encoder learns ONLY from BCE; E lookups inside the
    # encoder are detached so BCE does not reach E via the encoder.
    p.add_argument(
        "--use-walk-encoder", action="store_true",
        help="Enable the walk encoder for the link head. h_seed "
             "(per-edge MLP + GRU + mean-pool + MLP_seed) replaces "
             "E[seed] at link_head inputs only. InfoNCE remains on E. "
             "Default OFF.",
    )
    p.add_argument("--d-te",   default=32,  type=int, help="time2vec dim")
    p.add_argument("--d-he",   default=16,  type=int, help="hop embedding dim")
    p.add_argument("--d-edge", default=128, type=int, help="per-edge representation dim")
    p.add_argument("--d-walk", default=128, type=int, help="GRU/attn hidden / walk representation dim")
    p.add_argument(
        "--encoder-arch", choices=["gru", "attn"], default="gru",
        help="Walk encoder backbone: 'gru' (default) or 'attn' "
             "(transformer self-attention over walk edges).",
    )
    p.add_argument("--encoder-n-heads",  default=4, type=int,
                   help="attn-only: number of self-attention heads")
    p.add_argument("--encoder-n-layers", default=1, type=int,
                   help="attn-only: number of transformer-encoder layers")
    p.add_argument(
        "--encoder-exclude-seed", action="store_true",
        help="attn-only: drop E[seed] from both the last-edge tgt slot "
             "(replaced by a learned [SEED] marker) AND the final "
             "MLP_seed concat. h_seed becomes purely neighbourhood-derived.",
    )

    # Negatives.
    p.add_argument("--num-neg-per-pos", default=10, type=int)
    p.add_argument("--hist-neg-ratio", default=0.5, type=float)
    p.add_argument("--reservoir-size", default=32, type=int)

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
    p.add_argument("--batch-size", default=2000, type=int)
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
    p.add_argument("--skip-final-full-eval", action="store_true")
    p.add_argument("--monitor-sample-pct", default=1.0, type=float)

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
    if args.directed_override is not None:
        is_directed = args.directed_override
        directed_provenance = "CLI override"
    else:
        is_directed = loaded.is_directed
        directed_provenance = "dataset default"
    is_bipartite = detect_bipartite(loaded.train)
    dst_pool = np.unique(loaded.train.destinations).astype(np.int32)
    T_train = derive_t_train(loaded.train.timestamps)
    d_node_feat = (
        int(loaded.node_feat.shape[1])
        if loaded.node_feat is not None
        else None
    )
    # Edge feature dim is dataset-specific (None if absent). Only used
    # by the walk encoder when enabled; harmless otherwise.
    d_edge_feat = (
        int(loaded.train.edge_feat.shape[1])
        if loaded.train.edge_feat is not None
        else None
    )

    print(f"  num_nodes:     {num_nodes:,}")
    print(f"  directed:      {is_directed}  ({directed_provenance})")
    print(f"  bipartite:     {is_bipartite}")
    print(f"  dst_pool:      {len(dst_pool):,} unique destinations")
    print(f"  T_train:       {T_train:.0f}")
    print(f"  train edges:   {len(loaded.train.sources):,}")
    print(f"  val edges:     {len(loaded.val.sources):,}")
    print(f"  test edges:    {len(loaded.test.sources):,}")
    print(f"  has_node_feat: {loaded.node_feat is not None}"
          + (f" (d={d_node_feat})" if d_node_feat is not None else ""))

    # ─── Build batch factories ─────────────────────────────────────
    # create_batches consumes a SplitData and yields Batches in
    # chronological order. We wrap it in a lambda so the trainer can
    # re-iterate the split each epoch.
    train_batches_factory = (
        lambda: create_batches(loaded.train, args.batch_size)
    )
    val_batches_factory = (
        lambda: create_batches(loaded.val, args.batch_size)
    )
    test_batches_factory = (
        lambda: create_batches(loaded.test, args.batch_size)
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
        is_bipartite=is_bipartite,
        dst_pool=dst_pool,
        t_train_span=T_train,
        d_node_feat=d_node_feat,
        d_edge_feat=d_edge_feat,

        d_emb=args.d_emb,
        d_proj=args.d_proj,

        tau=args.tau,
        beta_time=args.beta_time,
        num_align_negatives=args.num_align_negatives,

        num_walks_per_node=args.num_walks_per_node,
        max_walk_len=args.max_walk_len,
        walk_bias=args.walk_bias,
        start_bias=args.start_bias,

        use_walk_encoder=args.use_walk_encoder,
        encoder_arch=args.encoder_arch,
        encoder_n_heads=args.encoder_n_heads,
        encoder_n_layers=args.encoder_n_layers,
        encoder_exclude_seed=args.encoder_exclude_seed,
        d_te=args.d_te,
        d_he=args.d_he,
        d_edge=args.d_edge,
        d_walk=args.d_walk,

        num_neg_per_pos=args.num_neg_per_pos,
        hist_neg_ratio=args.hist_neg_ratio,
        reservoir_size=args.reservoir_size,

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
        skip_final_full_eval=args.skip_final_full_eval,
        monitor_sample_pct=args.monitor_sample_pct,
    )

    print("\n=== Config ===")
    for k, v in vars(config).items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: <ndarray shape={v.shape} dtype={v.dtype}>")
        else:
            print(f"  {k}: {v}")

    # ─── Instantiate Trainer ───────────────────────────────────────
    trainer = Trainer(config=config, node_feat=loaded.node_feat, device=device)

    print("\n=== Parameter counts ===")
    n_E = sum(p.numel() for p in trainer.embedding_table.parameters())
    n_Pt = sum(p.numel() for p in trainer.p_target.parameters())
    n_Pc = sum(p.numel() for p in trainer.p_context.parameters())
    n_H = sum(p.numel() for p in trainer.link_head.parameters())
    n_W = (
        sum(p.numel() for p in trainer.walk_encoder.parameters())
        if trainer.walk_encoder is not None
        else 0
    )
    print(f"  embedding_table: {n_E:>12,}")
    print(f"  p_target:        {n_Pt:>12,}")
    print(f"  p_context:       {n_Pc:>12,}")
    print(f"  link_head:       {n_H:>12,}")
    if trainer.walk_encoder is not None:
        print(f"  walk_encoder:    {n_W:>12,}")
    print(f"  TOTAL trainable: {n_E + n_Pt + n_Pc + n_H + n_W:>12,}")

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

    return result


if __name__ == "__main__":
    main()
