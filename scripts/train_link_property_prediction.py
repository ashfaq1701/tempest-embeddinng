"""CLI entry point for tempest-embedding training.

Single-binary training script. Loads a TGB dataset via data.py,
constructs a Trainer, runs training, prints results. No experiment-
management logic — for parameter sweeps, invoke this script
repeatedly with different CLI args.

Hyperparameters exposed at CLI (and their grouping):
  Dataset:        --dataset, --tgb-root
  Model:          --d-emb
  Link/head:      --k-train
  Walks:          --num-walks-per-node, --max-walk-len, --walk-bias, --start-bias
                  (backward-only, undirected; source u → μ_u; candidate v via static E[v])
  Optimisation:   --lr, --weight-decay, --batch-size, --eval-batch-size,
                  --num-epochs, --early-stop-patience
                  (plain AdamW at a constant LR — no scheduler / decay / warmup)
  System:         --seed, --use-gpu, --use-gpu-tempest
  Analysis:       --stratify (post-train per-slice test-MRR stratification)

Derived from the dataset (not exposed): num_nodes, dst_pool, and the
train-split span (TrainStats) for the μ recency-λ init.

The full graph (train + val + test) is ingested into Tempest ONCE before
training; per-query cutoffs keep every walk causal (see Trainer.ingest_full_graph).
"""

import argparse
import pathlib
import sys
import time
from typing import Any, Dict

# Allow direct invocation (`python scripts/train_link_property_prediction.py ...`) by putting
# the project root on sys.path. `python -m scripts.train_link_property_prediction ...` works
# without this; the bootstrap is for the spec's first invocation form.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from link_property_prediction.data import Loaded, concat_splits, create_batches, load_tgb
from link_property_prediction.data_stats import compute_train_stats
from link_property_prediction.evaluator import Evaluator
from link_property_prediction.negatives import TGBNegativeSampler
from link_property_prediction.trainer import Trainer, TrainerConfig
from link_property_prediction.utils import seed_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tempest walks-supervised temporal embedding training"
    )

    # Dataset.
    p.add_argument("--dataset", required=True, type=str,
                   help="TGB dataset name, e.g. tgbl-wiki, tgbl-review")
    p.add_argument("--tgb-root", default="datasets", type=str)


    # Model — stateless NodeEncoding + walk-neighbourhood attention head.
    p.add_argument("--d-emb", default=128, type=int,
                   help="NodeEncoding random-feature width per hop-block (JL basis dim).")
    p.add_argument("--n-hops", default=3, type=int,
                   help="Diffusion depth: node_enc = [X0, ÂX0, …, Âⁿ X0], width (n_hops+1)*d_emb.")
    p.add_argument("--t2v-dim", default=16, type=int,
                   help="Time2Vec output dim for the per-token age feature.")

    # Link loss / head.
    p.add_argument(
        "--k-train", type=int, default=100,
        help="Per-query training negatives. The head sees [B, 1+K_train] "
             "candidates per query; positive at column 0.",
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

    # Walks (BACKWARD only, undirected) for the source side (u → μ_u). One-sided head: only the
    # source is walked; each candidate v enters through its static embedding E[v].
    p.add_argument("--num-walks-per-node", default=10, type=int,
                   help="K walks per source node u.")
    p.add_argument("--max-walk-len", default=5, type=int,
                   help="L, max walk length. (Sweep on wiki: shorter is better — 20→5 gave "
                        "+0.006 test, monotone, more stable.)")
    p.add_argument("--walk-bias", default="ExponentialWeight", type=str,
                   help="Per-hop edge bias for the backward walks.")
    p.add_argument("--start-bias", default="ExponentialWeight", type=str,
                   help="Initial-edge bias for the backward walks.")
    p.add_argument("--t2nv-p", default=4.0, type=float,
                   help="node2vec return param p (only used when the walk bias is "
                        "TemporalNode2Vec). Higher p => less immediate backtrack.")
    p.add_argument("--t2nv-q", default=0.25, type=float,
                   help="node2vec in-out param q (TemporalNode2Vec bias only). Lower q/p => "
                        "more outward exploration; p=4,q=0.25 = most diverse backward walks.")


    # Optimisation — plain AdamW at a constant LR (no scheduler / decay / warmup), like GraphMixer/TPNet.
    p.add_argument("--lr", default=1e-3, type=float,
                   help="Constant learning rate (no decay). GraphMixer/TPNet both use 1e-4.")
    p.add_argument("--weight-decay", default=1e-4, type=float,
                   help="AdamW weight decay (GraphMixer 1e-6, TPNet 0.0).")
    p.add_argument(
        "--batch-size", default=200, type=int,
        help="Train batch size. Under the per-query ranking link "
             "loss each batch does B*(1+K_train) link_head forwards.",
    )
    p.add_argument(
        "--eval-batch-size", default=20, type=int,
        help="Batch size for val/test eval batches. The link head "
             "materialises tensors of shape [eval_batch_size, 1+K_eval, "
             "d_emb] where K_eval is TGB's per-positive negative count "
             "(wiki=999, review=100, coin=20, comment=20). Comfortable "
             "values at d_emb=128 on 8 GB: wiki ~25-50, review ~200-500, "
             "coin/comment ~2000+. Default 200 fits review/coin/comment; "
             "wiki needs --eval-batch-size 25-50 explicitly.",
    )
    p.add_argument("--num-epochs", default=25, type=int)
    p.add_argument("--early-stop-patience", default=10, type=int)

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
    stats = compute_train_stats(train_sp.timestamps)

    print(f"  num_nodes:     {num_nodes:,}")
    print(f"  dst_pool:      {len(dst_pool):,} unique destinations")
    print(f"  t_min:         {stats.t_min}")
    print(f"  t_max:         {stats.t_max}")
    print(f"  T_train:       {stats.T_train:.0f}")
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
        n_hops=args.n_hops,
        d_ef=(int(train_sp.edge_feat.shape[1]) if train_sp.edge_feat is not None else 0),
        d_nf=(int(loaded.node_feat.shape[1]) if loaded.node_feat is not None else 0),
        node_feat=loaded.node_feat,

        t2v_dim=args.t2v_dim,

        K_train=args.k_train,

        num_walks_per_node=args.num_walks_per_node,
        max_walk_len=args.max_walk_len,
        walk_bias=args.walk_bias,
        start_bias=args.start_bias,
        t2nv_p=args.t2nv_p,
        t2nv_q=args.t2nv_q,
        lr=args.lr,
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
    n_total = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"  TOTAL trainable: {n_total:>12,}  (stateless: no learned node-embedding table)")

    # ─── Train ─────────────────────────────────────────────────────
    print("\n=== Training ===")
    result = trainer.train(
        train_batches_factory=train_batches_factory,
        full_graph=concat_splits(train_sp, val_sp, test_sp),
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
        from link_property_prediction.stratify import run_stratification
        meta = {
            "dataset": args.dataset, "seed": args.seed, "d_emb": args.d_emb,
            "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
            "head": type(trainer.model).__name__,
            "best_epoch": result["stopped_at_epoch"],
            "best_val": result["best_val_mrr"], "best_test": result["best_test_mrr"],
        }
        run_stratification(
            trainer, train_batches_factory, val_batches_factory,
            test_eval, test_batches_factory, num_nodes, meta)

    return result


if __name__ == "__main__":
    main()
