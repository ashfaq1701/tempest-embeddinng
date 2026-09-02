"""CLI entry point for tempest-embedding training.

Loads a dataset, constructs a Trainer, runs training, prints results.
"""

import argparse
import pathlib
import sys
import time
from typing import Any, Dict

# Put the project root on sys.path so direct invocation can import the package.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from link_property_prediction.data import Loaded, concat_splits, create_batches
from link_property_prediction.evaluator import make_suite
from link_property_prediction.trainer import Trainer, TrainerConfig
from link_property_prediction.utils import seed_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tempest walks-supervised temporal embedding training"
    )

    # ── Dataset ─────────────────────────────────────────────────────
    p.add_argument("--data-suite", default="tgb-seq", choices=["tgb-seq"],
                   help="Benchmark suite to load from.")
    p.add_argument("--dataset", required=True, type=str,
                   help="Dataset name within the suite.")
    p.add_argument("--data-root", default="datasets", type=str,
                   help="Data root directory.")
    p.add_argument("--is-bipartite", action="store_true",
                   help="Treat the graph as bipartite.")
    p.add_argument("--max-train-edges", default=0, type=int,
                   help="If >0, train on only the most-recent N train edges.")
    p.add_argument("--max-eval-edges", default=0, type=int,
                   help="If >0, eval on only the first N val/test edges.")

    # ── Model ───────────────────────────────────────────────────────
    p.add_argument("--d-emb", default=64, type=int,
                   help="Embedding dimension.")
    p.add_argument("--use-pop-bias", action="store_true",
                   help="Score a learned per-node popularity scalar (zero-init) alongside the distance, "
                        "mixed by the learned weight vector w.")

    # ── Walks ───────────────────────────────────────────────────────
    p.add_argument("--num-walks-per-node", default=5, type=int,
                   help="Walks per query node.")
    p.add_argument("--max-walk-len", default=5, type=int,
                   help="Max walk length.")
    p.add_argument("--walk-bias", default="ExponentialWeight", type=str,
                   help="Per-hop edge bias for the walks.")
    p.add_argument("--start-bias", default="ExponentialWeight", type=str,
                   help="Initial-edge bias for the walks.")
    p.add_argument("--t2nv-p", default=4.0, type=float,
                   help="node2vec return param p (TemporalNode2Vec bias only).")
    p.add_argument("--t2nv-q", default=0.25, type=float,
                   help="node2vec in-out param q (TemporalNode2Vec bias only).")

    # ── Negatives ───────────────────────────────────────────────────
    p.add_argument("--k-train", default=5, type=int,
                   help="Per-query training negatives.")
    p.add_argument("--k-eval", default=100, type=int,
                   help="Eval negatives per positive (tgb-seq val only).")

    # ── Optimisation / training ─────────────────────────────────────
    p.add_argument("--lr", default=1e-3, type=float,
                   help="Learning rate. One param group: the embedding tables, the distance "
                        "temperature and the NN pooler all step at this rate.")
    p.add_argument("--batch-size", default=1000, type=int,
                   help="Train batch size.")
    p.add_argument("--eval-batch-size", default=1000, type=int,
                   help="Val/test eval batch size.")
    p.add_argument("--num-epochs", default=50, type=int,
                   help="Max training epochs.")
    p.add_argument("--early-stop-patience", default=5, type=int,
                   help="Early-stop patience in epochs.")

    # ── System ──────────────────────────────────────────────────────
    p.add_argument("--seed", default=42, type=int,
                   help="Random seed.")
    p.add_argument("--use-gpu", action="store_true",
                   help="Place PyTorch tensors on CUDA.")
    p.add_argument("--use-gpu-tempest", action="store_true",
                   help="Run Tempest's walk sampler in GPU mode.")

    # ── Pooler widths ───────────────────────────────────────────────
    # Low-priority knobs: the defaults are the measured design and these are not swept.
    p.add_argument("--hidden-dim", default=32, type=int,
                   help="Hidden width of the pooler MLP. Pinned independently of the feature "
                        "count so a feature change does not also move pooler capacity.")

    # ── Post-training outputs ───────────────────────────────────────
    p.add_argument("--export-best-embedding-table", action="store_true",
                   help="After training, dump the best-val embedding table to disk.")

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

    # ─── Load dataset (native to the chosen suite) ─────────────────
    t0 = time.time()
    suite = make_suite(
        args.data_suite,
        name=args.dataset, root=args.data_root,
        is_bipartite=args.is_bipartite, k_eval=args.k_eval, seed=args.seed,
    )
    loaded: Loaded = suite.load()
    print(f"loaded ({args.data_suite}) in {time.time() - t0:.1f}s")

    # Derived dataset constants.
    num_nodes = loaded.max_node_count

    # Optional chronological subsample: recent suffix of train, prefix of val/test.
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

    # Negative-sampling pool, computed by the suite from the full train split.
    dst_pool = suite.dst_pool()

    print(f"  num_nodes:     {num_nodes:,}")
    _pool_kind = "destinations (bipartite)" if args.is_bipartite else "nodes (non-bipartite)"
    print(f"  neg_pool:      {len(dst_pool):,} unique {_pool_kind}")
    print(f"  train edges:   {len(train_sp.sources):,}")
    print(f"  val edges:     {len(val_sp.sources):,}")
    print(f"  test edges:    {len(test_sp.sources):,}")

    # ─── Build batch factories ─────────────────────────────────────
    # Wrapped in lambdas so the trainer can re-iterate each split every epoch.
    train_batches_factory = (
        lambda: create_batches(train_sp, args.batch_size)
    )
    val_batches_factory = (
        lambda: create_batches(val_sp, args.eval_batch_size)
    )
    test_batches_factory = (
        lambda: create_batches(test_sp, args.eval_batch_size)
    )

    # ─── Build evaluators (native to the suite) ────────────────────
    val_eval = suite.make_evaluator("val")
    test_eval = suite.make_evaluator("test")

    # ─── Build TrainerConfig ───────────────────────────────────────
    config = TrainerConfig(
        num_nodes=num_nodes,
        dst_pool=dst_pool,

        d_emb=args.d_emb,
        use_pop_bias=args.use_pop_bias,
        hidden_dim=args.hidden_dim,

        K_train=args.k_train,

        num_walks_per_node=args.num_walks_per_node,
        max_walk_len=args.max_walk_len,
        walk_bias=args.walk_bias,
        start_bias=args.start_bias,
        t2nv_p=args.t2nv_p,
        t2nv_q=args.t2nv_q,
        lr=args.lr,
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
    n_E = trainer.model.E.weight.numel()
    print(f"  model.E:         {n_E:>12,}")
    print(f"  head params:     {n_total - n_E:>12,}")
    print(f"  TOTAL trainable: {n_total:>12,}")

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

    # Optional: dump best-val embedding table for downstream analysis.
    if args.export_best_embedding_table:
        emb_dir = pathlib.Path("logs/embeddings")
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_path = emb_dir / (
            f"{args.dataset}_seed{args.seed}_demb{args.d_emb}"
            f"_ep{result['stopped_at_epoch']}.npy"
        )
        np.save(
            emb_path,
            trainer.model.E.weight.detach().cpu().numpy(),
        )
        print(f"  embedding_table:   saved to {emb_path}")

    return result


if __name__ == "__main__":
    main()
