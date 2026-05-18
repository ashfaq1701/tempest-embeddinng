"""Diagnostic: walk lengths actually achieved on a TGB dataset.

Replays the training schedule with the configured target_batch_size and,
AFTER each ingest, samples walks for that batch's seeds (union of src and
tgt — matching the trainer's seeding policy) and reports the lens stats
(avg, max). This is the same state that the NEXT batch's walks would see
during real training.

A final summary aggregates over all batches.
"""

import argparse

import numpy as np

from tempest_walks.data import create_batches, load_tgb
from tempest_walks.walks import WalkGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tgb-name", default="tgbl-wiki")
    p.add_argument("--tgb-root", default="datasets")
    p.add_argument("--target-batch-size", type=int, default=200)
    p.add_argument("--max-walk-len", type=int, default=20)
    p.add_argument("--num-walks-per-node", type=int, default=5)
    p.add_argument("--walk-bias", default="ExponentialWeight")
    p.add_argument("--is-directed", default=None,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--print-every", type=int, default=50,
                   help="Print a per-batch row every N batches.")
    args = p.parse_args()

    loaded = load_tgb(args.tgb_name, root=args.tgb_root)
    is_directed = args.is_directed if args.is_directed is not None else loaded.is_directed
    print(f"Dataset: {args.tgb_name}  N={loaded.max_node_count}  "
          f"train_edges={len(loaded.train.sources)}  is_directed={is_directed}")
    print(f"Walk config: max_walk_len={args.max_walk_len}  "
          f"num_walks_per_node={args.num_walks_per_node}  bias={args.walk_bias}  "
          f"target_batch_size={args.target_batch_size}")

    walk_gen = WalkGenerator(
        is_directed=is_directed,
        use_gpu=False,
        walk_bias=args.walk_bias,
        max_walk_len=args.max_walk_len,
        num_walks_per_node=args.num_walks_per_node,
    )

    batches = list(create_batches(loaded.train, args.target_batch_size))
    n_batches = len(batches)
    print(f"Total batches: {n_batches}\n")

    header = (
        f"{'batch':>6}  {'edges':>6}  {'seeds':>6}  {'walks':>7}  "
        f"{'avg':>6}  {'max':>4}  {'p50':>4}  {'p90':>4}  "
        f"{'%@cap':>6}  {'%cold':>6}"
    )
    print(header)
    print("-" * len(header))

    all_lens = []
    edges_ingested = 0

    for i, batch in enumerate(batches):
        # 1. Ingest this batch FIRST.
        walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
        edges_ingested += len(batch.src)

        # 2. Sample walks for this batch's seeds (union of src and tgt —
        #    same as the trainer's seeding policy).
        seeds = np.unique(np.concatenate([batch.src, batch.tgt]))
        walks = walk_gen.walks_for_nodes(seeds)
        lens = walks.lens.cpu().numpy().astype(np.int64)
        all_lens.append(lens)

        if (i + 1) % args.print_every == 0 or i == n_batches - 1:
            avg = float(lens.mean()) if lens.size else 0.0
            mx = int(lens.max()) if lens.size else 0
            p50 = int(np.percentile(lens, 50)) if lens.size else 0
            p90 = int(np.percentile(lens, 90)) if lens.size else 0
            cap = int((lens == args.max_walk_len).sum())
            cold = int((lens == 0).sum())
            print(
                f"{i+1:>6d}  {edges_ingested:>6d}  {len(seeds):>6d}  {lens.size:>7d}  "
                f"{avg:>6.2f}  {mx:>4d}  {p50:>4d}  {p90:>4d}  "
                f"{100*cap/max(lens.size,1):>5.1f}%  {100*cold/max(lens.size,1):>5.1f}%"
            )

    print()
    arr = np.concatenate(all_lens) if all_lens else np.array([], dtype=np.int64)
    if arr.size == 0:
        print("No walks sampled.")
        return

    print("=== Aggregate over all batches ===")
    print(f"  total walks:    {arr.size}")
    print(f"  mean lens:      {arr.mean():.3f}")
    print(f"  median lens:    {int(np.percentile(arr, 50))}")
    print(f"  max lens:       {int(arr.max())}")
    print(f"  p90 / p99:      {int(np.percentile(arr, 90))} / {int(np.percentile(arr, 99))}")
    cap = int((arr == args.max_walk_len).sum())
    cold = int((arr == 0).sum())
    solo = int((arr == 1).sum())
    print(f"  cold (lens=0):  {cold} ({100*cold/arr.size:.2f}%)")
    print(f"  solo (lens=1):  {solo} ({100*solo/arr.size:.2f}%)")
    print(f"  at cap (={args.max_walk_len}): {cap} ({100*cap/arr.size:.2f}%)")


if __name__ == "__main__":
    main()
