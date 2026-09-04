"""Effective walk-length statistics per TGB-Seq dataset.

Ingests the TRAIN split into Tempest, samples N random seed nodes, draws K backward
walks per seed at a generous max_walk_len, and reports the distribution of the
EFFECTIVE walk length Tempest returns (`WalkData.lens`) -- how deep a walk actually
gets before it runs out of causal history, as opposed to the cap it was allowed.

CPU Tempest only: this is meant to run alongside a GPU training job.
"""
import argparse
import json
import pathlib
import sys
import time

# Put the project root on sys.path so direct invocation can import the package.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from link_property_prediction.tgb_seq_eval import load_tgb_seq
from link_property_prediction.walks import WalkGenerator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-seeds", type=int, default=20_000)
    ap.add_argument("--mwl", type=int, default=80)
    ap.add_argument("--k", type=int, default=5, help="walks per seed node")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="write stats JSON here")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[{args.dataset}] loading ...", flush=True)
    loaded = load_tgb_seq(args.dataset)
    tr = loaded.train
    n_edges = int(tr.sources.shape[0])
    print(f"[{args.dataset}] train edges={n_edges:,}  nodes(max_id)={loaded.max_node_count:,}"
          f"  ({time.time()-t0:.1f}s)", flush=True)

    # Ingest the TRAIN split only. No edge features: they cost memory and do not
    # affect walk length.
    t1 = time.time()
    wg = WalkGenerator(
        use_gpu=False,
        walk_bias="ExponentialWeight",
        start_bias="ExponentialWeight",
        num_walks_per_node=args.k,
        max_walk_len=args.mwl,
    )
    wg.add_edges(tr.sources, tr.destinations, tr.timestamps, None)
    print(f"[{args.dataset}] ingested {n_edges:,} edges ({time.time()-t1:.1f}s)", flush=True)

    # Seed population = nodes that actually appear in the train split. Sampling from
    # the raw id range would draw ids with no history and floor the statistics.
    pop = np.unique(np.concatenate([tr.sources, tr.destinations]))
    rng = np.random.default_rng(args.seed)
    n_take = min(args.num_seeds, pop.shape[0])
    seeds = rng.choice(pop, size=n_take, replace=False)
    print(f"[{args.dataset}] active nodes={pop.shape[0]:,}  sampled seeds={n_take:,}", flush=True)

    t2 = time.time()
    wd = wg.walks_for_nodes(seeds, max_walk_len=args.mwl, num_walks_per_node=args.k)
    lens = wd.lens.numpy().astype(np.int64)
    walk_s = time.time() - t2

    # Cross-check `lens` against the padding mask; they must agree.
    mask_lens = (wd.nodes.numpy() != -1).sum(axis=1).astype(np.int64)
    agree = bool((mask_lens == lens).all())

    st = {
        "dataset": args.dataset,
        "train_edges": n_edges,
        "active_nodes": int(pop.shape[0]),
        "seeds": int(n_take),
        "k": args.k,
        "mwl": args.mwl,
        "n_walks": int(lens.shape[0]),
        "mean": float(lens.mean()),
        "std": float(lens.std()),
        "min": int(lens.min()),
        "max": int(lens.max()),
        "p50": float(np.percentile(lens, 50)),
        "p90": float(np.percentile(lens, 90)),
        "p99": float(np.percentile(lens, 99)),
        "frac_at_cap": float((lens >= args.mwl).mean()),
        "frac_len1": float((lens <= 1).mean()),
        "frac_ge5": float((lens >= 5).mean()),
        "lens_match_mask": agree,
        "walk_seconds": walk_s,
    }
    print(f"[{args.dataset}] walks={st['n_walks']:,} in {walk_s:.1f}s  "
          f"mean={st['mean']:.2f} std={st['std']:.2f} max={st['max']} "
          f"at_cap={st['frac_at_cap']:.4f} lens==mask:{agree}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(st, f, indent=2)
    print(f"[{args.dataset}] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
