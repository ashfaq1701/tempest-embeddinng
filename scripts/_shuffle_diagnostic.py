"""Temporary diagnostic — verify walks.nodes[i*K, lens[i*K]-1] == seeds[i]
under the shuffle_walk_order=False fix. Delete after running.

Run on a tiny graph: 5 nodes, 10 deterministic edges, then sample walks
for seeds [0, 1, 2, 3, 4] with K=3. Expect output rows 0..2 to have seed 0
at nodes[lens-1], rows 3..5 seed 1, ..., rows 12..14 seed 4.
"""

import numpy as np
from tempest_walks.walks import WalkGenerator


def main() -> None:
    gen = WalkGenerator(
        is_directed=False,
        use_gpu=False,
        walk_bias="ExponentialWeight",
        max_walk_len=5,
        num_walks_per_node=3,
        timescale_bound=-1,
    )

    # 5-node star + a couple of bridges, distinct timestamps per edge.
    src = np.array([0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 3, 4], dtype=np.int64)
    tgt = np.array([1, 0, 2, 0, 3, 0, 4, 0, 2, 1, 4, 3], dtype=np.int64)
    ts  = np.arange(1, len(src) + 1, dtype=np.int64)
    gen.add_edges(src, tgt, ts, edge_feat=None)

    seeds = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    walks = gen.walks_for_nodes(seeds)
    K = walks.K
    nodes = walks.nodes.numpy()
    lens = walks.lens.numpy()
    print(f"K={K}  W={nodes.shape[0]}  L={nodes.shape[1]}  lens={lens.tolist()}")
    print()
    ok = True
    for i, s in enumerate(seeds):
        for k in range(K):
            row = i * K + k
            seed_at_row = int(nodes[row, lens[row] - 1])
            match = "ok" if seed_at_row == int(s) else "MISMATCH"
            print(f"  row {row:2d} (seed expected={s}, K-slot={k}): "
                  f"nodes[{row}, lens-1={lens[row]-1}] = {seed_at_row}  {match}")
            if seed_at_row != int(s):
                ok = False
    print()
    print(f"Verdict: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
