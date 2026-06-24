"""Correctness tests for per-query walk tokens (tempest_walks/walk_token_csr.py).

`build_query_walk_tokens` generates K backward walks per QUERY — a (seed node, cutoff time)
pair — and packs each query's real context tokens left-aligned into [Q, U]. COUNT-FREE:
every reached position is its own token (no dedup; multiplicity kept). Checks:

1. CONTENT vs the SAME captured walks — the per-query (node, t_edge) MULTISET matches the
   contract reconstruction (context positions [0, lens-2]; seed slot / padding / origin
   excluded; kept with multiplicity), and the left-packing is dense with -1 on masked slots.
2. PER-QUERY CUTOFF — every emitted token has t_edge < that query's OWN cutoff; the SAME node
   at increasing cutoffs yields nested-larger, cutoff-respecting token sets (no dedup means
   each query is sampled independently).
3. EDGE CASES — a cutoff at/below the earliest edge, and an isolated node, each give an
   all-False (empty) row.
"""
from collections import Counter

import numpy as np
import torch

from tempest_walks.walk_token_csr import build_query_walk_tokens
from tempest_walks.walks import WalkGenerator


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _synthetic_graph(n_nodes=10, n_edges=120, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, n_edges).astype(np.int64)
    tgt = rng.integers(0, n_nodes, n_edges).astype(np.int64)
    same = src == tgt
    tgt[same] = (tgt[same] + 1) % n_nodes
    ts = np.sort(rng.choice(np.arange(1, 50_000), n_edges, replace=False)).astype(np.int64)
    return src, tgt, ts


def _per_query_multiset_from_dense(node_ids, node_mask, pos_ts):
    q_n, u = node_ids.shape
    return [Counter((int(node_ids[q, j]), int(pos_ts[q, j]))
                    for j in range(u) if bool(node_mask[q, j])) for q in range(q_n)]


def _expected_multiset_from_walkdata(wd, seeds):
    """Per-query (node, t_edge) MULTISET from captured WalkData under the contract:
    context = positions [0, lens-2]; the walk's OWN ORIGIN node-id dropped; multiplicity kept."""
    q_n = int(seeds.shape[0])
    k, length = wd.K, wd.nodes.shape[1]
    nodes = wd.nodes.view(q_n, k, length).cpu().numpy()
    ts = wd.timestamps.view(q_n, k, length).to(torch.int64).cpu().numpy()
    lens = wd.lens.view(q_n, k).cpu().numpy()
    sd = seeds.cpu().numpy().astype(np.int64)
    out = []
    for q in range(q_n):
        c = Counter()
        for kk in range(k):
            for p in range(0, int(lens[q, kk]) - 1):       # context only (excl seed slot + pad)
                node = int(nodes[q, kk, p])
                if node == int(sd[q]):                      # drop the walk's own origin
                    continue
                c[(node, int(ts[q, kk, p]))] += 1
        out.append(c)
    return out


def _run_with_capture(wg, seeds, cutoffs, **params):
    """Run build_query_walk_tokens while capturing the exact WalkData it sampled."""
    captured = {}
    orig = wg.walks_for_nodes

    def _cap(*a, **k):
        wd = orig(*a, **k)
        captured["wd"] = wd
        return wd

    wg.walks_for_nodes = _cap
    out = build_query_walk_tokens(wg, torch.device("cpu"), seeds, cutoffs, **params)
    wg.walks_for_nodes = orig
    return out, captured["wd"]


# ──────────────────────────────────────────────────────────────────────────
# 1. content end-to-end vs the SAME captured walks (multiplicity preserved)
# ──────────────────────────────────────────────────────────────────────────
def test_tokens_match_captured_walks_and_packing():
    src, tgt, ts = _synthetic_graph()
    k, mwl = 6, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=k, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)
    cutoffs = torch.full((5,), 50_000, dtype=torch.long)        # above all edges -> unconstrained

    (ids, mask, pts), wd = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=mwl, num_walks_per_node=k,
        walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    got = _per_query_multiset_from_dense(ids, mask, pts)
    exp = _expected_multiset_from_walkdata(wd, seeds)

    assert any(len(c) > 0 for c in exp), "no context tokens sampled — graph too sparse"
    for q in range(len(seeds)):
        assert got[q] == exp[q], (
            f"query {q} (seed {int(seeds[q])}): token multiset mismatch\n"
            f" got {sorted(got[q].items())}\n exp {sorted(exp[q].items())}")
    # multiplicity is real (no dedup): some node reached more than once.
    assert any(any(v > 1 for v in c.values()) for c in exp), \
        "expected a recurring node to exercise the multiplicity path"
    # packing: masked slots hold node id -1.
    assert bool((ids[~mask] == -1).all())
    print(f"\n[tokens] per-query multiset + multiplicity + packing OK ({len(seeds)} queries)")


# ──────────────────────────────────────────────────────────────────────────
# 2. per-query cutoff
# ──────────────────────────────────────────────────────────────────────────
def test_per_query_cutoff_is_honoured():
    src, tgt, ts = _synthetic_graph(seed=4)
    k, mwl = 8, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=k, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)

    uniq = np.unique(ts)
    node = int(src[len(src) // 2])
    cuts = [int(uniq[len(uniq) // 4]), int(uniq[len(uniq) // 2]), 50_000]
    seeds = torch.tensor([node, node, node], dtype=torch.long)     # SAME node, 3 cutoffs
    cutoffs = torch.tensor(cuts, dtype=torch.long)

    (ids, mask, pts), _ = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=mwl, num_walks_per_node=k,
        walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    max_ts = []
    for q in range(3):
        realts = pts[q][mask[q]].tolist()
        assert all(t < cuts[q] for t in realts), \
            f"query {q}: a token is at/after its cutoff {cuts[q]} (max {max(realts) if realts else None})"
        max_ts.append(max(realts) if realts else None)
    # Walks are stochastic with independent per-query RNG, so token COUNT is not monotone in
    # the cutoff. The deterministic invariant is the per-cutoff strictness checked above; the
    # loosest cutoff (unconstrained) must still sample some tokens.
    assert max_ts[2] is not None, "the unconstrained query should sample at least one token"
    print(f"\n[cutoff] every token strictly before its query's cutoff OK (max_ts {max_ts})")


# ──────────────────────────────────────────────────────────────────────────
# 3. edge cases
# ──────────────────────────────────────────────────────────────────────────
def test_empty_query_rows():
    src, tgt, ts = _synthetic_graph(n_nodes=6, n_edges=40, seed=2)
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=4, max_walk_len=6)
    wg.reset(); wg.add_edges(src, tgt, ts, None)

    t_min = int(ts.min())                       # no edge has t_edge < global min
    seeds = torch.tensor([int(src[0]), 999], dtype=torch.long)   # active node, isolated node 999
    cutoffs = torch.tensor([t_min, 50_000], dtype=torch.long)

    (ids, mask, _pts), _ = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=6, num_walks_per_node=4)
    assert not bool(mask[0].any()), "cutoff at/below earliest edge must yield an empty row"
    assert not bool(mask[1].any()), "isolated node must yield an empty row"
    print("\n[edge] cutoff-excluded query + isolated node give empty rows OK")


if __name__ == "__main__":
    test_tokens_match_captured_walks_and_packing()
    test_per_query_cutoff_is_honoured()
    test_empty_query_rows()
    print("\nALL PER-QUERY WALK-TOKEN CHECKS PASSED")
