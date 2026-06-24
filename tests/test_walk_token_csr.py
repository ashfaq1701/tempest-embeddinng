"""Correctness tests for the per-query dense walk tokens (tempest_walks/walk_token_csr.py).

`build_query_walk_tokens` runs K backward walks per QUERY — a (seed node, cutoff time) pair —
and returns a dense `WalkTokens`: a token bag [Q, U] (raw, count-free, drives μ via a per-row
softmax) and a neighbour bag [Q, Un] (per-query unique nodes + counts). Checks:

1. TOKEN BAG vs the SAME captured walks — per-query (node, t_edge) MULTISET (read via the mask)
   matches the contract reconstruction; padding slots are -1/False; every token's t_edge < its
   query's cutoff; cutoffs field round-trips.
2. NEIGHBOUR BAG — per query, masked `neighbors` are exactly the distinct token nodes,
   `neighbors_count` their occurrence counts, Σ counts == that query's token count.
3. EMPTY — a cutoff at/below the earliest edge and an isolated node give all-False rows.
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


def _query_token_multiset(wt, q):
    m = wt.node_mask[q]
    return Counter((int(n), int(t)) for n, t in zip(wt.node_ids[q][m], wt.pos_ts[q][m]))


def _expected_multiset_from_walkdata(wd, seeds):
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
    captured = {}
    orig = wg.walks_for_nodes

    def _cap(*a, **k):
        wd = orig(*a, **k)
        captured["wd"] = wd
        return wd

    wg.walks_for_nodes = _cap
    wt = build_query_walk_tokens(wg, torch.device("cpu"), seeds, cutoffs, **params)
    wg.walks_for_nodes = orig
    return wt, captured["wd"]


# ──────────────────────────────────────────────────────────────────────────
# 1. token bag
# ──────────────────────────────────────────────────────────────────────────
def test_token_bag_matches_captured_walks():
    src, tgt, ts = _synthetic_graph()
    k, mwl = 6, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=k, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)
    cutoffs = torch.full((5,), 50_000, dtype=torch.long)

    wt, wd = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=mwl, num_walks_per_node=k,
        walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    exp = _expected_multiset_from_walkdata(wd, seeds)

    assert torch.equal(wt.cutoffs, cutoffs), "cutoffs field must round-trip"
    assert bool((wt.node_ids[~wt.node_mask] == -1).all()), "padding slots must be -1"
    assert any(len(c) > 0 for c in exp), "graph too sparse"
    for q in range(len(seeds)):
        assert _query_token_multiset(wt, q) == exp[q], f"query {q}: token multiset mismatch"
    assert any(any(v > 1 for v in c.values()) for c in exp), "expected a multiplicity case"
    print(f"\n[token-bag] per-query multiset + multiplicity + packing OK "
          f"({len(seeds)} queries, U={wt.node_ids.shape[1]})")


def test_per_query_cutoff_is_honoured():
    src, tgt, ts = _synthetic_graph(seed=4)
    k, mwl = 8, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=k, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    uniq = np.unique(ts)
    node = int(src[len(src) // 2])
    cuts = [int(uniq[len(uniq) // 4]), int(uniq[len(uniq) // 2]), 50_000]
    seeds = torch.tensor([node, node, node], dtype=torch.long)
    cutoffs = torch.tensor(cuts, dtype=torch.long)

    wt, _ = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=mwl, num_walks_per_node=k,
        walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    for q in range(3):
        real = wt.pos_ts[q][wt.node_mask[q]].tolist()
        assert all(int(t) < cuts[q] for t in real), f"query {q}: token at/after cutoff"
    print("\n[cutoff] every token strictly before its query's cutoff OK")


# ──────────────────────────────────────────────────────────────────────────
# 2. neighbour bag
# ──────────────────────────────────────────────────────────────────────────
def test_neighbour_bag_matches_token_bag():
    src, tgt, ts = _synthetic_graph(seed=1)
    k, mwl = 6, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=k, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([2, 4, 6, 8], dtype=torch.long)
    cutoffs = torch.full((4,), 50_000, dtype=torch.long)

    wt, _ = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=mwl, num_walks_per_node=k,
        walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    assert bool((wt.neighbors_count[~wt.neighbors_mask] == 0).all()), "padding counts must be 0"

    saw_dedup = False
    for q in range(len(seeds)):
        tm = wt.node_mask[q]
        token_counts = Counter(int(n) for n in wt.node_ids[q][tm])     # node -> occurrences
        nm = wt.neighbors_mask[q]
        nbr = {int(n): int(c) for n, c in zip(wt.neighbors[q][nm], wt.neighbors_count[q][nm])}
        assert nbr == dict(token_counts), f"query {q}: neighbour bag != token-bag dedup"
        assert sum(nbr.values()) == int(tm.sum())
        if int(nm.sum()) < int(tm.sum()):
            saw_dedup = True
    assert saw_dedup, "expected at least one query whose tokens dedup to fewer neighbours"
    print(f"\n[neighbour-bag] unique nodes + counts match token bag OK "
          f"(Un={wt.neighbors.shape[1]})")


# ──────────────────────────────────────────────────────────────────────────
# 3. empty rows
# ──────────────────────────────────────────────────────────────────────────
def test_empty_query_rows():
    src, tgt, ts = _synthetic_graph(n_nodes=6, n_edges=40, seed=2)
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=4, max_walk_len=6)
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    t_min = int(ts.min())
    seeds = torch.tensor([int(src[0]), 999], dtype=torch.long)
    cutoffs = torch.tensor([t_min, 50_000], dtype=torch.long)

    wt, _ = _run_with_capture(wg, seeds, cutoffs, max_walk_len=6, num_walks_per_node=4)
    for q in (0, 1):
        assert not bool(wt.node_mask[q].any()), f"query {q}: token row not empty"
        assert not bool(wt.neighbors_mask[q].any()), f"query {q}: neighbour row not empty"
    print("\n[empty] cutoff-excluded + isolated queries → all-False rows OK")


if __name__ == "__main__":
    test_token_bag_matches_captured_walks()
    test_per_query_cutoff_is_honoured()
    test_neighbour_bag_matches_token_bag()
    test_empty_query_rows()
    print("\nALL DENSE WALK-TOKEN CHECKS PASSED")
