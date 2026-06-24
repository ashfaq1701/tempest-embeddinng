"""Correctness tests for the per-query walk CSR (tempest_walks/walk_token_csr.py).

`build_query_walk_tokens` runs K backward walks per QUERY — a (seed node, cutoff time) pair —
and returns a WalkTokenCSR: a token-stream CSR (raw, count-free, drives μ) and a neighbour CSR
(per-query unique nodes + counts). Checks:

1. TOKEN STREAM vs the SAME captured walks — per-query (node, t_edge) MULTISET (read via
   node_ids_ptr slices) matches the contract reconstruction; node_ids_ptr is a valid CSR
   pointer; every token's t_edge < its query's cutoff.
2. NEIGHBOUR CSR — per query, `neighbors` are exactly the distinct token nodes, `neighbors_count`
   their occurrence counts, Σ counts == that query's token length; neighbors_ptr is valid.
3. EMPTY — a cutoff at/below the earliest edge and an isolated node give empty CSR segments.
4. μ-EQUIVALENCE — the head's segmented `_mu_from_token_csr` over the CSR equals the proven
   dense `_mu_from_csr` on the same tokens (so switching layout cannot move results).
"""
from collections import Counter

import numpy as np
import torch

from tempest_walks.link_pred_head import GeometricPointHead
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


def _query_token_multiset(csr, q):
    s, e = int(csr.node_ids_ptr[q]), int(csr.node_ids_ptr[q + 1])
    return Counter((int(csr.node_ids[i]), int(csr.pos_ts[i])) for i in range(s, e))


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
    csr = build_query_walk_tokens(wg, torch.device("cpu"), seeds, cutoffs, **params)
    wg.walks_for_nodes = orig
    return csr, captured["wd"]


def _is_valid_ptr(ptr, total, q):
    return (int(ptr.shape[0]) == q + 1 and int(ptr[0]) == 0 and int(ptr[-1]) == total
            and bool((ptr[1:] >= ptr[:-1]).all()))


# ──────────────────────────────────────────────────────────────────────────
# 1. token stream
# ──────────────────────────────────────────────────────────────────────────
def test_token_stream_matches_captured_walks():
    src, tgt, ts = _synthetic_graph()
    k, mwl = 6, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=k, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)
    cutoffs = torch.full((5,), 50_000, dtype=torch.long)

    csr, wd = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=mwl, num_walks_per_node=k,
        walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    exp = _expected_multiset_from_walkdata(wd, seeds)

    assert _is_valid_ptr(csr.node_ids_ptr, int(csr.node_ids.shape[0]), len(seeds))
    assert any(len(c) > 0 for c in exp), "graph too sparse"
    for q in range(len(seeds)):
        assert _query_token_multiset(csr, q) == exp[q], f"query {q}: token multiset mismatch"
    assert any(any(v > 1 for v in c.values()) for c in exp), "expected a multiplicity case"
    print(f"\n[token-stream] per-query multiset + multiplicity OK ({len(seeds)} queries, "
          f"T={int(csr.node_ids.shape[0])})")


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

    csr, _ = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=mwl, num_walks_per_node=k,
        walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    for q in range(3):
        s, e = int(csr.node_ids_ptr[q]), int(csr.node_ids_ptr[q + 1])
        assert all(int(t) < cuts[q] for t in csr.pos_ts[s:e]), f"query {q}: token at/after cutoff"
    print("\n[cutoff] every token strictly before its query's cutoff OK")


# ──────────────────────────────────────────────────────────────────────────
# 2. neighbour CSR
# ──────────────────────────────────────────────────────────────────────────
def test_neighbour_csr_matches_token_stream():
    src, tgt, ts = _synthetic_graph(seed=1)
    k, mwl = 6, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=k, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([2, 4, 6, 8], dtype=torch.long)
    cutoffs = torch.full((4,), 50_000, dtype=torch.long)

    csr, _ = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=mwl, num_walks_per_node=k,
        walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    assert _is_valid_ptr(csr.neighbors_ptr, int(csr.neighbors.shape[0]), len(seeds))

    saw_dedup = False
    for q in range(len(seeds)):
        ts_s, ts_e = int(csr.node_ids_ptr[q]), int(csr.node_ids_ptr[q + 1])
        token_counts = Counter(int(n) for n in csr.node_ids[ts_s:ts_e])     # node -> occurrences
        ns, ne = int(csr.neighbors_ptr[q]), int(csr.neighbors_ptr[q + 1])
        nbr = {int(csr.neighbors[i]): int(csr.neighbors_count[i]) for i in range(ns, ne)}
        assert nbr == dict(token_counts), f"query {q}: neighbour CSR != token-stream dedup"
        # Σ counts == token length; uniques ≤ tokens
        assert sum(nbr.values()) == (ts_e - ts_s)
        if (ne - ns) < (ts_e - ts_s):
            saw_dedup = True
    assert saw_dedup, "expected at least one query whose tokens dedup to fewer neighbours"
    print(f"\n[neighbour-csr] unique nodes + counts match token stream OK "
          f"(Tn={int(csr.neighbors.shape[0])})")


# ──────────────────────────────────────────────────────────────────────────
# 3. empty segments
# ──────────────────────────────────────────────────────────────────────────
def test_empty_query_segments():
    src, tgt, ts = _synthetic_graph(n_nodes=6, n_edges=40, seed=2)
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=4, max_walk_len=6)
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    t_min = int(ts.min())
    seeds = torch.tensor([int(src[0]), 999], dtype=torch.long)
    cutoffs = torch.tensor([t_min, 50_000], dtype=torch.long)

    csr, _ = _run_with_capture(wg, seeds, cutoffs, max_walk_len=6, num_walks_per_node=4)
    for q in (0, 1):
        assert int(csr.node_ids_ptr[q]) == int(csr.node_ids_ptr[q + 1]), f"query {q} not empty"
        assert int(csr.neighbors_ptr[q]) == int(csr.neighbors_ptr[q + 1])
    print("\n[empty] cutoff-excluded + isolated queries → empty CSR segments OK")


# ──────────────────────────────────────────────────────────────────────────
# 4. μ-equivalence: segmented CSR μ == proven dense μ
# ──────────────────────────────────────────────────────────────────────────
def _densify(csr, t_query):
    q = int(csr.seeds.shape[0])
    counts = (csr.node_ids_ptr[1:] - csr.node_ids_ptr[:-1])
    u = max(int(counts.max()) if q else 1, 1)
    ids = torch.zeros((q, u), dtype=torch.long)
    nmask = torch.zeros((q, u), dtype=torch.bool)
    ages = torch.zeros((q, u), dtype=torch.float32)
    for qi in range(q):
        s, e = int(csr.node_ids_ptr[qi]), int(csr.node_ids_ptr[qi + 1])
        n = e - s
        if n:
            ids[qi, :n] = csr.node_ids[s:e]
            nmask[qi, :n] = True
            ages[qi, :n] = (t_query[qi] - csr.pos_ts[s:e]).clamp_min(0).float()
    return ids, nmask, ages


def test_mu_csr_equals_dense():
    torch.manual_seed(0)
    src, tgt, ts = _synthetic_graph(seed=3)
    k, mwl = 6, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=k, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    n_nodes = int(max(src.max(), tgt.max())) + 1
    seeds = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)
    t_query = torch.full((5,), 45_000, dtype=torch.long)
    cutoffs = t_query.clone()

    csr = build_query_walk_tokens(wg, torch.device("cpu"), seeds, cutoffs,
                                  max_walk_len=mwl, num_walks_per_node=k,
                                  walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    ids, nmask, ages = _densify(csr, t_query)

    d = 16
    head = GeometricPointHead(d_emb=d, t_train=50_000.0)
    e_weight = torch.randn(n_nodes, d)
    E_u = e_weight[seeds]                                          # base = E[seed]

    mu_dense = head._mu_from_csr(E_u, ids, nmask, ages, e_weight)  # [Q, d]
    mu_csr = head._mu_from_token_csr(
        E_u, csr.node_ids, csr.pos_ts, csr.node_ids_ptr, t_query, e_weight)
    assert torch.allclose(mu_dense, mu_csr, atol=1e-5, rtol=1e-4), \
        f"segmented CSR μ diverges from dense μ (max |Δ| = {(mu_dense - mu_csr).abs().max():.2e})"
    print(f"\n[μ-equiv] segmented CSR μ == dense μ (max |Δ| = "
          f"{(mu_dense - mu_csr).abs().max():.2e})")


if __name__ == "__main__":
    test_token_stream_matches_captured_walks()
    test_per_query_cutoff_is_honoured()
    test_neighbour_csr_matches_token_stream()
    test_empty_query_segments()
    test_mu_csr_equals_dense()
    print("\nALL WALK-CSR CHECKS PASSED")
