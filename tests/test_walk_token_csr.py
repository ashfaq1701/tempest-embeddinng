"""Correctness tests for the walk-neighbourhood CSR (tempest_walks/walk_token_csr.py).

Two layers:

1. `dedup_to_csr` — the pure grouping function — is checked on hand-built flat token
   sets with KNOWN repeats / padding / cold rows, so an off-by-one in the run-grouping,
   a dropped occurrence age, or a wrong count is caught exactly.

2. `walk_csr` END-TO-END against a live Tempest backward-walk batch: we capture the exact
   WalkData the routine sampled (walks are stochastic, so we can't re-roll them), then
   INDEPENDENTLY reconstruct the expected CSR from that same WalkData under the documented
   contract — context = positions [0, lens-2] (seed slot lens-1 and padding excluded), each
   distinct node carrying ALL its occurrence ages (age = clamp(t_query − t_edge, 0)) and its
   count — and assert the CSR matches node-set, per-node age-multiset, and count exactly.
"""
from collections import defaultdict

import numpy as np
import torch

from tempest_walks.walk_token_csr import dedup_to_csr, walk_csr
from tempest_walks.walks import WalkGenerator


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def _csr_to_dict(node_ids, node_mask, ages, age_mask):
    """CSR row -> {node_id: sorted([occurrence ages])} per seed row."""
    G, U = node_ids.shape
    kmax = ages.shape[-1]
    out = []
    for g in range(G):
        row = {}
        for u in range(U):
            if bool(node_mask[g, u]):
                nid = int(node_ids[g, u])
                a = sorted(float(ages[g, u, k]) for k in range(kmax) if bool(age_mask[g, u, k]))
                row[nid] = a
        out.append(row)
    return out


def _synthetic_graph(n_nodes=10, n_edges=120, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, n_edges).astype(np.int64)
    tgt = rng.integers(0, n_nodes, n_edges).astype(np.int64)
    same = src == tgt
    tgt[same] = (tgt[same] + 1) % n_nodes
    ts = np.sort(rng.choice(np.arange(1, 50_000), n_edges, replace=False)).astype(np.int64)
    return src, tgt, ts


# ──────────────────────────────────────────────────────────────────────────
# 1. dedup_to_csr — exact grouping on hand-built input
# ──────────────────────────────────────────────────────────────────────────
def test_dedup_to_csr_exact_grouping():
    # row0: ids [7,3,7,7,3,-1], ages [10,20,30,40,50,0], mask [1,1,1,1,1,0]
    #   -> node 7: count 3, ages {10,30,40} ; node 3: count 2, ages {20,50}
    # row1: ids [5,5,9,-1,-1,-1], ages [1,2,3,0,0,0], mask [1,1,1,0,0,0]
    #   -> node 5: count 2, ages {1,2} ; node 9: count 1, ages {3}
    flat_ids = torch.tensor([[7, 3, 7, 7, 3, -1], [5, 5, 9, -1, -1, -1]])
    flat_age = torch.tensor([[10., 20, 30, 40, 50, 0], [1., 2, 3, 0, 0, 0]])
    flat_msk = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)

    nid, nm, ag, am = dedup_to_csr(flat_ids, flat_age, flat_msk)
    got = _csr_to_dict(nid, nm, ag, am)

    assert got[0] == {7: [10., 30., 40.], 3: [20., 50.]}, got[0]
    assert got[1] == {5: [1., 2.], 9: [3.]}, got[1]
    # count == age_mask.sum, distinct node count == node_mask.sum
    assert int(am[0].sum()) == 5 and int(nm[0].sum()) == 2
    assert int(am[1].sum()) == 3 and int(nm[1].sum()) == 2
    print("\n[dedup] exact grouping (ids/ages/counts) OK")


def test_dedup_to_csr_cold_and_padding_rows():
    # row0 fully masked (cold: no valid tokens) ; row1 single node repeated.
    flat_ids = torch.tensor([[-1, -1, -1], [4, 4, 4]])
    flat_age = torch.tensor([[0., 0, 0], [7., 8, 9]])
    flat_msk = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.bool)
    nid, nm, ag, am = dedup_to_csr(flat_ids, flat_age, flat_msk)
    got = _csr_to_dict(nid, nm, ag, am)
    assert got[0] == {}, "cold row must yield no valid nodes"
    assert not bool(nm[0].any()), "cold row node_mask must be all-False"
    assert got[1] == {4: [7., 8., 9.]}, got[1]
    print("[dedup] cold row -> empty; single repeated node OK")


# ──────────────────────────────────────────────────────────────────────────
# 2. walk_csr END-TO-END vs an independent reconstruction of the SAME walks
# ──────────────────────────────────────────────────────────────────────────
def _expected_csr_from_walkdata(wd, seeds, t_query_per_seed):
    """Reconstruct {node: sorted ages} per seed from a captured WalkData, under the
    contract: context = positions [0, lens-2]; age = clamp(t_query − t_edge, 0); and the
    walk's OWN ORIGIN node-id is DROPPED (not just its seed slot)."""
    G = int(t_query_per_seed.shape[0])
    K, L = wd.K, wd.nodes.shape[1]
    nodes = wd.nodes.view(G, K, L).cpu().numpy()
    ts = wd.timestamps.view(G, K, L).to(torch.int64).cpu().numpy()
    lens = wd.lens.view(G, K).cpu().numpy()
    tq = t_query_per_seed.cpu().numpy().astype(np.int64)
    sd = seeds.cpu().numpy().astype(np.int64)
    ref = []
    for g in range(G):
        row = defaultdict(list)
        for k in range(K):
            for p in range(0, int(lens[g, k]) - 1):       # context only (excl. seed slot + pad)
                node = int(nodes[g, k, p])
                if node == int(sd[g]):                     # drop the walk's own origin
                    continue
                age = float(max(0, int(tq[g]) - int(ts[g, k, p])))
                row[node].append(age)
        ref.append({n: sorted(a) for n, a in row.items()})
    return ref


def test_walk_csr_matches_tempest_backward_walks():
    src, tgt, ts = _synthetic_graph()
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=6, max_walk_len=8,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset()
    wg.add_edges(src, tgt, ts, None)

    seeds = torch.tensor([1, 3, 5, 7, 9], dtype=torch.int64)
    # distinct, large per-seed query times so all ages are strictly positive.
    tq = torch.tensor([int(ts.max()) + 100 + i for i in range(len(seeds))], dtype=torch.int64)

    # Capture the EXACT WalkData walk_csr samples (walks are stochastic).
    captured = {}
    orig = wg.walks_for_nodes

    def _capturing(*a, **k):
        wd = orig(*a, **k)
        captured["wd"] = wd
        return wd

    wg.walks_for_nodes = _capturing
    csr = walk_csr(wg, torch.device("cpu"), seeds, tq,
                   max_walk_len=8, num_walks_per_node=6,
                   walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.walks_for_nodes = orig
    node_ids, node_mask, ages, age_mask = csr

    got = _csr_to_dict(node_ids, node_mask, ages, age_mask)
    exp = _expected_csr_from_walkdata(captured["wd"], seeds, tq)

    # sanity: at least some seed actually has walk-neighbours (graph dense enough)
    assert any(len(r) > 0 for r in exp), "no context tokens sampled — graph too sparse"
    # the fix: each seed's OWN origin node-id must NOT appear in its CSR.
    for g, s in enumerate(seeds.tolist()):
        assert s not in got[g], f"seed {s} (its own origin) leaked into its CSR: {sorted(got[g])}"

    for g in range(len(seeds)):
        assert set(got[g].keys()) == set(exp[g].keys()), \
            f"seed {int(seeds[g])}: node set mismatch\n got {sorted(got[g])}\n exp {sorted(exp[g])}"
        for nid in exp[g]:
            ga, ea = got[g][nid], exp[g][nid]
            assert len(ga) == len(ea), \
                f"seed {int(seeds[g])} node {nid}: count {len(ga)} != {len(ea)}"
            assert np.allclose(ga, ea, atol=1e-3), \
                f"seed {int(seeds[g])} node {nid}: ages {ga} != {ea}"
    n_nodes = sum(len(r) for r in exp)
    n_occ = sum(len(a) for r in exp for a in r.values())
    print(f"\n[walk_csr] {len(seeds)} seeds, {n_nodes} distinct nodes, {n_occ} occurrences — "
          f"node-set, per-node ages, and counts all match the raw walks")


def test_walk_csr_includes_exactly_the_context_positions():
    """The SEED SLOT (position lens-1, the INT64_MAX sentinel) and padding (>= lens) are
    excluded; every context position [0, lens-2] is included exactly once. Verified by an
    exact occurrence count: Σ_w age_mask == Σ_k (lens[g,k]-1) per seed.

    NOTE: the seed's NODE id CAN legitimately recur in its own CSR — a backward walk may
    revisit the seed node as a neighbour at a context position. Only the seed SLOT is
    excluded, not the seed's node-id wherever it appears. (Such a revisit contributes
    Log_{E[u]}(E[u]) = 0 to μ, so it is harmless.)"""
    src, tgt, ts = _synthetic_graph(seed=1)
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=6, max_walk_len=8)
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([2, 4, 6], dtype=torch.int64)
    tq = torch.tensor([int(ts.max()) + 100] * len(seeds), dtype=torch.int64)

    captured = {}
    orig = wg.walks_for_nodes
    wg.walks_for_nodes = lambda *a, **k: captured.setdefault("wd", orig(*a, **k))
    _, _, _, age_mask = walk_csr(
        wg, torch.device("cpu"), seeds, tq, max_walk_len=8, num_walks_per_node=6)
    wg.walks_for_nodes = orig

    wd = captured["wd"]
    G, K, L = len(seeds), wd.K, wd.nodes.shape[1]
    nodes = wd.nodes.view(G, K, L).cpu().numpy()
    lens = wd.lens.view(G, K).cpu().numpy()
    sd = seeds.cpu().numpy()
    for g in range(G):
        # context positions [0, lens-2] whose node-id is NOT the origin (the fix)
        n_ctx = int(sum(1 for k in range(K) for p in range(int(lens[g, k]) - 1)
                        if int(nodes[g, k, p]) != int(sd[g])))
        n_csr = int(age_mask[g].sum())
        assert n_csr == n_ctx, \
            f"seed {int(seeds[g])}: CSR has {n_csr} occurrences, expected {n_ctx} (non-origin context)"
    print("\n[walk_csr] CSR occurrences == non-origin context positions (slot+pad+origin excluded) OK")


if __name__ == "__main__":
    test_dedup_to_csr_exact_grouping()
    test_dedup_to_csr_cold_and_padding_rows()
    test_walk_csr_matches_tempest_backward_walks()
    test_walk_csr_includes_exactly_the_context_positions()
    print("\nALL WALK-CSR CHECKS PASSED")
