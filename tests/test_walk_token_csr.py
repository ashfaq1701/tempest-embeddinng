"""Correctness tests for the two-level packed walk CSR (tempest_walks/walk_token_csr.py).

The new layout is COUNT-FREE: every reached walk position is its own token (no dedup), packed
into a two-level CSR (seed → walk → position). Three things are checked:

1. `build_walk_batch` STRUCTURE — offset invariants (walk_csr / seed_csr monotone, lengths,
   uniform seed stride K), the packed arrays' length == P, and that the seed IS kept — it
   appears K times at its slot (seed id @ the INT64_MAX sentinel time).

2. `build_walk_batch` CONTENT, END-TO-END vs a live Tempest backward-walk batch — we capture
   the exact WalkData sampled (walks are stochastic) and INDEPENDENTLY reconstruct the expected
   per-seed token MULTISET under the contract (EVERY non-padding position [0, lens-1] kept WITH
   multiplicity — INCLUDING the seed slot and revisits; only padding p ≥ lens dropped), then
   assert the WalkBatch matches it exactly — node-with-ts multiset and total count. Multiplicity
   (no dedup) is verified here: a node reached k times must contribute k tokens.

3. `walk_batch_to_dense` + `gather_dense` — the per-seed dense token bag reproduces the same
   per-seed multiset; cold seeds give an all-False row; gather replicates rows onto a grid.
"""
from collections import Counter

import numpy as np
import torch

from tempest_walks.walk_token_csr import (build_walk_batch, gather_dense,
                                          walk_batch_to_dense)
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


def _seed_multiset_from_batch(wb):
    """Per-seed Counter of (node_id, t_edge) tokens, read via seed_csr ∘ walk_csr."""
    G = int(wb.seeds.shape[0])
    nodes = wb.walk_nodes.cpu().numpy()
    ts = wb.walk_pos_ts.cpu().numpy()
    wcsr = wb.walk_csr.cpu().numpy()
    scsr = wb.seed_csr.cpu().numpy()
    out = []
    for g in range(G):
        start = int(wcsr[int(scsr[g])])
        end = int(wcsr[int(scsr[g + 1])])
        out.append(Counter((int(nodes[i]), int(ts[i])) for i in range(start, end)))
    return out


def _seed_multiset_from_dense(node_ids, node_mask, pos_ts):
    G, U = node_ids.shape
    out = []
    for g in range(G):
        out.append(Counter((int(node_ids[g, u]), int(pos_ts[g, u]))
                           for u in range(U) if bool(node_mask[g, u])))
    return out


def _expected_multiset_from_walkdata(wd, seeds):
    """Reconstruct per-seed (node, t_edge) MULTISET from a captured WalkData: EVERY non-padding
    position [0, lens-1] is kept with multiplicity — INCLUDING the seed slot (seed at the
    INT64_MAX sentinel) and any seed revisits. Only padding (p ≥ lens) is dropped."""
    G = int(seeds.shape[0])
    K, L = wd.K, wd.nodes.shape[1]
    nodes = wd.nodes.view(G, K, L).cpu().numpy()
    ts = wd.timestamps.view(G, K, L).to(torch.int64).cpu().numpy()
    lens = wd.lens.view(G, K).cpu().numpy()
    out = []
    for g in range(G):
        c = Counter()
        for k in range(K):
            for p in range(0, int(lens[g, k])):           # all real positions (incl. seed slot)
                c[(int(nodes[g, k, p]), int(ts[g, k, p]))] += 1
        out.append(c)
    return out


def _walk_with_capture(wg, seeds, **params):
    """Run build_walk_batch while capturing the exact WalkData it sampled."""
    captured = {}
    orig = wg.walks_for_nodes

    def _capturing(*a, **k):
        wd = orig(*a, **k)
        captured["wd"] = wd
        return wd

    wg.walks_for_nodes = _capturing
    wb = build_walk_batch(wg, torch.device("cpu"), seeds, **params)
    wg.walks_for_nodes = orig
    return wb, captured["wd"]


# ──────────────────────────────────────────────────────────────────────────
# 1. structural invariants
# ──────────────────────────────────────────────────────────────────────────
def test_build_walk_batch_offset_invariants():
    src, tgt, ts = _synthetic_graph()
    K, mwl = 6, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=K, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([1, 3, 5, 7, 9], dtype=torch.int64)
    G = len(seeds)

    wb, _ = _walk_with_capture(wg, seeds, max_walk_len=mwl, num_walks_per_node=K,
                               walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    P = int(wb.walk_nodes.shape[0])

    # seed_csr: uniform stride K, length G+1, ends at W = G*K.
    assert wb.seed_csr.shape[0] == G + 1
    assert torch.equal(wb.seed_csr, torch.arange(G + 1, dtype=torch.int64) * K)
    # walk_csr: length W+1, monotone non-decreasing, starts at 0, ends at P.
    assert wb.walk_csr.shape[0] == G * K + 1
    assert int(wb.walk_csr[0]) == 0 and int(wb.walk_csr[-1]) == P
    assert bool((wb.walk_csr[1:] >= wb.walk_csr[:-1]).all()), "walk_csr must be monotone"
    # packed arrays agree on P; no edge features on this graph.
    assert wb.walk_pos_ts.shape[0] == P
    assert wb.walk_edge_feats is None
    # the seed IS now kept: every walk contributes a seed-slot token (seed id @ INT64_MAX), so
    # the seed appears K times per seed with the sentinel time.
    SENT = torch.iinfo(torch.int64).max
    per_seed = _seed_multiset_from_batch(wb)
    for g, s in enumerate(seeds.tolist()):
        assert per_seed[g][(s, SENT)] == K, \
            f"seed {s} should appear K={K} times at its slot (sentinel time); got {per_seed[g][(s, SENT)]}"
    print("\n[build] offset invariants + seed-inclusion OK "
          f"(G={G}, K={K}, P={P})")


# ──────────────────────────────────────────────────────────────────────────
# 2. content end-to-end vs the SAME captured walks (multiplicity preserved)
# ──────────────────────────────────────────────────────────────────────────
def test_build_walk_batch_matches_tempest_backward_walks():
    src, tgt, ts = _synthetic_graph()
    K, mwl = 6, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=K, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([1, 3, 5, 7, 9], dtype=torch.int64)

    wb, wd = _walk_with_capture(wg, seeds, max_walk_len=mwl, num_walks_per_node=K,
                                walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    got = _seed_multiset_from_batch(wb)
    exp = _expected_multiset_from_walkdata(wd, seeds)

    assert any(len(c) > 0 for c in exp), "no context tokens sampled — graph too sparse"
    for g in range(len(seeds)):
        assert got[g] == exp[g], (
            f"seed {int(seeds[g])}: token multiset mismatch\n"
            f" got {sorted(got[g].items())}\n exp {sorted(exp[g].items())}")
    # multiplicity is real: at least one seed has a node reached more than once.
    assert any(any(v > 1 for v in c.values()) for c in exp), \
        "expected some recurring node to exercise the no-dedup multiplicity path"
    total = sum(sum(c.values()) for c in exp)
    assert int(wb.walk_nodes.shape[0]) == total, "P must equal total non-origin context tokens"
    print(f"\n[build] {len(seeds)} seeds, P={total} tokens — node+ts multiset and "
          "multiplicity match the raw walks exactly")


# ──────────────────────────────────────────────────────────────────────────
# 3. dense view + gather
# ──────────────────────────────────────────────────────────────────────────
def test_walk_batch_to_dense_matches_packed():
    src, tgt, ts = _synthetic_graph(seed=1)
    K, mwl = 6, 8
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=K, max_walk_len=mwl)
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([2, 4, 6], dtype=torch.int64)

    wb, _ = _walk_with_capture(wg, seeds, max_walk_len=mwl, num_walks_per_node=K)
    packed = _seed_multiset_from_batch(wb)
    node_ids, node_mask, pos_ts = walk_batch_to_dense(wb)

    # shapes + U = max tokens per seed.
    G = len(seeds)
    seed_start = wb.walk_csr[wb.seed_csr[:-1]]
    seed_end = wb.walk_csr[wb.seed_csr[1:]]
    U_exp = max(int((seed_end - seed_start).max()), 1)
    assert node_ids.shape == (G, U_exp) and pos_ts.shape == (G, U_exp)
    # padded slots are masked, masked rows hold node id -1.
    assert bool((node_ids[~node_mask] == -1).all())

    dense = _seed_multiset_from_dense(node_ids, node_mask, pos_ts)
    for g in range(G):
        assert dense[g] == packed[g], f"seed {int(seeds[g])}: dense multiset != packed"
    print("\n[dense] per-seed dense token bag reproduces the packed multiset OK")


def test_walk_batch_to_dense_cold_seed_is_empty_row():
    # A seed with no ingested incident edges has no walk-neighbours → all-False dense row.
    src, tgt, ts = _synthetic_graph(n_nodes=6, n_edges=40, seed=2)
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=4, max_walk_len=6)
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    cold = 999                      # never appears as an edge endpoint
    seeds = torch.tensor([0, cold], dtype=torch.int64)
    wb, _ = _walk_with_capture(wg, seeds, max_walk_len=6, num_walks_per_node=4)
    node_ids, node_mask, _pos = walk_batch_to_dense(wb)
    assert not bool(node_mask[1].any()), "cold seed must yield an all-False dense row"
    print("\n[dense] cold seed -> empty (all-False) row OK")


def test_gather_dense_replicates_rows_onto_grid():
    src, tgt, ts = _synthetic_graph(seed=3)
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=5, max_walk_len=6)
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    seeds = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    wb, _ = _walk_with_capture(wg, seeds, max_walk_len=6, num_walks_per_node=5)
    dense = walk_batch_to_dense(wb)

    # [B] gather: index 0..G-1 plus a repeat -> the repeated cell equals the source row.
    index = torch.tensor([0, 2, 2, 3], dtype=torch.long)
    ids, mask, pts = gather_dense(dense, index, (4,))
    assert torch.equal(ids[1], ids[2]) and torch.equal(mask[1], mask[2])
    assert torch.equal(ids[0], dense[0][0]) and torch.equal(pts[3], dense[2][3])

    # [B,C] scatter: a flat index reshaped to a grid replicates per-seed rows.
    grid_index = torch.tensor([0, 1, 1, 0, 2, 3], dtype=torch.long)
    gi, gm, _gp = gather_dense(dense, grid_index, (2, 3))
    assert gi.shape[:2] == (2, 3)
    assert torch.equal(gi[0, 1], gi[0, 2])           # both name seed 1
    assert torch.equal(gm[0, 0], gm[1, 0])           # rows (0,0) and (1,0) both name seed 0
    print("\n[gather] dense rows replicate correctly onto [B] and [B,C] grids OK")


if __name__ == "__main__":
    test_build_walk_batch_offset_invariants()
    test_build_walk_batch_matches_tempest_backward_walks()
    test_walk_batch_to_dense_matches_packed()
    test_walk_batch_to_dense_cold_seed_is_empty_row()
    test_gather_dense_replicates_rows_onto_grid()
    print("\nALL WALK-CSR CHECKS PASSED")
