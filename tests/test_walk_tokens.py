"""Correctness tests for the raw per-query walk tensors (link_property_prediction/walk_tokens.py).

`build_query_walk_tokens` runs K backward walks per QUERY — a (seed node, cutoff time) pair —
and returns the RAW walks: nodes [Q, K, L], nodes_mask [Q, K, L], node-aligned timestamps
[Q, K, L] (seed = cutoff), cutoffs [Q]. Checks:

1. SHAPES + nodes match the captured raw walk output reshaped to [Q, K, L]; cutoffs round-trip;
   nodes_mask == (nodes != -1).
2. TIMESTAMPS — node-aligned, no INT64_MAX; the seed-slot sentinel is replaced by the query
   cutoff (== Tempest's timestamps with sentinel→cutoff); timestamps != -1 exactly where
   nodes_mask is True (so the mask is shared); every non-seed time < cutoff, every seed time == cutoff.
3. SEED — the last real node of every walk is that query's seed AND carries time == cutoff.
4. EMPTY — a cutoff at/below the earliest edge / an isolated node give fully empty walks (no
   traversable edge ⇒ all-False mask, all-padding nodes/timestamps).
"""
import numpy as np
import torch

from link_property_prediction.walk_tokens import _TS_SENTINEL, build_query_walk_tokens
from link_property_prediction.walks import WalkGenerator


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


def _build(seeds, cutoffs, k=6, mwl=8, gseed=0):
    src, tgt, ts = _synthetic_graph(seed=gseed)
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=k, max_walk_len=mwl,
                       walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    wg.add_edges(src, tgt, ts, None)
    wt, wd = _run_with_capture(
        wg, seeds, cutoffs, max_walk_len=mwl, num_walks_per_node=k,
        walk_bias="ExponentialWeight", start_bias="ExponentialWeight")
    return wt, wd


def _last_real_index(nodes_mask):
    """[Q, K] index of the last True position per walk (the seed slot)."""
    rev = nodes_mask.flip(-1).int().argmax(-1)
    return (nodes_mask.shape[-1] - 1) - rev


# ──────────────────────────────────────────────────────────────────────────
# 1. shapes + nodes + mask + cutoffs
# ──────────────────────────────────────────────────────────────────────────
def test_shapes_nodes_mask_cutoffs():
    k, mwl = 6, 8
    seeds = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)
    cutoffs = torch.full((5,), 50_000, dtype=torch.long)
    wt, wd = _build(seeds, cutoffs, k=k, mwl=mwl)
    q, length = len(seeds), int(wd.nodes.shape[1])

    for name, t in (("nodes", wt.nodes), ("nodes_mask", wt.nodes_mask), ("ages", wt.ages)):
        assert t.shape == (q, k, length), f"{name} shape {tuple(t.shape)}"
    assert wt.cutoffs.shape == (q,) and wt.seeds.shape == (q,)
    assert torch.equal(wt.cutoffs, cutoffs), "cutoffs must round-trip"
    assert torch.equal(wt.seeds, seeds), "seeds must round-trip"
    assert torch.equal(wt.nodes, wd.nodes.to(torch.int64).reshape(q, k, length)), \
        "nodes must equal the raw walk output reshaped to [Q, K, L]"
    assert torch.equal(wt.nodes_mask, wt.nodes != -1), "nodes_mask must be (nodes != -1)"
    print(f"\n[shapes] nodes/mask/ages {tuple(wt.nodes.shape)}, cutoffs {tuple(wt.cutoffs.shape)} OK")


# ──────────────────────────────────────────────────────────────────────────
# 2. node-aligned ages: seed=0, edges>=1, padding=-1, mask shared, causal
# ──────────────────────────────────────────────────────────────────────────
def test_ages_node_aligned_seed_is_zero():
    k, mwl = 8, 8
    seeds = torch.tensor([2, 4, 6, 8], dtype=torch.long)
    cutoffs = torch.tensor([12_000, 25_000, 40_000, 50_000], dtype=torch.long)
    wt, wd = _build(seeds, cutoffs, k=k, mwl=mwl, gseed=4)
    q, length = len(seeds), int(wd.nodes.shape[1])

    # ages == cutoff - (raw edge time, sentinel->cutoff), with padding -> -1.
    raw = wd.timestamps.to(torch.int64).reshape(q, k, length)
    edge_ts = torch.where(raw == _TS_SENTINEL, cutoffs.view(q, 1, 1), raw)
    expect = torch.where(wt.nodes_mask, cutoffs.view(q, 1, 1) - edge_ts, torch.full_like(raw, -1))
    assert torch.equal(wt.ages, expect), "ages must be cutoff - edge_time (seed 0), padding -1"

    # The mask is shared: a real age iff a real node (padding age == -1).
    assert torch.equal(wt.ages != -1, wt.nodes_mask), "ages != -1 must match nodes_mask"

    # Seed slot age == 0; every other real age strictly >= 1 (cutoff is exclusive).
    seed_idx = _last_real_index(wt.nodes_mask).unsqueeze(-1)               # [Q, K, 1]
    is_seed = torch.zeros_like(wt.nodes_mask)
    is_seed.scatter_(-1, seed_idx, wt.nodes_mask.any(-1, keepdim=True))    # only for non-empty walks
    assert bool((wt.ages[is_seed] == 0).all()), "seed age must == 0"
    non_seed = wt.nodes_mask & ~is_seed
    assert bool((wt.ages[non_seed] >= 1).all()), "non-seed ages must be >= 1"
    print("\n[ages] node-aligned, seed==0, edges>=1, mask shared OK")


# ──────────────────────────────────────────────────────────────────────────
# 3. seed = last real node, carrying cutoff
# ──────────────────────────────────────────────────────────────────────────
def test_seed_is_last_real_node():
    k, mwl = 6, 8
    seeds = torch.tensor([2, 4, 6, 8], dtype=torch.long)
    cutoffs = torch.tensor([20_000, 30_000, 40_000, 50_000], dtype=torch.long)
    wt, _ = _build(seeds, cutoffs, k=k, mwl=mwl, gseed=1)
    seed_idx = _last_real_index(wt.nodes_mask)                             # [Q, K]
    last_node = torch.gather(wt.nodes, -1, seed_idx.unsqueeze(-1)).squeeze(-1)        # [Q, K]
    last_age = torch.gather(wt.ages, -1, seed_idx.unsqueeze(-1)).squeeze(-1)          # [Q, K]
    assert torch.equal(last_node, seeds.view(-1, 1).expand_as(last_node)), \
        "the last real node of every walk must be that query's seed"
    assert bool((last_age == 0).all()), "the seed (last node) must carry age == 0"
    print("\n[seed] last real node == query seed, age == 0 OK")


# ──────────────────────────────────────────────────────────────────────────
# 4. empty walks (no traversable edge)
# ──────────────────────────────────────────────────────────────────────────
def test_empty_walks_when_no_predecessors():
    src, tgt, ts = _synthetic_graph(n_nodes=6, n_edges=40, seed=2)
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=4, max_walk_len=6)
    wg.add_edges(src, tgt, ts, None)
    t_min = int(ts.min())
    seeds = torch.tensor([int(src[0]), 999], dtype=torch.long)          # cutoff-excluded, isolated
    cutoffs = torch.tensor([t_min, 50_000], dtype=torch.long)

    wt, _ = _run_with_capture(wg, seeds, cutoffs, max_walk_len=6, num_walks_per_node=4)
    assert torch.equal(wt.seeds, seeds), "seeds must be kept even for cold/empty walks"
    for q in (0, 1):
        assert not bool(wt.nodes_mask[q].any()), f"query {q}: expected all-False mask"
        assert bool((wt.nodes[q] == -1).all()), f"query {q}: expected all-padding nodes"
        assert bool((wt.ages[q] == -1).all()), f"query {q}: expected all-padding ages"
    print("\n[empty] cutoff-excluded + isolated queries → fully empty walks (seeds kept) OK")


# ──────────────────────────────────────────────────────────────────────────
# 5. flatten_tokens — flat [Q, K*L] bag; padding always masked + seed-filter flags
# ──────────────────────────────────────────────────────────────────────────
def _synthetic_tokens():
    """One query, K=2 walks, L=4. Seed u=5, cutoff t=100. Ages = cutoff - t_edge (seed slot 0, pad -1).
    Walk1 nodes [7,5,3,5] ages [10,5,2,0]: pos1 = MID-WALK seed recurrence (age5),
                                           pos3 = seed SLOT (age0).
    Walk2 nodes [8,2,5,-1] ages [20,15,0,-1]: pos2 = seed SLOT (age0), pos3 = PADDING.
    Flat order is walk1's 4 positions then walk2's -> indices 0..7."""
    from link_property_prediction.walk_tokens import WalkTokens
    nodes = torch.tensor([[[7, 5, 3, 5], [8, 2, 5, -1]]], dtype=torch.long)
    ages = torch.tensor([[[10, 5, 2, 0], [20, 15, 0, -1]]], dtype=torch.long)
    return WalkTokens(seeds=torch.tensor([5], dtype=torch.long), nodes=nodes,
                      nodes_mask=(nodes != -1), ages=ages,
                      cutoffs=torch.tensor([100], dtype=torch.long))


def test_flatten_exclude_seed_positions_only():
    """exclude_seed_positions=True (default): only the walk-origin slot masked; mid-walk seed
    recurrences are KEPT (this is the whole slot-vs-token distinction)."""
    from link_property_prediction.walk_tokens import flatten_tokens
    wt = _synthetic_tokens()
    ids, mask, _ = flatten_tokens(wt, exclude_seed_positions=True)
    assert mask.int().tolist()[0] == [1, 1, 1, 0, 1, 1, 0, 0]  # slots pos3,6 (age 0) removed; pos1 KEPT
    seed_kept = (ids[0][mask[0]] == 5)
    assert bool(seed_kept.any()), "mid-walk seed recurrence must be KEPT"
    ages_flat = wt.ages.reshape(1, -1)                            # ages read from the instance now
    assert bool((ages_flat[0][mask[0]][seed_kept] > 0).all()), "surviving seed token is a recurrence (age>0)"
    print("\n[flatten] exclude_seed_positions keeps mid-walk recurrence, drops walk-origin slot OK")


def test_flatten_no_filtering_when_positions_false():
    """exclude_seed_positions=False: seed kept everywhere; only padding removed."""
    from link_property_prediction.walk_tokens import flatten_tokens
    wt = _synthetic_tokens()
    ids, mask, _ = flatten_tokens(wt, exclude_seed_positions=False)
    assert mask.int().tolist()[0] == [1, 1, 1, 1, 1, 1, 1, 0]  # only padding pos7 removed
    assert int(mask.sum()) == int(wt.nodes_mask.sum())
    print("\n[flatten] exclude_seed_positions False keeps the seed everywhere OK")


def test_flatten_shapes_and_padding_realdata():
    """On live Tempest walks: shapes correct, padding never leaks, default drops the seed's
    walk-origin slot (age 0) so every kept token has age > 0."""
    from link_property_prediction.walk_tokens import flatten_tokens
    k, mwl = 6, 8
    seeds = torch.tensor([2, 4, 6, 8], dtype=torch.long)
    cutoffs = torch.tensor([20_000, 30_000, 40_000, 50_000], dtype=torch.long)
    wt, _ = _build(seeds, cutoffs, k=k, mwl=mwl, gseed=1)
    q = len(seeds)
    ids, mask, pos = flatten_tokens(wt)                           # default: exclude seed's walk-origin slot
    assert ids.shape == (q, k * mwl) and mask.shape == (q, k * mwl) and pos.shape == (q, k * mwl)
    ages_flat = wt.ages.reshape(q, -1)                            # ages read from the instance now
    for i in range(q):
        kept = ids[i][mask[i]]
        assert bool((kept != -1).all()), f"padding leaked into kept tokens (q={i})"
        assert bool((ages_flat[i][mask[i]] > 0).all()), f"walk-origin slot (age 0) must be excluded (q={i})"
    print("\n[flatten] real-data: shapes OK, padding excluded, walk-origin slot excluded (default) OK")


# ──────────────────────────────────────────────────────────────────────────
# 6. edge features — [Q, K, L*d_ef], seed slot + padding zeroed
# ──────────────────────────────────────────────────────────────────────────
def test_edge_features_populated_and_seed_padding_zero():
    """build_query_walk_tokens carries per-position edge features [Q, K, L*d_ef]; the seed slot
    (age 0) and padding are forced to [0]*d_ef."""
    from link_property_prediction.walk_tokens import build_query_walk_tokens
    rng = np.random.default_rng(3)
    n_nodes, n_edges, d_ef = 8, 90, 4
    src = rng.integers(0, n_nodes, n_edges).astype(np.int64)
    tgt = rng.integers(0, n_nodes, n_edges).astype(np.int64)
    tgt[src == tgt] = (tgt[src == tgt] + 1) % n_nodes
    ts = np.sort(rng.choice(np.arange(1, 5000), n_edges, replace=False)).astype(np.int64)
    ef = rng.standard_normal((n_edges, d_ef)).astype(np.float32) + 5.0   # nonzero so a zero is meaningful
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=4, max_walk_len=6)
    wg.add_edges(src, tgt, ts, ef)
    seeds = torch.tensor([1, 3, 5], dtype=torch.long)
    cutoffs = torch.full((3,), 5000, dtype=torch.long)
    wt = build_query_walk_tokens(wg, torch.device("cpu"), seeds, cutoffs,
                                 max_walk_len=6, num_walks_per_node=4)
    q, k, L = wt.nodes.shape
    assert wt.edge_features is not None, "edge_features not populated"
    assert wt.edge_features.shape == (q, k, L * d_ef), \
        f"edge_features should be [Q, K, L*d_ef], got {tuple(wt.edge_features.shape)}"

    ef_pos = wt.edge_features.reshape(q, k, L, d_ef)              # unflatten to inspect per position
    assert bool((ef_pos[~wt.nodes_mask] == 0).all()), "padding edge features must be zero"
    seed_idx = _last_real_index(wt.nodes_mask)                    # [Q, K] last real slot = seed
    seed_ef = torch.gather(ef_pos, 2, seed_idx[..., None, None].expand(q, k, 1, d_ef)).squeeze(2)
    has_walk = wt.nodes_mask.any(-1)                              # only non-empty walks have a seed slot
    assert bool((seed_ef[has_walk] == 0).all()), "seed-slot edge features must be zero"
    print("\n[edge_features] shape [Q,K,L*d_ef], seed + padding zero OK")


if __name__ == "__main__":
    test_shapes_nodes_mask_cutoffs()
    test_ages_node_aligned_seed_is_zero()
    test_seed_is_last_real_node()
    test_empty_walks_when_no_predecessors()
    test_flatten_exclude_seed_positions_only()
    test_flatten_no_filtering_when_positions_false()
    test_flatten_shapes_and_padding_realdata()
    test_edge_features_populated_and_seed_padding_zero()
    print("\nALL RAW WALK-TENSOR CHECKS PASSED")
