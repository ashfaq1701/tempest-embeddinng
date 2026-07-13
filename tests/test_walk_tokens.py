"""Correctness tests for the raw per-query walk tensors (tempest_walks/walk_tokens.py).

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

from tempest_walks.walk_tokens import _TS_SENTINEL, build_query_walk_tokens
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
    wg.reset(); wg.add_edges(src, tgt, ts, None)
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

    for name, t in (("nodes", wt.nodes), ("nodes_mask", wt.nodes_mask), ("timestamps", wt.timestamps)):
        assert t.shape == (q, k, length), f"{name} shape {tuple(t.shape)}"
    assert wt.cutoffs.shape == (q,) and wt.seeds.shape == (q,)
    assert torch.equal(wt.cutoffs, cutoffs), "cutoffs must round-trip"
    assert torch.equal(wt.seeds, seeds), "seeds must round-trip"
    assert torch.equal(wt.nodes, wd.nodes.to(torch.int64).reshape(q, k, length)), \
        "nodes must equal the raw walk output reshaped to [Q, K, L]"
    assert torch.equal(wt.nodes_mask, wt.nodes != -1), "nodes_mask must be (nodes != -1)"
    print(f"\n[shapes] nodes/mask/timestamps {tuple(wt.nodes.shape)}, cutoffs {tuple(wt.cutoffs.shape)} OK")


# ──────────────────────────────────────────────────────────────────────────
# 2. node-aligned timestamps: seed=cutoff, no sentinel, mask shared, causal
# ──────────────────────────────────────────────────────────────────────────
def test_timestamps_node_aligned_seed_is_cutoff():
    k, mwl = 8, 8
    seeds = torch.tensor([2, 4, 6, 8], dtype=torch.long)
    cutoffs = torch.tensor([12_000, 25_000, 40_000, 50_000], dtype=torch.long)
    wt, wd = _build(seeds, cutoffs, k=k, mwl=mwl, gseed=4)
    q, length = len(seeds), int(wd.nodes.shape[1])

    # No sentinel survives, and it equals Tempest's timestamps with sentinel -> cutoff.
    assert not bool((wt.timestamps == _TS_SENTINEL).any()), "sentinel must be replaced"
    raw = wd.timestamps.to(torch.int64).reshape(q, k, length)
    expect = torch.where(raw == _TS_SENTINEL, cutoffs.view(q, 1, 1), raw)
    assert torch.equal(wt.timestamps, expect), "timestamps must be raw with sentinel->cutoff"

    # The mask is shared: a real time iff a real node.
    assert torch.equal(wt.timestamps != -1, wt.nodes_mask), "timestamps != -1 must match nodes_mask"

    # Seed slot == cutoff; every other real time strictly < cutoff.
    seed_idx = _last_real_index(wt.nodes_mask).unsqueeze(-1)               # [Q, K, 1]
    is_seed = torch.zeros_like(wt.nodes_mask)
    is_seed.scatter_(-1, seed_idx, wt.nodes_mask.any(-1, keepdim=True))    # only for non-empty walks
    cut = cutoffs.view(q, 1, 1).expand_as(wt.timestamps)
    assert bool((wt.timestamps[is_seed] == cut[is_seed]).all()), "seed time must == cutoff"
    non_seed = wt.nodes_mask & ~is_seed
    assert bool((wt.timestamps[non_seed] < cut[non_seed]).all()), "non-seed times must be < cutoff"
    print("\n[timestamps] node-aligned, seed==cutoff, edges<cutoff, mask shared OK")


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
    last_time = torch.gather(wt.timestamps, -1, seed_idx.unsqueeze(-1)).squeeze(-1)   # [Q, K]
    assert torch.equal(last_node, seeds.view(-1, 1).expand_as(last_node)), \
        "the last real node of every walk must be that query's seed"
    assert torch.equal(last_time, cutoffs.view(-1, 1).expand_as(last_time)), \
        "the seed (last node) must carry time == cutoff"
    print("\n[seed] last real node == query seed, time == cutoff OK")


# ──────────────────────────────────────────────────────────────────────────
# 4. empty walks (no traversable edge)
# ──────────────────────────────────────────────────────────────────────────
def test_empty_walks_when_no_predecessors():
    src, tgt, ts = _synthetic_graph(n_nodes=6, n_edges=40, seed=2)
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=4, max_walk_len=6)
    wg.reset(); wg.add_edges(src, tgt, ts, None)
    t_min = int(ts.min())
    seeds = torch.tensor([int(src[0]), 999], dtype=torch.long)          # cutoff-excluded, isolated
    cutoffs = torch.tensor([t_min, 50_000], dtype=torch.long)

    wt, _ = _run_with_capture(wg, seeds, cutoffs, max_walk_len=6, num_walks_per_node=4)
    assert torch.equal(wt.seeds, seeds), "seeds must be kept even for cold/empty walks"
    for q in (0, 1):
        assert not bool(wt.nodes_mask[q].any()), f"query {q}: expected all-False mask"
        assert bool((wt.nodes[q] == -1).all()), f"query {q}: expected all-padding nodes"
        assert bool((wt.timestamps[q] == -1).all()), f"query {q}: expected all-padding timestamps"
    print("\n[empty] cutoff-excluded + isolated queries → fully empty walks (seeds kept) OK")


# ──────────────────────────────────────────────────────────────────────────
# 5. flatten_tokens — flat [Q, K*L] bag; padding always masked + seed-filter flags
# ──────────────────────────────────────────────────────────────────────────
def _synthetic_tokens():
    """One query, K=2 walks, L=4. Seed u=5, cutoff t=100.
    Walk1 nodes [7,5,3,5] ts [90,95,98,100]: pos1 = MID-WALK seed recurrence (ts95, age5),
                                              pos3 = seed SLOT (ts==cutoff=100, age0).
    Walk2 nodes [8,2,5,-1] ts [80,85,100,-1]: pos2 = seed SLOT (ts100), pos3 = PADDING.
    Flat order is walk1's 4 positions then walk2's -> indices 0..7."""
    from tempest_walks.walk_tokens import WalkTokens
    nodes = torch.tensor([[[7, 5, 3, 5], [8, 2, 5, -1]]], dtype=torch.long)
    ts = torch.tensor([[[90, 95, 98, 100], [80, 85, 100, -1]]], dtype=torch.long)
    return WalkTokens(seeds=torch.tensor([5], dtype=torch.long), nodes=nodes,
                      nodes_mask=(nodes != -1), timestamps=ts,
                      cutoffs=torch.tensor([100], dtype=torch.long))


def test_flatten_exclude_seed_tokens_is_default():
    """Default (exclude_seed_tokens=True): EVERY occurrence of node u masked (slot + recurrence)."""
    from tempest_walks.walk_tokens import flatten_tokens
    wt = _synthetic_tokens()
    ids, mask, ages, _ = flatten_tokens(wt)                       # defaults True/True -> tokens wins
    assert mask.int().tolist()[0] == [1, 0, 1, 0, 1, 1, 0, 0]  # pos1(rec),3,6 (node5) + pad7 removed
    assert not bool((ids[0][mask[0]] == 5).any()), "no seed token may survive under exclude_seed_tokens"
    print("\n[flatten] exclude_seed_tokens (default) removes all seed occurrences OK")


def test_flatten_exclude_seed_positions_only():
    """exclude_seed_tokens=False, exclude_seed_positions=True: only the walk-origin slot masked;
    mid-walk seed recurrences are KEPT (this is the whole slot-vs-token distinction)."""
    from tempest_walks.walk_tokens import flatten_tokens
    wt = _synthetic_tokens()
    ids, mask, ages, _ = flatten_tokens(wt, exclude_seed_positions=True, exclude_seed_tokens=False)
    assert mask.int().tolist()[0] == [1, 1, 1, 0, 1, 1, 0, 0]  # slots pos3,6 (ts==cutoff) removed; pos1 KEPT
    seed_kept = (ids[0][mask[0]] == 5)
    assert bool(seed_kept.any()), "mid-walk seed recurrence must be KEPT"
    assert bool((ages[0][mask[0]][seed_kept] > 0).all()), "surviving seed token is a recurrence (age>0)"
    print("\n[flatten] exclude_seed_positions keeps mid-walk recurrence, drops walk-origin slot OK")


def test_flatten_no_filtering_when_both_false():
    """both False: seed kept everywhere; only padding removed."""
    from tempest_walks.walk_tokens import flatten_tokens
    wt = _synthetic_tokens()
    ids, mask, ages, _ = flatten_tokens(wt, exclude_seed_positions=False, exclude_seed_tokens=False)
    assert mask.int().tolist()[0] == [1, 1, 1, 1, 1, 1, 1, 0]  # only padding pos7 removed
    assert int(mask.sum()) == int(wt.nodes_mask.sum())
    print("\n[flatten] both flags False keeps the seed everywhere OK")


def test_flatten_tokens_takes_precedence_over_positions():
    """exclude_seed_tokens=True dominates regardless of exclude_seed_positions."""
    from tempest_walks.walk_tokens import flatten_tokens
    wt = _synthetic_tokens()
    m_pos_false = flatten_tokens(wt, exclude_seed_positions=False, exclude_seed_tokens=True)[1]
    m_pos_true = flatten_tokens(wt, exclude_seed_positions=True, exclude_seed_tokens=True)[1]
    assert bool((m_pos_false == m_pos_true).all()), "exclude_seed_tokens must ignore positions flag"
    assert m_pos_true.int().tolist()[0] == [1, 0, 1, 0, 1, 1, 0, 0]
    print("\n[flatten] exclude_seed_tokens precedence OK")


def test_flatten_shapes_and_padding_realdata():
    """On live Tempest walks: shapes correct, padding never leaks, default excludes the seed."""
    from tempest_walks.walk_tokens import flatten_tokens
    k, mwl = 6, 8
    seeds = torch.tensor([2, 4, 6, 8], dtype=torch.long)
    cutoffs = torch.tensor([20_000, 30_000, 40_000, 50_000], dtype=torch.long)
    wt, _ = _build(seeds, cutoffs, k=k, mwl=mwl, gseed=1)
    q = len(seeds)
    ids, mask, ages, _ = flatten_tokens(wt)                       # default: exclude seed tokens
    assert ids.shape == (q, k * mwl) and mask.shape == (q, k * mwl) and ages.shape == (q, k * mwl)
    for i in range(q):
        kept = ids[i][mask[i]]
        assert bool((kept != -1).all()), f"padding leaked into kept tokens (q={i})"
        assert not bool((kept == seeds[i]).any()), f"default must EXCLUDE seed {int(seeds[i])} (q={i})"
        assert bool((ages[i][mask[i]] >= 0).all()), f"kept token age must be >= 0 (q={i})"
    print("\n[flatten] real-data: shapes OK, padding excluded, seed excluded (default) OK")


if __name__ == "__main__":
    test_shapes_nodes_mask_cutoffs()
    test_timestamps_node_aligned_seed_is_cutoff()
    test_seed_is_last_real_node()
    test_empty_walks_when_no_predecessors()
    test_flatten_exclude_seed_tokens_is_default()
    test_flatten_exclude_seed_positions_only()
    test_flatten_no_filtering_when_both_false()
    test_flatten_tokens_takes_precedence_over_positions()
    test_flatten_shapes_and_padding_realdata()
    print("\nALL RAW WALK-TENSOR CHECKS PASSED")
