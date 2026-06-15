"""Edge-feature plumbing + pairing tests for WalkData.

Confirms that walk-step edge features come back out of Tempest correctly PAIRED
with their edges under the backward-walk contract (CLAUDE.md "Tempest walk
contract"): for p in [0, lens-2], edge_feats[i,p] must be the feature of the SAME
edge (nodes[i,p], nodes[i,p+1]) whose time is timestamps[i,p] — not a neighbouring
edge — and the seed slot + padding must be zero so the context mask lines up.

The synthetic graph encodes each edge's own identity into its feature vector
[src, tgt, ts, edge_id], so a mis-pairing (off-by-one, wrong row, transpose) is
caught exactly rather than statistically.
"""
import numpy as np
import torch

from tempest_walks.walks import WalkGenerator


def _synthetic_graph(n_nodes=8, n_edges=60, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, n_edges).astype(np.int64)
    tgt = rng.integers(0, n_nodes, n_edges).astype(np.int64)
    same = src == tgt
    tgt[same] = (tgt[same] + 1) % n_nodes
    ts = np.sort(rng.choice(np.arange(1, 5000), n_edges, replace=False)).astype(np.int64)
    eid = np.arange(n_edges)
    ef = np.stack([src, tgt, ts, eid], axis=1).astype(np.float32)   # [E, 4]
    return src, tgt, ts, ef


def test_edge_feats_present_and_aligned():
    src, tgt, ts, ef = _synthetic_graph()
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=4, max_walk_len=8)
    wg.reset()
    wg.add_edges(src, tgt, ts, ef)
    wd = wg.walks_for_nodes(np.array([1, 3, 5, 7], dtype=np.int64))

    assert wd.edge_feats is not None, "edge features were not plumbed into WalkData"
    NK, L = wd.nodes.shape
    assert wd.edge_feats.shape == (NK, L, ef.shape[1]), \
        f"edge_feats should be [NK, L, d_ef], got {tuple(wd.edge_feats.shape)}"
    assert wd.edge_feats.dtype == torch.float32

    nodes = wd.nodes.numpy(); wts = wd.timestamps.numpy()
    lens = wd.lens.numpy(); efs = wd.edge_feats.numpy()

    checked = ts_ok = pair_ok = 0
    for i in range(NK):
        Li = int(lens[i])
        for p in range(0, Li - 1):                       # real edge slots only
            f_src, f_tgt, f_ts, _ = efs[i, p]
            checked += 1
            ts_ok += int(round(f_ts)) == int(wts[i, p])  # same edge time
            ts_ok_pair = {int(round(f_src)), int(round(f_tgt))} == \
                {int(nodes[i, p]), int(nodes[i, p + 1])}  # same (unordered) endpoints
            pair_ok += ts_ok_pair
    assert checked > 0, "no real edges sampled — graph too sparse for the test"
    assert ts_ok == checked, f"edge_feats time mis-paired: {ts_ok}/{checked}"
    assert pair_ok == checked, f"edge_feats endpoints mis-paired: {pair_ok}/{checked}"
    print(f"\n[pairing] {checked} real-edge slots: ts-match {ts_ok}/{checked} "
          f"endpoint-match {pair_ok}/{checked}")


def test_edge_feats_seed_and_padding_are_zero():
    """Seed slot (p=lens-1) and padding (p>=lens) carry ZERO edge feature, so the
    context mask (positions < lens-1) selects exactly the real edge rows."""
    src, tgt, ts, ef = _synthetic_graph()
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=4, max_walk_len=8)
    wg.reset(); wg.add_edges(src, tgt, ts, ef)
    wd = wg.walks_for_nodes(np.array([1, 3, 5, 7], dtype=np.int64))

    NK, L = wd.nodes.shape
    lens = wd.lens.numpy(); efs = wd.edge_feats.numpy()
    pos = np.arange(L)[None, :]
    is_ctx = pos < (lens[:, None] - 1)                   # the model's context mask

    nz = np.abs(efs).sum(-1) > 0                          # rows with a real feature
    # real features must live exactly where the context mask is true
    assert np.array_equal(nz, is_ctx), \
        "edge_feats nonzero rows do not match the context mask (positions < lens-1)"
    assert np.abs(efs[~is_ctx]).max() == 0.0, "seed/padding edge_feats not exactly zero"
    print(f"\n[mask] edge_feats nonzero ⟺ context mask, over {NK}x{L} slots")


def test_edge_feats_none_when_dataset_has_none():
    """No edge features ingested -> WalkData.edge_feats is None (not a crash)."""
    src, tgt, ts, _ = _synthetic_graph()
    wg = WalkGenerator(use_gpu=False, num_walks_per_node=3, max_walk_len=6)
    wg.reset(); wg.add_edges(src, tgt, ts, None)          # no edge features
    wd = wg.walks_for_nodes(np.array([2, 4], dtype=np.int64))
    assert wd.edge_feats is None, "edge_feats should be None when none were ingested"
    print("\n[none] edge_feats is None when dataset has no edge features")


if __name__ == "__main__":
    test_edge_feats_present_and_aligned()
    test_edge_feats_seed_and_padding_are_zero()
    test_edge_feats_none_when_dataset_has_none()
    print("\nALL EDGE-FEAT CHECKS PASSED")
