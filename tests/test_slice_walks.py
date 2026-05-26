"""Pin the slice_walks_by_seeds contract.

The function carves a sub-WalkData from a larger WalkData by seed id.
It is load-bearing for the unified Tempest call in the trainer: ONE
walks_for_nodes(src ∪ tgt ∪ neg_tgt) per batch + slice for alignment_loss.

Three invariants this test pins:
  1. Row gather order: rows [j*K, (j+1)*K) of the slice belong to
     subset_seeds[j] (the K-contiguous grouping invariant that every
     downstream consumer assumes).
  2. Value fidelity: nodes / timestamps / lens / edge_feats for the
     selected blocks match the corresponding blocks in the source.
  3. edge_feats=None passthrough: if the source has no edge features,
     the slice has none either.
"""

import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from tempest_walks.walks import WalkData, slice_walks_by_seeds


def _make_walks(seed_ids, K, L, d_ef=4, with_ef=True):
    """Construct a WalkData where every cell encodes (block_index, position)
    so we can verify which block ended up where after slicing.

    For block i and position p:
      nodes[i*K+k, p]      = seed_ids[i] * 1000 + k * 10 + p
      timestamps[i*K+k, p] = nodes value + 1
      edge_feats[i*K+k, p] = float(nodes value) along d_ef
      lens[i*K+k]          = L - (k % 3)  # vary per row to test gather
    """
    seed_ids_np = np.asarray(seed_ids, dtype=np.int64)
    N = len(seed_ids_np)
    NK = N * K
    nodes = torch.zeros(NK, L, dtype=torch.int32)
    ts = torch.zeros(NK, L, dtype=torch.int64)
    lens = torch.zeros(NK, dtype=torch.int64)
    for i, sid in enumerate(seed_ids_np):
        for k in range(K):
            row = i * K + k
            for p in range(L):
                val = int(sid) * 1000 + k * 10 + p
                nodes[row, p] = val
                ts[row, p] = val + 1
            lens[row] = L - (k % 3)
    ef = None
    if with_ef:
        ef = torch.zeros(NK, L - 1, d_ef, dtype=torch.float32)
        for i, sid in enumerate(seed_ids_np):
            for k in range(K):
                row = i * K + k
                for p in range(L - 1):
                    val = float(int(sid) * 1000 + k * 10 + p)
                    ef[row, p, :] = val
    seeds_t = torch.from_numpy(seed_ids_np)
    return WalkData(
        nodes=nodes,
        timestamps=ts,
        lens=lens,
        edge_feats=ef,
        seeds=seeds_t,
        K=K,
    )


def test_slice_preserves_block_grouping_and_values():
    K, L = 5, 7
    all_seed_ids = np.array([3, 11, 42, 57, 99, 200, 314], dtype=np.int64)
    walks = _make_walks(all_seed_ids, K, L)

    subset = np.array([11, 99, 314], dtype=np.int64)
    sliced = slice_walks_by_seeds(walks, subset)

    # Shapes: 3 seeds × K = 15 rows.
    assert sliced.K == K
    assert sliced.nodes.shape == (3 * K, L)
    assert sliced.timestamps.shape == (3 * K, L)
    assert sliced.lens.shape == (3 * K,)
    assert sliced.edge_feats.shape == (3 * K, L - 1, 4)
    assert sliced.seeds.tolist() == [11, 99, 314]

    # Block grouping + value fidelity. For each subset seed j, rows
    # [j*K, (j+1)*K) of the slice must equal the matching original rows.
    for j, sid in enumerate(subset.tolist()):
        orig_block_idx = int(np.searchsorted(all_seed_ids, sid))
        for k in range(K):
            new_row = j * K + k
            orig_row = orig_block_idx * K + k
            assert torch.equal(sliced.nodes[new_row], walks.nodes[orig_row]), (
                f"node row mismatch at sid={sid} k={k}"
            )
            assert torch.equal(sliced.timestamps[new_row], walks.timestamps[orig_row])
            assert int(sliced.lens[new_row]) == int(walks.lens[orig_row])
            assert torch.equal(sliced.edge_feats[new_row], walks.edge_feats[orig_row])


def test_slice_handles_no_edge_feats():
    K, L = 4, 6
    all_seed_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    walks = _make_walks(all_seed_ids, K, L, with_ef=False)

    subset = np.array([2, 4], dtype=np.int64)
    sliced = slice_walks_by_seeds(walks, subset)

    assert sliced.edge_feats is None
    assert sliced.nodes.shape == (2 * K, L)
    assert sliced.seeds.tolist() == [2, 4]
    # Spot-check one row.
    sid_2_row0_orig = walks.nodes[1 * K + 0]   # index of seed 2 in original = 1
    assert torch.equal(sliced.nodes[0], sid_2_row0_orig)


def test_slice_full_set_equals_original():
    K, L = 3, 5
    all_seed_ids = np.array([7, 8, 9], dtype=np.int64)
    walks = _make_walks(all_seed_ids, K, L)

    # Slicing the full set should be a no-op (value-wise).
    sliced = slice_walks_by_seeds(walks, all_seed_ids)
    assert torch.equal(sliced.nodes, walks.nodes)
    assert torch.equal(sliced.timestamps, walks.timestamps)
    assert torch.equal(sliced.lens, walks.lens)
    assert torch.equal(sliced.edge_feats, walks.edge_feats)
    assert sliced.seeds.tolist() == all_seed_ids.tolist()


def test_slice_single_seed():
    K, L = 5, 4
    all_seed_ids = np.array([10, 20, 30], dtype=np.int64)
    walks = _make_walks(all_seed_ids, K, L)

    subset = np.array([20], dtype=np.int64)
    sliced = slice_walks_by_seeds(walks, subset)

    assert sliced.nodes.shape == (K, L)
    assert sliced.seeds.tolist() == [20]
    # Original rows for seed 20 are [K, 2K).
    for k in range(K):
        assert torch.equal(sliced.nodes[k], walks.nodes[K + k])


if __name__ == "__main__":
    test_slice_preserves_block_grouping_and_values()
    test_slice_handles_no_edge_feats()
    test_slice_full_set_equals_original()
    test_slice_single_seed()
    print("OK: all slice_walks_by_seeds tests passed")
