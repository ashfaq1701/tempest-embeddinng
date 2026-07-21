"""Fixed-basis NodeEncoding: each global node id gets a PERMANENT fingerprint, so a node's base
features (block 0) are identical across different batches — the property that makes the codes
MLP-usable and recurrence-aware. Contrast with the default fresh (anonymized) basis, which redraws
every call. Real graph + real Tempest walks over tests/data/sample_data.csv."""
import pathlib

import numpy as np
import torch

from link_property_prediction.model import NodeEncoding, StatelessLinkHead
from link_property_prediction.walk_tokens import build_query_walk_tokens
from link_property_prediction.walks import WalkGenerator

_CSV = pathlib.Path(__file__).parent / "data" / "sample_data.csv"
_DEV = torch.device("cpu")
_D = np.loadtxt(_CSV, delimiter=",", skiprows=1, dtype=np.int64)
SRC, DST, TS = _D[:, 0], _D[:, 1], _D[:, 2]
NUM_NODES = int(max(SRC.max(), DST.max())) + 1
CUTOFF = int(TS.max()) + 1


def _wg():
    wg = WalkGenerator(num_walks_per_node=6, max_walk_len=5)
    wg.add_edges(SRC, DST, TS, None)
    return wg


def _bag(wg, seeds):
    s = torch.as_tensor(seeds, dtype=torch.long)
    c = torch.full((len(seeds),), CUTOFF, dtype=torch.long)
    return build_query_walk_tokens(wg, _DEV, s, c, max_walk_len=5, num_walks_per_node=6)


def _block0_by_id(enc, tokens):
    """Map global id -> its block-0 (base) features from one encode."""
    dim = enc.dim
    assoc, node_enc = enc(tokens)
    present = (assoc >= 0).nonzero(as_tuple=True)[0]
    return {int(g): node_enc[assoc[g], :dim] for g in present}


def test_fixed_basis_stable_across_batches_and_matches_table():
    """Fixed basis: a node's base features are identical in two DIFFERENT bags (and equal x0_table[id]),
    even with the global RNG advanced between calls — i.e. keyed to identity, not to the batch."""
    wg = _wg()
    enc = NodeEncoding(NUM_NODES, dim=16, n_hops=2, fixed_basis=True)

    torch.manual_seed(1)
    b0_a = _block0_by_id(enc, _bag(wg, [0, 1, 2, 3]))
    torch.manual_seed(999)                                     # advance RNG — must not matter under fixed basis
    b0_b = _block0_by_id(enc, _bag(wg, [2, 3, 4, 5]))

    shared = set(b0_a) & set(b0_b)
    assert shared, "bags should share nodes"
    for g in shared:
        assert torch.equal(b0_a[g], b0_b[g])                  # same id → same fingerprint across batches
        assert torch.equal(b0_a[g], enc.x0_table[g])          # and it IS the permanent table row


def test_fresh_basis_differs_across_batches():
    """Default fresh basis: the same id gets DIFFERENT base features on two encodes (rotating basis)."""
    wg = _wg()
    enc = NodeEncoding(NUM_NODES, dim=16, n_hops=2)           # fixed_basis=False (default)
    assert not hasattr(enc, "x0_table")
    b0_a = _block0_by_id(enc, _bag(wg, [0, 1, 2, 3]))
    b0_b = _block0_by_id(enc, _bag(wg, [0, 1, 2, 3]))
    shared = set(b0_a) & set(b0_b)
    assert shared and all(not torch.equal(b0_a[g], b0_b[g]) for g in shared)


def test_head_runs_with_fixed_basis():
    """StatelessLinkHead(fixed_basis=True) produces finite, differentiable logits."""
    wg = _wg()
    head = StatelessLinkHead(NUM_NODES, d_emb=16, n_hops=3, fixed_basis=True)
    logits = head(_bag(wg, [3, 4]), _bag(wg, [5, 6, 7, 8]))
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    logits.sum().backward()
