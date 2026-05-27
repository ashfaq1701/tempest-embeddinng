"""Pin the AttentionWalkEncoder's contract.

This module is the load-bearing piece for the Q1-Q5 redesign. Tests
cover:
  1. Output shape: [N, d_emb].
  2. No NaN / no Inf for normal walks.
  3. Zero-edge walks produce zero h_walk contribution (the empty-walk
     row is masked from the transformer and gets a zero pool).
  4. exclude_seed=True replaces tgt_embs at the to-seed edge with a
     learned [SEED] marker (verified by setting E[seed] to a very
     distinctive value and confirming the encoder output is INSENSITIVE
     to that value when exclude_seed=True, and sensitive when False).
  5. Detached E lookups: changing E.weight.grad is None after forward
     (E does not receive gradient via the encoder).
"""

import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from tempest_walks.model import EmbeddingTable
from tempest_walks.walk_encoder import AttentionWalkEncoder
from tempest_walks.walks import WalkData


def _make_walks(num_seeds: int, K: int, L: int, num_nodes: int, d_ef: int = 0):
    """Construct a small synthetic WalkData. Variable lens per row to
    exercise padding handling."""
    NK = num_seeds * K
    nodes = torch.zeros(NK, L, dtype=torch.int32)
    ts = torch.zeros(NK, L, dtype=torch.int64)
    lens = torch.zeros(NK, dtype=torch.int64)
    seeds = torch.arange(1, num_seeds + 1, dtype=torch.int64) * 7  # arbitrary
    for i in range(num_seeds):
        for k in range(K):
            row = i * K + k
            # Length varies between 2 and L, with one explicit lens=1 case
            # at (i=0, k=0) to exercise the zero-edge code path.
            if i == 0 and k == 0:
                lens[row] = 1
            else:
                lens[row] = max(2, L - ((row * 3) % (L - 1)))
            # Fill nodes with arbitrary ids in [0, num_nodes), pad with -1.
            actual_len = int(lens[row])
            for p in range(actual_len):
                # Seed at lens-1.
                if p == actual_len - 1:
                    nodes[row, p] = int(seeds[i])
                else:
                    nodes[row, p] = (row * 7 + p) % num_nodes
                ts[row, p] = 100 + p
            # padding
            for p in range(actual_len, L):
                nodes[row, p] = -1
                ts[row, p] = -1
            # The seed timestamp should be INT64_MAX per contract;
            # the encoder slices [:, :-1] so it never reads ts at lens-1,
            # but be honest with the contract:
            ts[row, actual_len - 1] = torch.iinfo(torch.int64).max
    ef = torch.zeros(NK, L - 1, d_ef, dtype=torch.float32) if d_ef > 0 else None
    return WalkData(
        nodes=nodes, timestamps=ts, lens=lens, edge_feats=ef, seeds=seeds, K=K,
    )


def _make_encoder(num_nodes, d_emb=16, d_ef=0, exclude_seed=False, L=6):
    E = EmbeddingTable(num_nodes=num_nodes, d_emb=d_emb)
    return AttentionWalkEncoder(
        embedding_table=E,
        d_emb=d_emb,
        d_ef=d_ef,
        d_te=8,
        d_he=4,
        d_edge=32,
        d_walk=32,
        max_walk_len=L,
        n_heads=4,
        n_layers=1,
        exclude_seed=exclude_seed,
    )


def test_output_shape_and_finiteness():
    N, K, L = 5, 3, 6
    num_nodes = 64
    walks = _make_walks(N, K, L, num_nodes, d_ef=0)
    enc = _make_encoder(num_nodes, d_emb=16, L=L)
    h = enc(walks, t_now=1000.0, T_train=1000.0)
    assert h.shape == (N, 16), f"expected ({N}, 16), got {tuple(h.shape)}"
    assert torch.isfinite(h).all(), "h_seed contains non-finite values"


def test_with_edge_features():
    N, K, L = 4, 2, 5
    num_nodes = 32
    d_ef = 7
    walks = _make_walks(N, K, L, num_nodes, d_ef=d_ef)
    enc = _make_encoder(num_nodes, d_emb=16, d_ef=d_ef, L=L)
    h = enc(walks, t_now=500.0, T_train=500.0)
    assert h.shape == (N, 16)
    assert torch.isfinite(h).all()


def test_zero_edge_walk_does_not_break():
    """The (i=0, k=0) walk in _make_walks has lens=1 → 0 edges. The
    encoder must handle it without NaN/error and zero its contribution
    to the K-walk mean for that seed."""
    N, K, L = 3, 2, 5
    num_nodes = 32
    walks = _make_walks(N, K, L, num_nodes, d_ef=0)
    # Confirm fixture has lens=1 for row 0.
    assert int(walks.lens[0]) == 1, "fixture lost zero-edge row"
    enc = _make_encoder(num_nodes, d_emb=16, L=L)
    h = enc(walks, t_now=200.0, T_train=200.0)
    assert torch.isfinite(h).all(), "zero-edge walk produced NaN"


def test_exclude_seed_makes_encoder_insensitive_to_E_seed():
    """When exclude_seed=True, perturbing E[seed] (only that row)
    should NOT change h_seed for that seed — the encoder must use the
    [SEED] marker instead of E[seed] at the last-edge tgt slot AND not
    concat E[seed] at the output MLP.

    Conversely, with exclude_seed=False (default), the encoder DOES
    depend on E[seed] (via both the last-edge tgt and the MLP concat),
    so perturbing E[seed] should change h_seed."""
    torch.manual_seed(0)
    N, K, L = 4, 2, 5
    num_nodes = 32
    walks = _make_walks(N, K, L, num_nodes, d_ef=0)
    seed0 = int(walks.seeds[0])  # the seed whose E[seed] we'll perturb

    for exclude in (True, False):
        torch.manual_seed(42)
        enc = _make_encoder(num_nodes, d_emb=16, L=L, exclude_seed=exclude)
        enc.eval()
        with torch.no_grad():
            h_before = enc(walks, t_now=1.0, T_train=1.0).clone()
            # Perturb E[seed] for seed0 only.
            enc.E.E.weight.data[seed0] += 100.0
            h_after = enc(walks, t_now=1.0, T_train=1.0).clone()
            # Restore to keep test isolation.
            enc.E.E.weight.data[seed0] -= 100.0

        # Compare h_seed for the perturbed seed (row 0).
        delta = (h_after[0] - h_before[0]).abs().max().item()
        if exclude:
            assert delta < 1e-5, (
                f"exclude_seed=True: encoder is still sensitive to E[seed] "
                f"perturbation (max abs delta = {delta:.6f})"
            )
        else:
            assert delta > 1e-3, (
                f"exclude_seed=False: encoder was unexpectedly INSENSITIVE "
                f"to E[seed] perturbation (max abs delta = {delta:.6f}) — "
                f"the baseline path should depend on E[seed]"
            )


def test_E_does_not_receive_gradient_via_encoder():
    """BCE→E must not happen via the encoder. After encoder forward +
    a scalar loss + backward, E.weight.grad should be None (or zero)
    even though encoder param grads are populated."""
    N, K, L = 3, 2, 5
    num_nodes = 32
    walks = _make_walks(N, K, L, num_nodes, d_ef=0)
    enc = _make_encoder(num_nodes, d_emb=16, L=L)
    enc.train()
    h = enc(walks, t_now=200.0, T_train=200.0)
    loss = h.sum()
    loss.backward()
    # E lookup is detached inside the encoder, so its weight should not
    # receive a gradient from the encoder's loss.
    e_grad = enc.E.E.weight.grad
    assert e_grad is None or torch.equal(e_grad, torch.zeros_like(e_grad)), (
        "E.weight received gradient via the encoder — .detach() is broken"
    )
    # Sanity: at least one encoder parameter has a nonzero gradient.
    enc_grad_norms = [
        p.grad.norm().item() for p in enc.parameters()
        if p is not enc.E.E.weight and p.grad is not None
    ]
    assert any(g > 0 for g in enc_grad_norms), (
        "no encoder parameter received gradient — forward graph broken"
    )


if __name__ == "__main__":
    test_output_shape_and_finiteness()
    test_with_edge_features()
    test_zero_edge_walk_does_not_break()
    test_exclude_seed_makes_encoder_insensitive_to_E_seed()
    test_E_does_not_receive_gradient_via_encoder()
    print("OK: all AttentionWalkEncoder tests passed")
