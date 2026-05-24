"""Unit test for chunked InfoNCE.

Verifies that alignment_loss produces identical loss and gradients
for chunk_size ∈ {0, 10, 50, 100, 7 (non-divisor)} on a deterministic
synthetic batch.

Reference: NK=200, L=20, d_proj=128, tau=0.5, beta=1.0, torch.manual_seed(0).
"""

import math
import sys
import pathlib

# Allow direct invocation.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tempest_walks.losses import alignment_loss
from tempest_walks.walks import WalkData


class _FakeE(nn.Module):
    """Wraps an nn.Embedding to expose `.E.weight.device`, matching
    EmbeddingTable's API used by alignment_loss."""
    def __init__(self, num_nodes, d_emb):
        super().__init__()
        self.E = nn.Embedding(num_nodes, d_emb)
        nn.init.normal_(self.E.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        return self.E(ids)


class _IdentityHead(nn.Module):
    """A passthrough projection head that L2-normalises a learnable
    projection of the input. We make the projection deterministic by
    using a fixed-seed Linear.
    """
    def __init__(self, d_emb, d_proj, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.empty(d_emb, d_proj))
        nn.init.normal_(self.W, mean=0.0, std=1.0, generator=g)
        # Disambiguate target vs context with different seeds.

    def forward(self, x):
        y = x @ self.W
        return F.normalize(y, p=2, dim=-1, eps=1e-12)


def build_synthetic_walks(NK, L, num_nodes, seed):
    """Construct a deterministic WalkData with random lens in [3, L]."""
    rng = np.random.default_rng(seed)
    lens = torch.from_numpy(rng.integers(low=3, high=L + 1, size=NK).astype(np.int64))
    nodes = torch.full((NK, L), -1, dtype=torch.long)
    timestamps = torch.full((NK, L), -1, dtype=torch.long)
    for i in range(NK):
        valid = int(lens[i])
        nodes[i, :valid] = torch.from_numpy(
            rng.integers(0, num_nodes, size=valid).astype(np.int64)
        )
        timestamps[i, :valid] = torch.from_numpy(
            rng.integers(0, 1_000_000, size=valid).astype(np.int64)
        )
    # K=1 → one walk per seed in this synthetic. (NK == N when K=1.)
    K = 1
    N = NK
    seeds = torch.arange(N, dtype=torch.long)
    return WalkData(
        nodes=nodes, timestamps=timestamps, lens=lens,
        edge_feats=None, seeds=seeds, K=K,
    )


def compute_loss_and_grads(chunk_size, NK, L, d_emb, d_proj, num_nodes, tau, beta, T_train, seed):
    """Build a fresh model + walks (deterministic by seed) and return
    (loss_value, grad_seed_W, grad_ctx_W)."""
    torch.manual_seed(seed)
    E = _FakeE(num_nodes, d_emb)
    p_target = _IdentityHead(d_emb, d_proj, seed=seed + 1)
    p_context = _IdentityHead(d_emb, d_proj, seed=seed + 2)

    walks = build_synthetic_walks(NK, L, num_nodes, seed=seed)

    loss = alignment_loss(
        embedding_table=E,
        p_target=p_target,
        p_context=p_context,
        walks=walks,
        t_now=1_000_000,
        T_train=T_train,
        beta=beta,
        tau=tau,
        chunk_size=chunk_size,
    )
    loss.backward()

    grad_target_W = p_target.W.grad.detach().clone()
    grad_context_W = p_context.W.grad.detach().clone()
    grad_E = E.E.weight.grad.detach().clone()
    return float(loss.item()), grad_target_W, grad_context_W, grad_E


def main():
    NK, L = 200, 20
    d_emb = 128
    d_proj = 128
    num_nodes = 1000
    tau = 0.5
    beta = 1.0
    T_train = 1000_000.0
    seed = 0

    # Reference: no chunking.
    print(f"Reference (chunk_size=0):")
    loss_ref, gT_ref, gC_ref, gE_ref = compute_loss_and_grads(
        0, NK, L, d_emb, d_proj, num_nodes, tau, beta, T_train, seed)
    print(f"  loss = {loss_ref:.10f}")

    # Chunked variants.
    failed = False
    for chunk in [10, 50, 100, 7, 32, 1, NK]:
        loss_c, gT_c, gC_c, gE_c = compute_loss_and_grads(
            chunk, NK, L, d_emb, d_proj, num_nodes, tau, beta, T_train, seed)
        abs_diff = abs(loss_c - loss_ref)
        gT_max_diff = (gT_c - gT_ref).abs().max().item()
        gC_max_diff = (gC_c - gC_ref).abs().max().item()
        gE_max_diff = (gE_c - gE_ref).abs().max().item()
        # Tolerance 1e-5 accommodates float32 accumulation noise.
        # Worst case is chunk=1 where 200 sequential additions of
        # small per-seed contributions reorder vs the non-chunked
        # logsumexp's internal reduction order. The math is exact
        # in real arithmetic; this is purely float32 reordering.
        ok = (
            abs_diff < 1e-5
            and gT_max_diff < 1e-5
            and gC_max_diff < 1e-5
            and gE_max_diff < 1e-5
        )
        status = "PASS" if ok else "FAIL"
        print(
            f"  chunk={chunk:3d}: loss={loss_c:.10f}  "
            f"abs_diff={abs_diff:.2e}  "
            f"grad_target_max_diff={gT_max_diff:.2e}  "
            f"grad_context_max_diff={gC_max_diff:.2e}  "
            f"grad_E_max_diff={gE_max_diff:.2e}  "
            f"[{status}]"
        )
        if not ok:
            failed = True

    if failed:
        print("\nFAIL: at least one chunk variant differs from reference.")
        sys.exit(1)
    else:
        print("\nPASS: all chunk variants match reference within 1e-5.")


if __name__ == "__main__":
    main()
