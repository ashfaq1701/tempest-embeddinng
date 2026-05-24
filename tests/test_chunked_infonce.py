"""Unit tests for chunked InfoNCE.

Coverage:
1. chunk_vs_full_K1 — K=1, NK=200: every chunk size in
   {0, 10, 50, 100, 7, 32, 1, NK, 500} matches the chunk=0 reference
   on loss and on gradients (target W, context W, embedding E).
2. chunk_vs_full_Kgt1 — K=4 multi-walk: pool_walk_idx groups by
   ROW, not by seed; each walk only sees its own positions as
   positives even when seeds repeat across rows. A bug that swapped
   row-index for seed-index would silently lump positives across
   sibling walks. Exercises chunk ∈ {0, 1, 7, 32, NK}.
3. chunk_vs_full_node_feat — node_feat path (alternate projection
   head signature) is also exact under chunking.
4. chunked_matches_naive_reference — compares chunk_size=0 AND
   chunk_size=7 against an independent triple-loop implementation
   of the InfoNCE math. This is the gold standard: it doesn't trust
   the production no-chunk path either, so a bug shared by both
   paths would still be caught.

Single entry point: run `python tests/test_chunked_infonce.py`.
Exits non-zero on any failure.
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


# Tolerance: 1e-5 accommodates float32 reordering across chunk boundaries.
# Chunk=1 is the worst case — NK sequential additions of small per-seed
# contributions vs the non-chunked logsumexp's internal reduction.
TOL = 1e-5

# Module-level device. Functions read this directly so pytest can
# discover and run the tests without needing to inject a fixture.
# Override by setting env CUDA_VISIBLE_DEVICES="" to force CPU.
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Fixtures ───────────────────────────────────────────────────────


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
    """Deterministic Linear → L2-normalise."""
    def __init__(self, d_emb, d_proj, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.empty(d_emb, d_proj))
        nn.init.normal_(self.W, mean=0.0, std=1.0, generator=g)

    def forward(self, x):
        y = x @ self.W
        return F.normalize(y, p=2, dim=-1, eps=1e-12)


class _NodeFeatHead(nn.Module):
    """Projection head that accepts an extra `node_feat` kwarg."""
    def __init__(self, d_emb, d_nf, d_proj, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.empty(d_emb + d_nf, d_proj))
        nn.init.normal_(self.W, mean=0.0, std=1.0, generator=g)

    def forward(self, x, node_feat=None):
        z = torch.cat([x, node_feat], dim=-1) if node_feat is not None else x
        return F.normalize(z @ self.W, p=2, dim=-1, eps=1e-12)


def build_synthetic_walks(N, K, L, num_nodes, seed):
    """Construct a WalkData with N seeds × K walks of random length.
    Each row [i*K + k, :] is a walk for seed i. CPU tensors —
    alignment_loss moves them to its target device automatically.
    """
    rng = np.random.default_rng(seed)
    NK = N * K
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
    seeds = torch.arange(N, dtype=torch.long)
    return WalkData(
        nodes=nodes, timestamps=timestamps, lens=lens,
        edge_feats=None, seeds=seeds, K=K,
    )


def make_model(num_nodes, d_emb, d_proj, seed, *, device, d_nf=None):
    """Fresh deterministic embedding + two projection heads on `device`."""
    torch.manual_seed(seed)
    E = _FakeE(num_nodes, d_emb)
    if d_nf is None:
        p_t = _IdentityHead(d_emb, d_proj, seed=seed + 1)
        p_c = _IdentityHead(d_emb, d_proj, seed=seed + 2)
    else:
        p_t = _NodeFeatHead(d_emb, d_nf, d_proj, seed=seed + 1)
        p_c = _NodeFeatHead(d_emb, d_nf, d_proj, seed=seed + 2)
    return E.to(device), p_t.to(device), p_c.to(device)


def call_loss_and_grads(E, p_t, p_c, walks, *, tau, beta, T_train, chunk_size,
                        node_feat=None):
    """Reset grads, run loss + backward, return (loss, gT, gC, gE) snapshots."""
    for p in list(E.parameters()) + list(p_t.parameters()) + list(p_c.parameters()):
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    loss = alignment_loss(
        embedding_table=E,
        p_target=p_t,
        p_context=p_c,
        walks=walks,
        t_now=1_000_000,
        T_train=T_train,
        beta=beta,
        tau=tau,
        node_feat=node_feat,
        chunk_size=chunk_size,
    )
    loss.backward()
    return (
        float(loss.item()),
        p_t.W.grad.detach().clone(),
        p_c.W.grad.detach().clone(),
        E.E.weight.grad.detach().clone(),
    )


def _assert_match(label, ref, val):
    """ref/val are (loss, gT, gC, gE) tuples. Asserts within TOL."""
    loss_ref, gT_ref, gC_ref, gE_ref = ref
    loss_v, gT_v, gC_v, gE_v = val
    diffs = {
        "loss": abs(loss_v - loss_ref),
        "gT":   (gT_v - gT_ref).abs().max().item(),
        "gC":   (gC_v - gC_ref).abs().max().item(),
        "gE":   (gE_v - gE_ref).abs().max().item(),
    }
    ok = all(d < TOL for d in diffs.values())
    status = "PASS" if ok else "FAIL"
    detail = "  ".join(f"{k}={v:.2e}" for k, v in diffs.items())
    print(f"  {label:40s}  {detail}  [{status}]")
    return ok


# ─── Naive reference implementation ─────────────────────────────────


def naive_alignment_loss(E, p_t, p_c, walks, *, t_now, T_train, beta, tau,
                         node_feat=None):
    """Textbook InfoNCE: triple-loop over seeds, no chunking, no broadcasting
    tricks. Mirrors the docstring formula in losses.py exactly:

        L_i = - (Σ_p w[i,p] · log p(n_p^+ | s_i)) / (Σ_p w[i,p])
        log p(n | s) = -||p_t(s) - p_c(n)||² / τ
                       - logsumexp_j (-||p_t(s) - p_c(n_j)||² / τ)
        j ranges over all VALID batch contexts (every other walk's
        positions). Padding/seed slots excluded.

    Returns scalar mean over seeds-with-positives.
    """
    device = E.E.weight.device
    nodes = walks.nodes.to(device).long()
    timestamps = walks.timestamps.to(device).long()
    lens = walks.lens.to(device).long()
    seeds_t = walks.seeds.to(device).long()
    K = walks.K
    NK, L = nodes.shape
    M = NK * L

    # Validity mask.
    positions = torch.arange(L, device=device).unsqueeze(0)
    seed_pos = (lens - 1).unsqueeze(1)
    is_context = positions < seed_pos                 # [NK, L]
    valid_flat = is_context.reshape(M)                # [M]

    # Project everything.
    seed_per_row = seeds_t.repeat_interleave(K)
    e_seed = E(seed_per_row)
    nodes_safe = nodes.clamp_min(0)
    e_ctx = E(nodes_safe.reshape(-1))
    if node_feat is not None:
        nf_seed = node_feat[seed_per_row]
        nf_ctx = node_feat[nodes_safe.reshape(-1)]
        p_seed = p_t(e_seed, node_feat=nf_seed)       # [NK, d]
        p_ctx = p_c(e_ctx, node_feat=nf_ctx)          # [M, d]
    else:
        p_seed = p_t(e_seed)
        p_ctx = p_c(e_ctx)

    # Hop/time weights (same formula as production).
    hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()
    dt = (float(t_now) - timestamps.float()).clamp_min(0.0)
    w = 1.0 / hop_dist + (1.0 + dt / max(T_train, 1.0)).pow(-beta)   # [NK, L]

    valid_idx = torch.nonzero(valid_flat, as_tuple=False).squeeze(1)  # [V]
    p_ctx_valid = p_ctx[valid_idx]                                    # [V, d]

    losses = []
    for i in range(NK):
        # Per-seed similarity to ALL valid contexts.
        sq = ((p_seed[i].unsqueeze(0) - p_ctx_valid) ** 2).sum(dim=1)  # [V]
        sim = -sq / tau                                                # [V]
        log_Z = torch.logsumexp(sim, dim=0)                            # scalar

        # Positives: valid contexts j whose row in nodes is i.
        pos_loss_terms = []
        pos_w = []
        for pos_in_walk in range(L):
            if not bool(is_context[i, pos_in_walk]):
                continue
            j_pool = i * L + pos_in_walk
            sq_p = ((p_seed[i] - p_ctx[j_pool]) ** 2).sum()
            sim_p = -sq_p / tau
            log_p = sim_p - log_Z
            pos_loss_terms.append(w[i, pos_in_walk] * log_p)
            pos_w.append(w[i, pos_in_walk])

        if not pos_w:
            continue
        num = torch.stack(pos_loss_terms).sum()
        den = torch.stack(pos_w).sum()
        losses.append(-num / den)

    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


# ─── Test cases ─────────────────────────────────────────────────────


def test_chunk_vs_full_K1():
    device = _DEVICE
    print(f"Test 1: chunk vs full, K=1, NK=200  [{device}]")
    N, K, L = 200, 1, 20
    d_emb, d_proj, num_nodes = 64, 64, 1000
    tau, beta, T_train = 0.5, 1.0, 1_000_000.0
    seed = 0

    walks = build_synthetic_walks(N, K, L, num_nodes, seed)
    NK = N * K

    E, p_t, p_c = make_model(num_nodes, d_emb, d_proj, seed, device=device)
    ref = call_loss_and_grads(E, p_t, p_c, walks, tau=tau, beta=beta,
                              T_train=T_train, chunk_size=0)
    print(f"  reference loss = {ref[0]:.10f}")

    failed = False
    for cs in [10, 50, 100, 7, 32, 1, NK, 500]:
        E, p_t, p_c = make_model(num_nodes, d_emb, d_proj, seed, device=device)
        val = call_loss_and_grads(E, p_t, p_c, walks, tau=tau, beta=beta,
                                  T_train=T_train, chunk_size=cs)
        if not _assert_match(f"chunk={cs}", ref, val):
            failed = True
    assert not failed, "at least one chunk variant differs from reference"


def test_chunk_vs_full_Kgt1():
    device = _DEVICE
    print(f"Test 2: chunk vs full, K=4 (multi-walk per seed)  [{device}]")
    N, K, L = 50, 4, 16
    d_emb, d_proj, num_nodes = 64, 64, 500
    tau, beta, T_train = 0.3, 1.5, 500_000.0
    seed = 1

    walks = build_synthetic_walks(N, K, L, num_nodes, seed)
    NK = N * K

    E, p_t, p_c = make_model(num_nodes, d_emb, d_proj, seed, device=device)
    ref = call_loss_and_grads(E, p_t, p_c, walks, tau=tau, beta=beta,
                              T_train=T_train, chunk_size=0)
    print(f"  reference loss = {ref[0]:.10f}  (NK={NK})")

    failed = False
    for cs in [1, 7, 32, NK, 1000]:
        E, p_t, p_c = make_model(num_nodes, d_emb, d_proj, seed, device=device)
        val = call_loss_and_grads(E, p_t, p_c, walks, tau=tau, beta=beta,
                                  T_train=T_train, chunk_size=cs)
        if not _assert_match(f"chunk={cs}", ref, val):
            failed = True
    assert not failed, "at least one chunk variant differs from reference"


def test_chunk_vs_full_node_feat():
    device = _DEVICE
    print(f"Test 3: chunk vs full, node_feat path  [{device}]")
    N, K, L = 60, 2, 14
    d_emb, d_proj, num_nodes, d_nf = 48, 48, 400, 8
    tau, beta, T_train = 0.5, 1.0, 1_000_000.0
    seed = 2

    walks = build_synthetic_walks(N, K, L, num_nodes, seed)
    NK = N * K
    g = torch.Generator().manual_seed(seed + 99)
    node_feat = torch.randn(num_nodes, d_nf, generator=g).to(device)

    E, p_t, p_c = make_model(num_nodes, d_emb, d_proj, seed, device=device, d_nf=d_nf)
    ref = call_loss_and_grads(E, p_t, p_c, walks, tau=tau, beta=beta,
                              T_train=T_train, chunk_size=0,
                              node_feat=node_feat)
    print(f"  reference loss = {ref[0]:.10f}")

    failed = False
    for cs in [1, 13, NK]:
        E, p_t, p_c = make_model(num_nodes, d_emb, d_proj, seed, device=device, d_nf=d_nf)
        val = call_loss_and_grads(E, p_t, p_c, walks, tau=tau, beta=beta,
                                  T_train=T_train, chunk_size=cs,
                                  node_feat=node_feat)
        if not _assert_match(f"chunk={cs}", ref, val):
            failed = True
    assert not failed, "at least one chunk variant differs from reference"


def test_chunked_matches_naive_reference():
    """Both chunked and non-chunked production paths match an
    independent triple-loop implementation. Catches bugs shared by
    both code paths."""
    device = _DEVICE
    print(f"Test 4: chunked + full match naive triple-loop reference  [{device}]")
    N, K, L = 8, 3, 10                       # small enough for the loop
    d_emb, d_proj, num_nodes = 32, 32, 50
    tau, beta, T_train = 0.4, 1.2, 250_000.0
    seed = 3

    walks = build_synthetic_walks(N, K, L, num_nodes, seed)
    NK = N * K

    E_n, p_t_n, p_c_n = make_model(num_nodes, d_emb, d_proj, seed, device=device)
    naive = naive_alignment_loss(
        E_n, p_t_n, p_c_n, walks,
        t_now=1_000_000, T_train=T_train, beta=beta, tau=tau,
    )
    naive.backward()
    naive_ref = (
        float(naive.item()),
        p_t_n.W.grad.detach().clone(),
        p_c_n.W.grad.detach().clone(),
        E_n.E.weight.grad.detach().clone(),
    )
    print(f"  naive loss = {naive_ref[0]:.10f}")

    failed = False
    for cs in [0, 7, 1, NK]:
        E, p_t, p_c = make_model(num_nodes, d_emb, d_proj, seed, device=device)
        val = call_loss_and_grads(E, p_t, p_c, walks, tau=tau, beta=beta,
                                  T_train=T_train, chunk_size=cs)
        if not _assert_match(f"production chunk={cs} vs naive", naive_ref, val):
            failed = True
    assert not failed, "production path diverges from naive reference"


def main():
    if _DEVICE.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu (no CUDA available)")

    failed_any = False
    for fn in (
        test_chunk_vs_full_K1,
        test_chunk_vs_full_Kgt1,
        test_chunk_vs_full_node_feat,
        test_chunked_matches_naive_reference,
    ):
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL: {fn.__name__} — {e}")
            failed_any = True
    if failed_any:
        sys.exit(1)
    print(f"\nPASS: all chunked InfoNCE tests match reference within {TOL:.0e}.")


if __name__ == "__main__":
    main()
