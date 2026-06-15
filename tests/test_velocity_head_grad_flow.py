"""Gradient-flow tests for GeometricVelocityPerWalkAvgHead.

Tests the head the way it is actually wired in the trainer: E[u] (source), E[v]
(candidates) and the walk-neighbour tokens are all GATHERED ROWS of ONE shared
nn.Embedding, so the gradient that matters is the one landing on `embedding.weight`.
These checks catch the realistic bugs: a `.detach()` on one gather path, an inverted
context mask, or a valid neighbour silently dropped from the trajectory fit.

  1. Shared-table gradient — every role (source / candidate / neighbour) deposits
     nonzero gradient on the single embedding parameter the optimizer updates.
  2. Per-token masking — every VALID walk token receives gradient; every MASKED /
     padded token receives EXACTLY zero (w=0 in the recency softmax must zero it).
  3. Numerical correctness — autograd.gradcheck on (E_u, E_v, tok) in double precision,
     so the log-map + LS-fit gradients are verified correct, not merely present.

Inputs are nonzero random and the loss is a plain .sum(), so an accidental exact-zero
gradient (e.g. a candidate sitting exactly on the prediction) cannot masquerade as a
healthy path.
"""
import torch

from tempest_walks.link_pred_head import GeometricVelocityPerWalkAvgHead


def _head(d, pf=False, double=False):
    h = GeometricVelocityPerWalkAvgHead(d_emb=d, use_pair_features=pf)
    return h.double() if double else h


# ── Check 1 ────────────────────────────────────────────────────────────────────
def test_shared_table_gradient_per_role():
    """Gradient reaches embedding.weight through ALL THREE roles separately."""
    torch.manual_seed(0)
    B, C, K, L, d = 3, 4, 3, 5, 8
    # DISJOINT id blocks so each role's grad on embedding.weight is isolated.
    u_idx = torch.arange(B)                                   # source rows   [0,B)
    v_idx = 100 + torch.arange(B * C).view(B, C)             # candidate rows [100,…)
    tok_idx = 1000 + torch.arange(B * K * L).view(B, K, L)   # neighbour rows [1000,…)
    N = 1000 + B * K * L + 1
    emb = torch.nn.Embedding(N, d)

    E_u = emb(u_idx)                                          # [B,d]
    E_v = emb(v_idx)                                          # [B,C,d]
    tok = emb(tok_idx)                                        # [B,K,L,d]
    age = torch.rand(B, K, L) * 3 + 0.1                      # nonzero, modest
    mask = torch.ones(B, K, L, dtype=torch.bool)            # all neighbours valid
    rec = torch.rand(B, C)

    out = _head(d)(tok, age, mask, E_u, E_v, rec)
    assert out.shape == (B, C)
    assert torch.isfinite(out).all()
    out.sum().backward()

    g = emb.weight.grad
    assert g is not None, "embedding.weight got no gradient at all"
    src = g[u_idx].abs().sum().item()
    cand = g[v_idx.reshape(-1)].abs().sum().item()
    neigh = g[tok_idx.reshape(-1)].abs().sum().item()
    print(f"\n[check1] |grad| source={src:.4e} candidate={cand:.4e} neighbour={neigh:.4e}")
    assert src > 0, "SOURCE rows E[u] received no gradient (detached source path?)"
    assert cand > 0, "CANDIDATE rows E[v] received no gradient (detached candidate path?)"
    assert neigh > 0, "NEIGHBOUR rows received no gradient (walk tokens detached/dropped?)"


# ── Check 2 ────────────────────────────────────────────────────────────────────
def test_per_token_masking():
    """Valid tokens get gradient; masked/padded tokens get exactly none."""
    torch.manual_seed(1)
    B, C, K, L, d = 4, 3, 3, 6, 8
    tok = torch.randn(B, K, L, d, requires_grad=True)        # leaf neighbour tensor
    E_u = torch.randn(B, d)
    E_v = torch.randn(B, C, d)
    # every walk keeps token 0 (so the fit is never empty); rest random, with
    # ages nonzero EVERYWHERE so only the mask — not a zero age — can kill a token.
    mask = torch.rand(B, K, L) > 0.5
    mask[:, :, 0] = True
    assert (~mask).any(), "test needs at least one masked token"
    age = torch.rand(B, K, L) * 3 + 0.1
    rec = torch.rand(B, C)

    out = _head(d)(tok, age, mask, E_u, E_v, rec)
    out.sum().backward()

    gper = tok.grad.abs().sum(-1)                            # [B,K,L]
    valid_min = gper[mask].min().item()
    masked_max = gper[~mask].abs().max().item()
    print(f"\n[check2] min|grad| over VALID tokens = {valid_min:.4e}  (want > 0)")
    print(f"[check2] max|grad| over MASKED tokens = {masked_max:.4e}  (want ~0)")
    assert (gper[mask] > 0).all(), "a VALID walk token received zero gradient (dropped?)"
    assert masked_max < 1e-6, "a MASKED/padded token received gradient (inverted mask / leak?)"


# ── Check 3 ────────────────────────────────────────────────────────────────────
def test_gradcheck_logmap_and_fit():
    """Numerical gradient correctness of the log-map + per-walk LS fit (double)."""
    torch.manual_seed(2)
    B, C, K, L, d = 2, 3, 2, 4, 6
    head = _head(d, double=True)
    E_u = torch.randn(B, d, dtype=torch.double, requires_grad=True)
    E_v = torch.randn(B, C, d, dtype=torch.double, requires_grad=True)
    tok = torch.randn(B, K, L, d, dtype=torch.double, requires_grad=True)
    age = (torch.rand(B, K, L, dtype=torch.double) * 3 + 0.1)   # fixed, distinct
    mask = torch.ones(B, K, L, dtype=torch.bool)               # all valid (differentiable)
    rec = torch.rand(B, C, dtype=torch.double)

    def f(eu, ev, t):
        return head(t, age, mask, eu, ev, rec)

    ok = torch.autograd.gradcheck(f, (E_u, E_v, tok), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"\n[check3] gradcheck passed = {ok}")
    assert ok


if __name__ == "__main__":
    test_shared_table_gradient_per_role()
    test_per_token_masking()
    test_gradcheck_logmap_and_fit()
    print("\nALL CHECKS PASSED")
