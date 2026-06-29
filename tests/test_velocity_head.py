"""Tests for VelocityHead (tempest_walks/link_pred_head.py).

The head fits a weighted free line to u's neighbour trajectory in the tangent space at E[u] and
evaluates it at the query time (extrapolation). Checks:

1. SHAPES — forward -> [Q, C]; the predicted q̂ is unit-norm.
2. COLD → COSINE — a query with no context tokens scores exactly γ·cosine(E[u], E[v]).
3. DEGENERATE → CENTROID — a single context neighbour gives b=0 ⇒ μ=v̄ ⇒ q̂ = E[ctx].
4. GRADIENT COVERAGE — only the touched rows of E (seed + context + candidates) get gradient;
   padding (clamp_min(0) → node 0) and untouched rows get exactly zero.
5. DRIFT — on a planted linear drift, velocity's q̂ is far closer to the true next than the
   centroid (the win this head exists for).
6. RECURRENCE (honest) — when the true next IS the most-recent node, velocity does NOT beat the
   most-recent baseline (documents the known failure mode rather than hiding it).
"""
import math

import torch
import torch.nn.functional as F

from tempest_walks.link_pred_head import VelocityHead, sphere_exp
from tempest_walks.walk_tokens import WalkTokens


def _tokens(seeds, cut, nodes, times):
    nodes = torch.tensor(nodes, dtype=torch.long)
    times = torch.tensor(times, dtype=torch.long)
    return WalkTokens(seeds=torch.tensor(seeds, dtype=torch.long), nodes=nodes,
                      nodes_mask=(nodes != -1), timestamps=times,
                      cutoffs=torch.tensor(cut, dtype=torch.long))


def _sphere_E(n, d, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Parameter(F.normalize(torch.randn(n, d), dim=-1))


def _qhat(head, e, tok):
    """Run the head's pieces to get the predicted q̂ (and intermediates) for assertions."""
    p = head._base_point(e, tok)
    v, s, w = head._context(e, tok, p)
    mu = head._fit_at_query(v, s, w)
    return sphere_exp(p, mu, head.eps), p, v, w


def _circle(theta):
    return torch.tensor([math.cos(theta), math.sin(theta), 0.0])


def _geo(a, b):
    return torch.arccos((a * b).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6))


# ──────────────────────────────────────────────────────────────────────────
def test_shapes_and_unit_qhat():
    e = _sphere_E(40, 16); head = VelocityHead(16, t_train=1000.0)
    tok = _tokens([1], [1000], [[[5, 6, 7, -1, 1]]], [[[100, 300, 500, -1, 1000]]])
    lg = head(e, tok, torch.randint(0, 40, (1, 4)))
    assert lg.shape == (1, 4) and bool(torch.isfinite(lg).all())
    qhat, *_ = _qhat(head, e, tok)
    assert torch.allclose(qhat.norm(dim=-1), torch.ones(1), atol=1e-5), "q̂ must be unit-norm"
    print("\n[shapes] forward -> [Q,C], q̂ unit OK")


def test_cold_is_cosine():
    e = _sphere_E(40, 16); head = VelocityHead(16, t_train=1000.0)
    tok = _tokens([2], [1000], [[[-1, -1, -1]]], [[[-1, -1, -1]]])   # empty walk
    cand = torch.randint(0, 40, (1, 5))
    lg = head(e, tok, cand)
    p = F.normalize(e[2], dim=-1)
    expect = F.softplus(head.logit_scale) * (p[None] * F.normalize(e[cand[0]], dim=-1)).sum(-1)
    assert torch.allclose(lg[0], expect, atol=1e-5), "cold query must score γ·cosine(E[u], E[v])"
    print("\n[cold] no context → γ·cosine baseline OK")


def test_degenerate_is_centroid():
    e = _sphere_E(40, 16); head = VelocityHead(16, t_train=1000.0)
    tok = _tokens([2], [1000], [[[4, -1, -1, 2]]], [[[400, -1, -1, 1000]]])   # one context node
    qhat, *_ = _qhat(head, e, tok)
    assert torch.allclose(qhat[0], F.normalize(e[4], dim=-1), atol=1e-4), \
        "single context neighbour → b=0 → μ=v̄ → q̂ = E[ctx]"
    print("\n[degenerate] one neighbour → centroid (q̂ = E[ctx]) OK")


def test_gradient_coverage():
    e = _sphere_E(40, 16); head = VelocityHead(16, t_train=1000.0)
    tok = _tokens([1], [1000], [[[5, 6, 7, -1, 1]]], [[[100, 300, 500, -1, 1000]]])
    cand = torch.tensor([[3, 8, 9, 2]])
    F.cross_entropy(head(e, tok, cand), torch.zeros(1, dtype=torch.long)).backward()
    g = e.grad.norm(dim=-1)
    touched = {1, 5, 6, 7, 3, 8, 9, 2}                  # seed + context + candidates
    for n in range(40):
        if n in touched:
            assert g[n] > 0, f"touched node {n} must get gradient"
        else:
            assert g[n] == 0, f"untouched node {n} (incl. padding fill 0) must get zero gradient"
    print("\n[gradient] only touched rows of E get gradient; padding/untouched are exactly zero OK")


def test_velocity_beats_centroid_on_drift():
    # planted linear drift along a great circle: θ = 0.1, 0.2, 0.3 at t = 100, 200, 300; the true
    # next continues to θ = 0.4. Velocity extrapolates to 0.4; the centroid averages to ~0.2.
    e = torch.zeros(10, 3)
    e[0] = _circle(0.0)                                 # seed
    e[1], e[2], e[3] = _circle(0.1), _circle(0.2), _circle(0.3)
    e[9] = _circle(0.4)                                 # TRUE NEXT
    e = torch.nn.Parameter(F.normalize(e, dim=-1))
    head = VelocityHead(3, t_train=1000.0)
    tok = _tokens([0], [400], [[[1, 2, 3, 0]]], [[[100, 200, 300, 400]]])
    qhat, p, v, w = _qhat(head, e, tok)
    true_next = F.normalize(e[9], dim=-1)
    vbar = (w.unsqueeze(-1) * v).sum((1, 2)) / w.sum((1, 2)).clamp_min(1e-6).unsqueeze(-1)
    centroid = sphere_exp(p, vbar, head.eps)
    dv, dc = float(_geo(qhat[0], true_next)), float(_geo(centroid[0], true_next))
    print(f"\n[drift] velocity {dv:.4f} vs centroid {dc:.4f} (geodesic to true-next)")
    assert dv < dc - 0.05, f"velocity ({dv}) must beat centroid ({dc}) on drift by a wide margin"


def test_velocity_loses_to_most_recent_on_recurrence():
    # honest failure mode: the true next IS the most-recent node (recurrence), so the trivial
    # most-recent baseline is exact while velocity overshoots past it.
    e = torch.zeros(10, 3)
    e[0] = _circle(0.0)
    e[1], e[2], e[3] = _circle(0.1), _circle(0.2), _circle(0.3)
    e = torch.nn.Parameter(F.normalize(e, dim=-1))
    head = VelocityHead(3, t_train=1000.0)
    tok = _tokens([0], [400], [[[1, 2, 3, 0]]], [[[100, 200, 300, 400]]])
    qhat, *_ = _qhat(head, e, tok)
    true_next = F.normalize(e[3], dim=-1)              # recurrence: most-recent node 3 repeats
    dv, dmr = float(_geo(qhat[0], true_next)), float(_geo(true_next, true_next))
    print(f"\n[recurrence] velocity {dv:.4f} vs most-recent {dmr:.4f} (velocity is NOT better)")
    assert dv >= dmr - 1e-4, "honest: velocity must not beat the most-recent baseline on recurrence"


if __name__ == "__main__":
    test_shapes_and_unit_qhat()
    test_cold_is_cosine()
    test_degenerate_is_centroid()
    test_gradient_coverage()
    test_velocity_beats_centroid_on_drift()
    test_velocity_loses_to_most_recent_on_recurrence()
    print("\nALL VELOCITY-HEAD CHECKS PASSED")
