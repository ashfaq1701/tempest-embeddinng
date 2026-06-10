"""DeepSphereSimpleHead correctness — the runaway-proof contract.

Pins the clauses that make the head unable to blow up:
  1. φ output is unit-norm for any input (re-normalised every block).
  2. φ is the identity AND inert at init (ReZero α=0) → no first-step kick.
  3. Bounded score: |logit| ≤ 1/τ (cosine in [-1,1], τ the only scale knob).
  4. Cold-start (all-padding) → finite (learned prior direction).
  5. E is detached inside the head → no gradient reaches E even if E_v/E_w
     arrive with requires_grad; the head's own params do get gradient.
  6. The head can represent / learn a planted positive.
"""
import math

import torch
import torch.nn.functional as F

from tempest_walks.link_pred_head import DeepSphereSimpleHead


def _inputs(B=2, C=9, L=12, d=16, valid=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    E_v = F.normalize(torch.randn(B, C, d, generator=g), dim=-1)
    E_w = F.normalize(torch.randn(B, L, d, generator=g), dim=-1)
    elapsed = torch.rand(B, L, generator=g)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, :valid] = True
    return E_v, E_w, elapsed, mask


def test_phi_output_unit_norm():
    head = DeepSphereSimpleHead(d=16, depth=3)
    y = head.phi(torch.randn(5, 7, 16))                 # arbitrary, NOT unit
    assert torch.allclose(y.norm(dim=-1), torch.ones(5, 7), atol=1e-5)


def test_phi_identity_and_inert_at_init():
    head = DeepSphereSimpleHead(d=16, depth=3)
    E = F.normalize(torch.randn(8, 16), dim=-1)
    assert torch.allclose(head.phi(E), E, atol=1e-6)     # α=0 ⇒ φ = identity
    assert all(float(b.alpha) == 0.0 for b in head.phi_blocks)


def test_score_bounded_by_tau():
    head = DeepSphereSimpleHead(d=16)
    E_v, E_w, el, m = _inputs()
    logit = head(E_v, E_w, el, m)
    tau = head.diagnostics()["tau"]
    assert logit.shape == (E_v.shape[0], E_v.shape[1])
    assert logit.abs().max().item() <= 1.0 / tau + 1e-4   # |cos| ≤ 1


def test_cold_start_no_nan():
    head = DeepSphereSimpleHead(d=16)
    E_v, E_w, el, m = _inputs()
    m2 = m.clone(); m2[0] = False                         # one empty query
    assert torch.isfinite(head(E_v, E_w, el, m2)).all()
    m2[:] = False                                         # whole batch empty
    assert torch.isfinite(head(E_v, E_w, el, m2)).all()


def test_E_detached_inside_head():
    head = DeepSphereSimpleHead(d=16)
    E_v, E_w, el, m = _inputs()
    E_v = E_v.clone().requires_grad_(True)               # even if E wants grad
    E_w = E_w.clone().requires_grad_(True)
    out = head(E_v, E_w, el, m)
    F.cross_entropy(out, torch.zeros(out.shape[0], dtype=torch.long)).backward()
    assert E_v.grad is None and E_w.grad is None          # head detaches E
    g = sum(p.grad.abs().sum().item() for p in head.parameters()
            if p.grad is not None)
    assert g > 0                                          # head params learn


def test_learns_planted_positive():
    torch.manual_seed(0)
    d = 16
    head = DeepSphereSimpleHead(d=d)
    E_v, E_w, el, m = _inputs(B=8, C=101, L=12, d=d, valid=8)
    with torch.no_grad():                                # plant pos = pooled dir
        E_v[:, 0] = head.pool_history(E_w, el, m)
    opt = torch.optim.Adam(head.parameters(), lr=3e-2)
    last = None
    for _ in range(150):
        loss = F.cross_entropy(head(E_v, E_w, el, m),
                               torch.zeros(8, dtype=torch.long))
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    assert last < 0.5 * math.log(101)
