"""Verification suite for LorentzManifold.

Every claim made in the module's docstrings is checked here. Run with:

    python -m pytest tests/test_lorentz.py -v
"""

import math

import geoopt
import pytest
import torch

from link_property_prediction.lorentz import LorentzManifold

torch.manual_seed(0)
KS = [1.0, 0.37, 2.7]
DIM = 6
TOL = 1e-11


# ----------------------------------------------------------------------
# helpers: the ambient formulas exactly as printed in the paper
# ----------------------------------------------------------------------
def amb_tangent(m, x, u):
    """Ambient lift of a tangent vector: v0 = (x.u)/x0 fixes the time part."""
    v0 = (x * u).sum(-1, keepdim=True) / m._x0(x)
    return torch.cat([v0, u.expand(*v0.shape[:-1], u.shape[-1])], -1)


def lip(a, b):
    """Eq. 2."""
    return -a[..., 0] * b[..., 0] + (a[..., 1:] * b[..., 1:]).sum(-1)


def lift(xp, k):
    """Eq. 6."""
    x0 = torch.sqrt(k + (xp * xp).sum(-1, keepdim=True))
    return torch.cat([x0, xp], -1)


def paper_dist(X, Y, k):
    """Eq. 5, verbatim: sqrt(k) arcosh(-<x,y>_L / k)."""
    return math.sqrt(k) * torch.acosh((-lip(X, Y) / k).clamp_min(1.0))


def paper_expmap(X, V, k):
    """Eq. 9, generalised to curvature -1/k."""
    nrm = torch.sqrt(lip(V, V)).unsqueeze(-1)
    sk = math.sqrt(k)
    return torch.cosh(nrm / sk) * X + sk * torch.sinh(nrm / sk) * V / nrm


def rand_points(n, dim=DIM, scale=1.0, dtype=torch.float64):
    return torch.randn(n, dim, dtype=dtype) * scale


def rand_tangent(m, x, scale=1.0):
    """Any vector in R^n is tangent in this chart."""
    return torch.randn_like(x) * scale


# ======================================================================
# 1. the chart reproduces the paper's ambient definitions
# ======================================================================
@pytest.mark.parametrize("k", KS)
def test_lift_lands_on_manifold_at_every_radius(k):
    m = LorentzManifold(k=k)
    eps = torch.finfo(torch.float64).eps
    for scale in (1e-6, 1.0, 1e3, 1e6, 1e9):
        X = m._lift(rand_points(64, scale=scale))
        resid = (lip(X, X) + k).abs()
        assert (resid / (X[..., 0] ** 2)).max() < 8 * eps


@pytest.mark.parametrize("k", KS)
def test_inner_is_pullback_of_ambient_form(k):
    m = LorentzManifold(k=k)
    x = rand_points(32, scale=3.0)
    u = rand_tangent(32, x)
    V = amb_tangent(m, x, u)
    assert torch.allclose(m.inner(x, u), lip(V, V), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("k", KS)
def test_lift_tangent_is_tangent(k):
    m = LorentzManifold(k=k)
    x = rand_points(32, scale=3.0)
    X, V = m._lift(x), amb_tangent(m, x, rand_tangent(32, x))
    assert lip(X, V).abs().max() < 1e-10


@pytest.mark.parametrize("k", KS)
def test_inner_positive_definite(k):
    m = LorentzManifold(k=k)
    x = rand_points(64, scale=50.0)
    assert m.inner(x, rand_tangent(64, x)).min() > 0


@pytest.mark.parametrize("k", KS)
def test_egrad2rgrad_matches_papers_ambient_recipe(k):
    """THE CORE CLAIM. Paper: h = g_l^-1 grad f, then proj_x(h).
    Chart: g^-1 grad, after the chain rule through Eq. 6. Must agree."""
    m = LorentzManifold(k=k)
    x = rand_points(32, scale=4.0)
    X = m._lift(x)
    Ea = torch.randn(32, DIM + 1, dtype=torch.float64)

    h = Ea.clone()
    h[..., 0] *= -1
    paper = h + lip(X, h).unsqueeze(-1) * X / k

    x0 = X[..., :1]
    Ec = Ea[..., 1:] + Ea[..., :1] * x / x0
    chart = m.egrad2rgrad(x, Ec)

    assert torch.allclose(paper[..., 1:], chart, rtol=1e-9, atol=1e-9)


# ======================================================================
# 2. exponential map
# ======================================================================
@pytest.mark.parametrize("k", KS)
def test_expmap_matches_equation_9(k):
    m = LorentzManifold(k=k)
    x = rand_points(32, scale=3.0)
    u = rand_tangent(32, x, scale=0.7)
    mine = m._lift(m.expmap(x, u))
    theirs = paper_expmap(m._lift(x), amb_tangent(m, x, u), k)
    assert torch.allclose(mine, theirs, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("k", KS)
def test_expmap_stays_on_manifold(k):
    m = LorentzManifold(k=k)
    for scale in (1.0, 1e3, 1e6):
        x = rand_points(32, scale=scale)
        y = m.expmap(x, rand_tangent(32, x, scale=2.0))
        Y = m._lift(y)
        assert torch.isfinite(Y).all()
        assert ((lip(Y, Y) + k).abs() / (Y[..., 0] ** 2)).max() < 8 * torch.finfo(torch.float64).eps


@pytest.mark.parametrize("k", KS)
def test_expmap_step_length_is_tangent_norm(k):
    m = LorentzManifold(k=k)
    x = rand_points(32, scale=3.0)
    u = rand_tangent(32, x, scale=1.3)
    assert torch.allclose(m.dist(x, m.expmap(x, u)), m.inner(x, u).sqrt(),
                          rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("k", KS)
def test_expmap_zero_is_identity(k):
    m = LorentzManifold(k=k)
    x = rand_points(16, scale=3.0)
    z = torch.zeros_like(x)
    assert torch.allclose(m.expmap(x, z), x, rtol=0, atol=1e-12)


@pytest.mark.parametrize("k", KS)
def test_expmap_small_norm_is_euclidean(k):
    m = LorentzManifold(k=k)
    x = rand_points(16, scale=1.0)
    u = rand_tangent(16, x) * 1e-9
    assert torch.allclose(m.expmap(x, u), x + u, rtol=0, atol=1e-15)


# ======================================================================
# 3. logarithmic map
# ======================================================================
@pytest.mark.parametrize("k", KS)
def test_logmap_inverts_expmap(k):
    m = LorentzManifold(k=k)
    x = rand_points(32, scale=3.0)
    u = rand_tangent(32, x, scale=1.1)
    y = m.expmap(x, u)
    assert torch.allclose(m.logmap(x, y), u, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("k", KS)
def test_expmap_inverts_logmap(k):
    m = LorentzManifold(k=k)
    x, y = rand_points(32, scale=3.0), rand_points(32, scale=3.0)
    assert torch.allclose(m.expmap(x, m.logmap(x, y)), y, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("k", KS)
def test_logmap_norm_equals_distance(k):
    m = LorentzManifold(k=k)
    x, y = rand_points(32, scale=3.0), rand_points(32, scale=3.0)
    assert torch.allclose(m.inner(x, m.logmap(x, y)).sqrt(), m.dist(x, y),
                          rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("k", KS)
def test_logmap_of_self_is_zero(k):
    m = LorentzManifold(k=k)
    x = rand_points(16, scale=3.0)
    assert m.logmap(x, x).abs().max() < 1e-12


# ======================================================================
# 4. distance
# ======================================================================
@pytest.mark.parametrize("k", KS)
def test_dist_matches_equation_5_verbatim(k):
    m = LorentzManifold(k=k)
    x, y = rand_points(64, scale=3.0), rand_points(64, scale=3.0)
    assert torch.allclose(m.dist(x, y), paper_dist(m._lift(x), m._lift(y), k),
                          rtol=1e-10, atol=1e-11)


def test_dist_matches_geoopt_lorentz():
    for k in KS:
        m = LorentzManifold(k=k)
        g = geoopt.Lorentz(k=torch.tensor(k, dtype=torch.float64))
        x, y = rand_points(64, scale=3.0), rand_points(64, scale=3.0)
        assert torch.allclose(m.dist(x, y), g.dist(m._lift(x), m._lift(y)),
                              rtol=1e-12, atol=1e-13)


def test_geoopt_downcasts_a_python_float_curvature_to_float32():
    assert geoopt.Lorentz(k=2.7).k.dtype == torch.float32
    assert isinstance(LorentzManifold(k=2.7).k, float)
    assert LorentzManifold(k=2.7).k == 2.7

    m = LorentzManifold(k=2.7)
    g32 = geoopt.Lorentz(k=2.7)
    g64 = geoopt.Lorentz(k=torch.tensor(2.7, dtype=torch.float64))
    x, y = rand_points(64, scale=3.0), rand_points(64, scale=3.0)
    X, Y = m._lift(x), m._lift(y)
    gap32 = (m.dist(x, y) - g32.dist(X, Y)).abs().max()
    gap64 = (m.dist(x, y) - g64.dist(X, Y)).abs().max()
    assert gap32 > 1e-8, "geoopt may have fixed the float32 k downcast"
    assert gap64 < 1e-12
    m1, g1 = LorentzManifold(k=1.0), geoopt.Lorentz(k=1.0)
    assert (m1.dist(x, y) - g1.dist(m1._lift(x), m1._lift(y))).abs().max() < 1e-13


@pytest.mark.parametrize("k", KS)
def test_dist_is_a_metric(k):
    m = LorentzManifold(k=k)
    x, y, z = (rand_points(64, scale=2.0) for _ in range(3))
    assert (m.dist(x, y) >= 0).all()
    assert torch.allclose(m.dist(x, y), m.dist(y, x), rtol=1e-12, atol=1e-12)
    assert (m.dist(x, z) <= m.dist(x, y) + m.dist(y, z) + 1e-9).all()
    assert m.dist(x, x).abs().max() == 0.0


@pytest.mark.parametrize("k", KS)
def test_dist0_matches_dist_to_origin(k):
    m = LorentzManifold(k=k)
    x = rand_points(64, scale=100.0)
    assert torch.allclose(m.dist0(x), m.dist(x, torch.zeros_like(x)),
                          rtol=1e-10, atol=1e-11)


def test_dist_stays_nonnegative_where_geoopt_float32_does_not():
    g = geoopt.Lorentz(k=1.0)
    m = LorentzManifold(k=1.0)
    for scale, expect_neg in ((1.0, True), (139.0, True), (1390.0, True)):
        xp = rand_points(200, dim=3, scale=scale)
        X32 = lift(xp, torch.tensor(1.0, dtype=torch.float64)).float()
        d_geo = g.dist(X32, X32)
        assert (d_geo < 0).any() == expect_neg, scale
        d_mine = m.dist(xp.float(), xp.float())
        assert (d_mine >= 0).all()
        assert torch.isfinite(d_mine).all()


# ======================================================================
# 5. parallel transport
# ======================================================================
@pytest.mark.parametrize("k", KS)
def test_transp_is_tangent_and_isometric(k):
    m = LorentzManifold(k=k)
    x, y = rand_points(32, scale=3.0), rand_points(32, scale=3.0)
    v = rand_tangent(32, x)
    pv = m.transp(x, y, v)
    Y, PV = m._lift(y), amb_tangent(m, y, pv)
    assert lip(Y, PV).abs().max() < 1e-9
    assert torch.allclose(m.inner(y, pv), m.inner(x, v), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("k", KS)
def test_broadcasting_shapes(k):
    m = LorentzManifold(k=k)
    cases = [((5, DIM), (5, DIM)), ((3, 5, DIM), (3, 5, DIM)),
             ((1, 5, DIM), (7, 5, DIM)), ((5, DIM), (1, DIM)),
             ((1, DIM), (5, DIM))]
    for xs, ys in cases:
        a = torch.randn(*xs, dtype=torch.float64)
        b = torch.randn(*ys, dtype=torch.float64)
        v = torch.randn(*ys, dtype=torch.float64)
        batch = torch.broadcast_shapes(xs[:-1], ys[:-1])
        assert m.dist(a, b).shape == batch
        assert m.logmap(a, b).shape == batch + (DIM,)
        assert m.transp(a, b, v).shape == batch + (DIM,)
        assert m.expmap(a, v).shape == batch + (DIM,)
        assert torch.isfinite(m.transp(a, b, v)).all()


@pytest.mark.parametrize("k", KS)
def test_transp_to_self_is_identity(k):
    m = LorentzManifold(k=k)
    x = rand_points(16, scale=3.0)
    v = rand_tangent(16, x)
    assert torch.allclose(m.transp(x, x, v), v, rtol=1e-11, atol=1e-12)


# ======================================================================
# 6. midpoint
# ======================================================================
@pytest.mark.parametrize("k", KS)
def test_midpoint_on_manifold_and_scale_invariant(k):
    m = LorentzManifold(k=k)
    x = rand_points(5 * 8, scale=2.0).reshape(8, 5, DIM)
    w = torch.rand(8, 5, dtype=torch.float64)
    w_norm = w / w.sum(-1, keepdim=True)

    mu = m.midpoint(x, w_norm)
    MU = m._lift(mu)
    assert torch.allclose(lip(MU, MU), torch.full((8,), -k, dtype=torch.float64),
                          rtol=1e-11, atol=1e-11)
    assert torch.allclose(m.midpoint(x, w), mu, rtol=1e-10, atol=1e-11)
    assert torch.allclose(m.midpoint(x, 0.01 * w), mu, rtol=1e-10, atol=1e-11)


def test_midpoint_of_one_point_is_that_point():
    for k in KS:
        m = LorentzManifold(k=k)
        x = rand_points(4, scale=3.0).reshape(4, 1, DIM)
        w = torch.ones(4, 1, dtype=torch.float64)
        assert torch.allclose(m.midpoint(x, w), x.squeeze(1), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("k", KS)
def test_poincare_round_trip(k):
    m = LorentzManifold(k=k)
    x = rand_points(64, scale=3.0)
    u = m.to_poincare(x)
    assert (u.norm(dim=-1) < 1).all()
    assert torch.allclose(m.from_poincare(u), x, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("k", KS)
def test_poincare_map_is_an_isometry(k):
    m = LorentzManifold(k=k)
    x, y = rand_points(64, scale=2.0), rand_points(64, scale=2.0)
    u, v = m.to_poincare(x), m.to_poincare(y)
    dp = torch.acosh(1 + 2 * (u - v).pow(2).sum(-1)
                     / ((1 - u.pow(2).sum(-1)) * (1 - v.pow(2).sum(-1))))
    assert torch.allclose(math.sqrt(k) * dp, m.dist(x, y), rtol=1e-8, atol=1e-9)


# ======================================================================
# 8. gradient safety -- the torch.where NaN trap
# ======================================================================
@pytest.mark.parametrize("k", KS)
def test_expmap_gradient_finite_at_zero_tangent(k):
    m = LorentzManifold(k=k)
    x = rand_points(8, scale=2.0).requires_grad_(True)
    u = torch.zeros(8, DIM, dtype=torch.float64, requires_grad=True)
    m.expmap(x, u).sum().backward()
    assert torch.isfinite(x.grad).all(), "NaN gradient through expmap at ||v||=0"
    assert torch.isfinite(u.grad).all()


@pytest.mark.parametrize("k", KS)
def test_logmap_gradient_finite_at_coincident_points(k):
    m = LorentzManifold(k=k)
    x = rand_points(8, scale=2.0).requires_grad_(True)
    y = x.detach().clone().requires_grad_(True)
    m.logmap(x, y).sum().backward()
    assert torch.isfinite(x.grad).all(), "NaN gradient through logmap at x == y"
    assert torch.isfinite(y.grad).all()


@pytest.mark.parametrize("k", KS)
def test_logmap_gradient_finite_in_mixed_batch(k):
    m = LorentzManifold(k=k)
    x = rand_points(8, scale=2.0)
    y = rand_points(8, scale=2.0)
    y[3] = x[3]
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    m.logmap(x, y).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()


def test_output_masking_would_nan_but_input_masking_does_not():
    TINY = 1e-30

    def output_masked(w):
        num = 2.0 * torch.asinh(torch.sqrt(w / 2.0))
        den = torch.sqrt(w * (w + 2.0))
        return torch.where(w > TINY, num / den.clamp_min(TINY), torch.ones_like(w))

    def input_masked(w):
        safe = w > TINY
        ws = torch.where(safe, w, torch.ones_like(w))
        num = 2.0 * torch.asinh(torch.sqrt(ws * 0.5))
        den = torch.sqrt(ws * (ws + 2.0))
        return torch.where(safe, num / den, 1.0 - w / 3.0)

    grads = {}
    for name, fn in (("output", output_masked), ("input", input_masked)):
        w = torch.zeros(4, 1, dtype=torch.float64, requires_grad=True)
        val = fn(w)
        val.sum().backward()
        assert torch.allclose(val, torch.ones_like(val))
        grads[name] = w.grad.clone()

    assert torch.isnan(grads["output"]).all(), "expected the known where() trap"
    assert torch.isfinite(grads["input"]).all()
    assert torch.allclose(grads["input"], torch.full_like(grads["input"], -1 / 3))


@pytest.mark.parametrize("k", KS)
def test_expmap_jacobian_at_zero_is_the_identity(k):
    m = LorentzManifold(k=k)
    x = rand_points(1, scale=2.0)
    J = torch.autograd.functional.jacobian(
        lambda u: m.expmap(x, u).squeeze(0),
        torch.zeros(1, DIM, dtype=torch.float64),
    ).reshape(DIM, DIM)
    assert torch.allclose(J, torch.eye(DIM, dtype=torch.float64), rtol=0, atol=1e-12)


@pytest.mark.parametrize("k", KS)
def test_gradients_finite_on_ordinary_input(k):
    m = LorentzManifold(k=k)
    x = rand_points(16, scale=3.0).requires_grad_(True)
    y = rand_points(16, scale=3.0).requires_grad_(True)
    loss = (m.dist(x, y).sum() + m.logmap(x, y).sum()
            + m.expmap(x, m.egrad2rgrad(x, y)).sum() + m.dist0(x).sum())
    loss.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()


# ======================================================================
# 9. dtype and device robustness -- the reason k is not a buffer
# ======================================================================
def test_k_survives_module_float():
    m = LorentzManifold(k=2.7)
    net = torch.nn.Module()
    net.manifold = m
    net.p = geoopt.ManifoldParameter(m.random(8, DIM, dtype=torch.float64), manifold=m)
    net.float()
    net.half()
    net.double()
    assert m.k == 2.7 and isinstance(m.k, float)
    assert m._sqrt_k == math.sqrt(2.7)


def test_k_not_in_state_dict_is_documented_tradeoff():
    m = LorentzManifold(k=2.7)
    net = torch.nn.Module()
    net.manifold = m
    assert "manifold.k" not in net.state_dict()


# ======================================================================
# 10. geoopt integration -- optimizers must run unmodified
# ======================================================================
@pytest.mark.parametrize("k", KS)
def test_manifold_is_instantiable_and_complete(k):
    m = LorentzManifold(k=k)
    assert not getattr(type(m), "__abstractmethods__", frozenset())
    x, y = rand_points(4, scale=1.0), rand_points(4, scale=1.0)
    for name in ("retr", "expmap", "logmap", "transp", "projx", "proju",
                 "egrad2rgrad", "inner", "dist"):
        assert hasattr(m, name)
    m.logmap(x, y)


@pytest.mark.parametrize("k", KS)
def test_base_class_composites_work(k):
    m = LorentzManifold(k=k)
    x = rand_points(8, scale=2.0)
    u = rand_tangent(8, x, scale=0.1)
    v = rand_tangent(8, x)
    y, pv = m.retr_transp(x, u, v)
    assert torch.allclose(y, m.expmap(x, u), rtol=1e-12, atol=1e-12)
    assert torch.allclose(pv, m.transp(x, y, v), rtol=1e-12, atol=1e-12)
    ci = m.component_inner(x, u)
    assert torch.isfinite(ci).all()


@pytest.mark.parametrize("opt_cls", [geoopt.optim.RiemannianSGD,
                                     geoopt.optim.RiemannianAdam])
@pytest.mark.parametrize("k", KS)
def test_optimizer_reduces_a_hyperbolic_loss(opt_cls, k):
    torch.manual_seed(7)
    m = LorentzManifold(k=k)
    p = geoopt.ManifoldParameter(m.random(32, DIM, dtype=torch.float64), manifold=m)
    target = m.expmap(torch.zeros(DIM, dtype=torch.float64),
                      torch.randn(DIM, dtype=torch.float64))
    lr = 0.2 if opt_cls is geoopt.optim.RiemannianSGD else 0.02
    opt = opt_cls([p], lr=lr, stabilize=10)

    first = last = None
    for step in range(200):
        opt.zero_grad()
        loss = m.dist(p, target.expand_as(p)).pow(2).mean()
        loss.backward()
        opt.step()
        assert torch.isfinite(p).all(), f"non-finite iterate at step {step}"
        if first is None:
            first = float(loss)
        last = float(loss)

    assert last < first * 0.02, f"{opt_cls.__name__}: {first:.4f} -> {last:.4f}"
    P = m._lift(p.detach())
    assert ((lip(P, P) + k).abs() / (P[..., 0] ** 2)).max() < 8 * torch.finfo(torch.float64).eps


@pytest.mark.parametrize("k", KS)
def test_momentum_path_uses_transport(k):
    torch.manual_seed(11)
    m = LorentzManifold(k=k)
    p = geoopt.ManifoldParameter(m.random(16, DIM, dtype=torch.float64), manifold=m)
    opt = geoopt.optim.RiemannianSGD([p], lr=5e-2, momentum=0.9, stabilize=5)
    target = torch.randn(DIM, dtype=torch.float64)
    for _ in range(60):
        opt.zero_grad()
        m.dist(p, target.expand_as(p)).pow(2).mean().backward()
        opt.step()
        assert torch.isfinite(p).all()


# ======================================================================
# 11. gradcheck and end-to-end
# ======================================================================
@pytest.mark.parametrize("k", KS)
def test_gradcheck_against_numerical_jacobians(k):
    from torch.autograd import gradcheck
    m = LorentzManifold(k=k)
    x = (rand_points(3, dim=4, scale=1.5)).requires_grad_(True)
    y = (rand_points(3, dim=4, scale=1.5)).requires_grad_(True)
    u = (rand_points(3, dim=4, scale=0.4)).requires_grad_(True)
    for fn, args in [
        (lambda a, b: m.dist(a, b), (x, y)),
        (lambda a: m.dist0(a), (x,)),
        (lambda a, b: m.logmap(a, b), (x, y)),
        (lambda a, c: m.expmap(a, c), (x, u)),
        (lambda a, b, c: m.transp(a, b, c), (x, y, u)),
        (lambda a, c: m.egrad2rgrad(a, c), (x, u)),
        (lambda a, c: m.inner(a, c), (x, u)),
        (lambda a: m.to_poincare(a), (x,)),
    ]:
        assert gradcheck(fn, args, eps=1e-6, atol=1e-8, rtol=1e-6)


def test_float32_end_to_end_training_stays_finite():
    """Includes exact self-pairs, which is how the coincident-point failure
    actually appeared: Patent has the shortest walks in the suite, so
    near-empty bags pool to identical points and dist(x, x) is routine. Before
    _safe_sqrt this died within four steps."""
    torch.manual_seed(1)
    m = LorentzManifold(k=1.0)
    p = geoopt.ManifoldParameter(m.random(64, 8, dtype=torch.float32), manifold=m)
    target = torch.randn(8, dtype=torch.float32) * 3
    opt = geoopt.optim.RiemannianAdam([p], lr=0.05, stabilize=10)
    first = last = None
    for _ in range(300):
        opt.zero_grad()
        q = p.clone()
        q[:16] = p[:16]                                   # 25% exact self-pairs
        loss = (m.dist(p, target.expand_as(p)).pow(2).mean()
                + m.dist(p, q).pow(2).mean())
        loss.backward()
        opt.step()
        assert torch.isfinite(p).all()
        if first is None:
            first = float(loss)
        last = float(loss)
    assert p.dtype == torch.float32
    assert last < first * 0.02


@pytest.mark.parametrize("k", KS)
def test_dist0_is_accurate_at_the_papers_init_radius(k):
    m = LorentzManifold(k=k)
    x32 = (rand_points(4096, scale=1e-3)).float()
    exact = m.dist0(x32.double())

    def naive(xx):
        w = (m._x0(xx) / m._sqrt_k - 1.0).clamp_min(0.0)
        return (2.0 * m._sqrt_k * torch.asinh(torch.sqrt(w / 2))).squeeze(-1)

    rel = lambda a: float(((a.double() - exact) / exact.clamp_min(1e-30)).abs().max())
    assert rel(naive(x32)) > 1e-3
    assert rel(m.dist0(x32)) < 1e-5


@pytest.mark.parametrize("k", KS)
def test_inner_does_not_regress_on_random_tangents(k):
    m = LorentzManifold(k=k)
    for scale in (1.0, 1e4, 1e8):
        x = (rand_points(1024, scale=scale)).float()
        u = rand_points(1024, scale=1.0).float()
        exact = m.inner(x.double(), u.double(), keepdim=True)
        rel = float(((m.inner(x, u, keepdim=True).double() - exact) / exact.abs()).abs().max())
        assert rel < 1e-5


@pytest.mark.parametrize("k", KS)
def test_nearby_pairs_at_radius_survive_float32(k):
    m = LorentzManifold(k=k)

    def naive_gap(xx, yy):
        d = xx - yy
        d0 = m._x0(xx) - m._x0(yy)
        return (((d * d).sum(-1, keepdim=True)) - d0 * d0) * m._half_inv_k

    def naive_dist(xx, yy):
        return (2 * m._sqrt_k * torch.asinh(
            torch.sqrt(naive_gap(xx, yy).clamp_min(0) * 0.5))).squeeze(-1)

    for scale, sep in ((1e2, 1e-4), (1e4, 1e-4), (1e2, 1e-6)):
        x64 = rand_points(2000, scale=scale)
        y64 = x64 + torch.randn_like(x64) * sep
        x, y = x64.float(), y64.float()
        exact = m.dist(x.double(), y.double())
        den = exact.abs().clamp_min(1e-30)
        e_naive = float(((naive_dist(x, y).double() - exact) / den).abs().max())
        e_mine = float(((m.dist(x, y).double() - exact) / den).abs().max())
        assert e_naive > 1e-2, (scale, sep, e_naive)
        assert e_mine < 1e-5, (scale, sep, e_mine)
        distinct = (x != y).any(-1)
        if distinct.any():
            assert (m.dist(x, y)[distinct] > 0).all(), (scale, sep)


@pytest.mark.parametrize("k", KS)
def test_transp_stays_isometric_in_float32(k):
    m = LorentzManifold(k=k)
    for scale in (1.0, 1e2, 1e4):
        x = rand_points(512, scale=scale).float()
        y = rand_points(512, scale=scale).float()
        v = rand_points(512, scale=1.0).float()
        pv = m.transp(x, y, v)
        ratio = m.inner(y.double(), pv.double()) / m.inner(x.double(), v.double())
        assert (ratio - 1).abs().max() < 1e-4


def test_random_matches_paper_init():
    m = LorentzManifold(k=1.0)
    x = m.random(4096, DIM, dtype=torch.float64)
    assert isinstance(x, geoopt.ManifoldTensor)
    assert x.abs().max() <= 1e-3
    assert m.dist0(x).max() < 1e-2


def test_origin_is_the_vertex():
    for k in KS:
        m = LorentzManifold(k=k)
        o = m.origin(3, DIM, dtype=torch.float64)
        assert (o == 0).all()
        assert m.dist0(o).abs().max() == 0.0
        assert torch.allclose(m._lift(o)[..., 0].squeeze(-1),
                              torch.full((3,), math.sqrt(k), dtype=torch.float64))


# ======================================================================
# 13. dtype passthrough
# ======================================================================
DTYPES = [torch.float64, torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("k", KS)
def test_every_op_is_dtype_passthrough(dt, k):
    """No internal upcast, no dtype branch: what goes in comes out."""
    m = LorentzManifold(k=k)
    x = rand_points(16, scale=2.0).to(dt)
    y = rand_points(16, scale=2.0).to(dt)
    u = rand_points(16, scale=0.3).to(dt)
    bag = rand_points(4 * 5, scale=2.0).reshape(4, 5, DIM).to(dt)
    wts = torch.rand(4, 5).to(dt)
    for name, out in [
        ("dist", m.dist(x, y)), ("dist0", m.dist0(x)), ("logmap", m.logmap(x, y)),
        ("expmap", m.expmap(x, u)), ("transp", m.transp(x, y, u)),
        ("egrad2rgrad", m.egrad2rgrad(x, u)), ("inner", m.inner(x, u)),
        ("to_poincare", m.to_poincare(x)), ("projx", m.projx(x)),
        ("midpoint", m.midpoint(bag, wts)),
    ]:
        assert out.dtype == dt, f"{name} returned {out.dtype} for {dt} input"
        assert torch.isfinite(out).all(), f"{name} non-finite in {dt}"


@pytest.mark.parametrize("dt", DTYPES)
def test_accuracy_lands_at_the_dtypes_own_resolution(dt, k=1.0):
    m = LorentzManifold(k=k)
    x = rand_points(500, scale=3.0)
    y = rand_points(500, scale=3.0)
    exact = m.dist(x.to(dt).double(), y.to(dt).double())
    got = m.dist(x.to(dt), y.to(dt)).double()
    rel = ((got - exact).abs() / exact.abs().clamp_min(1e-6)).max()
    assert rel < 20 * torch.finfo(dt).eps, (dt, float(rel))


def test_no_hardcoded_epsilon_breaks_in_float16():
    """A fixed 1e-30 floor underflows to exactly 0.0 in float16, and 1/0 is
    inf, so the floor must come from finfo(dtype).

    Exercises the two sites that actually USE the floor -- midpoint's rsqrt and
    from_poincare's division. An earlier version of this test called inner,
    expmap, dist0 and logmap, none of which touch _tiny, so replacing it with a
    hardcoded 1e-30 passed."""
    assert float(torch.tensor(1e-30, dtype=torch.float16)) == 0.0
    assert torch.finfo(torch.float16).tiny > 0
    m = LorentzManifold(k=1.0)

    # midpoint with a degenerate bag: -<s,s>_L/k -> 0, so rsqrt hits the floor
    bag = torch.zeros(2, 4, DIM, dtype=torch.float16)
    for w in (torch.zeros(2, 4, dtype=torch.float16),
              torch.rand(2, 4, dtype=torch.float16)):
        assert torch.isfinite(m.midpoint(bag, w)).all()

    # from_poincare at and beyond the boundary: 1 - ||u||^2 -> 0
    for val in (0.0, 0.999, 1.0, 1.5):
        u = torch.full((4, DIM), val / DIM ** 0.5, dtype=torch.float16)
        assert torch.isfinite(m.from_poincare(u)).all(), val


@pytest.mark.parametrize("k", KS)
def test_inner_needs_no_division_floor(k):
    m = LorentzManifold(k=k)
    at_origin = torch.zeros(4, DIM, dtype=torch.float64)
    u = rand_points(4, scale=1.0)
    assert torch.allclose(m.inner(at_origin, u), (u * u).sum(-1), rtol=0, atol=1e-14)

    x = (rand_points(500, scale=1e-3)).half()
    uu = rand_points(500, scale=1.0).half()
    exact = m.inner(x.double(), uu.double(), keepdim=True)
    got = m.inner(x, uu, keepdim=True).double()
    rel = float(((got - exact) / exact.abs()).abs().max())
    assert rel < 20 * torch.finfo(torch.float16).eps, rel


@pytest.mark.parametrize("k", KS)
def test_inner_beats_the_printed_form_at_both_widths(k):
    m = LorentzManifold(k=k)

    def printed(x, u):
        return ((u * u).sum(-1, keepdim=True)
                - (x * u).sum(-1, keepdim=True) ** 2
                / (m.k + (x * x).sum(-1, keepdim=True)))

    for dt, scale, bound in ((torch.float32, 1e2, 1e-5), (torch.float32, 1e4, 1e-3),
                             (torch.float64, 1e4, 1e-12), (torch.float64, 1e6, 1e-12)):
        x = (rand_points(1000, scale=scale)).to(dt)
        u = (x.double() / x.double().norm(dim=-1, keepdim=True)).to(dt)
        x0sq = m.k + (x.double() * x.double()).sum(-1, keepdim=True)
        q = (x.double() * u.double()).sum(-1, keepdim=True) / x0sq
        exact = ((u.double() - q * x.double()) ** 2).sum(-1, keepdim=True) + q * q * m.k
        rel = lambda a: float(((a.double() - exact) / exact.abs()).abs().max())
        assert rel(m.inner(x, u, keepdim=True)) < bound, (dt, scale)
        assert rel(printed(x, u)) > rel(m.inner(x, u, keepdim=True))


def test_midpoint_float32_loss_is_small_and_stays_on_manifold():
    m = LorentzManifold(k=1.0)
    for scale in (1.0, 1e2, 1e4, 1e6):
        for T in (8, 64):
            x = (rand_points(64 * T, scale=scale).reshape(64, T, DIM)).float()
            w = torch.rand(64, T)
            w = w / w.sum(-1, keepdim=True)
            a = m.midpoint(x, w)
            b = m.midpoint(x.double(), w.double())
            assert float(((a.double() - b).abs() / b.abs().clamp_min(1e-12)).max()) < 5e-3
            P = m._lift(a.double())
            assert ((lip(P, P) + 1.0).abs() / (P[..., 0] ** 2)).max() < 1e-6


# ======================================================================
# 14. degenerate-input gradients
#
# The class of bug this section exists for: sqrt(t) has an INFINITE derivative
# at t == 0, so any operation whose sqrt argument can be exactly zero returns
# a finite value with a NaN gradient, and one such pair poisons the whole
# batch. Zero is reachable from real data here -- coincident points, a zero
# tangent vector, the origin -- so it is the common case, not a corner.
#
# This is a SWEEP rather than a handful of cases on purpose. dist() shipped
# broken while logmap() and expmap() were already guarded, because the earlier
# suite tested those two and not dist. Enumerating (operation, degenerate
# input) for every public op is what makes that omission impossible.
# ======================================================================
def _deg_cases(k=1.0):
    """(label, callable, tensors-that-must-have-finite-grads) at each op's
    degenerate input."""
    m = LorentzManifold(k=k)
    D = 8
    def pt(s=1.0):
        return (torch.randn(6, D, dtype=torch.float64) * s).requires_grad_(True)
    out = []

    x = pt(); y = x.detach().clone().requires_grad_(True)
    out.append(("dist(x, x)", lambda: m.dist(x, y), (x, y)))

    z = torch.zeros(6, D, dtype=torch.float64, requires_grad=True)
    out.append(("dist0(origin)", lambda: m.dist0(z), (z,)))

    a = pt(); b = a.detach().clone().requires_grad_(True)
    out.append(("logmap(x, x)", lambda: m.logmap(a, b), (a, b)))

    c = pt(); u0 = torch.zeros(6, D, dtype=torch.float64, requires_grad=True)
    out.append(("expmap(x, 0)", lambda: m.expmap(c, u0), (c, u0)))

    e = pt(); f = e.detach().clone().requires_grad_(True); v = pt()
    out.append(("transp(x, x, v)", lambda: m.transp(e, f, v), (e, f, v)))

    g = torch.zeros(6, D, dtype=torch.float64, requires_grad=True); h = pt()
    out.append(("inner(origin, u)", lambda: m.inner(g, h), (g, h)))

    i = pt(); j = torch.zeros(6, D, dtype=torch.float64, requires_grad=True)
    out.append(("inner(x, 0)", lambda: m.inner(i, j), (i, j)))

    bag = (torch.randn(4, 5, D, dtype=torch.float64)).requires_grad_(True)
    zw = torch.zeros(4, 5, dtype=torch.float64, requires_grad=True)
    out.append(("midpoint(bag, zeros)", lambda: m.midpoint(bag, zw), (bag, zw)))

    o1 = torch.zeros(6, D, dtype=torch.float64, requires_grad=True)
    out.append(("to_poincare(origin)", lambda: m.to_poincare(o1), (o1,)))
    o2 = torch.zeros(6, D, dtype=torch.float64, requires_grad=True)
    out.append(("from_poincare(origin)", lambda: m.from_poincare(o2), (o2,)))
    return out


@pytest.mark.parametrize("k", KS)
def test_no_operation_nans_its_gradient_at_a_degenerate_input(k):
    for label, fn, tensors in _deg_cases(k):
        fn().sum().backward()
        for t in tensors:
            assert t.grad is not None, label
            assert torch.isfinite(t.grad).all(), f"{label}: NaN/inf gradient"


@pytest.mark.parametrize("k", KS)
def test_dist_self_pair_is_zero_with_zero_gradient(k):
    """The specific regression: dist(x, x) used to return 0.0 with a NaN
    gradient. Zero is the correct subgradient -- a self-distance is already at
    the function's minimum."""
    m = LorentzManifold(k=k)
    for dt in (torch.float32, torch.float64):
        x = (torch.randn(64, 16, dtype=dt) * 1e-3).requires_grad_(True)
        y = x.detach().clone()
        d = m.dist(x, y)
        d.sum().backward()
        assert float(d.abs().max()) == 0.0, dt
        assert torch.isfinite(x.grad).all(), dt
        assert float(x.grad.abs().max()) == 0.0, dt


@pytest.mark.parametrize("k", KS)
def test_gradient_is_bounded_approaching_coincidence(k):
    """d(dist)/dw diverges as w -> 0, but dw/dx vanishes proportionally, so the
    gradient w.r.t. the POINTS stays O(1). That is why `w > 0` is a sufficient
    predicate and no _tiny floor is needed here."""
    m = LorentzManifold(k=k)
    for sep in (1e-3, 1e-6, 1e-9, 1e-12, 1e-15):
        x = (torch.randn(256, 16, dtype=torch.float64) * 1e-3).requires_grad_(True)
        y = x.detach() + sep
        m.dist(x, y).sum().backward()
        assert torch.isfinite(x.grad).all(), sep
        assert float(x.grad.abs().max()) < 10.0, (sep, float(x.grad.abs().max()))




# ----------------------------------------------------------------------
# projx: outlier shave (stat-driven fence, dtype-capped)
#
# projx is no longer the identity. geoopt calls it at `stabilize` intervals; it
# pulls the handful of nodes that have drifted past the float32 resolution wall
# -- where transp's residual reignites an Adam feedback loop and expmap
# overflows (ML-20M seed 5) -- back to the cloud edge, and leaves every other
# row exactly where it was. Fence = p50 + K*(p99.9 - p50), capped by the dtype's
# resolution limit. See the projx docstring.
# ----------------------------------------------------------------------
from link_property_prediction.lorentz import _SHAVE_K, _SHAVE_PULL_Q


def _points_at_radii(m, radii, dim=DIM, dtype=torch.float64):
    """Points sitting at exactly the given dist0 radii, in random directions.
    ||x'|| = sqrt(k) sinh(r / sqrt(k)) is the exact inverse of dist0."""
    radii = torch.as_tensor(radii, dtype=dtype)
    dirs = torch.randn(radii.numel(), dim, dtype=dtype)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    norms = m._sqrt_k * torch.sinh(radii / m._sqrt_k)
    return dirs * norms.unsqueeze(-1)


@pytest.mark.parametrize("k", KS)
def test_projx_protects_small_graphs(k):
    """No row-count threshold: a small graph (a few hundred nodes) is handled
    too. Run in float32, the training dtype. A healthy small cloud is returned
    untouched (its max is below the fence), but a small graph with a runaway
    node past the wall still has it caught -- by the dtype cap, the backstop
    where too few points make the stat fence unreliable -- so it can never sit
    past the wall."""
    m = LorentzManifold(k=k)
    healthy = _points_at_radii(m, torch.rand(200) * 2.0, dtype=torch.float32)
    assert torch.equal(m.projx(healthy.clone()), healthy)

    with_runaway = healthy.clone()
    with_runaway[5] = _points_at_radii(m, torch.tensor([20.0]), dtype=torch.float32)[0]
    out = m.projx(with_runaway.clone())
    moved = (with_runaway - out).norm(dim=-1) > 0
    assert torch.nonzero(moved).flatten().tolist() == [5]
    assert torch.equal(out[~moved], with_runaway[~moved])
    assert float(m.dist0(out).max()) <= float(m._dtype_radius_limit(out)) + 1e-2


@pytest.mark.parametrize("k", KS)
def test_projx_is_identity_on_a_healthy_cloud(k):
    """A cloud with no detached tail -- every radius below the fence -- is
    returned bit-identical. projx only ever moves genuine outliers."""
    m = LorentzManifold(k=k)
    torch.manual_seed(1)
    radii = torch.rand(4000) * 2.0                           # tight, in [0, 2], no tail
    x = _points_at_radii(m, radii)
    out = m.projx(x.clone())
    assert torch.equal(out, x)


@pytest.mark.parametrize("k", KS)
def test_projx_shaves_only_the_detached_tail_and_preserves_the_cloud(k):
    """The core behaviour: a few detached outliers are pulled back; every other
    row is left exactly where it was (behaviour-preserving). The moved rows land
    at the cloud-edge radius, which verifies the sqrt(k) sinh(r/sqrt(k)) inverse
    under curvature."""
    m = LorentzManifold(k=k)
    torch.manual_seed(2)
    n = 4000
    x = _points_at_radii(m, torch.rand(n) * 2.0)             # cloud in [0, 2]
    outliers = {10: 9.0, 100: 12.0, 500: 15.0}              # detached, well past the tail
    for row, r in outliers.items():
        x[row] = _points_at_radii(m, torch.tensor([r]))[0]

    r0 = m.dist0(x)
    out = m.projx(x.clone())
    r1 = m.dist0(out)
    moved = (x - out).norm(dim=-1) > 0

    # exactly the injected outliers moved, and the cloud is bit-identical
    assert set(torch.nonzero(moved).flatten().tolist()) == set(outliers)
    assert torch.equal(out[~moved], x[~moved])

    # they were pulled INWARD, to the p99.9 cloud edge (= the pull target)
    p50 = torch.quantile(r0, 0.5)
    p999 = torch.quantile(r0, _SHAVE_PULL_Q)
    fence = torch.minimum(p50 + _SHAVE_K * (p999 - p50), m._dtype_radius_limit(x))
    target = torch.minimum(p999, fence)
    assert torch.allclose(r1[moved], target.to(r1.dtype), atol=1e-4)
    assert float(r1[moved].max()) < float(r0[moved].min())


@pytest.mark.parametrize("k", KS)
def test_projx_is_idempotent(k):
    """Applying projx twice equals applying it once: the shaved table is a fixed
    point (nothing sits past the fence after the first pass)."""
    m = LorentzManifold(k=k)
    torch.manual_seed(3)
    n = 4000
    x = _points_at_radii(m, torch.rand(n) * 2.0)
    x[7] = _points_at_radii(m, torch.tensor([13.0]))[0]
    once = m.projx(x.clone())
    twice = m.projx(once.clone())
    assert torch.equal(once, twice)


def test_dtype_radius_limit_is_dtype_derived_and_scales_with_k():
    """The hard ceiling is read from the dtype, not hardcoded: float64 resolves
    a larger radius than float32, and both scale with sqrt(k)."""
    for k in KS:
        m = LorentzManifold(k=k)
        f32 = m._dtype_radius_limit(torch.zeros(1, dtype=torch.float32))
        f64 = m._dtype_radius_limit(torch.zeros(1, dtype=torch.float64))
        assert float(f64) > float(f32)
        # sqrt(k) * asinh(sqrt(k/eps)/sqrt(k)), checked against a direct recompute
        eps = torch.finfo(torch.float32).eps
        expect = m._sqrt_k * math.asinh(math.sqrt(m.k / eps) / m._sqrt_k)
        assert abs(float(f32) - expect) < 1e-4
    assert float(LorentzManifold(1.0)._dtype_radius_limit(torch.zeros(1))) == pytest.approx(8.664, abs=1e-2)


def test_projx_never_leaves_a_row_past_the_dtype_wall():
    """Even a heavy tail whose stat fence would land past the resolution wall is
    capped: after projx no row exceeds _dtype_radius_limit, so a wholesale drift
    cannot place a point on the light cone. Run in float32, where the wall bites."""
    m = LorentzManifold(k=1.0)
    torch.manual_seed(4)
    n = 4000
    # cloud in [0,2] plus a heavy tail at 15-20 -> spread fence >> dtype limit
    radii = torch.rand(n) * 2.0
    radii[:300] = 15.0 + torch.rand(300) * 5.0
    x = _points_at_radii(m, radii, dtype=torch.float32)
    out = m.projx(x.clone())
    limit = float(m._dtype_radius_limit(x))
    assert float(m.dist0(out).max()) <= limit + 1e-3
    assert torch.isfinite(out).all()
