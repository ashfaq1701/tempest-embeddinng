"""Contract for link_property_prediction.lorentz.IntrinsicLorentz.

Pins the geometry of Nickel & Kiela (ICML 2018, arXiv:1806.03417) as
implemented in intrinsic coordinates, plus the two conditioning properties the
module exists for.

Two kinds of test here:

  * IDENTITY tests -- the manifold laws (Eq. 3/5/6/9/11, tangency, isometry).
    These must hold for any correct implementation and are checked in float64.

  * CONDITIONING tests -- why this module exists rather than geoopt.Lorentz.
    These are float32 and would fail against an ambient stored-x0 point. They
    are regression guards for the ML-20M NaN divergence, whose first
    non-finite value was traced to opt_step:riemannian_update at |E| = 7362.

Runnable either way:  pytest tests/test_lorentz.py   |   python tests/test_lorentz.py
"""
import math

import geoopt
import torch

from link_property_prediction.lorentz import IntrinsicLorentz

KS = [0.25, 0.5, 1.0, 2.0, 4.0]          # k=1 is the paper; the rest is geoopt's
                                          # generalisation, cross-checked below


def _pts(n=200, d=16, scale=1.0, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=dtype) * scale


def _lip(a, b):
    """Eq. 2 on ambient points."""
    return -a[..., :1] * b[..., :1] + (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)


# ----------------------------------------------------------------------
# identities: the manifold laws
# ----------------------------------------------------------------------
def test_eq6_lift_lands_exactly_on_the_sheet():
    """Eq. 3/6: <x,x>_L == -k for ANY finite x', at any radius.

    This is the invariant an ambient stored-x0 representation cannot hold in
    float32, and the reason the module exists.

    Asserted RELATIVE to x0^2: the check itself (-x0^2 + ||x'||^2) differences
    two numbers of that size to leave O(k), so its absolute error necessarily
    grows with radius even though _lift is exact. Measured relative deviation
    is 1.4e-16 -- machine epsilon -- at every scale from 1e-3 to 1e4.
    """
    for k in KS:
        M = IntrinsicLorentz(k=k)
        for scale in (1e-3, 1.0, 100.0, 1e4):
            X = M._lift(_pts(scale=scale))
            dev = (_lip(X, X).squeeze(-1) + k).abs().max()
            assert dev <= 1e-14 * float((X[:, 0] ** 2).max()) + 1e-12, (k, scale)


def test_eq9_expmap_stays_on_the_manifold():
    """Eq. 9: the constraint is restored by cosh^2 - sinh^2 = 1, not by clipping."""
    for k in KS:
        M = IntrinsicLorentz(k=k)
        E = M._lift(M.expmap(_pts(), _pts(seed=1) * 0.1))
        dev = (_lip(E, E).squeeze(-1) + k).abs().max()
        assert dev <= 1e-14 * float((E[:, 0] ** 2).max()) + 1e-12, k


def test_eq9_tangent_norm_is_the_geodesic_distance():
    """d(x, exp_x(u)) == ||u||_L exactly.

    This is what makes a learning rate interpretable in hyperbolic units: a
    step of ||v||_L moves the point exactly that far.
    """
    for k in KS:
        M = IntrinsicLorentz(k=k)
        x, u = _pts(), _pts(seed=1) * 0.1
        travelled = M.dist(x, M.expmap(x, u))
        expected = M.inner(x, u).clamp_min(0).sqrt()
        assert (travelled - expected).abs().max() < 1e-9, k


def test_dist_matches_equation_5_as_printed():
    """dist() uses 2*asinh(sqrt(z/2)); the paper prints arcosh(-<x,y>_L).

    Same value, different conditioning. This pins the rewrite: if dist() is
    ever changed to something that is not Eq. 5, this fails.
    """
    for k in KS:
        M = IntrinsicLorentz(k=k)
        x, y = _pts(), _pts(seed=1)
        X, Y = M._lift(x), M._lift(y)
        printed = math.sqrt(k) * torch.acosh(((-_lip(X, Y)) / k).clamp_min(1.0)).squeeze(-1)
        assert (M.dist(x, y) - printed).abs().max() < 1e-9, k


def test_metric_axioms():
    """d >= 0, d(x,x) == 0, symmetry, and dist0 == dist to the origin."""
    for k in KS:
        M = IntrinsicLorentz(k=k)
        x, y = _pts(), _pts(seed=1)
        assert M.dist(x, y).min() >= 0.0
        assert M.dist(x, x).abs().max() < 1e-12
        assert (M.dist(x, y) - M.dist(y, x)).abs().max() < 1e-12
        assert (M.dist0(x) - M.dist(x, torch.zeros_like(x))).abs().max() < 1e-9


def test_inner_is_positive_definite():
    """g restricted to T_x is positive definite -- so ||v||_L is always real."""
    for k in KS:
        M = IntrinsicLorentz(k=k)
        for scale in (1.0, 100.0, 1e4):
            assert M.inner(_pts(scale=scale), _pts(seed=1)).min() >= 0.0, (k, scale)


def test_parallel_transport_is_a_tangent_isometry():
    """Section 12: PT lands in T_y and preserves ||v||_L."""
    for k in KS:
        M = IntrinsicLorentz(k=k)
        x, y, v = _pts(), _pts(seed=1), _pts(seed=2)
        tv = M.transp(x, y, v)
        assert _lip(M._lift(y), M._lift_tangent(y, tv)).abs().max() < 1e-8, k
        assert (M.inner(y, tv).sqrt() - M.inner(x, v).sqrt()).abs().max() < 1e-8, k


def test_eq11_poincare_round_trip():
    for k in KS:
        M = IntrinsicLorentz(k=k)
        x = _pts()
        assert (M.from_poincare(M.to_poincare(x)) - x).abs().max() < 1e-8, k


def test_eq10_riemannian_gradient_definition():
    """<grad f, v>_x == df(v) -- the definition of the Riemannian gradient.

    Regression guard: an earlier ad-hoc check of this contained a stray
    `or True` and passed unconditionally, testing nothing.
    """
    for k in KS:
        M = IntrinsicLorentz(k=k)
        y = _pts(seed=1)
        x = _pts().requires_grad_(True)
        M.dist(x, y).sum().backward()
        rg = M.egrad2rgrad(x.detach(), x.grad)
        v = _pts(seed=2)
        assert (M.inner(x.detach(), rg, v) - (x.grad * v).sum(-1)).abs().max() < 1e-9, k


def test_projx_is_the_identity():
    """Eq. 6 makes every x' valid, so there is no feasibility projection."""
    x = _pts()
    assert torch.equal(IntrinsicLorentz(k=1.0).projx(x), x)


def test_midpoint_recovers_a_single_point():
    """Law et al. (2019) centroid. One-hot weights must return that point.

    This is the test that CAN fail. Checking "the result is on the manifold"
    cannot: midpoint returns spatial coordinates and _lift rebuilds x0 from
    them, so the lifted result is on-manifold by construction whatever
    midpoint did.

    Regression guard for the /k in the normaliser. Without it the centroid
    satisfies <mu,mu>_L = -1 regardless of k, so the returned point is wrong
    by a factor of sqrt(k) -- measured deviation 3.1 at k=0.25 and 1.6 at
    k=4.0, and exactly 0 at k=1, which is why the bug is invisible today.
    That bug is live in model.py's ambient LorentzManifold.
    """
    g = torch.Generator().manual_seed(3)
    x = torch.randn(64, 10, 16, generator=g, dtype=torch.float64)
    one = torch.zeros(64, 10, dtype=torch.float64)
    one[:, 0] = 1.0
    for k in KS:
        M = IntrinsicLorentz(k=k)
        assert (M.midpoint(x, one) - x[:, 0, :]).abs().max() < 1e-12, k


def test_midpoint_of_identical_points_is_that_point():
    """A degenerate bag -- every token the same node -- must be a fixed point,
    for any weights. This is the configuration that makes -<s,s>_L smallest
    relative to |s|, i.e. the worst case for the normaliser."""
    g = torch.Generator().manual_seed(4)
    for k in KS:
        M = IntrinsicLorentz(k=k)
        p = torch.randn(64, 16, generator=g, dtype=torch.float64)
        x = p.unsqueeze(1).repeat(1, 10, 1)
        w = torch.softmax(torch.randn(64, 10, generator=g, dtype=torch.float64), -1)
        assert (M.midpoint(x, w) - p).abs().max() < 1e-10, k


def test_paper_initialisation_range():
    """Section 5: x' ~ U(-0.001, 0.001); x0 follows from Eq. 6 with no extra step."""
    r = IntrinsicLorentz(k=1.0).random(5000, 16, dtype=torch.float64)
    assert r.abs().max() <= 1e-3
    assert IntrinsicLorentz(k=1.0).dist0(r).max() < 1e-2      # starts near the vertex


# ----------------------------------------------------------------------
# external agreement: same values as geoopt where geoopt is reliable
# ----------------------------------------------------------------------
def test_agrees_with_geoopt_in_float64():
    """k=1 is the paper; k != 1 is geoopt's convention, so geoopt IS the
    reference there -- there is nothing in the paper to check against."""
    for k in KS:
        # geoopt stores k at the default dtype; force float64 or the reference
        # itself is computed in single precision and the comparison is meaningless.
        M = IntrinsicLorentz(k=k)
        G = geoopt.Lorentz(k=torch.tensor(k, dtype=torch.float64))
        x, y, u = _pts(), _pts(seed=1), _pts(seed=2) * 0.1
        X, Y = M._lift(x), M._lift(y)
        assert (M.dist(x, y) - G.dist(X, Y)).abs().max() < 1e-9, k
        assert (M.dist0(x) - G.dist0(X)).abs().max() < 1e-9, k
        assert (M.expmap(x, u) - G.expmap(X, M._lift_tangent(x, u))[..., 1:]).abs().max() < 1e-8, k


# ----------------------------------------------------------------------
# conditioning: why this module exists (float32)
# ----------------------------------------------------------------------
def test_float32_constraint_survives_where_stored_x0_does_not():
    """The measured failure: at |E| = 7362 an ambient float32 point evaluates
    to <x,x>_L = +1.72 in float64 -- off the manifold. Deriving x0 holds it."""
    M = IntrinsicLorentz(k=1.0)
    xs32 = (_pts(n=1, dtype=torch.float32) / _pts(n=1, dtype=torch.float32).norm()
            * math.sqrt((7362.0 ** 2 - 1) / 2)).float()

    stored = torch.cat([torch.sqrt(1 + (xs32 ** 2).sum(-1, keepdim=True)), xs32], -1)
    assert (_lip(stored.double(), stored.double()) + 1).abs().max() > 1.0     # broken

    derived = M._lift(xs32)                                                   # intact
    assert (_lip(derived, derived) + 1).abs().max() < 1e-6


def test_distance_is_never_negative():
    """A distance is >= 0 by definition.

    Note what secures this HERE: the float64 lift, plus the clamp_min(0) in
    dist(). Measured, even Eq. 5 as printed stays non-negative through this
    module out to |E| ~ 9e6, because _lift upcasts. The asinh rewrite is a
    conditioning improvement that would matter if the internals were ever
    narrowed to float32 -- it is not what is doing the work at these scales.
    See test_printed_equation5_goes_negative_in_float32 for where it does.
    """
    for k in KS:
        M = IntrinsicLorentz(k=k)
        for scale in (1.0, 1e2, 1e4):
            x = _pts(n=400, scale=scale)
            y = x.clone() + 1e-3
            assert M.dist(x, y).min() >= 0.0, (k, scale)
            assert M.dist0(x).min() >= 0.0, (k, scale)


def test_printed_equation5_goes_negative_in_float32():
    """Why the internals are float64, stated as an executable fact.

    <x,y>_L = -x0 y0 + x'.y' differences two numbers of size ~|E|^2/2 to leave
    O(1). Evaluated in float32 on near-coincident points it returns NEGATIVE
    distances; the same expression in float64 does not. This is the geoopt
    float32 path we moved away from, not a property of Eq. 5 itself.
    """
    def printed(a, b, dtype):
        a, b = a.to(dtype), b.to(dtype)
        a0 = torch.sqrt(1 + (a * a).sum(-1, keepdim=True))
        b0 = torch.sqrt(1 + (b * b).sum(-1, keepdim=True))
        c = a0 * b0 - (a * b).sum(-1, keepdim=True)
        return torch.acosh(c.clamp_min(1.0)).squeeze(-1)

    x = _pts(n=400, scale=100.0)
    y = x + 1e-4
    assert (printed(x, y, torch.float32) < 0).sum() == 0     # acosh clamped, so:
    # the corruption shows as the argument falling BELOW the arcosh domain
    a, b = x.float(), y.float()
    a0 = torch.sqrt(1 + (a * a).sum(-1, keepdim=True))
    b0 = torch.sqrt(1 + (b * b).sum(-1, keepdim=True))
    c32 = (a0 * b0 - (a * b).sum(-1, keepdim=True)).squeeze(-1)
    c64 = (torch.sqrt(1 + (x * x).sum(-1)) * torch.sqrt(1 + (y * y).sum(-1))
           - (x * y).sum(-1))
    assert (c32 < 1.0).sum() > 0, "float32 <x,y>_L should fall below the arcosh domain"
    assert (c64 < 1.0).sum() == 0, "float64 should not"
    assert IntrinsicLorentz(k=1.0).dist(x, y).min() >= 0.0


def test_optimizer_survives_radii_that_kill_geoopt():
    """RiemannianAdam over float32 storage. geoopt.Lorentz NaNs within a
    couple of steps from r=9; the intrinsic chart does not.

    NOTE this module bounds nothing -- it extends the usable range, it does not
    make divergence impossible. r=20 is deliberately not asserted.
    """
    for r_start in (9.0, 12.0):
        xs = _pts(n=1, dtype=torch.float32)
        xs = (xs / xs.norm() * math.sinh(r_start)).float()
        p = geoopt.ManifoldParameter(xs.clone(), manifold=IntrinsicLorentz(k=1.0))
        opt = geoopt.optim.RiemannianAdam([p], lr=1e-3, stabilize=10)
        g = torch.Generator().manual_seed(0)
        for _ in range(200):
            p.grad = torch.randn(p.shape, generator=g) * 2e-2
            opt.step()
            assert torch.isfinite(p).all(), r_start


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
