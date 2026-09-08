"""Lorentz model of hyperbolic geometry, faithful to Nickel & Kiela (ICML 2018).

    Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry
    Maximilian Nickel, Douwe Kiela.  PMLR 80.  arXiv:1806.03417

Equation numbers below refer to that paper.

WHY THIS EXISTS
---------------
`geoopt.Lorentz` stores a point as the full ambient (n+1)-vector, including the
time coordinate x0, and trusts what is stored. Eq. 6 says x0 carries no free
information -- it is determined by x'. Storing it anyway creates an invariant
that float32 cannot hold: <x,x>_L = -k requires resolving a "+k" inside
x0^2 = k + ||x'||^2, and once x0^2 exceeds 2^24 that term is below one ULP.
Measured, at ||E|| = 7362 (hyperbolic radius 9.25) a stored-x0 point evaluates
to <x,x>_L = +1.72 in float64 -- the point has silently left the manifold, and
expmap on it returns NaN.

This module removes the redundant coordinate. A point IS x' in R^n; x0 is
derived in float64 wherever it is needed and never persists. The constraint
then holds by construction at any radius, which is exactly Eq. 6 read as a
parameterisation rather than as a check.

Being a `geoopt.Manifold`, it drives geoopt's RiemannianSGD / RiemannianAdam
unchanged: `retr_transp`, `expmap_transp` and `component_inner` come from the
base class, so no optimizer code is needed.

WHAT IS THE PAPER'S AND WHAT IS OURS
------------------------------------
The paper specifies no epsilons, no clamps, no domain guards and no
hyperparameters; it argues only that d_l has no fraction and so avoids the
Poincare boundary blow-up. Every guard below is therefore OURS, and each is
marked `# GUARD (not in the paper)` with the failure it prevents.

SCOPE -- what this does NOT do
------------------------------
This class is the geometry, exactly. It bounds nothing. Deriving x0 makes the
constraint hold at any radius, which extends the usable range (measured: fp32
storage survives r=12 where geoopt.Lorentz NaNs at r=9), but a diverging
optimizer still walks out of float64's range near r ~ 19. Bounding the
trajectory is a separate concern and is deliberately not handled here.

CURVATURE
---------
The paper is k=1 throughout: it defines only <x,x>_L = -1, mentions curvature
once in passing ("constant negative sectional curvature"), and never
parameterises, tunes or ablates it. The `k` argument below is geoopt's
convention (<x,x>_L = -k, curvature -1/k), carried over because the existing
LorentzManifold exposes it. At k != 1 there is nothing in the paper to be
faithful to; those formulas are cross-validated against geoopt.Lorentz(k)
instead (dist/dist0/expmap/egrad2rgrad agree to <= 4e-15 for k in
[0.25, 4.0]).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import geoopt
import torch
from torch import Tensor

__all__ = ["IntrinsicLorentz"]

# cosh/sinh overflow guards; float64 cosh overflows near 710.
_MAX_SINH_ARG = 700.0
_TINY = 1e-30


class IntrinsicLorentz(geoopt.Manifold):
    """Lorentz model in intrinsic coordinates. Points are x' in R^n.

    Parameters
    ----------
    k : curvature parameter; the sheet is <x,x>_L = -k with sectional
        curvature -1/k. k=1 is the paper. See CURVATURE above.
    """

    name = "IntrinsicLorentz"
    ndim = 1
    reversible = False

    def __init__(self, k: float = 1.0) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        # kept in float64: every internal computation runs at this precision
        self.register_buffer("k", torch.as_tensor(float(k), dtype=torch.float64))

    # ------------------------------------------------------------------
    # internals: the lift x' -> (x0, x') that Eq. 6 defines
    # ------------------------------------------------------------------
    def _x0(self, x64: Tensor) -> Tensor:
        """Eq. 6: x0 = sqrt(k + ||x'||^2). float64 in, float64 out."""
        return torch.sqrt(self.k + (x64 * x64).sum(-1, keepdim=True))

    def _lift(self, x: Tensor) -> Tensor:
        """[..., n] -> [..., n+1] ambient point in float64.

        POST: <lift(x), lift(x)>_L == -k to float64 precision, for any finite x.
        """
        x64 = x.double()
        return torch.cat([self._x0(x64), x64], -1)

    def _lift_tangent(self, x: Tensor, u: Tensor) -> Tensor:
        """Tangent [..., n] -> ambient [..., n+1] in float64.

        Tangency <x,v>_L = 0 fixes the time component: v0 = (x'.u)/x0.
        """
        x64, u64 = x.double(), u.double()
        return torch.cat([(x64 * u64).sum(-1, keepdim=True) / self._x0(x64), u64], -1)

    @staticmethod
    def _lip(a: Tensor, b: Tensor) -> Tensor:
        """Eq. 2, ambient: <a,b>_L = -a0 b0 + sum_i a_i b_i."""
        return -a[..., :1] * b[..., :1] + (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)

    # ------------------------------------------------------------------
    # geoopt.Manifold API (the abstract set)
    # ------------------------------------------------------------------
    def projx(self, x: Tensor) -> Tensor:
        """Point onto the manifold.

        The IDENTITY: by Eq. 6 every x' in R^n is a valid point, so unlike the
        Poincare ball there is no feasibility projection to make. geoopt calls
        this at `stabilize` intervals to repair drift; here there is no drift
        to repair, because x0 is never stored.
        """
        return x

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Vector onto T_x. Identity: in this chart T_x is all of R^n, because
        the constraint <x,v>_L = 0 is already absorbed by the parameterisation.
        """
        return u

    def inner(self, x: Tensor, u: Tensor, v: Optional[Tensor] = None,
              *, keepdim: bool = False) -> Tensor:
        """Induced metric: <u,v>_x = u.v - (x.u)(x.v)/x0^2.

        Pull back ds^2 = -dx0^2 + ||dx'||^2 through Eq. 6, using
        dx0 = (x'.dx')/x0. Positive definite since ||x'||^2/x0^2 < 1.
        """
        x64, u64 = x.double(), u.double()
        v64 = u64 if v is None else v.double()
        x0sq = self.k + (x64 * x64).sum(-1, keepdim=True)
        res = ((u64 * v64).sum(-1, keepdim=True)
               - (x64 * u64).sum(-1, keepdim=True) * (x64 * v64).sum(-1, keepdim=True) / x0sq)
        if not keepdim:
            res = res.squeeze(-1)
        return res.to(x.dtype)

    def egrad2rgrad(self, x: Tensor, u: Tensor) -> Tensor:
        """Eq. 10 in this chart: g^-1 grad = u + x (x.u)/k.

        By Sherman-Morrison on g = I - x'x'^T/x0^2, using x0^2 - ||x'||^2 = k.
        The paper's two steps (flip the sign of component 0, then project onto
        T_x) are both absorbed here -- there is nothing to project.
        """
        x64, u64 = x.double(), u.double()
        return (u64 + x64 * ((x64 * u64).sum(-1, keepdim=True) / self.k)).to(x.dtype)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """Eq. 9: exp_x(v) = cosh(||v||_L) x + sinh(||v||_L) v/||v||_L,
        generalised to curvature -1/k, evaluated on the lifted float64 point.

        POST: d(x, expmap(x,u)) == sqrt(inner(x,u,u)) -- the tangent norm IS
        the geodesic distance travelled, which is what makes a learning rate
        interpretable in hyperbolic units.
        """
        x64, u64 = x.double(), u.double()
        X, U = self._lift(x64), self._lift_tangent(x64, u64)
        sk = self.k.sqrt()
        nrm = self.inner(x64, u64, keepdim=True).clamp_min(0.0).sqrt()
        # GUARD (not in the paper): clamp cosh/sinh argument; float64 cosh
        # overflows near 710 and would return inf, then NaN.
        t = (nrm / sk).clamp(max=_MAX_SINH_ARG)
        # GUARD (not in the paper): v/||v||_L is 0/0 at ||v||=0. The limit of
        # sinh(t)*sk/nrm as nrm->0 is 1, giving the Euclidean step x + v.
        coef = torch.where(nrm > _TINY,
                           torch.sinh(t) * sk / nrm.clamp_min(_TINY),
                           torch.ones_like(nrm))
        out = torch.cosh(t) * X + coef * U
        return out[..., 1:].to(x.dtype)

    def retr(self, x: Tensor, u: Tensor) -> Tensor:
        """The exponential map is closed form, so it IS the retraction. This is
        the 2018 paper's improvement over the 2017 first-order retraction."""
        return self.expmap(x, u)

    def _check_point_on_manifold(self, x: Tensor, *, atol: float = 1e-5,
                                 rtol: float = 1e-5) -> Tuple[bool, Optional[str]]:
        """Only finiteness can fail: the chart cannot represent an invalid point."""
        if not torch.isfinite(x).all():
            return False, "point contains non-finite values"
        return True, None

    def _check_vector_on_tangent(self, x: Tensor, u: Tensor, *, atol: float = 1e-5,
                                 rtol: float = 1e-5) -> Tuple[bool, Optional[str]]:
        """T_x = R^n in this chart, so any finite u is tangent."""
        if not torch.isfinite(u).all():
            return False, "tangent vector contains non-finite values"
        return True, None

    # ------------------------------------------------------------------
    # overridden for conditioning (base class would route through ambient)
    # ------------------------------------------------------------------
    def dist(self, x: Tensor, y: Tensor, *, keepdim: bool = False) -> Tensor:
        """Eq. 5: d_l(x,y) = sqrt(k) arcosh(-<x,y>_L / k).

        Evaluated through the algebraically IDENTICAL form
            -<x,y>_L - k = (||x'-y'||^2 - (x0-y0)^2) / 2
            arcosh(1 + w)  = 2 asinh(sqrt(w/2))
        which differences the coordinates BEFORE squaring. The naive form
        subtracts two numbers of size ~||E||^2/2 to leave O(1); measured, it
        returns NEGATIVE distances (149/400 pairs at ||E||=139) once the radius
        grows. Same value, different conditioning. See `dist_naive`.
        """
        X, Y = self._lift(x), self._lift(y)
        d0 = X[..., :1] - Y[..., :1]
        ds = X[..., 1:] - Y[..., 1:]
        z = ((ds * ds).sum(-1, keepdim=True) - d0 * d0) / 2.0
        # GUARD (not in the paper): z >= 0 exactly; float error can make it
        # slightly negative for coincident points, which would NaN the sqrt.
        res = 2.0 * self.k.sqrt() * torch.asinh(torch.sqrt((z / self.k).clamp_min(0.0) / 2.0))
        if not keepdim:
            res = res.squeeze(-1)
        return res.to(x.dtype)

    def dist_naive(self, x: Tensor, y: Tensor, *, keepdim: bool = False) -> Tensor:
        """Eq. 5 exactly as printed. Kept for verification against `dist`; not
        used in training because of the cancellation documented above."""
        X, Y = self._lift(x), self._lift(y)
        c = (-self._lip(X, Y)) / self.k
        # GUARD (not in the paper): arcosh's domain is [1, inf).
        res = self.k.sqrt() * torch.acosh(c.clamp_min(1.0))
        if not keepdim:
            res = res.squeeze(-1)
        return res.to(x.dtype)

    def dist0(self, x: Tensor, *, keepdim: bool = False) -> Tensor:
        """Distance to the origin (sqrt(k), 0, ..., 0). Reduces to
        z = x0 sqrt(k) - k in the stable form above."""
        x64 = x.double()
        z = self._x0(x64) * self.k.sqrt() - self.k
        res = 2.0 * self.k.sqrt() * torch.asinh(torch.sqrt((z / self.k).clamp_min(0.0) / 2.0))
        if not keepdim:
            res = res.squeeze(-1)
        return res.to(x.dtype)

    def transp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport T_x -> T_y along the connecting geodesic:

            PT(v) = v + <y,v>_L / (k - <x,y>_L) * (x + y)

        Not in the paper (Algorithm 1 carries no state), but required by any
        optimizer with momentum -- the buffer is a tangent vector at a point
        that moves. The denominator satisfies k - <x,y>_L >= 2k, so it never
        degenerates. POST: <y, PT(v)>_L = 0 and ||PT(v)||_L = ||v||_L.
        """
        X, Y = self._lift(x), self._lift(y)
        V = self._lift_tangent(x, v)
        coef = self._lip(Y, V) / (self.k - self._lip(X, Y))
        return (V + coef * (X + Y))[..., 1:].to(x.dtype)

    # ------------------------------------------------------------------
    # extensions used by this project (NOT in the paper)
    # ------------------------------------------------------------------
    def midpoint(self, x: Tensor, w: Tensor) -> Tensor:
        """Weighted Lorentzian centroid of a bag. Law et al. (ICML 2019).

        x [..., T, n] points, w [..., T] weights. Returns [..., n].

            mu = s / sqrt(-<s,s>_L / k),   s = sum_t w_t x_t

        NOTE this minimises squared LORENTZIAN distance, not squared geodesic
        distance -- it is not the Riemannian barycenter (which has no closed
        form). The /k is load-bearing: without it the result satisfies
        <mu,mu>_L = -1 regardless of k, i.e. it is off-manifold for k != 1.
        """
        X = self._lift(x)                                   # [..., T, n+1]
        s = (w.double().unsqueeze(-1) * X).sum(dim=-2)      # [..., n+1]
        neg = -self._lip(s, s) / self.k
        # GUARD (not in the paper): neg >= 1 in exact arithmetic for convex
        # weights; clamp so a degenerate bag cannot sqrt a non-positive value.
        return (s / neg.clamp_min(_TINY).sqrt())[..., 1:].to(x.dtype)

    def to_poincare(self, x: Tensor) -> Tensor:
        """Eq. 11: p(x) = x' / (x0 + 1), on the unit-normalised hyperboloid.
        Returns coordinates in the unit Poincare ball."""
        X = self._lift(x) / self.k.sqrt()
        return (X[..., 1:] / (X[..., :1] + 1.0)).to(x.dtype)

    def from_poincare(self, u: Tensor) -> Tensor:
        """Inverse of Eq. 11, returning intrinsic coordinates x'."""
        u64 = u.double()
        n2 = (u64 * u64).sum(-1, keepdim=True)
        # GUARD (not in the paper): the ball is open; 1 - ||u||^2 -> 0 at the
        # boundary. This is precisely the blow-up the Lorentz model avoids.
        xs = 2.0 * u64 / (1.0 - n2).clamp_min(_TINY)
        return (xs * self.k.sqrt()).to(u.dtype)

    # ------------------------------------------------------------------
    # initialisation (paper, section 5)
    # ------------------------------------------------------------------
    def random(self, *size, dtype=None, device=None, irange: float = 1e-3) -> Tensor:
        """The paper's init: x' ~ U(-0.001, 0.001), x0 then set by Eq. 6.
        Here only x' is materialised, so Eq. 6 needs no separate step."""
        return torch.empty(*size, dtype=dtype, device=device).uniform_(-irange, irange)

    def origin(self, *size, dtype=None, device=None, seed: int = 42) -> Tensor:
        """The vertex (sqrt(k), 0, ..., 0), i.e. x' = 0."""
        return torch.zeros(*size, dtype=dtype, device=device)
