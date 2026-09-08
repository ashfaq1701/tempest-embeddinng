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
Measured, at ||E|| = 7362 (hyperbolic radius 9.25) a point whose x0 was STORED
in float32, then EVALUATED in float64, gives <x,x>_L = +1.72 -- the point has
silently left the manifold, and expmap on it returns NaN. The error is in the
storage, not the evaluation: float64 resolves the "+k" there with room to
spare (x0^2 = 2.7e7, one float64 ULP is 6.0e-9), whereas one float32 ULP at
x0 = 5206 is 6.2e-4 and 2*x0*ULP ~ 6.5, which is the magnitude observed.

This module removes the redundant coordinate. A point IS x' in R^n; x0 is
derived in float64 wherever it is needed and never persists. The constraint
then holds by construction at any radius, which is exactly Eq. 6 read as a
parameterisation rather than as a check.

Being a `geoopt.Manifold`, it drives geoopt's RiemannianSGD / RiemannianAdam
unchanged: `retr_transp`, `expmap_transp` and `component_inner` come from the
base class, so no optimizer code is needed.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import geoopt
import torch
from torch import Tensor

__all__ = ["LorentzManifold"]

_LOG_MAX_F64 = math.log(torch.finfo(torch.float64).max)   # ~709.78
_TINY = 1e-30


class LorentzManifold(geoopt.Manifold):
    """Lorentz model in intrinsic coordinates. Points are x' in R^n.

    Parameters
    ----------
    k : curvature parameter; the sheet is <x,x>_L = -k with sectional
        curvature -1/k. k=1 is the paper, which never parameterises
        curvature; k != 1 follows geoopt's convention.
    """

    name = "LorentzManifold"
    ndim = 1
    reversible = False

    def __init__(self, k: float = 1.0) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        # NOT a registered buffer. geoopt.Manifold is an nn.Module and does not
        # override _apply, so a float buffer is silently converted by
        # model.float() / .half() / .to(float32) -- which would demote the
        # float64 arithmetic this class exists to protect (measured: 2.0e-7
        # error in dist at k=2.7 after a stray .float()). A 0-dim CPU tensor
        # promotes correctly against CUDA operands, so this is device-safe.
        self.k = torch.tensor(float(k), dtype=torch.float64)

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
        t = nrm / sk
        # GUARD (not in the paper): cosh(t) multiplies X, so the overflow bound
        # depends on |X|, not on cosh's own limit -- cosh(700) is 5.1e303 and
        # float64 tops out at 1.8e308, so cosh(700)*X already overflows once
        # |X| > 3.5e4. Bound t by the operand instead of a fixed constant.
        #
        # This RAISES rather than clamping. Clamping would truncate the step
        # and silently break this method's postcondition
        # d(x, exp_x(u)) == ||u||_L, which is the one guarantee that makes a
        # learning rate meaningful in hyperbolic units. A step this large means
        # the caller has already diverged; say so.
        t_max = _LOG_MAX_F64 - torch.log(X.abs().amax(dim=-1, keepdim=True).clamp_min(1.0))
        if bool((t > t_max).any()):
            raise RuntimeError(
                f"expmap: geodesic step {float(t.max()):.3e} would overflow "
                f"cosh(t)*x (limit {float(t_max.min()):.3e} at |x|max="
                f"{float(X.abs().max()):.3e}). The step, not the guard, is the "
                f"problem: ||v||_L IS the hyperbolic distance travelled.")
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
        grows. Same value, different conditioning -- verified against Eq. 5 as
        printed, and against geoopt.Lorentz.dist, to 8.9e-16 in float64.

        To be precise about those negatives, since mathematical arcosh has
        range [0, inf) and cannot return one: they come from geoopt's arcosh,
        which is log(x + sqrt(clamp_min(x^2 - 1, 1e-15))). It clamps the sqrt
        argument but not x, so once cancellation pushes x below 1 it returns
        log(x) < 0. Measured on geoopt float32: 96/400 near-coincident pairs
        negative, min -1.34e-1. A literal acosh(clamp_min(x, 1)) would return
        0 there instead, and Eq. 5 in exact arithmetic returns neither.

        Calibration: this is a float32 phenomenon. Because _lift upcasts, Eq. 5
        as printed also stays non-negative THROUGH THIS MODULE out to
        |E| ~ 9e6 -- so at the radii we reach it is the float64 lift, not this
        rewrite, doing the work. The rewrite is insurance for a narrower
        internal dtype, not the load-bearing part.
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

    def dist0(self, x: Tensor, *, keepdim: bool = False) -> Tensor:
        """Distance to the origin (sqrt(k), 0, ..., 0). Reduces to
        z = x0 sqrt(k) - k in the stable form above."""
        x64 = x.double()
        z = self._x0(x64) * self.k.sqrt() - self.k
        res = 2.0 * self.k.sqrt() * torch.asinh(torch.sqrt((z / self.k).clamp_min(0.0) / 2.0))
        if not keepdim:
            res = res.squeeze(-1)
        return res.to(x.dtype)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Inverse of Eq. 9: the tangent vector at x pointing to y.

        Not in the paper -- Algorithm 1 never needs it -- but the base class
        raises NotImplementedError, so without it anything reaching for a
        tangent-space representation fails at runtime: Riemannian means,
        interpolation, geoopt.optim.RiemannianLineSearch, and the map-to-T_0 /
        aggregate / map-back pattern used by most hyperbolic GNN layers.

            log_x(y) = arcosh(a) / sqrt(a^2 - 1) * (Y - a X),   a = -<X,Y>_L / k

        Tangency and length both check out: <X, Y - aX>_L = 0 and
        ||Y - aX||_L = sqrt(k(a^2 - 1)), so ||log_x(y)||_L = d(x,y), and
        exp_x(log_x(y)) = y.

        The prefactor is evaluated through the same stable substitution as
        dist(): with w = a - 1 = z/k, arcosh(a) = 2 asinh(sqrt(w/2)) and
        sqrt(a^2 - 1) = sqrt(w(w + 2)), which avoids forming a^2 - 1 as a
        difference of large numbers. Its limit as w -> 0 (coincident points)
        is 1, handled explicitly.
        """
        X, Y = self._lift(x), self._lift(y)
        # w = a - 1, computed by differencing coordinates before squaring
        d0 = X[..., :1] - Y[..., :1]
        ds = X[..., 1:] - Y[..., 1:]
        w = (((ds * ds).sum(-1, keepdim=True) - d0 * d0) / 2.0 / self.k).clamp_min(0.0)
        a = w + 1.0
        num = 2.0 * torch.asinh(torch.sqrt(w / 2.0))          # == arcosh(a)
        den = torch.sqrt(w * (w + 2.0))                        # == sqrt(a^2 - 1)
        coef = torch.where(w > _TINY, num / den.clamp_min(_TINY), torch.ones_like(w))
        return (coef * (Y - a * X))[..., 1:].to(x.dtype)

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
        # GUARD (not in the paper): for convex weights neg >= 1 exactly, so a
        # value near zero means the caller passed weights this operation is not
        # defined for (all-zero, or near-cancelling). Flooring it would return
        # a finite, plausible-looking, arbitrary point -- with all-zero weights,
        # the origin. Refuse instead.
        if bool((neg < 0.5).any()):
            raise ValueError(
                f"midpoint: -<s,s>_L/k = {float(neg.min()):.3e}, expected >= 1. "
                f"Weights must be a convex combination over a non-empty bag.")
        return (s / neg.sqrt())[..., 1:].to(x.dtype)

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
        """The paper's init, quoted: "sampling from the uniform distribution
        U(-0.001, 0.001) and by setting x0 according to Equation 6". Here only
        x' is materialised, so Eq. 6 needs no separate step."""
        return torch.empty(*size, dtype=dtype, device=device).uniform_(-irange, irange)

    def origin(self, *size, dtype=None, device=None, seed: int = 42) -> Tensor:
        """The vertex (sqrt(k), 0, ..., 0), i.e. x' = 0. `seed` is unused but
        kept: it is part of geoopt.Manifold.origin's signature."""
        return torch.zeros(*size, dtype=dtype, device=device)
