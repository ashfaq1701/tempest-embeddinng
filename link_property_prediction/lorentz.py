"""Lorentz model of hyperbolic geometry in AMBIENT coordinates, numerically stable.

A point is (x0, x') in R^{n+1} on the sheet <x,x>_L = -k, x0 > 0, with the Minkowski product
<x,y>_L = -x0 y0 + x'.y'. The stored x0 is a convenience for callers that want geoopt's
(n+1)-vector convention; it is NEVER trusted in arithmetic. Every operation re-derives
x0 = sqrt(k + ||x'||^2) from the spatial part and evaluates its formula in a form that does
not subtract nearly equal large numbers. That is the whole difference from geoopt.Lorentz,
whose dist/logmap/inner evaluate -<x,y>_L = x0 y0 - x'.y' directly: past hyperbolic radius ~8
in float32 the "+k" in x0^2 = k + ||x'||^2 is below one ULP, the stored constraint is lost,
arcosh receives arguments below 1, and the result is NaN. Here the same quantities are built
from coordinate differences (see _gap) and sums of positive terms (see dist0), so they stay
finite and accurate at any radius float32 can represent at all.

Conventions: k is the curvature parameter (curvature -1/k; k = 1 is the usual sheet).
Tangent vectors are ambient vectors with <x,u>_L = 0, i.e. u0 = (x'.u')/x0. Dtype is passed
through unchanged; nothing upcasts.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import geoopt
import torch
from torch import Tensor

__all__ = ["LorentzManifold"]

_THIRD = 1.0 / 3.0
_SIXTH = 1.0 / 6.0


def _safe_sqrt(t: Tensor) -> Tensor:
    """sqrt with a zero subgradient at t == 0. The dummy is substituted BEFORE the sqrt:
    torch.where evaluates both arms, and sqrt(0) has an infinite backward that survives masking."""
    pos = t > 0
    out = torch.where(pos, torch.sqrt(torch.where(pos, t, torch.ones_like(t))), torch.zeros_like(t))
    return torch.where(torch.isnan(t), t, out)


def _tiny(t: Tensor) -> float:
    """Smallest positive normal of t's dtype; a fixed 1e-30 underflows to 0 in float16."""
    return torch.finfo(t.dtype).tiny


class LorentzManifold(geoopt.Manifold):
    """Lorentz model, points (x0, x') in R^{n+1}. x0 is always recomputed from x'."""

    name = "LorentzManifold"
    ndim = 1
    reversible = False

    def __init__(self, k: float = 1.0) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = float(k)
        self._sqrt_k = math.sqrt(k)
        self._inv_k = 1.0 / k
        self._inv_sqrt_k = 1.0 / math.sqrt(k)
        self._half_inv_k = 0.5 / k

    def extra_repr(self) -> str:
        return f"k={self.k}"

    # ------------------------------------------------------------------
    # coordinate helpers: the spatial part is the state, x0 is derived
    # ------------------------------------------------------------------
    @staticmethod
    def _sp(x: Tensor) -> Tensor:
        """Spatial part x' of an ambient vector."""
        return x[..., 1:]

    def _x0(self, xs: Tensor) -> Tensor:
        """x0 = sqrt(k + ||x'||^2): a sum of positive terms, exact at any width."""
        return torch.sqrt(self.k + (xs * xs).sum(-1, keepdim=True))

    def _point(self, xs: Tensor) -> Tensor:
        """Ambient point from a spatial part."""
        return torch.cat([self._x0(xs), xs], -1)

    def _tangent(self, xs: Tensor, us: Tensor) -> Tensor:
        """Ambient tangent at the point with spatial part xs, from its spatial part us:
        u0 = (x'.u')/x0 is the Minkowski-orthogonality condition solved for u0, a
        well-conditioned division."""
        u0 = (xs * us).sum(-1, keepdim=True) / self._x0(xs)
        return torch.cat([u0, us], -1)

    def _gap(self, xs: Tensor, ys: Tensor) -> Tensor:
        """w = -<X,Y>_L/k - 1 >= 0 from the spatial parts, via
            -<x,y>_L - k = (||x'-y'||^2 - (x0-y0)^2) / 2
                 x0 - y0 = (x'-y').(x'+y') / (x0 + y0)
        Differences of coordinates are taken before squaring, and x0 - y0 is formed
        without subtracting two large x0. This is the rearrangement geoopt lacks."""
        x0, y0 = self._x0(xs), self._x0(ys)
        d = xs - ys
        d0 = (d * (xs + ys)).sum(-1, keepdim=True) / (x0 + y0)
        w = ((d * d).sum(-1, keepdim=True) - d0 * d0) * self._half_inv_k
        return w.clamp_min(0.0)

    # ------------------------------------------------------------------
    # geoopt.Manifold API
    # ------------------------------------------------------------------
    def projx(self, x: Tensor) -> Tensor:
        """Recompute x0 from x'. Every x' is a valid point, so this only repairs a stale
        time coordinate; geoopt's `stabilize` calls it periodically."""
        return self._point(self._sp(x))

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Project an ambient vector onto T_x: keep u', re-derive u0 from the constraint.
        The input's u0 is discarded, so a vector built in the wrong tangent space is
        repaired rather than trusted."""
        return self._tangent(self._sp(x), self._sp(u))

    def inner(self, x: Tensor, u: Tensor, v: Optional[Tensor] = None,
              *, keepdim: bool = False) -> Tensor:
        """Minkowski product of two tangent vectors at x, evaluated on spatial parts:
            <u,v>_x = u'.v' - (x'.u')(x'.v')/x0^2
        computed through the split (u' - q_u x').(v' - q_v x') + q_u q_v k with
        q = (x'.u')/x0^2, whose terms cannot cancel. The printed form, and the ambient
        -u0 v0 + u'.v', both cancel catastrophically as u nears the radial direction."""
        xs = self._sp(x)
        us = self._sp(u)
        same = v is None or v is u
        x0sq = self.k + (xs * xs).sum(-1, keepdim=True)
        qu = (xs * us).sum(-1, keepdim=True) / x0sq
        if same:
            vs, qv = us, qu
        else:
            vs = self._sp(v)
            qv = (xs * vs).sum(-1, keepdim=True) / x0sq
        res = ((us - qu * xs) * (vs - qv * xs)).sum(-1, keepdim=True) + qu * qv * self.k
        return res if keepdim else res.squeeze(-1)

    def egrad2rgrad(self, x: Tensor, g: Tensor) -> Tensor:
        """Riemannian gradient from an ambient Euclidean gradient (g0, g').

        x0 is a function of x', so a loss that touched the stored x0 contributes to x'
        through the chain rule d x0 / d x' = x' / x0. Fold g0 in, then apply the intrinsic
        metric inverse g' + x'(x'.g')/k (Sherman-Morrison on I - x'x'^T/x0^2), then lift as
        a tangent vector. Nothing is projected after the fact."""
        xs = self._sp(x)
        gs = self._sp(g) + g[..., :1] * xs / self._x0(xs)
        rs = gs + xs * ((xs * gs).sum(-1, keepdim=True) * self._inv_k)
        return self._tangent(xs, rs)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """exp_x(u) = cosh(t) x + sinh(t)/t u with t = ||u||_x / sqrt(k), returned as an
        ambient point with x0 re-derived (not cosh(t) x0 + ... which drifts).
        POST: d(x, exp_x(u)) == sqrt(inner(x, u, u))."""
        xs = self._sp(x)
        us = self._sp(u)
        qq = self.inner(x, u, keepdim=True).clamp_min(0.0)   # ||u||_x^2
        tsq = qq * self._inv_k
        safe = qq > _tiny(qq)
        nrm = torch.where(safe, qq, torch.ones_like(qq)).sqrt()
        t = nrm * self._inv_sqrt_k
        cosh_t = torch.where(safe, torch.cosh(t), 1.0 + tsq * 0.5)
        coef = torch.where(safe, torch.sinh(t) * (self._sqrt_k / nrm), 1.0 + tsq * _SIXTH)
        ys = torch.addcmul(cosh_t * xs, coef, us)
        return self._point(ys)

    def retr(self, x: Tensor, u: Tensor) -> Tensor:
        """The exponential map is closed form, so it is the retraction."""
        return self.expmap(x, u)

    def _check_point_on_manifold(self, x: Tensor, *, atol: float = 1e-5,
                                 rtol: float = 1e-5) -> Tuple[bool, Optional[str]]:
        """Finite, x0 > 0, and x0 consistent with x' to within tolerance. A stale x0 is
        reported rather than silently accepted; projx repairs it."""
        if not torch.isfinite(x).all():
            return False, "point contains non-finite values"
        x0 = x[..., :1]
        if not bool((x0 > 0).all()):
            return False, "x0 must be positive"
        ok = torch.allclose(x0, self._x0(self._sp(x)), atol=atol, rtol=rtol)
        return (True, None) if ok else (False, "x0 inconsistent with ||x'||; call projx")

    def _check_vector_on_tangent(self, x: Tensor, u: Tensor, *, atol: float = 1e-5,
                                 rtol: float = 1e-5) -> Tuple[bool, Optional[str]]:
        if not torch.isfinite(u).all():
            return False, "tangent vector contains non-finite values"
        xs = self._sp(x)
        want = (xs * self._sp(u)).sum(-1, keepdim=True) / self._x0(xs)
        ok = torch.allclose(u[..., :1], want, atol=atol, rtol=rtol)
        return (True, None) if ok else (False, "u0 != (x'.u')/x0; call proju")

    # ------------------------------------------------------------------
    # distances
    # ------------------------------------------------------------------
    def dist(self, x: Tensor, y: Tensor, *, keepdim: bool = False) -> Tensor:
        """sqrt(k) arcosh(1 + w) via 2 sqrt(k) asinh(sqrt(w/2)), w from _gap. No arcosh of a
        value that can fall below 1, no clamp floor, exact 0 for coincident points, zero
        subgradient there instead of an infinite one."""
        res = (2.0 * self._sqrt_k) * torch.asinh(_safe_sqrt(self._gap(self._sp(x), self._sp(y)) * 0.5))
        return res if keepdim else res.squeeze(-1)

    def dist0(self, x: Tensor, *, keepdim: bool = False) -> Tensor:
        """Distance to the origin with w = x0/sqrt(k) - 1 formed as
        ||x'||^2 / (sqrt(k)(x0 + sqrt(k))): no subtraction, exact near the origin."""
        xs = self._sp(x)
        n2 = (xs * xs).sum(-1, keepdim=True)
        x0 = torch.sqrt(self.k + n2)
        w = (n2 / (self._sqrt_k * (x0 + self._sqrt_k))).clamp_min(0.0)
        res = (2.0 * self._sqrt_k) * torch.asinh(_safe_sqrt(w * 0.5))
        return res if keepdim else res.squeeze(-1)

    # ------------------------------------------------------------------
    # log map and transport
    # ------------------------------------------------------------------
    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """log_x(y) = arcosh(a)/sqrt(a^2 - 1) (Y - a X), a = 1 + w, on spatial parts, lifted
        as a tangent vector. The 0/0 at w = 0 is handled on the INPUT of the sqrt."""
        xs, ys = self._sp(x), self._sp(y)
        w = self._gap(xs, ys)
        a = w + 1.0
        safe = w > _tiny(w)
        w_safe = torch.where(safe, w, torch.ones_like(w))
        num = 2.0 * torch.asinh(torch.sqrt(w_safe * 0.5))     # arcosh(a)
        den = torch.sqrt(w_safe * (w_safe + 2.0))             # sqrt(a^2 - 1)
        coef = torch.where(safe, num / den, 1.0 - w * _THIRD)
        return self._tangent(xs, coef * (ys - a * xs))

    def transp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport T_x -> T_y:  PT(v) = v + <Y-X,V>_L / (k(2+w)) (X + Y), with
        <Y-X,V>_L evaluated through a radial/tangential split about n = x'/||x'|| and the
        rationalised difference
            <Y-X,V>_L = v_r [k d_r (2b + d_r) - b^2 ||d_perp||^2] / (x0 (x0 a + y0 b))
                        + d_perp . v_perp
        which is exact for radial and for perpendicular steps and never a difference of two
        near-copies in between. THE GROUPING IS LOAD-BEARING; the printed form loses the
        answer entirely once ||x'|| ||d|| > 2k/eps. Spatial parts only; the result is lifted."""
        xs, ys, vs = self._sp(x), self._sp(y), self._sp(v)
        x0, y0 = self._x0(xs), self._x0(ys)
        d = ys - xs
        d0 = (d * (xs + ys)).sum(-1, keepdim=True) / (y0 + x0)
        dd = (d * d).sum(-1, keepdim=True)
        w = ((dd - d0 * d0) * self._half_inv_k).clamp_min(0.0)
        nx = xs.norm(dim=-1, keepdim=True)
        n = xs / nx.clamp_min(math.sqrt(_tiny(nx)))           # unit radial direction; 0 at the origin
        vr = (n * vs).sum(-1, keepdim=True)
        vperp = vs - vr * n
        dr = (n * d).sum(-1, keepdim=True)
        dperp = d - dr * n
        dp2 = (dperp * dperp).sum(-1, keepdim=True)
        a = nx + dr
        conj = x0 * a + y0 * nx
        direct = x0 * a - y0 * nx
        num = self.k * dr * (2.0 * nx + dr) - nx * nx * dp2
        pos = conj > 0
        rational = num / torch.where(pos, conj, torch.ones_like(conj))
        yv = vr * (torch.where(pos, rational, direct) / x0) + (dperp * vperp).sum(-1, keepdim=True)
        coef = yv / (self.k * (2.0 + w))
        return self._tangent(ys, torch.addcmul(vs, coef, xs + ys))

    # ------------------------------------------------------------------
    # extensions
    # ------------------------------------------------------------------
    def weighted_midpoint(self, x: Tensor, w: Tensor) -> Tensor:
        """Weighted Lorentzian centroid (Law et al., 2019): mu = s / sqrt(-<s,s>_L / k),
        s = sum_t w_t X_t. x [..., T, n+1], w [..., T] -> [..., n+1]. -<s,s>_L/k is formed as
        s0^2 - ||s'||^2 with s0 built from re-derived x0's; it is the one O(Tn) expression here
        that cancels in float32 for a bag concentrated at radius. Floor at (sum w)^2, the
        smallest legitimate value, so it never fires on a valid bag."""
        xs = self._sp(x)
        wu = w.unsqueeze(-1)
        s0 = (wu * self._x0(xs)).sum(dim=-2)
        s = (wu * xs).sum(dim=-2)
        neg = (s0 * s0 - (s * s).sum(-1, keepdim=True)) * self._inv_k
        wsum = w.sum(-1, keepdim=True)
        ms = s * neg.clamp_min((wsum * wsum).clamp_min(_tiny(neg))).rsqrt()
        return self._point(ms)

    def to_poincare(self, x: Tensor) -> Tensor:
        """x' / (x0 + sqrt(k)), unit-ball coordinates; x0 re-derived."""
        xs = self._sp(x)
        return xs / (self._x0(xs) + self._sqrt_k)

    # ------------------------------------------------------------------
    # initialisation
    # ------------------------------------------------------------------
    def random(self, *size, dtype=None, device=None, irange: float = 1e-3) -> Tensor:
        """x' ~ U(-irange, irange), x0 derived. `size` is the ambient shape [..., n+1];
        the spatial part is drawn with n = size[-1] - 1."""
        *lead, n1 = size
        xs = torch.empty(*lead, n1 - 1, dtype=dtype, device=device).uniform_(-irange, irange)
        return geoopt.ManifoldTensor(self._point(xs), manifold=self)
