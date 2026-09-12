"""Lorentz model of hyperbolic geometry, in intrinsic coordinates.

    Nickel & Kiela, Learning Continuous Hierarchies in the Lorentz Model of
    Hyperbolic Geometry. ICML 2018, PMLR 80. arXiv:1806.03417

Equation numbers refer to that paper. A point IS x' in R^n: Eq. 6 determines
the time coordinate x0, so it is derived where needed and never stored.
`geoopt.Lorentz` stores it and trusts it, which float32 cannot support --
resolving the "+k" in x0^2 = k + ||x'||^2 needs it above one ULP, so past
x0^2 ~ 2^24 the constraint <x,x>_L = -k is silently lost.

Dtype passthrough: every operation computes in the dtype it is handed and
returns it. Nothing upcasts. That is only safe because the expressions below
are restructured to avoid the cancellations the printed forms suffer; each
rearrangement is pinned by a test in tests/test_lorentz.py, which is where the
supporting measurements live. Pass float64 tensors if you need more precision
than float32 gives -- past hyperbolic radius ~12, `inner` with u nearly
parallel to x' is the first thing to degrade.

Two geoopt notes. `geoopt.Lorentz(k=2.7)` stores k as float32 via
torch.as_tensor, so it carries a curvature error into float64 arithmetic; k
here is a plain Python float, which adopts the input dtype and cannot be
demoted by model.float(). It is therefore absent from state_dict() and must be
passed when reloading. And the base-class `component_inner` is
inner(..., keepdim=True), a scalar per point, so RiemannianAdam here is
per-embedding adaptive rather than per-coordinate -- not comparable to a
coordinatewise Euclidean Adam.
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
    """sqrt(t) with a zero gradient at t == 0 instead of an infinite one.

    1/(2 sqrt t) diverges at 0, poisoning the batch's gradient with NaN, and
    t == 0 is legitimate here: coincident points (dist) or the origin (dist0).
    Zero is the right subgradient -- t == 0 is the minimum. The dummy must be
    substituted BEFORE the sqrt: torch.where evaluates both arms, so masking
    the output still runs sqrt(0), whose backward is inf and 0 * inf = nan
    survives. A real NaN in t is propagated, not sent to the zero branch.
    """
    pos = t > 0
    out = torch.where(pos, torch.sqrt(torch.where(pos, t, torch.ones_like(t))), torch.zeros_like(t))
    return torch.where(torch.isnan(t), t, out)


def _tiny(t: Tensor) -> float:
    """Smallest positive normal of t's dtype, as a division floor. Read from
    the dtype because a fixed 1e-30 underflows to 0.0 in float16."""
    return torch.finfo(t.dtype).tiny


# Outlier-shave policy for projx (invoked by geoopt's `stabilize`). A few rare
# nodes drift to a radius where float32 stops resolving the tangential
# displacements parallel transport needs; there transp's residual reignites an
# Adam momentum feedback loop and expmap overflows (the ML-20M seed-5 crash).
# Those positions are already numerically meaningless -- past the wall the
# stored x' no longer satisfies <x,x>_L = -k -- so projx pulls the detached tail
# back to the cloud edge and leaves every other row untouched.
_SHAVE_K = 2.0           # fence = p50 + K*(p99.9 - p50): robust z-score, cloud body as scale
_SHAVE_PULL_Q = 0.999    # pull flagged rows back to this radius quantile (the cloud edge)


class LorentzManifold(geoopt.Manifold):
    """Lorentz model in intrinsic coordinates. Points are x' in R^n.

    k is the curvature parameter: the sheet is <x,x>_L = -k, curvature -1/k.
    k=1 is the paper, which never parameterises curvature; k != 1 follows
    geoopt's convention.
    """

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
    # internals
    #
    # No ambient vector is ever built: every operation needs only the space
    # part of its ambient answer, which depends on the ambient vectors solely
    # through x' and the scalar x0. `_lift` exists for callers that want
    # geoopt.Lorentz's convention; nothing here uses it.
    # ------------------------------------------------------------------
    def _x0(self, x: Tensor) -> Tensor:
        """Eq. 6: x0 = sqrt(k + ||x'||^2). A sum of positive terms, so well
        conditioned at any width."""
        return torch.sqrt(self.k + (x * x).sum(-1, keepdim=True))

    def _lift(self, x: Tensor) -> Tensor:
        """[..., n] -> [..., n+1] ambient point, for interop."""
        return torch.cat([self._x0(x), x], -1)

    def _gap(self, x: Tensor, y: Tensor) -> Tensor:
        """w = -<X,Y>_L/k - 1 >= 0, needed by dist and logmap.

        Two rearrangements, both load-bearing:
            -<x,y>_L - k = (||x'-y'||^2 - (x0-y0)^2) / 2
                 x0 - y0 = (x'-y').(x'+y') / (x0 + y0)
        The first differences coordinates before squaring; the second avoids
        subtracting two nearly equal large x0. Without them, nearby points at
        radius collapse to distance 0 in float32 -- and nearby pairs are what
        the paper's N(i,j) ranking loss compares.
        """
        x0, y0 = self._x0(x), self._x0(y)
        d = x - y
        d0 = (d * (x + y)).sum(-1, keepdim=True) / (x0 + y0)
        w = ((d * d).sum(-1, keepdim=True) - d0 * d0) * self._half_inv_k
        return w.clamp_min(0.0)

    # ------------------------------------------------------------------
    # geoopt.Manifold API (the abstract set)
    # ------------------------------------------------------------------
    def _dtype_radius_limit(self, x: Tensor) -> Tensor:
        """Radius past which `k + ||x'||^2 == ||x'||^2` in x's dtype: x0 then
        collapses onto ||x'|| and the point sits on the light cone numerically,
        not the hyperboloid. That happens once ||x'||^2 ~ k/eps. This is the
        hard ceiling the stat-driven fence is capped by, so a table that has
        drifted wholesale can never place the fence itself past the wall. It is
        dtype-derived, not a float32 constant: ~8.7 in float32, ~18.7 in
        float64, so the class stays dtype-agnostic."""
        norm_limit = math.sqrt(self.k / torch.finfo(x.dtype).eps)
        r = self._sqrt_k * math.asinh(norm_limit * self._inv_sqrt_k)
        return torch.tensor(r, dtype=x.dtype, device=x.device)

    def projx(self, x: Tensor) -> Tensor:
        """Outlier shave -- not identity. geoopt calls projx at `stabilize`
        intervals to repair points that have drifted off the manifold. In this
        chart the drift is not a lost x0 (x0 is derived, never stored) but a
        radius past the float32 resolution wall, where transp's residual
        reignites an Adam feedback loop and expmap overflows (ML-20M seed 5).
        Rows out there are already meaningless -- <x,x>_L = -k no longer holds
        numerically -- so we pull the detached tail back to the cloud edge and
        leave every other row exactly where it was.

        The fence is stat-driven, no fixed radius: p50 + K*(p99.9 - p50) over
        the table's own radii (a robust z-score using the cloud body as scale),
        capped by the dtype's resolution limit. It is identity on a healthy
        table -- if nothing sits past the fence, x is returned unchanged -- so
        the only rows it ever moves are a genuinely detached tail.

        No row-count threshold, so small graphs are protected too: a healthy
        cloud of any size keeps its max below the fence (fence = 2 p99.9 - p50 >
        p99.9), and a small graph with a runaway node still has it caught by the
        dtype cap even where too few points make the stat fence unreliable.
        """
        # torch.quantile supports only float32/64, so estimate the fence there;
        # never downcast float64. The rescale and the output stay in x's dtype,
        # so this is still a pure dtype passthrough.
        radius = self.dist0(x)
        r = radius if radius.dtype in (torch.float32, torch.float64) else radius.float()
        median = torch.quantile(r, 0.5)
        cloud_edge = torch.quantile(r, _SHAVE_PULL_Q)             # p99.9
        fence = median + _SHAVE_K * (cloud_edge - median)
        fence = torch.minimum(fence, self._dtype_radius_limit(x).to(r.dtype))

        outlier = r > fence
        if not bool(outlier.any()):
            return x

        # Pull each flagged row inward along its own ray to the cloud-edge
        # radius. ||x'|| = sqrt(k) sinh(r / sqrt(k)) inverts dist0 exactly, so
        # rescaling the norm sets the radius with no change of direction.
        target_radius = torch.minimum(cloud_edge, fence)
        target_norm = self._sqrt_k * torch.sinh(target_radius * self._inv_sqrt_k)
        norm = x.norm(dim=-1, keepdim=True).clamp_min(_tiny(x))
        pulled = x * (target_norm.to(x.dtype) / norm)
        return torch.where(outlier.unsqueeze(-1), pulled, x)

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        """Identity. T_x is all of R^n here: the constraint <x,v>_L = 0 is
        absorbed by the parameterisation."""
        return u

    def inner(self, x: Tensor, u: Tensor, v: Optional[Tensor] = None,
              *, keepdim: bool = False) -> Tensor:
        """Induced metric, the pullback of ds^2 = -dx0^2 + ||dx'||^2 through
        Eq. 6:  <u,v>_x = u.v - (x'.u)(x'.v)/x0^2.

        Evaluated through the identical split, with q_u = (x'.u)/x0^2:

            <u,v>_x = (u - q_u x').(v - q_v x') + q_u q_v k

        The printed form is the one badly conditioned expression in this model:
        as u nears parallel to x' its two terms cancel to a result of size
        ||u||^2 k/x0^2. For u = v the split's terms are both non-negative, so
        they cannot. Used at both widths -- the printed form is unsafe in
        float64 too. Dividing by x0^2 >= k > 0 rather than ||x'||^2 avoids
        needing any floor at the origin.
        """
        same = v is None or v is u
        x0sq = self.k + (x * x).sum(-1, keepdim=True)
        qu = (x * u).sum(-1, keepdim=True) / x0sq
        if same:
            v, qv = u, qu
        else:
            qv = (x * v).sum(-1, keepdim=True) / x0sq
        res = (((u - qu * x) * (v - qv * x)).sum(-1, keepdim=True) + qu * qv * self.k)
        return res if keepdim else res.squeeze(-1)

    def egrad2rgrad(self, x: Tensor, u: Tensor) -> Tensor:
        """Eq. 10 in this chart: g^-1 grad = u + x' (x'.u)/k, by
        Sherman-Morrison on g = I - x'x'^T/x0^2. The paper's sign flip and
        tangent projection are both absorbed: there is nothing to project."""
        return u + x * ((x * u).sum(-1, keepdim=True) * self._inv_k)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """Eq. 9, generalised to curvature -1/k. Returns the space part, which
        is cosh(t) x' + coef u.

        POST: d(x, expmap(x,u)) == sqrt(inner(x,u,u)). The tangent norm IS the
        geodesic distance travelled, which is what makes a learning rate
        interpretable in hyperbolic units.

        A step large enough to overflow cosh(t)*x0 gives inf rather than
        raising; `_check_point_on_manifold` reports non-finite values, and a
        device-side guard would stall the pipeline on every call.
        """
        qq = self.inner(x, u, keepdim=True).clamp_min(0.0)   # ||v||_L^2
        tsq = qq * self._inv_k                               # t^2

        # v/||v||_L is 0/0 at ||v||=0. Sanitise the INPUT of sqrt, not the
        # output of the branch: torch.where evaluates both arms, and sqrt's
        # backward is grad/(2*result), so a 0 in the untaken arm yields NaN
        # that survives the mask. The Taylor heads (cosh t = 1 + t^2/2,
        # sinh(t)/t = 1 + t^2/6) only run below finfo.tiny and are immaterial
        # there; they are kept because they are the correct limits.
        safe = qq > _tiny(qq)
        nrm = torch.where(safe, qq, torch.ones_like(qq)).sqrt()
        t = nrm * self._inv_sqrt_k
        cosh_t = torch.where(safe, torch.cosh(t), 1.0 + tsq * 0.5)
        coef = torch.where(safe, torch.sinh(t) * (self._sqrt_k / nrm),
                           1.0 + tsq * _SIXTH)
        return torch.addcmul(cosh_t * x, coef, u)

    def retr(self, x: Tensor, u: Tensor) -> Tensor:
        """The exponential map is closed form, so it IS the retraction -- the
        2018 paper's improvement over the 2017 first-order retraction."""
        return self.expmap(x, u)

    def _check_point_on_manifold(self, x: Tensor, *, atol: float = 1e-5,
                                 rtol: float = 1e-5) -> Tuple[bool, Optional[str]]:
        """Only finiteness can fail: the chart cannot represent an invalid point."""
        if not torch.isfinite(x).all():
            return False, "point contains non-finite values"
        return True, None

    def _check_vector_on_tangent(self, x: Tensor, u: Tensor, *, atol: float = 1e-5,
                                 rtol: float = 1e-5) -> Tuple[bool, Optional[str]]:
        """T_x = R^n, so any finite u is tangent."""
        if not torch.isfinite(u).all():
            return False, "tangent vector contains non-finite values"
        return True, None

    # ------------------------------------------------------------------
    # distance
    # ------------------------------------------------------------------
    def dist(self, x: Tensor, y: Tensor, *, keepdim: bool = False) -> Tensor:
        """Eq. 5, via arcosh(1 + w) = 2 asinh(sqrt(w/2)) with w from _gap.

        The substitution avoids geoopt's arcosh, which is
        log(x + sqrt(clamp_min(x^2 - 1, 1e-15))): it clamps the sqrt argument
        but not x, so cancellation below 1 makes it return log(x) < 0, a
        negative distance. It also removes that 1e-15 clamp floor, so self-
        pairs return exactly 0.

        AT w = 0 the true derivative diverges: the gradient of d has unit
        norm everywhere but undefined direction at coincident points. Rather
        than requiring callers to exclude self-pairs -- which our single
        callsite cannot, since it scores a source against every candidate and
        the candidate pool may contain the source -- _safe_sqrt selects the
        zero subgradient there. Values where w > 0 are bit-identical.
        """
        res = (2.0 * self._sqrt_k) * torch.asinh(_safe_sqrt(self._gap(x, y) * 0.5))
        return res if keepdim else res.squeeze(-1)

    def dist0(self, x: Tensor, *, keepdim: bool = False) -> Tensor:
        """Distance to the origin. w = x0/sqrt(k) - 1, formed as
        ||x'||^2 / (sqrt(k)(x0 + sqrt(k))) to remove the subtraction: near the
        origin x0 -> sqrt(k), and the paper initialises at U(-0.001, 0.001),
        so that is where every run starts."""
        n2 = (x * x).sum(-1, keepdim=True)
        x0 = torch.sqrt(self.k + n2)
        w = (n2 / (self._sqrt_k * (x0 + self._sqrt_k))).clamp_min(0.0)
        res = (2.0 * self._sqrt_k) * torch.asinh(_safe_sqrt(w * 0.5))
        return res if keepdim else res.squeeze(-1)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Inverse of Eq. 9:

            log_x(y) = arcosh(a)/sqrt(a^2 - 1) * (Y - a X),  a = -<X,Y>_L/k

        Not in the paper (Algorithm 1 never needs it), but the base class
        raises NotImplementedError, which breaks Riemannian means,
        interpolation, RiemannianLineSearch and any map-to-T_0 / aggregate /
        map-back layer. Same substitution as dist(); the mask is on the input
        of sqrt, or one coincident pair NaNs the whole batch's gradient.
        """
        w = self._gap(x, y)
        a = w + 1.0                          # linear in w, no branch needed
        safe = w > _tiny(w)
        w_safe = torch.where(safe, w, torch.ones_like(w))
        num = 2.0 * torch.asinh(torch.sqrt(w_safe * 0.5))    # == arcosh(a)
        den = torch.sqrt(w_safe * (w_safe + 2.0))            # == sqrt(a^2 - 1)
        coef = torch.where(safe, num / den, 1.0 - w * _THIRD)
        return coef * (y - a * x)

    def transp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport T_x -> T_y:

            PT(v) = v + <Y-X, V>_L / (k (2 + w)) * (x + y)

        with k - <X,Y>_L = k(2 + w) and <Y,V>_L = <Y-X,V>_L since <X,V>_L = 0.

        <Y-X,V>_L is NOT formed as (d.v) - d0 (x.v)/x0. For momentum with a
        radial component v_r and a radial step delta both terms are delta*v_r
        and the true difference is delta*v_r*k/x0^2; at r = 11.4 that is 250x
        below float32 rounding, so the computed value is noise. That noise
        multiplies (x + y) ~ 2||x'|| and lands in the radial momentum, ~3% per
        transport at r = 11.4, systematically, and the resulting longer step
        makes the next error larger. That loop is how a 6.7e-3 step became
        149.8 and overflowed cosh in expmap.

        Split v and d into radial and tangential parts about n = x'/||x'||.
        With P = d.(x+y) = 2||x'|| d_r + ||d||^2 (difference first), d0 = y0 - x0
        = P/(y0 + x0), and the identity k + x0 y0 - ||x'||^2 = 2k + x0 d0,

            <Y-X,V>_L = v_r [d_r (2k + x0 d0) - ||x'|| ||d||^2] / (x0 (x0 + y0))
                        + d_perp . v_perp

        The bracket is exact to ~eps ||x'|| delta / (2k) relative (0.8% at
        r = 11.4, delta = 3), which enters the transported momentum scaled by
        the step and is immaterial. d_perp.v_perp is used rather than d.v_perp
        because v_perp computed as v - v_r n carries an n-component of size
        eps v_r, and d_r times that is again the noise term. Measured over 300
        radial transports at r = 11.4 in float32 the Riemannian norm of the
        momentum is preserved to 0.3%, against 0 or +12% for the previous form.
        Agrees with the previous form to 1e-9 in float64 at r <= 5.

        At the origin n = 0, so v_r = d_r = 0, the bracket term vanishes and
        yv = d.v, which is the correct limit.
        """
        x0, y0 = self._x0(x), self._x0(y)
        d = y - x
        big_p = (d * (x + y)).sum(-1, keepdim=True)
        d0 = big_p / (y0 + x0)
        dd = (d * d).sum(-1, keepdim=True)
        w = ((dd - d0 * d0) * self._half_inv_k).clamp_min(0.0)
        nx = x.norm(dim=-1, keepdim=True)
        n = x / nx.clamp_min(_tiny(nx))
        vr = (n * v).sum(-1, keepdim=True)
        vperp = v - vr * n
        dr = (n * d).sum(-1, keepdim=True)
        dperp = d - dr * n
        bracket = (dr * (2.0 * self.k + x0 * d0) - nx * dd) / (x0 * (x0 + y0))
        yv = vr * bracket + (dperp * vperp).sum(-1, keepdim=True)
        coef = yv / (self.k * (2.0 + w))
        return torch.addcmul(v, coef, x + y)

    # ------------------------------------------------------------------
    # extensions used by this project (NOT in the paper)
    # ------------------------------------------------------------------
    def midpoint(self, x: Tensor, w: Tensor) -> Tensor:
        """Weighted Lorentzian centroid (Law et al., ICML 2019):

            mu = s / sqrt(-<s,s>_L / k),   s = sum_t w_t x_t

        x [..., T, n], w [..., T] -> [..., n]. Minimises squared Lorentzian
        distance, not squared geodesic distance, so it is not the Riemannian
        barycenter (no closed form). The /k is load-bearing for k != 1.

        neg = -<s,s>_L / k is formed as s0^2 - ||s'||^2, which cancels: it is
        Eq. 6 applied to a weighted sum, and for a bag whose weight sits on
        one point at radius r the two terms are ~ (w sinh r)^2 while the
        result is O((sum w)^2). This is the one O(T n) operation that loses
        accuracy in float32; the pairwise form is exact but O(T^2 n).

        Floor. For valid points and non-negative weights,
        neg = sum_tu w_t w_u (1 + gap_tu) >= (sum_t w_t)^2, with equality iff
        all points coincide. That is the smallest legitimate value, so the
        floor is (sum w)^2: a no-op on every valid input at every scale, and
        a bound of 1/(sum w) on the multiplier when cancellation drives the
        computed neg to 0 or below. A floor at 1 is wrong for weights that do
        not sum to 1: neg scales as c^2 and s as c, so the ratio is invariant
        only while the floor does not fire; one point with weight 0.5 has
        neg = 0.25, would be floored to 1, and would come back at half its
        radius, off the manifold. The inner clamp handles the all-zero bag
        (0/0). With the floor the degenerate result is s/(sum w), the
        Euclidean mean of the x'_t: finite, near the bag, off-manifold.
        """
        wu = w.unsqueeze(-1)
        s0 = (wu * self._x0(x)).sum(dim=-2)
        s = (wu * x).sum(dim=-2)
        neg = (s0 * s0 - (s * s).sum(-1, keepdim=True)) * self._inv_k
        wsum = w.sum(-1, keepdim=True)
        return s * neg.clamp_min((wsum * wsum).clamp_min(_tiny(neg))).rsqrt()

    def to_poincare(self, x: Tensor) -> Tensor:
        """Eq. 11: x'/(x0 + sqrt(k)), giving unit Poincare ball coordinates."""
        return x / (self._x0(x) + self._sqrt_k)

    def from_poincare(self, u: Tensor) -> Tensor:
        """Inverse of Eq. 11. The ball is open, so 1 - ||u||^2 -> 0 at the
        boundary -- precisely the blow-up the Lorentz model avoids."""
        n2 = (u * u).sum(-1, keepdim=True)
        return (2.0 * self._sqrt_k) * u / (1.0 - n2).clamp_min(_tiny(u))

    # ------------------------------------------------------------------
    # initialisation (paper, section 3.2.2)
    # ------------------------------------------------------------------
    def random(self, *size, dtype=None, device=None, irange: float = 1e-3) -> Tensor:
        """The paper's init: x' ~ U(-0.001, 0.001), with x0 following from
        Eq. 6 and so needing no separate step."""
        return geoopt.ManifoldTensor(
            torch.empty(*size, dtype=dtype, device=device).uniform_(-irange, irange),
            manifold=self,
        )

    def origin(self, *size, dtype=None, device=None, seed: int = 42) -> Tensor:
        """The vertex, i.e. x' = 0. `seed` is unused but part of
        geoopt.Manifold.origin's signature."""
        return geoopt.ManifoldTensor(
            torch.zeros(*size, dtype=dtype, device=device), manifold=self
        )
