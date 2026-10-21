"""Euclidean x Lorentz product manifold with the LorentzManifold interface: dist, dist0,
weighted_midpoint, random. geoopt.ProductManifold supplies the optimiser side, factor by factor."""

import geoopt
import torch
from torch import Tensor

from .euclidean import EuclideanManifold
from .lorentz import LorentzManifold

__all__ = ["ProductManifold"]

_SQ_DIST_FLOOR = 1e-12


class ProductManifold(geoopt.ProductManifold):
    """R^a x H^b: a point is [x_E | x_H] on the last axis; distances combine as sqrt(d_E^2 + d_H^2)."""

    def __init__(self, d_euclidean: int, d_hyperbolic: int) -> None:
        super().__init__((EuclideanManifold(), d_euclidean), (LorentzManifold(), d_hyperbolic))

    def _parts(self, x: Tensor):
        """The factor slices of x: [x_E, x_H]."""
        return [self.take_submanifold_value(x, i) for i in range(len(self.manifolds))]

    def dist(self, x: Tensor, y: Tensor, *, keepdim: bool = False) -> Tensor:
        """Product geodesic distance; the floor keeps the gradient finite at x == y."""
        sq = sum(m.dist(a, b, keepdim=keepdim) ** 2
                 for m, a, b in zip(self.manifolds, self._parts(x), self._parts(y)))
        return sq.clamp_min(_SQ_DIST_FLOOR).sqrt()

    def dist0(self, x: Tensor, *, keepdim: bool = False) -> Tensor:
        """Distance to the product origin."""
        return torch.stack(self.factor_radii(x, keepdim=keepdim)).pow(2).sum(0).clamp_min(_SQ_DIST_FLOOR).sqrt()

    def factor_radii(self, x: Tensor, *, keepdim: bool = False):
        """Each factor's distance to its origin: [r_E, r_H]."""
        return [m.dist0(a, keepdim=keepdim) for m, a in zip(self.manifolds, self._parts(x))]

    def weighted_midpoint(self, x: Tensor, w: Tensor) -> Tensor:
        """Each factor's midpoint (weighted mean, Lorentzian centroid), concatenated."""
        return torch.cat([m.weighted_midpoint(a, w) for m, a in zip(self.manifolds, self._parts(x))], -1)

    def random(self, *size, dtype=None, device=None, radius: float = 1e-3) -> Tensor:
        """Each factor's own init at radius * sqrt(d_i / d): the per-coordinate scale of a single
        geometry at width d, and a product radius concentrating at `radius`."""
        d = size[-1]
        parts = [m.random(*size[:-1], shape[0], dtype=dtype, device=device,
                          radius=radius * (shape[0] / d) ** 0.5)
                 for m, shape in zip(self.manifolds, self.shapes)]
        return geoopt.ManifoldTensor(torch.cat(parts, -1), manifold=self)
