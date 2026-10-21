"""Euclidean geometry with the LorentzManifold interface: dist, dist0, weighted_midpoint, random."""

import geoopt
import torch
from torch import Tensor

__all__ = ["EuclideanManifold"]

_SQ_DIST_FLOOR = 1e-12


class EuclideanManifold(geoopt.Euclidean):
    """Flat R^n. geoopt.Euclidean supplies the optimiser side (identity gradient, retr = x + u)."""

    def __init__(self) -> None:
        super().__init__(ndim=1)

    def dist(self, x: Tensor, y: Tensor, *, keepdim: bool = False) -> Tensor:
        """||x - y||; the floor on the square keeps the gradient finite at x == y."""
        return ((x - y) ** 2).sum(-1, keepdim=keepdim).clamp_min(_SQ_DIST_FLOOR).sqrt()

    def dist0(self, x: Tensor, *, keepdim: bool = False) -> Tensor:
        """Distance to the origin, ||x||."""
        return x.norm(dim=-1, keepdim=keepdim)

    def weighted_midpoint(self, x: Tensor, w: Tensor) -> Tensor:
        """Weighted mean of x [..., T, n] under weights w [..., T] summing to 1."""
        return (w.unsqueeze(-1) * x).sum(-2)

    def random(self, *size, dtype=None, device=None, radius: float = 1e-3) -> Tensor:
        """Isotropic normal with ||x|| concentrating at `radius`: LorentzManifold.random without the expmap."""
        x = torch.randn(*size, dtype=dtype, device=device) * (radius / size[-1] ** 0.5)
        return geoopt.ManifoldTensor(x, manifold=self)
