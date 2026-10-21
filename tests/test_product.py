"""ProductManifold: Euclidean x Lorentz, each factor behaving exactly as on its own."""
import geoopt
import torch

from link_property_prediction.euclidean import EuclideanManifold
from link_property_prediction.lorentz import LorentzManifold
from link_property_prediction.model import LinkPredHead
from link_property_prediction.product import ProductManifold

D_E, D_H = 6, 5


def _points(n: int, scale: float = 0.5) -> torch.Tensor:
    return ProductManifold(D_E, D_H).random(n, D_E + D_H, radius=scale)


def test_dist_and_dist0_combine_the_factors():
    m, euc, hyp = ProductManifold(D_E, D_H), EuclideanManifold(), LorentzManifold()
    x, y = _points(9), _points(9)
    expected = (euc.dist(x[:, :D_E], y[:, :D_E]) ** 2 + hyp.dist(x[:, D_E:], y[:, D_E:]) ** 2).sqrt()
    assert torch.allclose(m.dist(x, y), expected)
    r_E, r_H = m.factor_radii(x)
    assert torch.allclose(r_E, euc.dist0(x[:, :D_E])) and torch.allclose(r_H, hyp.dist0(x[:, D_E:]))
    assert torch.allclose(m.dist0(x), (r_E ** 2 + r_H ** 2).sqrt())


def test_dist_gradient_finite_at_coincident_points():
    x = _points(4).clone().requires_grad_(True)
    ProductManifold(D_E, D_H).dist(x, x.detach()).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_weighted_midpoint_is_per_factor():
    m = ProductManifold(D_E, D_H)
    x = _points(12).view(3, 4, D_E + D_H)
    w = torch.softmax(torch.randn(3, 4), dim=-1)
    mid = m.weighted_midpoint(x, w)
    assert torch.allclose(mid[:, :D_E], EuclideanManifold().weighted_midpoint(x[..., :D_E], w))
    assert torch.allclose(mid[:, D_E:], LorentzManifold().weighted_midpoint(x[..., D_E:], w))


def test_random_matches_single_geometry_scale():
    torch.manual_seed(0)
    m = ProductManifold(64, 64)
    x = m.random(20000, 128, radius=1e-3)
    assert isinstance(x, geoopt.ManifoldTensor) and x.manifold is m
    r_E, r_H = m.factor_radii(x)
    assert abs(float(m.dist0(x).mean()) - 1e-3) < 2e-5
    assert abs(float(r_E.mean()) - 1e-3 / 2 ** 0.5) < 2e-5
    assert abs(float(r_H.mean()) - 1e-3 / 2 ** 0.5) < 2e-5


def test_riemannian_adam_steps_each_factor_as_on_its_own():
    torch.manual_seed(0)
    init = _points(10)
    grads = [torch.randn(10, D_E + D_H) for _ in range(5)]
    prod = geoopt.ManifoldParameter(init.clone(), manifold=ProductManifold(D_E, D_H))
    euc = geoopt.ManifoldParameter(init[:, :D_E].clone(), manifold=EuclideanManifold())
    hyp = geoopt.ManifoldParameter(init[:, D_E:].clone(), manifold=LorentzManifold())
    opt_p = geoopt.optim.RiemannianAdam([prod], lr=1e-2, stabilize=10)
    opt_s = geoopt.optim.RiemannianAdam([euc, hyp], lr=1e-2, stabilize=10)
    for g in grads:
        prod.grad, euc.grad, hyp.grad = g.clone(), g[:, :D_E].clone(), g[:, D_E:].clone()
        opt_p.step(); opt_s.step()
    assert torch.allclose(prod[:, :D_E], euc, atol=1e-6)
    assert torch.allclose(prod[:, D_E:], hyp, atol=1e-6)


def test_head_splits_d_emb_between_the_factors():
    head = LinkPredHead(num_nodes=50, T_train=100, max_walk_len=5, d_emb=8)
    assert isinstance(head.geom, ProductManifold) and head.E.weight.manifold is head.geom
    assert [s[0] for s in head.geom.shapes] == [4, 4]
