"""EuclideanManifold: the LorentzManifold interface on flat R^n."""
import geoopt
import torch

from link_property_prediction.euclidean import EuclideanManifold


def test_dist_is_norm_and_finite_at_coincident_points():
    m = EuclideanManifold()
    x, y = torch.randn(7, 5), torch.randn(7, 5)
    assert torch.allclose(m.dist(x, y), (x - y).norm(dim=-1))
    assert torch.allclose(m.dist0(x), x.norm(dim=-1))

    x.requires_grad_(True)
    m.dist(x, x.detach()).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_weighted_midpoint_is_weighted_mean():
    m = EuclideanManifold()
    x = torch.randn(3, 4, 6)
    w = torch.softmax(torch.randn(3, 4), dim=-1)
    expected = torch.einsum("qt,qtd->qd", w, x)
    assert torch.allclose(m.weighted_midpoint(x, w), expected, atol=1e-6)


def test_random_concentrates_at_radius():
    torch.manual_seed(0)
    x = EuclideanManifold().random(20000, 128, radius=1e-3)
    assert isinstance(x, geoopt.ManifoldTensor)
    assert abs(float(x.norm(dim=-1).mean()) - 1e-3) < 2e-5


def test_riemannian_adam_matches_adam():
    torch.manual_seed(0)
    init, grads = torch.randn(10, 4), [torch.randn(10, 4) for _ in range(5)]
    p_r = geoopt.ManifoldParameter(init.clone(), manifold=EuclideanManifold())
    p_a = torch.nn.Parameter(init.clone())
    opt_r = geoopt.optim.RiemannianAdam([p_r], lr=1e-2, stabilize=10)
    opt_a = torch.optim.Adam([p_a], lr=1e-2)
    for g in grads:
        p_r.grad, p_a.grad = g.clone(), g.clone()
        opt_r.step(); opt_a.step()
    assert torch.allclose(p_r, p_a, atol=1e-6)

