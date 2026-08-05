"""Correctness tests for the monotone min/mean/max metric head (link_property_prediction/model.py).

The head's whole design rests on ONE property — ds/dd <= 0 for every geodesic distance it consumes — so
that property is ASSERTED here rather than trusted. The rest pins pairwise_dist against geoopt, the
[min, mean, max] contents of bag_stats (including exclusion of masked slots), the simplex mix, and the
cold-bag fallback to the identity distance.
"""
import geoopt
import torch

from link_property_prediction.model import (
    LinkPredHead, PoincareManifold, bag_stats, bag_weights)
from link_property_prediction.walk_tokens import WalkTokens


def _ball_points(*shape, std=0.3, seed=0):
    torch.manual_seed(seed)
    ball = geoopt.PoincareBall(c=1.0)
    return ball.random_normal(*shape, std=std).to(torch.float64)


def _make_tokens(nodes, ages, positions, seeds):
    """Build a WalkTokens from explicit [Q, T] arrays; mask/seed masks derived as in the real builder."""
    nodes = torch.as_tensor(nodes, dtype=torch.long)
    ages = torch.as_tensor(ages, dtype=torch.long)
    positions = torch.as_tensor(positions, dtype=torch.long)
    seeds = torch.as_tensor(seeds, dtype=torch.long)
    mask = nodes != -1
    seed_mask = mask & (ages == 0)
    seed_node_mask = mask & (nodes == seeds[:, None])
    return WalkTokens(seeds=seeds, cutoffs=torch.zeros_like(seeds), nodes=nodes, ages=ages,
                      positions=positions, mask=mask, seed_mask=seed_mask,
                      seed_node_mask=seed_node_mask)


def test_pairwise_dist_matches_geoopt():
    """Closed-form arcosh distance == geoopt.PoincareBall.dist, including broadcast leading dims."""
    ball = geoopt.PoincareBall(c=1.0)
    x = _ball_points(4, 7, 5, seed=1)          # [4, 7, d]
    y = _ball_points(4, 6, 5, seed=2)          # [4, 6, d]
    got = PoincareManifold.pairwise_dist(x, y)                       # [4, 7, 6]
    want = ball.dist(x.unsqueeze(-2), y.unsqueeze(-3))               # [4, 7, 6]
    assert got.shape == want.shape, f"{got.shape} vs {want.shape}"
    assert (got - want).abs().max().item() < 1e-8

    # Broadcast: a single anchor row against a [B, C, T, d] bag.
    a = _ball_points(3, 1, 1, 5, seed=3)       # [3, 1, 1, d]
    bag = _ball_points(3, 4, 6, 5, seed=4)     # [3, 4, 6, d]
    got_b = PoincareManifold.pairwise_dist(a, bag).squeeze(-2)       # [3, 4, 6]
    want_b = ball.dist(a, bag)                                       # [3, 4, 6]
    assert got_b.shape == want_b.shape
    assert (got_b - want_b).abs().max().item() < 1e-8
    print("\n[pairwise_dist] matches geoopt (batched + broadcast) OK")


def test_score_is_monotone_decreasing_in_every_distance():
    """THE load-bearing property: ds/dd_p <= 0 for every distance entering the score, at a random mix.

    Differentiates the score aggregate w.r.t. synthetic distance tensors, so the check is on the head's
    ALGEBRA ([min, mean, max] + simplex softmax mix), independent of the geometry."""
    torch.manual_seed(0)
    mix = torch.softmax(torch.randn(3, dtype=torch.float64), dim=-1)            # simplex weights

    d_id = torch.rand(6, 4, dtype=torch.float64).requires_grad_(True)           # [B, C]
    d_bag = (torch.rand(6, 4, 9, dtype=torch.float64) * 3).requires_grad_(True)   # [B, C, T]
    log_w = torch.log_softmax(torch.randn(6, 1, 9, dtype=torch.float64), dim=-1)

    stats = bag_stats(d_bag, log_w)                                             # [B, C, 3]
    s = -(d_id + stats @ mix)                                                   # [B, C]
    s.sum().backward()

    assert bool((d_id.grad <= 0).all()), "score must be non-increasing in the identity distance"
    assert bool((d_bag.grad <= 1e-12).all()), "score must be non-increasing in every bag distance"
    # Every bag distance receives SOME non-positive gradient share; total over T equals -(sum of mix) = -1.
    assert (d_bag.grad.sum(dim=-1) + 1.0).abs().max().item() < 1e-9, \
        "per-anchor bag gradients must sum to -1 (min+mean+max are convex combinations, mix sums to 1)"
    print("[monotone] ds/dd <= 0 everywhere; bag gradients sum to -1 (convex combination) OK")


def test_bag_stats_is_min_mean_max():
    """bag_stats returns exactly [weighted min, weighted mean, weighted max] over the LIVE slots, and a
    masked (-inf log_w) slot is excluded from all three even if its distance is extreme."""
    torch.manual_seed(1)
    d = torch.tensor([[0.2, 0.5, 1.3, 99.0]], dtype=torch.float64)              # slot 3 is an outlier
    # Exclude slot 3 (log_w = -inf); weights over the first three.
    raw = torch.tensor([[0.3, 0.5, 0.2, float("-inf")]], dtype=torch.float64)
    log_w = torch.log_softmax(raw, dim=-1)

    out = bag_stats(d, log_w)                                                   # [1, 3]
    d_min, mean, d_max = out[0].tolist()
    assert abs(d_min - 0.2) < 1e-12, "min must be the nearest LIVE token (outlier excluded)"
    assert abs(d_max - 1.3) < 1e-12, "max must be the farthest LIVE token (99.0 outlier excluded)"
    w = log_w.exp()
    want_mean = (w[0, :3] * d[0, :3]).sum().item()
    assert abs(mean - want_mean) < 1e-12, "mean must be the weighted average over live slots"
    # Sanity: unequal weights actually move the mean off the arithmetic mean.
    assert abs(mean - float(d[0, :3].mean())) > 1e-3, "mean must be WEIGHTED, not plain"
    print("[bag_stats] returns [min, mean, max]; masked slots excluded; mean is weighted OK")


def test_bag_weights_mask_and_mass():
    """Weights sum to 1 per row; own-seed slots (origin AND revisits) excluded; padding excluded."""
    #                seed 5: slots = [seed(5), 7, 5(revisit), 9, pad]
    tokens = _make_tokens(nodes=[[5, 7, 5, 9, -1]], ages=[[0, 3, 8, 1, -1]],
                          positions=[[1, 2, 3, 4, 0]], seeds=[5])
    nodes, log_w = bag_weights(tokens)
    w = log_w.exp()
    assert abs(w.sum().item() - 1.0) < 1e-6, "weights must sum to 1"
    assert w[0, 0].item() == 0.0 and w[0, 2].item() == 0.0, "own-seed slots must carry zero weight"
    assert w[0, 4].item() == 0.0, "padding must carry zero weight"
    assert abs(w[0, 1].item() - w[0, 3].item()) < 1e-6, "equal (age,hop) log-mass -> equal weight"
    print("[bag_weights] mass=1, own-seed + padding excluded, prior applied OK")


def test_cold_bag_falls_back_to_identity():
    """A bag with no context -> {(seed, 1)}, so min = mean = max = the identity distance. End to end with
    BOTH sides cold, the whole score collapses to -3 * d(E_u, E_v) (d_id + [d_id]*3 . mix + [d_id]*3 . mix,
    mix summing to 1 -> d_id + d_id + d_id)."""
    tokens = _make_tokens(nodes=[[5, -1, -1], [6, 8, 9]], ages=[[0, -1, -1], [0, 2, 5]],
                          positions=[[1, 0, 0], [1, 2, 3]], seeds=[5, 6])
    nodes, log_w = bag_weights(tokens)
    assert nodes[0, 0].item() == 5 and abs(log_w[0, 0].item()) < 1e-9, "cold row -> slot0 = (seed, w=1)"
    assert bool(torch.isinf(log_w[0, 1:]).all()), "cold row -> every other slot excluded"

    head = LinkPredHead(num_nodes=10, d_emb=4)
    with torch.no_grad():
        head.E.weight.copy_(head.geom.manifold.random_normal(10, 4, std=0.3))
    src = _make_tokens(nodes=[[1, -1, -1]], ages=[[0, -1, -1]], positions=[[1, 0, 0]], seeds=[1])
    cand = _make_tokens(nodes=[[2, -1, -1], [3, -1, -1]], ages=[[0, -1, -1], [0, -1, -1]],
                        positions=[[1, 0, 0], [1, 0, 0]], seeds=[2, 3])
    logits = head(src, cand)                                                    # [1, 2]
    e = head.E.weight.detach()
    d_id = PoincareManifold.pairwise_dist(e[1:2].unsqueeze(0), e[2:4].unsqueeze(0)).squeeze()
    want = -3.0 * d_id
    assert logits.shape == (1, 2)
    assert (logits.squeeze(0) - want).abs().max().item() < 1e-5, f"{logits} vs {want}"
    print("[cold bag] both sides cold -> score == -3 * identity distance OK")


def test_mix_is_simplex_and_init_equal():
    """The only head parameter theta inits at zeros, so mix = softmax = (1/3, 1/3, 1/3); >= 0, sums to 1."""
    head = LinkPredHead(num_nodes=8, d_emb=4)
    mix = head.mix
    assert mix.shape == (3,)
    assert bool((mix >= 0).all()) and abs(mix.sum().item() - 1.0) < 1e-6, "mix must lie on the simplex"
    assert (mix - 1.0 / 3.0).abs().max().item() < 1e-6, "init must be the equal mix (1/3, 1/3, 1/3)"
    print("[mix] softmax(theta) is a simplex, init equal (1/3,1/3,1/3) OK")


def test_forward_shapes_and_grad_flow():
    """forward returns [B, C] and backprops into BOTH E and theta (no detach anywhere)."""
    torch.manual_seed(2)
    b, c, t, n = 3, 4, 6, 40
    head = LinkPredHead(num_nodes=n, d_emb=8)
    seeds_u = torch.randint(0, n, (b,))
    seeds_v = torch.randint(0, n, (b * c,))
    nodes_u = torch.randint(0, n, (b, t)); nodes_u[:, -1] = -1
    nodes_v = torch.randint(0, n, (b * c, t)); nodes_v[:, -1] = -1
    ages_u = torch.randint(1, 50, (b, t)); ages_u[:, 0] = 0; ages_u[:, -1] = -1
    ages_v = torch.randint(1, 50, (b * c, t)); ages_v[:, 0] = 0; ages_v[:, -1] = -1
    pos_u = torch.arange(1, t + 1).repeat(b, 1); pos_u[:, -1] = 0
    pos_v = torch.arange(1, t + 1).repeat(b * c, 1); pos_v[:, -1] = 0
    nodes_u[:, 0] = seeds_u; nodes_v[:, 0] = seeds_v

    src = _make_tokens(nodes_u, ages_u, pos_u, seeds_u)
    cand = _make_tokens(nodes_v, ages_v, pos_v, seeds_v)
    logits = head(src, cand)
    assert logits.shape == (b, c), logits.shape
    assert torch.isfinite(logits).all(), "logits must be finite"

    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(b, dtype=torch.long))
    loss.backward()
    assert head.E.weight.grad is not None and torch.isfinite(head.E.weight.grad).all()
    assert head.theta.grad is not None and torch.isfinite(head.theta.grad).all()
    assert head.E.weight.grad.abs().sum() > 0, "link loss must reach E (end-to-end, no detach)"
    print("[forward] shape [B,C], finite logits, gradients reach BOTH E and theta OK")


if __name__ == "__main__":
    test_pairwise_dist_matches_geoopt()
    test_score_is_monotone_decreasing_in_every_distance()
    test_bag_stats_is_min_mean_max()
    test_bag_weights_mask_and_mass()
    test_cold_bag_falls_back_to_identity()
    test_mix_is_simplex_and_init_equal()
    test_forward_shapes_and_grad_flow()
    print("\nall head tests passed")
