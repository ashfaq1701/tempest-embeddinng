"""Correctness tests for the walk-neighbour alignment loss (link_property_prediction/alignment_loss.py).

The loss is InfoNCE rows over one shared distance matrix, so what can go wrong is bookkeeping — which
rows exist, which nodes are excluded, whether occurrences aggregate — plus the one behavioural claim
that matters: a step on it must pull each seed toward its own walk context.
"""
import geoopt
import torch

from link_property_prediction.alignment_loss import _extract_context, alignment_loss
from link_property_prediction.model import PoincareManifold
from link_property_prediction.walk_tokens import WalkTokens


def _tokens(nodes, ages, positions, seeds):
    nodes = torch.as_tensor(nodes, dtype=torch.long)
    ages = torch.as_tensor(ages, dtype=torch.long)
    positions = torch.as_tensor(positions, dtype=torch.long)
    seeds = torch.as_tensor(seeds, dtype=torch.long)
    mask = nodes != -1
    return WalkTokens(seeds=seeds, cutoffs=torch.zeros_like(seeds), nodes=nodes, ages=ages,
                      positions=positions, mask=mask, seed_mask=mask & (ages == 0),
                      seed_node_mask=mask & (nodes == seeds[:, None]))


def _emb(n, d=6, std=0.3, seed=0):
    torch.manual_seed(seed)
    return geoopt.PoincareBall(c=1.0).random_normal(n, d, std=std).requires_grad_(True)


# 2 queries, C = 2 candidates each (column 0 = the true edge, but this loss does not care).
def _fixture():
    src = _tokens(nodes=[[1, 5, 6, -1], [3, 7, 8, -1]], ages=[[0, 2, 9, -1], [0, 1, 4, -1]],
                  positions=[[1, 2, 3, 0], [1, 2, 3, 0]], seeds=[1, 3])
    cand = _tokens(nodes=[[2, 9, 5, -1], [10, 11, -1, -1], [4, 12, 7, -1], [13, 14, -1, -1]],
                   ages=[[0, 3, 5, -1], [0, 2, -1, -1], [0, 1, 6, -1], [0, 4, -1, -1]],
                   positions=[[1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 3, 0], [1, 2, 0, 0]],
                   seeds=[2, 10, 4, 13])
    return src, cand


def test_context_excludes_seed_and_padding():
    """Positives are real slots that are NOT the bag's own seed node — the origin slot, any mid-walk
    revisit of the seed, and padding are all dropped."""
    #                seed 5: [seed(5), 7, 5(revisit), 9, pad]
    bag = _tokens([[5, 7, 5, 9, -1]], [[0, 3, 8, 1, -1]], [[1, 2, 3, 4, 0]], [5])
    ctx = _extract_context(bag, torch.float32)
    live = ctx.is_context[0].tolist()
    assert live == [False, True, False, True, False], live
    print("\n[context] origin slot, seed revisit and padding excluded OK")


def test_both_bags_contribute_anchors():
    """Both 'sides' are just both bags contributing rows: every seed of src AND of cand is an anchor,
    and the pool spans the context of both."""
    src, cand = _fixture()
    e = _emb(20)
    s, c = _extract_context(src, e.dtype), _extract_context(cand, e.dtype)
    anchors = set(s.seeds.tolist()) | set(c.seeds.tolist())
    pool = set(torch.cat([s.nodes[s.is_context], c.nodes[c.is_context]]).tolist())
    assert anchors == {1, 3, 2, 10, 4, 13}, sorted(anchors)
    assert {5, 6, 7, 8} <= pool and {9, 11, 12, 14} <= pool, sorted(pool)
    print("[anchors] source AND candidate seeds both anchor rows; pool spans both bags OK")


def test_pool_aggregates_repeated_occurrences():
    """A node appearing in several bags gets the SUM of its occurrence weights in the denominator —
    exact, because a geodesic depends only on the node."""
    geom = PoincareManifold()
    e = _emb(20)
    # Node 5 occurs twice (src row 0 and cand row 0); node 6 once. Perturbing 5 must move the loss more.
    src, cand = _fixture()
    base = float(alignment_loss(src, cand, e, geom).detach())
    deltas = {}
    for node in (5, 6):
        e2 = e.detach().clone()
        e2[node] = e2[node] * 0.2
        deltas[node] = abs(float(alignment_loss(src, cand, e2, geom).detach()) - base)
    assert deltas[5] > deltas[6], f"repeated node must carry more mass: {deltas}"
    print(f"[pool] repeated occurrences aggregate ({deltas[5]:.4f} vs {deltas[6]:.4f}) OK")


def test_self_node_excluded_from_both_terms():
    """A seed that also appears as a context token of ANOTHER bag must not score against itself: the
    degenerate dist(E[u], E[u]) = 0 is dropped from the numerator and from the denominator alike."""
    geom = PoincareManifold()
    e = _emb(20)
    # Node 5 is the seed of row 1 AND a context token of row 0 -> the self term must be excluded.
    src = _tokens([[1, 5, 6, -1], [5, 1, 6, -1]], [[0, 2, 3, -1], [0, 2, 3, -1]],
                  [[1, 2, 3, 0], [1, 2, 3, 0]], [1, 5])
    cand = _tokens([[2, 9, -1, -1], [10, 11, -1, -1]], [[0, 2, -1, -1], [0, 2, -1, -1]],
                   [[1, 2, 0, 0], [1, 2, 0, 0]], [2, 10])
    loss = alignment_loss(src, cand, e, geom)
    assert torch.isfinite(loss), "self-node overlap must not produce inf/nan"
    loss.backward()
    assert torch.isfinite(e.grad).all()
    print("[self] anchor node excluded from both numerator and denominator OK")


def test_gradient_pulls_seed_toward_its_own_context():
    """THE behavioural claim: a step on the alignment loss must DECREASE the geodesic distance from
    each seed to its own walk context."""
    geom = PoincareManifold()
    src, cand = _fixture()
    e = _emb(20, std=0.4, seed=3)

    def seed_to_ctx(emb):
        d = 0.0
        for s, ctx in ((1, [5, 6]), (3, [7, 8]), (2, [9, 5]), (4, [12, 7])):
            for q in ctx:
                d = d + geom.pairwise_dist(emb[s][None], emb[q][None]).squeeze()
        return d

    before = float(seed_to_ctx(e.detach()))
    alignment_loss(src, cand, e, geom).backward()
    stepped = geoopt.PoincareBall(c=1.0).projx(e.detach() - 0.5 * e.grad)
    after = float(seed_to_ctx(stepped))
    assert after < before, f"seeds must be pulled toward their context: {before:.4f} -> {after:.4f}"
    print(f"[gradient] seed->own-context distance {before:.4f} -> {after:.4f} OK")


def test_matches_a_naive_reference_implementation():
    """Ground truth: recompute the whole loss with explicit Python loops — no unique(), no scatter_add,
    no shared matrix — and require the vectorised version to match. This is the test that would catch a
    wrong index, a mis-aggregated pool weight, or a self-term that slipped through."""
    import math

    geom = PoincareManifold()
    src, cand = _fixture()
    e = _emb(20, std=0.35, seed=7).detach().double()

    def d(a, b):
        return float(geom.pairwise_dist(e[a][None], e[b][None]).squeeze())

    # Gather every (row seed, [(node, log_weight), ...]) pair the loss should see.
    rows = []
    for bag in (src, cand):
        ctx = _extract_context(bag, torch.float64)
        for q in range(ctx.nodes.shape[0]):
            live = [(int(ctx.nodes[q, t]), float(ctx.log_weight[q, t]))
                    for t in range(ctx.nodes.shape[1]) if bool(ctx.is_context[q, t])]
            rows.append((int(ctx.seeds[q]), live))

    # Pool weight = SUM of exp(log_weight) over ALL occurrences of a node, across every row.
    pool_w = {}
    for _, live in rows:
        for node, lw in live:
            pool_w[node] = pool_w.get(node, 0.0) + math.exp(lw)

    per_row = []
    for seed, live in rows:
        if not live:
            continue
        log_num = torch.logsumexp(
            torch.tensor([-d(seed, n) + lw for n, lw in live], dtype=torch.float64), dim=0)
        log_den = torch.logsumexp(torch.tensor(
            [-d(seed, x) + math.log(w) for x, w in pool_w.items() if x != seed],
            dtype=torch.float64), dim=0)
        per_row.append(float(log_den - log_num))
    want = sum(per_row) / len(per_row)

    got = float(alignment_loss(src, cand, e, geom).detach())
    assert abs(got - want) < 1e-9, f"vectorised {got:.10f} vs naive reference {want:.10f}"
    print(f"[reference] matches a naive loop implementation ({got:.6f}) OK")


def test_no_context_returns_exact_zero():
    """All-cold bags contribute no rows: exact zero, no nan, no crash."""
    geom = PoincareManifold()
    e = _emb(20)
    src = _tokens([[1, -1], [3, -1]], [[0, -1], [0, -1]], [[1, 0], [1, 0]], [1, 3])
    cand = _tokens([[2, -1], [10, -1]], [[0, -1], [0, -1]], [[1, 0], [1, 0]], [2, 10])
    out = alignment_loss(src, cand, e, geom)
    assert float(out.detach()) == 0.0
    out.backward()
    assert e.grad is None or float(e.grad.abs().sum()) == 0.0

    # Partially cold: finite, and gradients still flow.
    e2 = _emb(20)
    src2 = _tokens([[1, -1, -1], [3, 7, -1]], [[0, -1, -1], [0, 1, -1]], [[1, 0, 0], [1, 2, 0]], [1, 3])
    cand2 = _tokens([[2, 9, -1], [10, -1, -1]], [[0, 2, -1], [0, -1, -1]], [[1, 2, 0], [1, 0, 0]], [2, 10])
    out2 = alignment_loss(src2, cand2, e2, geom)
    assert torch.isfinite(out2)
    out2.backward()
    assert torch.isfinite(e2.grad).all()
    print("[cold] all-cold gives exact zero; partially cold stays finite OK")


if __name__ == "__main__":
    test_context_excludes_seed_and_padding()
    test_both_bags_contribute_anchors()
    test_pool_aggregates_repeated_occurrences()
    test_self_node_excluded_from_both_terms()
    test_gradient_pulls_seed_toward_its_own_context()
    test_matches_a_naive_reference_implementation()
    test_no_context_returns_exact_zero()
    print("\nall alignment-loss tests passed")
