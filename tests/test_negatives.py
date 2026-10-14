import numpy as np

from link_property_prediction.data import Batch, SplitData
from link_property_prediction.negatives import (
    CollisionFreeNegativeSampler, SameTimeDestinations, UniformNegativeSampler)


def _edges(src, dst, ts):
    return SplitData(sources=np.asarray(src, dtype=np.int64),
                     destinations=np.asarray(dst, dtype=np.int64),
                     timestamps=np.asarray(ts, dtype=np.int64), edge_feat=None)


def _batch(split):
    return Batch(src=split.sources, tgt=split.destinations, ts=split.timestamps, edge_feat=None)


def _random_edges(n_edges, n_src, pool, n_times, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.integers(0, n_times, n_edges))
    return _edges(rng.integers(0, n_src, n_edges), rng.choice(pool, n_edges), ts)


def _same_time_destinations(edges):
    groups = {}
    for s, d, t in zip(edges.sources, edges.destinations, edges.timestamps):
        groups.setdefault((s, t), set()).add(d)
    return groups


def test_same_time_destinations_matches_brute_force():
    edges = _edges([1, 1, 1, 2, 1], [10, 11, 10, 10, 12], [5, 5, 5, 5, 6])
    lookup = SameTimeDestinations(edges, num_nodes=13)
    src, ts = np.array([1, 1, 2, 2, 3]), np.array([5, 6, 5, 7, 5])
    dst = np.array([[10, 11, 12]] * 5)
    expected = [[True, True, False], [False, False, True], [True, False, False],
                [False, False, False], [False, False, False]]
    np.testing.assert_array_equal(lookup.contains(src, ts, dst), expected)
    np.testing.assert_array_equal(lookup.count(src, ts), [2, 1, 1, 0, 0])


def test_uniform_sampler_draws_unchanged():
    pool = np.arange(100, 150, dtype=np.int32)
    split = _random_edges(64, 10, pool, 20)
    neg = UniformNegativeSampler(7, pool, seed=3).sample(_batch(split))
    expected = pool[np.random.default_rng(3).integers(0, len(pool), (64, 7))]
    np.testing.assert_array_equal(neg, expected)


def test_collision_free_excludes_same_time_destinations_and_repeats():
    pool = np.arange(1, 41, dtype=np.int32)
    edges = _random_edges(2000, 30, pool, 50)
    neg = CollisionFreeNegativeSampler(30, pool, edges, seed=0).sample(_batch(edges))
    excluded = _same_time_destinations(edges)
    for s, t, row in zip(edges.sources, edges.timestamps, neg):
        assert not set(row.tolist()) & excluded[(s, t)]
        if len(pool) - len(excluded[(s, t)]) >= len(row):
            assert len(set(row.tolist())) == len(row)


def test_collision_free_small_pool_still_excludes_but_repeats():
    pool = np.arange(1, 98, dtype=np.int32)
    edges = _random_edges(500, 50, pool, 400)
    neg = CollisionFreeNegativeSampler(100, pool, edges, seed=0).sample(_batch(edges))
    excluded = _same_time_destinations(edges)
    for s, t, row in zip(edges.sources, edges.timestamps, neg):
        assert not set(row.tolist()) & excluded[(s, t)]
        assert len(set(row.tolist())) < len(row)


def test_collision_free_is_seeded():
    pool = np.arange(1, 200, dtype=np.int32)
    edges = _random_edges(300, 20, pool, 30)
    a = CollisionFreeNegativeSampler(50, pool, edges, seed=7).sample(_batch(edges))
    b = CollisionFreeNegativeSampler(50, pool, edges, seed=7).sample(_batch(edges))
    np.testing.assert_array_equal(a, b)
