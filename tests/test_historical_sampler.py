"""HistoricalNegativeSampler correctness — fixed-pool Vitter-R reservoir."""
import numpy as np

from tempest_walks.negatives import HistoricalNegativeSampler


def _obs(s, src, dst):
    s.observe(np.array([src]), np.array([dst]))


def test_fill_phase_exact():
    s = HistoricalNegativeSampler(num_nodes=10, dst_pool=np.arange(100),
                                  reservoir_size=8, seed=0)
    for d in range(8):
        _obs(s, 3, d)
    assert set(s.reservoir[3].tolist()) == set(range(8))   # fill = exactly observed
    assert s.count[3] == 8


def test_reservoir_is_subset_of_history():
    s = HistoricalNegativeSampler(num_nodes=10, dst_pool=np.arange(1000),
                                  reservoir_size=16, seed=1)
    observed = set()
    for d in range(200):
        _obs(s, 5, d); observed.add(d)
    res = [x for x in s.reservoir[5].tolist() if x >= 0]
    assert len(res) == 16 and set(res).issubset(observed)


def test_vitter_uniformity():
    """Algorithm R guarantee: after N items each is retained with prob M/N."""
    M, N, trials = 16, 64, 400
    counts = np.zeros(N)
    for t in range(trials):
        s = HistoricalNegativeSampler(num_nodes=1, dst_pool=np.arange(N),
                                      reservoir_size=M, seed=t)
        for d in range(N):
            _obs(s, 0, d)
        for x in s.reservoir[0]:
            if x >= 0:
                counts[x] += 1
    freq = counts / trials                                 # ~ M/N = 0.25
    assert abs(freq.mean() - M / N) < 0.02
    assert freq.std() < 0.05                               # roughly uniform


def test_sample_shape_and_historical_only():
    s = HistoricalNegativeSampler(num_nodes=10, dst_pool=np.arange(1000),
                                  reservoir_size=16, seed=2)
    for d in range(50):
        _obs(s, 7, d)
    neg = s.sample(np.array([7, 7, 7]), num_neg=20)
    assert neg.shape == (3, 20)
    assert (neg >= 0).all()                                # reservoir full → no -1
    assert set(neg.flatten().tolist()).issubset(set(range(50)))


def test_cold_start_random_fallback():
    s = HistoricalNegativeSampler(num_nodes=10, dst_pool=np.arange(200, 260),
                                  reservoir_size=16, seed=3)
    neg = s.sample(np.array([4]), num_neg=10)              # node 4 never observed
    assert neg.shape == (1, 10) and (neg >= 0).all()
    assert set(neg.flatten().tolist()).issubset(set(range(200, 260)))  # fallback


def test_batch_observe_and_reset():
    s = HistoricalNegativeSampler(num_nodes=10, dst_pool=np.arange(100),
                                  reservoir_size=8, seed=4)
    # batch with a repeated source — must not crash; counts are exact
    s.observe(np.array([1, 1, 2]), np.array([10, 11, 12]))
    assert s.count[1] == 2 and s.count[2] == 1
    s.reset()
    assert (s.reservoir == -1).all() and (s.count == 0).all()
