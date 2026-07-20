"""Contract for the TGB-identical fixed-size batch iterator (data.create_batches).

Pins: consecutive fixed-size chunks, order-preserving, drop_last=False, and a
timestamp group larger than batch_size IS split across batches (matching
torch_geometric.TemporalDataLoader / TPNet's DataLoader-over-indices).
"""
import numpy as np

from link_property_prediction.data import SplitData, create_batches


def _split(n, ts=None, with_ef=False):
    src = np.arange(n, dtype=np.int64)
    dst = np.arange(n, dtype=np.int64) + 1000
    if ts is None:
        ts = np.arange(n, dtype=np.int64)          # one edge per timestamp
    ef = np.arange(n * 3, dtype=np.float32).reshape(n, 3) if with_ef else None
    return SplitData(src, dst, np.asarray(ts, dtype=np.int64), ef)


def test_concat_reproduces_split_in_order():
    sp = _split(2050, with_ef=True)
    batches = list(create_batches(sp, 200))
    assert np.array_equal(np.concatenate([b.src for b in batches]), sp.sources)
    assert np.array_equal(np.concatenate([b.tgt for b in batches]), sp.destinations)
    assert np.array_equal(np.concatenate([b.ts for b in batches]), sp.timestamps)
    assert np.array_equal(
        np.concatenate([b.edge_feat for b in batches]), sp.edge_feat)


def test_exact_sizes_drop_last_false():
    sp = _split(2050)
    batches = list(create_batches(sp, 200))
    assert len(batches) == 11                       # ceil(2050/200): 10 full + 1 partial
    assert all(len(b.src) == 200 for b in batches[:-1])
    assert len(batches[-1].src) == 50               # final partial batch kept


def test_divides_evenly():
    sp = _split(400)
    sizes = [len(b.src) for b in create_batches(sp, 200)]
    assert sizes == [200, 200]


def test_large_timestamp_group_is_split():
    # All 500 edges share timestamp 7 — exact mode must STILL split 200/200/100,
    # unlike the old timestamp-grouped iterator which would emit one 500-edge batch.
    sp = _split(500, ts=np.full(500, 7))
    batches = list(create_batches(sp, 200))
    assert [len(b.src) for b in batches] == [200, 200, 100]
    assert all((b.ts == 7).all() for b in batches)  # same group spans all batches


def test_batch_size_larger_than_split():
    sp = _split(120)
    batches = list(create_batches(sp, 200))
    assert len(batches) == 1 and len(batches[0].src) == 120


def test_empty_split_yields_nothing():
    assert list(create_batches(_split(0), 200)) == []


def test_edge_feat_none_passthrough():
    batches = list(create_batches(_split(300, with_ef=False), 200))
    assert all(b.edge_feat is None for b in batches)


def test_batch_count_matches_ceil_div():
    # The leaderboard-comparability claim: #batches == ceil(n / batch_size).
    # tgbl-review train has 3,413,837 edges => 17,070 batches at bs=200.
    import math
    assert math.ceil(3_413_837 / 200) == 17_070
    sp = _split(3_413_837 // 1000)                  # scaled proxy, same formula
    assert len(list(create_batches(sp, 200))) == math.ceil((3_413_837 // 1000) / 200)
