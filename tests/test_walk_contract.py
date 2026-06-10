"""Pin Tempest's walk-output contract.

Tempest's walk sampler returns walks with a specific tensor layout that
the rest of the codebase (alignment_loss, eventual walk encoder)
depends on. If a future Tempest version changes any of these, the
tests below will fail loudly so the dependency surface is reviewed.

Verified contract (see also tempest_walks/walks.py docstring):
  nodes        [NK, L]            int32   ; padding = -1
  timestamps   [NK, L]            int64   ; sentinel INT64_MAX at lens-1; padding = -1
  edge_feats   [NK, L-1, d_ef]    float32 ; one column SHORTER than nodes
  lens         [NK]               int64
  seeds        [N]                int64
  K            (int)              walks per seed
  NK == N * K
  Row grouping: rows [i*K, (i+1)*K) belong to seeds[i].

Walk semantics:
  Direction "Backward_In_Time" — chronologically OLDEST node at pos 0,
  SEED at pos lens[i]-1.
  timestamps[i, p] for p in [0, lens[i]-2] is the timestamp of the
  edge (nodes[i, p], nodes[i, p+1]).
  timestamps[i, lens[i]-1] = INT64_MAX (seed has no outgoing edge in
  the walk; sentinel exists "for parity" with nodes' shape).

Run with `python tests/test_walk_contract.py` or `pytest`.
"""

import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from tempest_walks.data import load_tgb
from tempest_walks.walks import WalkGenerator, WalkData


_INT64_MAX = (1 << 63) - 1

# Module-level fixture: load tgbl-wiki once, ingest first 10k edges,
# sample walks from 8 seeds. All tests share this state.
_N_INGEST = 10_000
_N_SEEDS = 8
_MAX_WALK_LEN = 20
_NUM_WALKS_PER_NODE = 5


def _build_walks() -> tuple[WalkData, dict, int]:
    """Return (walks, edge_set, ingested_count). edge_set is a dict
    (src, tgt) → list[ts] over the ingested edges (with reverse
    edges added for undirected graphs)."""
    loaded = load_tgb("tgbl-wiki", root="datasets")
    train = loaded.train
    src = train.sources[:_N_INGEST]
    tgt = train.destinations[:_N_INGEST]
    ts = train.timestamps[:_N_INGEST]
    ef = train.edge_feat[:_N_INGEST] if train.edge_feat is not None else None

    # Wiki edges are bipartite user→page action streams; the
    # contract test treats them as undirected (Tempest is configured
    # the same way at the trainer's default --is-directed=False).
    is_directed = False
    wg = WalkGenerator(
        is_directed=is_directed,
        use_gpu=False,
        embedding_num_walks_per_node=_NUM_WALKS_PER_NODE,
        embedding_max_walk_len=_MAX_WALK_LEN,
    )
    wg.add_edges(src, tgt, ts, ef)

    # Seeds: deterministic sample from the ingested node set.
    seed_pool = np.unique(np.concatenate([src, tgt]))
    rng = np.random.default_rng(42)
    seeds = rng.choice(seed_pool, size=_N_SEEDS, replace=False)
    walks = wg.walks_for_nodes_embedding_backward(seeds)

    edge_set: dict = {}
    for s, t, time in zip(src, tgt, ts):
        edge_set.setdefault((int(s), int(t)), []).append(int(time))
    if not is_directed:
        for s, t, time in zip(src, tgt, ts):
            edge_set.setdefault((int(t), int(s)), []).append(int(time))

    return walks, edge_set, len(src)


# Build once, share across tests.
_WALKS, _EDGE_SET, _N_INGESTED = _build_walks()


def test_shapes():
    """Shape and dtype contract."""
    NK = _N_SEEDS * _NUM_WALKS_PER_NODE
    assert _WALKS.nodes.shape == (NK, _MAX_WALK_LEN)
    assert _WALKS.nodes.dtype == torch.int32
    assert _WALKS.timestamps.shape == (NK, _MAX_WALK_LEN)
    assert _WALKS.timestamps.dtype == torch.int64
    assert _WALKS.lens.shape == (NK,)
    assert _WALKS.lens.dtype == torch.int64
    assert _WALKS.seeds.shape == (_N_SEEDS,)
    assert _WALKS.seeds.dtype == torch.int64
    assert _WALKS.K == _NUM_WALKS_PER_NODE
    # edge_feats has one fewer position than nodes (a real edge connects
    # consecutive nodes, so there are L-1 edges per L-length walk).
    if _WALKS.edge_feats is not None:
        assert _WALKS.edge_feats.shape[:2] == (NK, _MAX_WALK_LEN - 1)
        assert _WALKS.edge_feats.dtype == torch.float32


def test_seed_at_lens_minus_1():
    """nodes[i, lens[i]-1] is the seed for walk i, matching seeds[i // K]."""
    K = _WALKS.K
    for i in range(_WALKS.nodes.shape[0]):
        L_i = int(_WALKS.lens[i])
        if L_i == 0:
            continue
        seed_at_tail = int(_WALKS.nodes[i, L_i - 1])
        expected_seed = int(_WALKS.seeds[i // K])
        assert seed_at_tail == expected_seed, (
            f"walk {i}: nodes[{L_i-1}]={seed_at_tail} != expected seed "
            f"{expected_seed} (= seeds[{i // K}])"
        )


def test_timestamp_alignment_with_ingested_edges():
    """For every p in [0, lens[i]-2]: timestamps[i, p] equals the
    timestamp of edge (nodes[i, p], nodes[i, p+1]) in the ingested set."""
    n_checked = 0
    for i in range(_WALKS.nodes.shape[0]):
        L_i = int(_WALKS.lens[i])
        for p in range(L_i - 1):
            u = int(_WALKS.nodes[i, p])
            v = int(_WALKS.nodes[i, p + 1])
            t = int(_WALKS.timestamps[i, p])
            assert (u, v) in _EDGE_SET, (
                f"walk {i} pos {p}: edge ({u}, {v}) absent from ingested set"
            )
            assert t in _EDGE_SET[(u, v)], (
                f"walk {i} pos {p}: edge ({u}, {v}) exists but ts {t} "
                f"not among its ingested timestamps {_EDGE_SET[(u, v)]}"
            )
            n_checked += 1
    assert n_checked > 0, "no (u, v, t) pairs to check — fixture degenerate"


def test_seed_position_timestamp_sentinel():
    """timestamps[i, lens[i]-1] == INT64_MAX. Documents the
    'for parity' sentinel — seed has no associated outgoing edge."""
    for i in range(_WALKS.nodes.shape[0]):
        L_i = int(_WALKS.lens[i])
        if L_i == 0:
            continue
        ts_at_seed = int(_WALKS.timestamps[i, L_i - 1])
        assert ts_at_seed == _INT64_MAX, (
            f"walk {i}: timestamps[{L_i-1}]={ts_at_seed} != INT64_MAX. "
            f"Loss code and walk-encoder design assume this sentinel."
        )


def test_padding_sentinels():
    """For positions p >= lens[i], both nodes and timestamps are -1.
    Edge feats (where present) have zero row-norm in the unused tail."""
    for i in range(_WALKS.nodes.shape[0]):
        L_i = int(_WALKS.lens[i])
        if L_i >= _WALKS.nodes.shape[1]:
            continue
        node_pad = _WALKS.nodes[i, L_i:].tolist()
        ts_pad = _WALKS.timestamps[i, L_i:].tolist()
        assert all(x == -1 for x in node_pad), (
            f"walk {i} node padding != -1: {node_pad}"
        )
        assert all(x == -1 for x in ts_pad), (
            f"walk {i} ts padding != -1: {ts_pad}"
        )
    # edge_feats padding: rows at index >= lens[i] - 1 are zeros.
    if _WALKS.edge_feats is not None:
        for i in range(_WALKS.edge_feats.shape[0]):
            L_i = int(_WALKS.lens[i])
            # Edge-feat valid indices are [0, lens-2], i.e. < lens-1.
            valid_end = max(L_i - 1, 0)
            tail = _WALKS.edge_feats[i, valid_end:]
            if tail.numel() > 0:
                assert torch.all(tail == 0), (
                    f"walk {i} edge_feats tail (pos >= {valid_end}) not all zero"
                )


def main():
    failed = []
    for fn in (
        test_shapes,
        test_seed_at_lens_minus_1,
        test_timestamp_alignment_with_ingested_edges,
        test_seed_position_timestamp_sentinel,
        test_padding_sentinels,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed.append(fn.__name__)
    if failed:
        print(f"\n{len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    print(f"\nAll walk-contract tests pass "
          f"(ingested {_N_INGESTED} edges, sampled walks from {_N_SEEDS} seeds, "
          f"NK={_WALKS.nodes.shape[0]} walks).")


if __name__ == "__main__":
    main()
