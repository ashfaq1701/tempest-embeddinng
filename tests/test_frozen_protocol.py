"""Frozen-encoder node-classification protocol, on a small synthetic graph (CPU, offline).

  - causality: no walk token reaches an edge at or after the query time, including an edge
    that shares the query's timestamp and the query edge itself
  - freezing: the optimiser holds no encoder parameter and training leaves the encoder's
    weights bit-identical
  - replay: two encoding passes over the same split give bitwise-identical features
  - checkpoint round trip: the rebuilt encoder carries the trained weights exactly
  - the 4 geometric columns match direct recomputation, and pool()[0] is forward() exactly
  - the 2 Tempest history columns match a brute-force scan of strictly earlier edges
"""
import numpy as np
import pytest
import torch

from link_property_prediction.data import SplitData
from link_property_prediction.model import LinkPredHead
from link_property_prediction.walk_tokens import build_query_walk_tokens
from node_classification.classifier import NodeClassifier
from node_classification.encoder import N_FEATURES, N_GEOMETRY, FrozenEncoder
from node_classification.train import fit_classifier

N_NODES, D_EMB, D_EF = 20, 8, 6
CKPT_ARGS = {"hidden_dim": 8, "n_layers_pooler": 2, "seed": 3, "num_walks_per_node": 4,
             "max_walk_len": 5, "walk_bias": "ExponentialWeight",
             "start_bias": "ExponentialWeight", "t2nv_p": 4.0, "t2nv_q": 0.25}


def _graph(n_edges: int = 400, seed: int = 0) -> SplitData:
    rng = np.random.default_rng(seed)
    src = rng.integers(1, N_NODES, n_edges).astype(np.int64)
    dst = rng.integers(1, N_NODES, n_edges).astype(np.int64)
    dst[src == dst] = (dst[src == dst] % (N_NODES - 1)) + 1
    ts = np.sort(rng.integers(1, 200, n_edges)).astype(np.int64)    # many shared timestamps
    ef = rng.normal(size=(n_edges, D_EF)).astype(np.float32)
    return SplitData(sources=src, destinations=dst, timestamps=ts, edge_feat=ef)


def _trained_model() -> LinkPredHead:
    model = LinkPredHead(num_nodes=N_NODES, d_emb=D_EMB, hidden_dim=8, n_layers_pooler=2, seed=3)
    with torch.no_grad():                         # move E off its init so a lost load shows
        model.E.weight.copy_(model.geom.random(N_NODES, D_EMB, std=0.5))
    return model


@pytest.fixture
def encoder(tmp_path) -> FrozenEncoder:
    path = tmp_path / "ckpt.pt"
    torch.save({"state_dict": _trained_model().state_dict(), "args": CKPT_ARGS}, path)
    return FrozenEncoder.from_checkpoint(str(path), _graph(), device=torch.device("cpu"),
                                         use_gpu_tempest=False, seed=7)


def test_walks_see_only_edges_strictly_before_the_query(encoder):
    graph = encoder.graph
    walker = encoder._fresh_walker()
    queries = slice(150, 400)                     # queries ARE graph edges: own edge + ties exist
    tokens = build_query_walk_tokens(
        walker, torch.device("cpu"),
        torch.from_numpy(graph.sources[queries]), torch.from_numpy(graph.timestamps[queries]),
        max_walk_len=5, num_walks_per_node=4,
        start_bias="ExponentialWeight", walk_bias="ExponentialWeight")
    real_edge = tokens.mask & ~tokens.seed_mask
    assert real_edge.any()
    # age = cutoff - t_edge, so age >= 1 everywhere means t_edge < t: the query edge and every
    # edge sharing its timestamp are excluded.
    assert int(tokens.ages[real_edge].min()) >= 1


def test_optimiser_excludes_encoder_and_training_leaves_it_unchanged(encoder):
    graph = encoder.graph
    cut = [0, 250, 320, 400]
    splits = {name: SplitData(*(arr[a:b] for arr in graph))
              for name, a, b in zip(("train", "val", "test"), cut[:-1], cut[1:])}
    rng = np.random.default_rng(1)
    labels = {name: (rng.random(len(s.sources)) < 0.3).astype(np.float32)
              for name, s in splits.items()}

    before = encoder.state_hash()
    features = {name: encoder.encode(split) for name, split in splits.items()}
    classifier = NodeClassifier(n_in=N_FEATURES)
    encoder_params = {id(p) for p in encoder.model.parameters()}
    assert not any(id(p) in encoder_params for p in classifier.parameters())
    assert not any(p.requires_grad for p in encoder.model.parameters())

    fit_classifier(classifier, features, labels,
                   batch_size=50, lr=1e-3, num_epochs=2, patience=5, seed=0)
    assert encoder.state_hash() == before


def test_two_passes_replay_identical_features(encoder):
    first = encoder.encode(encoder.graph, batch_size=64)
    second = encoder.encode(encoder.graph, batch_size=64)
    assert first.shape == (len(encoder.graph.sources), N_FEATURES)
    assert torch.equal(first, second)


def test_checkpoint_round_trip_restores_trained_weights(encoder):
    reference = _trained_model().state_dict()
    for name, tensor in encoder.model.state_dict().items():
        assert torch.equal(tensor, reference[name]), name


def test_geometric_features_match_direct_recomputation(encoder):
    walker = encoder._fresh_walker()
    graph = encoder.graph
    src = torch.from_numpy(graph.sources[200:260])
    tokens = build_query_walk_tokens(
        walker, torch.device("cpu"), src, torch.from_numpy(graph.timestamps[200:260]),
        max_walk_len=5, num_walks_per_node=4,
        start_bias="ExponentialWeight", walk_bias="ExponentialWeight")
    geom = encoder.model.geom
    with torch.no_grad():
        p_u, w, x_tokens = encoder.model.bag_weights.pool(tokens)
        feats = encoder._geometry(tokens, src)
        assert torch.equal(p_u, encoder.model.bag_weights(tokens))      # pool()[0] is forward()
        valid = w > 0
        rows = []
        for q in range(len(p_u)):
            x = x_tokens[q, valid[q]]
            d = geom.dist(x, p_u[q])
            a = torch.softmax(-d, -1)
            rows.append(torch.stack([geom.dist0(p_u[q]),
                                     geom.dist0(encoder.model.E.weight[src[q]]),
                                     (a * d).sum(), -(a * a.log()).sum()]))
        expected = torch.stack(rows)
    assert feats.shape == (len(p_u), N_GEOMETRY)
    torch.testing.assert_close(feats, expected, rtol=1e-5, atol=1e-5)


def test_tempest_history_matches_brute_force(encoder):
    stream = encoder.graph
    q = slice(150, 400)
    src, ts = stream.sources[q], stream.timestamps[q]
    got = encoder._history(encoder._fresh_walker(), src, ts).numpy()
    assert got.shape == (len(src), N_FEATURES - N_GEOMETRY)
    nodes_all = np.concatenate([stream.sources, stream.destinations])
    times_all = np.concatenate([stream.timestamps, stream.timestamps])
    for i, (u, t) in enumerate(zip(src, ts)):
        times = times_all[(nodes_all == u) & (times_all < t)]   # undirected: every incident edge
        assert got[i, 0] == np.float32(np.log1p(len(times)))
        expected = np.log1p(t - times.max()) if len(times) else encoder.no_event_log_gap
        np.testing.assert_allclose(got[i, 1], expected, rtol=1e-6)
