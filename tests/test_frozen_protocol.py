"""Frozen-encoder node-classification protocol, on a small synthetic graph (CPU, offline).

  - causality: no walk token reaches an edge at or after the query time, including an edge
    that shares the query's timestamp and the query edge itself
  - freezing: the optimiser holds no encoder parameter and training leaves the encoder's
    weights bit-identical
  - replay: two encoding passes over the same split give bitwise-identical features
  - checkpoint round trip: the rebuilt encoder carries the trained weights exactly
  - geo features are [d0(p_u), sum_i w_i d(x_i, p_u)], and pool()[0] is forward() exactly
  - ef is the mean feature of u's real walk edges (seed slot and padding excluded)
"""
import numpy as np
import pytest
import torch

from link_property_prediction.data import SplitData
from link_property_prediction.model import LinkPredHead
from link_property_prediction.walk_tokens import build_query_walk_tokens
from node_classification.classifier import NodeClassifier
from node_classification.encoder import FrozenEncoder
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

    classifier = NodeClassifier(n_geo=encoder.n_geo, d_ef=encoder.d_ef)
    encoder_params = {id(p) for p in encoder.model.parameters()}
    assert not any(id(p) in encoder_params for p in classifier.parameters())
    assert not any(p.requires_grad for p in encoder.model.parameters())

    before = encoder.state_hash()
    fit_classifier(encoder, classifier, splits, labels,
                   batch_size=50, lr=1e-3, num_epochs=2, patience=5)
    assert encoder.state_hash() == before


def test_two_passes_replay_identical_features(encoder):
    first = list(encoder.encode(encoder.graph, batch_size=64))
    second = list(encoder.encode(encoder.graph, batch_size=64))
    assert len(first) == len(second)
    for (g1, e1), (g2, e2) in zip(first, second):
        assert torch.equal(g1, g2) and torch.equal(e1, e2)


def test_checkpoint_round_trip_restores_trained_weights(encoder):
    reference = _trained_model().state_dict()
    for name, tensor in encoder.model.state_dict().items():
        assert torch.equal(tensor, reference[name]), name


def test_features_are_radius_and_weighted_spread(encoder):
    walker = encoder._fresh_walker()
    graph = encoder.graph
    tokens = build_query_walk_tokens(
        walker, torch.device("cpu"),
        torch.from_numpy(graph.sources[200:260]), torch.from_numpy(graph.timestamps[200:260]),
        max_walk_len=5, num_walks_per_node=4,
        start_bias="ExponentialWeight", walk_bias="ExponentialWeight")
    geom = encoder.model.geom
    with torch.no_grad():
        p_u, w, x_tokens = encoder.model.bag_weights.pool(tokens)
        feats = encoder._geo_features(tokens)
        assert torch.equal(p_u, encoder.model.bag_weights(tokens))      # pool()[0] is forward()
        valid = w > 0
        spread = torch.stack([(w[q, valid[q]] * geom.dist(x_tokens[q, valid[q]], p_u[q])).sum()
                              for q in range(len(p_u))])
    assert feats.shape == (len(p_u), 2)
    torch.testing.assert_close(feats[:, 0], geom.dist0(p_u))
    torch.testing.assert_close(feats[:, 1], spread)


def test_edge_features_are_the_mean_over_real_walk_edges(encoder):
    walker = encoder._fresh_walker()
    graph = encoder.graph
    tokens = build_query_walk_tokens(
        walker, torch.device("cpu"),
        torch.from_numpy(graph.sources[200:260]), torch.from_numpy(graph.timestamps[200:260]),
        max_walk_len=5, num_walks_per_node=4,
        start_bias="ExponentialWeight", walk_bias="ExponentialWeight")
    ef = encoder._edge_features(tokens)
    real_edge = (tokens.mask & ~tokens.seed_mask).flatten(1)
    flat = tokens.edge_features.flatten(1, 2)
    for q in range(len(ef)):
        expected = (flat[q, real_edge[q]].mean(dim=0) if real_edge[q].any()
                    else torch.zeros(flat.shape[-1]))
        torch.testing.assert_close(ef[q], expected)
