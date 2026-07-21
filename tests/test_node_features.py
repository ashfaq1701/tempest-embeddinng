"""Node-feature passthrough: build_query_walk_tokens attaches per-token static node features to
WalkTokens, and StatelessLinkHead consumes them in the attention logit. Real graph + real Tempest
walks over tests/data/sample_data.csv; an independent gather is the oracle."""
import pathlib

import numpy as np
import torch

from link_property_prediction.model import StatelessLinkHead
from link_property_prediction.walk_tokens import build_query_walk_tokens
from link_property_prediction.walks import WalkGenerator

_CSV = pathlib.Path(__file__).parent / "data" / "sample_data.csv"
_DEV = torch.device("cpu")


def _load():
    d = np.loadtxt(_CSV, delimiter=",", skiprows=1, dtype=np.int64)
    return d[:, 0], d[:, 1], d[:, 2]


SRC, DST, TS = _load()
NUM_NODES = int(max(SRC.max(), DST.max())) + 1
CUTOFF = int(TS.max()) + 1


def _walk_gen():
    wg = WalkGenerator(num_walks_per_node=6, max_walk_len=5)
    wg.add_edges(SRC, DST, TS, None)
    return wg


def _bag(wg, seeds, node_feat, n_walks=6, max_len=5):
    seeds_t = torch.as_tensor(seeds, dtype=torch.long)
    cut_t = torch.full((len(seeds),), CUTOFF, dtype=torch.long)
    return build_query_walk_tokens(wg, _DEV, seeds_t, cut_t,
                                   max_walk_len=max_len, num_walks_per_node=n_walks,
                                   node_feat=node_feat)


def test_node_features_match_independent_gather():
    """WalkTokens.node_features [Q, K, L*d_nf] equals node_feat[node_id] at every real token and is
    zero at padding — checked against a direct numpy gather (a different implementation)."""
    rng = np.random.default_rng(0)
    d_nf = 7
    node_feat = rng.standard_normal((NUM_NODES, d_nf)).astype(np.float32)
    seeds = rng.choice(NUM_NODES, 5, replace=False)
    wg = _walk_gen()
    tok = _bag(wg, seeds, node_feat)

    q, k, length = tok.nodes.shape
    assert tok.node_features.shape == (q, k, length * d_nf)
    got = tok.node_features.reshape(q, k, length, d_nf).numpy()

    nodes = tok.nodes.numpy()
    mask = tok.nodes_mask.numpy()
    # real tokens carry their node's feature row; padding is zeroed.
    ref = node_feat[np.clip(nodes, 0, None)] * mask[..., None]
    assert np.allclose(got, ref, atol=1e-6), np.abs(got - ref).max()
    # sanity: some real tokens exist and carry non-zero features.
    assert mask.sum() > 0 and np.abs(got[mask]).sum() > 0


def test_seed_node_features_match_gather():
    """WalkTokens.seed_node_features [Q, d_nf] is the query seed's own node-feature row."""
    rng = np.random.default_rng(2)
    d_nf = 5
    node_feat = rng.standard_normal((NUM_NODES, d_nf)).astype(np.float32)
    seeds = rng.choice(NUM_NODES, 6, replace=False)
    tok = _bag(_walk_gen(), seeds, node_feat)
    assert tok.seed_node_features.shape == (len(seeds), d_nf)
    assert np.allclose(tok.seed_node_features.numpy(), node_feat[seeds], atol=1e-6)


def test_none_when_no_node_feat():
    """No node_feat table → both node-feature fields are None (nothing attached)."""
    wg = _walk_gen()
    tok = _bag(wg, np.array([0, 1, 2]), node_feat=None)
    assert tok.node_features is None
    assert tok.seed_node_features is None


def test_head_consumes_node_features():
    """StatelessLinkHead(d_nf>0) runs on bags carrying node features and produces finite, differentiable
    logits. Node features enter as the ⟨seed_nf, token_nf⟩ affinity (weighted by nf_weight), NOT through
    the time-bias MLP — so the MLP width is unchanged and nf_weight receives gradient."""
    rng = np.random.default_rng(1)
    d_nf = 4
    node_feat = rng.standard_normal((NUM_NODES, d_nf)).astype(np.float32)
    wg = _walk_gen()
    src = _bag(wg, np.array([3, 4]), node_feat)
    cand = _bag(wg, np.array([5, 6, 7, 8]), node_feat)          # 2 queries × 2 candidates

    head = StatelessLinkHead(NUM_NODES, d_emb=16, n_hops=3, t2v_dim=16, d_ef=0, d_nf=d_nf)
    logits = head(src, cand)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    # node features stay OUT of the time-bias MLP (input width == t2v(16) + hop(1) + d_ef(0)); they enter
    # via the affinity term, so nf_weight must have received a gradient.
    assert head.encoder.attn_bias_mlp[0].in_features == 16 + 1 + 0
    assert head.encoder.nf_weight.grad is not None and head.encoder.nf_weight.grad.abs().item() >= 0.0
