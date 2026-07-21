"""Correctness tests for NodeEncoding — the stateless batch-local node embedding.

EVERY test runs on the real graph in tests/data/sample_data.csv, walked by real Tempest
(build_query_walk_tokens). No synthetic tokens. The oracle is an INDEPENDENT dense reconstruction of
the diffusion, built by explicit per-walk edge enumeration (a genuinely different implementation from
the module's vectorized slicing) so shared-recipe bugs are caught.

Properties verified against real bags:
  - exact diffusion  node_enc == [X0, ÂX0, …, ÂⁿX0]  (Â = D⁻¹A over the walk-induced graph),
  - recency weighting from the OLDER endpoint using real ages,
  - co-reachability statistics  E[<Xk[i],Xk[j]>] = dim·<Âᵏ[i,:],Âᵏ[j,:]>  (JL),
  - anonymization  (relabelling node ids leaves the structural inner-products unchanged),
  - joint two-bag encoding == diffusion over the UNION graph, incl. cross-bag co-reachability,
  - directed mode, shape/assoc/determinism, empty and seed-only (isolated) bags.
"""
import pathlib

import numpy as np
import torch

from link_property_prediction.model import NodeEncoding
from link_property_prediction.walk_tokens import WalkTokens, build_query_walk_tokens
from link_property_prediction.walks import WalkGenerator

_CSV = pathlib.Path(__file__).parent / "data" / "sample_data.csv"
_DEV = torch.device("cpu")


# ── real graph + real walks ──────────────────────────────────────────────────
def load_edges():
    d = np.loadtxt(_CSV, delimiter=",", skiprows=1, dtype=np.int64)
    return d[:, 0], d[:, 1], d[:, 2]                       # src, dst, ts


SRC, DST, TS = load_edges()
NUM_NODES = int(max(SRC.max(), DST.max())) + 1             # 112
CUTOFF = int(TS.max()) + 1                                 # permissive — walks may use every edge


def new_walk_gen():
    wg = WalkGenerator(num_walks_per_node=8, max_walk_len=6)
    wg.add_edges(SRC, DST, TS, None)
    return wg


_WG = new_walk_gen()                                       # module-level: ingest once, reuse
# NB: NodeEncoding.undirected controls how the adjacency is built FROM the walk edges (symmetrise or
# not) — independent of the walk sampler, so the directed-mode test reuses these undirected walks.


def real_bag(seeds, n_walks, max_len, wg=None, cutoff=CUTOFF):
    """A real WalkTokens bag: n_walks per seed on the sample graph (permissive cutoff — no causality)."""
    wg = wg or _WG
    seeds_t = torch.as_tensor(seeds, dtype=torch.long)
    cut_t = torch.full((len(seeds),), cutoff, dtype=torch.long)
    return build_query_walk_tokens(wg, _DEV, seeds_t, cut_t,
                                   max_walk_len=max_len, num_walks_per_node=n_walks)


# ── INDEPENDENT dense oracle (explicit enumeration; NOT the module's slicing) ─
def independent_graph(bags, recency_lambda, undirected):
    """Rebuild the batch-local weighted adjacency from the walk bags by looping every consecutive
    (position l, l+1) real pair — a different implementation from the module. Returns (present, assoc,
    A) with A [U,U] numpy, rows/cols in sorted-global-id order (matches torch.unique)."""
    present = set()
    for t in bags:
        nd, nm = t.nodes.numpy(), t.nodes_mask.numpy()
        present.update(nd[nm].tolist())
        present.update(t.seeds.numpy().tolist())
    present = sorted(present)
    loc = {g: i for i, g in enumerate(present)}
    u = len(present)
    A = np.zeros((u, u), dtype=np.float64)
    for t in bags:
        nd, nm, ag = t.nodes.numpy(), t.nodes_mask.numpy(), t.ages.numpy()
        q, k, length = nd.shape
        for iq in range(q):
            for ik in range(k):
                for il in range(length - 1):
                    if nm[iq, ik, il] and nm[iq, ik, il + 1]:
                        a, b = loc[nd[iq, ik, il]], loc[nd[iq, ik, il + 1]]
                        w = float(np.exp(-recency_lambda * ag[iq, ik, il]))   # OLDER endpoint's age
                        A[a, b] += w
                        if undirected:
                            A[b, a] += w
    return present, loc, A


def dense_diffuse(A, x0, n_hops):
    """[X0, ÂX0, …, ÂⁿX0] and Â, with exact D⁻¹ (0 rows for isolated nodes)."""
    deg = A.sum(1)
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)   # exact 1/deg; 0 if isolated
    Ahat = A * inv[:, None]
    blocks, xk = [x0], x0
    for _ in range(n_hops):
        xk = Ahat @ xk
        blocks.append(xk)
    return np.concatenate(blocks, axis=-1), Ahat


def module_matches_dense(bags, dim, n_hops, recency_lambda, undirected):
    """Run NodeEncoding on `bags`, extract its own X0 (block 0), and assert every diffused block equals
    the independent dense reference. Returns (assoc, node_enc, present, Ahat) for further checks."""
    enc = NodeEncoding(NUM_NODES, dim, n_hops, recency_lambda=recency_lambda, undirected=undirected)
    assoc, node_enc = enc(bags[0], bags[1]) if len(bags) == 2 else enc(bags[0])
    present, loc, A = independent_graph(bags, recency_lambda, undirected)
    x0 = node_enc[:, :dim].numpy()
    ref, Ahat = dense_diffuse(A, x0, n_hops)
    # rows align: module uses torch.unique (sorted) == our sorted `present`.
    assert torch.equal(assoc[torch.tensor(present)], torch.arange(len(present)))
    assert np.allclose(node_enc.numpy(), ref, atol=1e-4), np.abs(node_enc.numpy() - ref).max()
    return assoc, node_enc, present, Ahat


# ══════════════════════════════════════════════════════════════════════════════
# ① exact diffusion vs independent dense oracle — the hard correctness gate
# ══════════════════════════════════════════════════════════════════════════════
def test_diffusion_matches_independent_dense_undirected():
    torch.manual_seed(0)
    np.random.seed(0)
    seeds = np.random.choice(NUM_NODES, 4, replace=False)
    bag = real_bag(seeds, n_walks=8, max_len=6)
    module_matches_dense([bag], dim=16, n_hops=3, recency_lambda=0.0, undirected=True)


def test_diffusion_matches_independent_dense_directed():
    torch.manual_seed(1)
    np.random.seed(1)
    seeds = np.random.choice(NUM_NODES, 4, replace=False)
    bag = real_bag(seeds, n_walks=8, max_len=6)
    module_matches_dense([bag], dim=16, n_hops=3, recency_lambda=0.0, undirected=False)


def test_recency_weighting_from_real_ages():
    # real ages are large (permissive cutoff), so a small lambda gives a non-degenerate weight gradient
    # (recent edges ~1, oldest ~exp(-2)); the module's older-endpoint weighting must match the oracle.
    torch.manual_seed(2)
    np.random.seed(2)
    seeds = np.random.choice(NUM_NODES, 4, replace=False)
    bag = real_bag(seeds, n_walks=8, max_len=6)
    _, _, _, Ahat = module_matches_dense([bag], dim=16, n_hops=2, recency_lambda=1e-5, undirected=True)
    # sanity: the reconstructed transition is row-stochastic on non-isolated rows.
    rs = Ahat.sum(1)
    assert np.allclose(rs[rs > 0], 1.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# ② co-reachability statistics (JL): E[<Xk[i],Xk[j]>] = dim·<Âᵏ[i,:],Âᵏ[j,:]>
# ══════════════════════════════════════════════════════════════════════════════
def test_coreachability_statistics_hold():
    np.random.seed(3)
    seeds = np.random.choice(NUM_NODES, 3, replace=False)        # small bag -> small U, meaningful values
    bag = real_bag(seeds, n_walks=5, max_len=4)
    dim, n_hops, draws = 64, 2, 500
    present, loc, A = independent_graph([bag], 0.0, True)
    _, Ahat = dense_diffuse(A, np.zeros((len(present), dim)), n_hops)
    An = np.linalg.matrix_power(Ahat, n_hops)
    C = torch.from_numpy(An @ An.T).float()                     # true n-hop co-reachability Gram

    enc = NodeEncoding(NUM_NODES, dim, n_hops, recency_lambda=0.0)
    torch.manual_seed(0)
    gram = torch.zeros(len(present), len(present))
    for _ in range(draws):
        _, ne = enc(bag)
        xk = ne[:, n_hops * dim:(n_hops + 1) * dim]
        gram += (xk @ xk.t()) / dim
    gram /= draws

    assert torch.allclose(gram, C, atol=0.04), (gram - C).abs().max()
    # and the empirical structure is genuinely non-trivial (not all ~0)
    assert C.max() > 0.05


# ══════════════════════════════════════════════════════════════════════════════
# ③ anonymization: relabelling node ids leaves structural inner-products unchanged
# ══════════════════════════════════════════════════════════════════════════════
def test_anonymization_relabel_invariant():
    np.random.seed(4)
    seeds = np.random.choice(NUM_NODES, 3, replace=False)
    bag = real_bag(seeds, n_walks=5, max_len=4)
    dim, n_hops, draws = 48, 2, 500

    perm = torch.randperm(NUM_NODES)
    nodes2 = torch.where(bag.nodes_mask, perm[bag.nodes.clamp_min(0)], bag.nodes)
    bag2 = WalkTokens(seeds=perm[bag.seeds], nodes=nodes2, nodes_mask=bag.nodes_mask,
                      ages=bag.ages, cutoffs=bag.cutoffs)

    enc = NodeEncoding(NUM_NODES, dim, n_hops, recency_lambda=0.0)

    def global_gram(b, seed):
        torch.manual_seed(seed)
        g = torch.zeros(NUM_NODES, NUM_NODES)
        for _ in range(draws):
            assoc, ne = enc(b)
            xk = ne[:, n_hops * dim:(n_hops + 1) * dim]
            present = (assoc >= 0).nonzero(as_tuple=True)[0]
            local = assoc[present]
            g[present[:, None], present[None, :]] += (xk[local] @ xk[local].t()) / dim
        return g / draws

    g1 = global_gram(bag, 0)
    g2 = global_gram(bag2, 0)
    # g2 is g1 with rows/cols permuted by `perm`; undo it and compare on the original present set.
    present = (g1.abs().sum(1) > 0).nonzero(as_tuple=True)[0]
    for a in present.tolist():
        for b in present.tolist():
            assert abs(g1[a, b].item() - g2[perm[a], perm[b]].item()) < 0.03


# ══════════════════════════════════════════════════════════════════════════════
# joint two-bag encoding: subset-A seeds vs subset-B seeds → one joint encoding
# ══════════════════════════════════════════════════════════════════════════════
def test_joint_two_bags_match_union_diffusion():
    torch.manual_seed(5)
    np.random.seed(5)
    picks = np.random.choice(NUM_NODES, 8, replace=False)
    bag_a = real_bag(picks[:4], n_walks=6, max_len=5)           # bag 1: subset A
    bag_b = real_bag(picks[4:], n_walks=6, max_len=5)           # bag 2: subset B
    # jointly encoded == diffusion over the UNION of both bags' walk graphs
    module_matches_dense([bag_a, bag_b], dim=16, n_hops=3, recency_lambda=0.0, undirected=True)


def test_joint_cross_bag_coreachability():
    """Bag-A and bag-B jointly encoded: the cross-bag co-reachability (an A-side node vs a B-side node)
    must match the dense UNION Â² Gram, and be non-zero wherever the two bags' walks share nodes — the
    exact property that makes source/candidate scoring work (separate encodings would give ~0)."""
    np.random.seed(6)
    picks = np.random.choice(NUM_NODES, 8, replace=False)
    bag_a = real_bag(picks[:4], n_walks=6, max_len=5)
    bag_b = real_bag(picks[4:], n_walks=6, max_len=5)
    dim, n_hops, draws = 64, 2, 500

    present, loc, A = independent_graph([bag_a, bag_b], 0.0, True)
    _, Ahat = dense_diffuse(A, np.zeros((len(present), dim)), n_hops)
    An = np.linalg.matrix_power(Ahat, n_hops)
    C = torch.from_numpy(An @ An.T).float()

    enc = NodeEncoding(NUM_NODES, dim, n_hops, recency_lambda=0.0)
    torch.manual_seed(0)
    gram = torch.zeros(len(present), len(present))
    for _ in range(draws):
        a, ne = enc(bag_a, bag_b)                               # JOINT
        idx = a[torch.tensor(present)]
        xk = ne[idx][:, n_hops * dim:(n_hops + 1) * dim]
        gram += (xk @ xk.t()) / dim
    gram /= draws

    assert torch.allclose(gram, C, atol=0.04), (gram - C).abs().max()
    # the two bags actually overlap in reachability -> some cross entries are non-zero.
    a_local = [loc[g] for g in bag_a.seeds.tolist() if g in loc]
    b_local = [loc[g] for g in bag_b.seeds.tolist() if g in loc]
    cross = C[a_local][:, b_local]
    assert cross.max() > 0.02                                   # A-side and B-side ARE co-reachable


# ══════════════════════════════════════════════════════════════════════════════
# structural invariants on real bags
# ══════════════════════════════════════════════════════════════════════════════
def test_shape_assoc_determinism():
    np.random.seed(7)
    seeds = np.random.choice(NUM_NODES, 5, replace=False)
    bag = real_bag(seeds, n_walks=6, max_len=5)
    dim, n_hops = 8, 3
    enc = NodeEncoding(NUM_NODES, dim, n_hops)

    torch.manual_seed(9); assoc, ne1 = enc(bag)
    torch.manual_seed(9); _, ne2 = enc(bag)
    u = int((assoc >= 0).sum())
    assert ne1.shape == (u, (n_hops + 1) * dim)                 # fixed width
    assert torch.equal(ne1, ne2)                               # determinism given the seed
    present = torch.unique(torch.cat([bag.nodes[bag.nodes_mask], bag.seeds]))
    assert torch.equal(present[assoc[present]], present)       # assoc round-trip
    assert torch.equal(assoc[bag.seeds].ge(0), torch.ones_like(bag.seeds, dtype=torch.bool))  # seeds present


def test_empty_batch():
    tok = WalkTokens(seeds=torch.empty(0, dtype=torch.long),
                     nodes=torch.empty(0, 6, 5, dtype=torch.long),
                     nodes_mask=torch.empty(0, 6, 5, dtype=torch.bool),
                     ages=torch.empty(0, 6, 5, dtype=torch.long),
                     cutoffs=torch.empty(0, dtype=torch.long))
    _, ne = NodeEncoding(NUM_NODES, 8, 3)(tok)
    assert ne.shape == (0, 4 * 8)


def test_seed_only_walks_are_isolated():
    # max_walk_len=1 -> each walk is just the seed, no edges -> every node's diffusion blocks are 0.
    np.random.seed(8)
    seeds = np.random.choice(NUM_NODES, 5, replace=False)
    bag = real_bag(seeds, n_walks=4, max_len=1)
    dim, n_hops = 8, 2
    _, ne = NodeEncoding(NUM_NODES, dim, n_hops)(bag)
    assert torch.count_nonzero(ne[:, dim:]) == 0               # all diffused blocks zero
    assert torch.isfinite(ne).all()
