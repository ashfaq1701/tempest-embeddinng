"""Correctness tests for NodeEncoding — the stateless batch-local node embedding.

EVERY test runs on the real graph in tests/data/sample_data.csv, walked by real Tempest
(build_walk_nodes). No synthetic tokens. The oracle is an INDEPENDENT dense reconstruction of the
diffusion, built by explicit per-walk edge enumeration (a genuinely different implementation from the
module's vectorized slicing) so shared-recipe bugs are caught.

NodeEncoding consumes a single WalkNodes bag (seeds + walk connectivity, no ages/features): the batch's
unique node union walked at ONE shared cutoff. Edges are UNWEIGHTED — the bare WalkNodes carries no
ages, so there is no recency weighting.

The basis is FIXED: one permanent random fingerprint per global node id (x0_table). Tests that need the
JL EXPECTATION average over many re-seeded tables (the randomness is over the basis draw, not per batch).

Properties verified against real bags:
  - exact diffusion  node_enc == [X0, ÂX0, …, ÂⁿX0]  (Â = D⁻¹A over the walk-induced graph, unweighted),
  - co-reachability statistics  E[<Xk[i],Xk[j]>] = dim·<Âᵏ[i,:],Âᵏ[j,:]>  (JL, over re-seeded bases),
  - fixed basis: a node's code is keyed to its global id — stable across batches (identity-keyed),
  - directed mode, shape/assoc/determinism, empty and seed-only (isolated) bags.
"""
import pathlib

import numpy as np
import torch

from link_property_prediction.model import NodeEncoding
from link_property_prediction.walk_tokens import WalkNodes, build_walk_nodes
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


def real_walk_nodes(seeds, n_walks, max_len, cutoff=CUTOFF):
    """A real WalkNodes bag: n_walks per seed on the sample graph, one shared cutoff (permissive)."""
    seeds_t = torch.as_tensor(seeds, dtype=torch.long)
    return build_walk_nodes(_WG, _DEV, seeds_t, cutoff,
                            max_walk_len=max_len, num_walks_per_node=n_walks)


# ── INDEPENDENT dense oracle (explicit enumeration; NOT the module's slicing) ─
def independent_graph(wn, undirected):
    """Rebuild the batch-local UNWEIGHTED adjacency from a WalkNodes bag by looping every consecutive
    (position l, l+1) real pair — a different implementation from the module. Returns (present, loc, A)
    with A [U,U] numpy, rows/cols in sorted-global-id order (matches torch.unique)."""
    nd, nm = wn.nodes.numpy(), wn.nodes_mask.numpy()
    present = sorted(set(nd[nm].tolist()) | set(wn.seeds.numpy().tolist()))
    loc = {g: i for i, g in enumerate(present)}
    u = len(present)
    A = np.zeros((u, u), dtype=np.float64)
    q, k, length = nd.shape
    for iq in range(q):
        for ik in range(k):
            for il in range(length - 1):
                if nm[iq, ik, il] and nm[iq, ik, il + 1]:
                    a, b = loc[nd[iq, ik, il]], loc[nd[iq, ik, il + 1]]
                    A[a, b] += 1.0                          # unweighted: one count per traversal
                    if undirected:
                        A[b, a] += 1.0
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


def module_matches_dense(wn, dim, n_hops, undirected):
    """Run NodeEncoding on `wn`, extract its own X0 (block 0), and assert every diffused block equals
    the independent dense reference. Returns (assoc, node_enc, present, Ahat) for further checks."""
    enc = NodeEncoding(NUM_NODES, dim, n_hops, undirected=undirected)
    assoc, node_enc = enc(wn)
    present, loc, A = independent_graph(wn, undirected)
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
    wn = real_walk_nodes(seeds, n_walks=8, max_len=6)
    module_matches_dense(wn, dim=16, n_hops=3, undirected=True)


def test_diffusion_matches_independent_dense_directed():
    torch.manual_seed(1)
    np.random.seed(1)
    seeds = np.random.choice(NUM_NODES, 4, replace=False)
    wn = real_walk_nodes(seeds, n_walks=8, max_len=6)
    _, _, _, Ahat = module_matches_dense(wn, dim=16, n_hops=3, undirected=False)
    # sanity: the reconstructed transition is row-stochastic on non-isolated rows.
    rs = Ahat.sum(1)
    assert np.allclose(rs[rs > 0], 1.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# ② co-reachability statistics (JL): E[<Xk[i],Xk[j]>] = dim·<Âᵏ[i,:],Âᵏ[j,:]>
# ══════════════════════════════════════════════════════════════════════════════
def test_coreachability_statistics_hold():
    np.random.seed(3)
    seeds = np.random.choice(NUM_NODES, 3, replace=False)        # small bag -> small U, meaningful values
    wn = real_walk_nodes(seeds, n_walks=5, max_len=4)
    dim, n_hops, draws = 64, 2, 500
    present, loc, A = independent_graph(wn, True)
    _, Ahat = dense_diffuse(A, np.zeros((len(present), dim)), n_hops)
    An = np.linalg.matrix_power(Ahat, n_hops)
    C = torch.from_numpy(An @ An.T).float()                     # true n-hop co-reachability Gram

    # Average over the RANDOM BASIS: a fresh fingerprint table each draw (fixed within a draw). The
    # empirical Gram then estimates E over the basis of dim·<Âⁿ[i,:], Âⁿ[j,:]>.
    gram = torch.zeros(len(present), len(present))
    for d in range(draws):
        _, ne = NodeEncoding(NUM_NODES, dim, n_hops, basis_seed=d)(wn)
        xk = ne[:, n_hops * dim:(n_hops + 1) * dim]
        gram += (xk @ xk.t()) / dim
    gram /= draws

    assert torch.allclose(gram, C, atol=0.04), (gram - C).abs().max()
    # and the empirical structure is genuinely non-trivial (not all ~0)
    assert C.max() > 0.05


# ══════════════════════════════════════════════════════════════════════════════
# ③ fixed basis: a node's code is keyed to its GLOBAL id — stable across batches
# ══════════════════════════════════════════════════════════════════════════════
def test_fixed_basis_stable_across_batches():
    """The same node in two DIFFERENT bags gets the SAME base fingerprint (block 0), equal to its
    permanent x0_table row — i.e. keyed to identity, not to the batch (this is what makes the codes
    MLP-usable / recurrence-aware). NB: relabelling ids WOULD change it — the fixed basis is deliberately
    identity-keyed, not anonymized."""
    np.random.seed(4)
    dim, n_hops = 32, 2
    picks = np.random.choice(NUM_NODES, 8, replace=False)
    enc = NodeEncoding(NUM_NODES, dim, n_hops)
    a_assoc, a_enc = enc(real_walk_nodes(picks[:5], n_walks=5, max_len=4))
    b_assoc, b_enc = enc(real_walk_nodes(picks[3:], n_walks=5, max_len=4))    # overlaps on picks[3:5] (both seeds)

    shared = ((a_assoc >= 0) & (b_assoc >= 0)).nonzero(as_tuple=True)[0]
    assert shared.numel() > 0
    for g in shared.tolist():
        base_a, base_b = a_enc[a_assoc[g], :dim], b_enc[b_assoc[g], :dim]
        assert torch.equal(base_a, base_b)                             # same id → same fingerprint across batches
        assert torch.equal(base_a, enc.x0_table[g])                    # and it IS the permanent table row


# ══════════════════════════════════════════════════════════════════════════════
# structural invariants on real bags
# ══════════════════════════════════════════════════════════════════════════════
def test_shape_assoc_determinism():
    np.random.seed(7)
    seeds = np.random.choice(NUM_NODES, 5, replace=False)
    wn = real_walk_nodes(seeds, n_walks=6, max_len=5)
    dim, n_hops = 8, 3
    enc = NodeEncoding(NUM_NODES, dim, n_hops)

    torch.manual_seed(9); assoc, ne1 = enc(wn)
    torch.manual_seed(9); _, ne2 = enc(wn)
    u = int((assoc >= 0).sum())
    assert ne1.shape == (u, (n_hops + 1) * dim)                 # fixed width
    assert torch.equal(ne1, ne2)                               # determinism given the seed
    present = torch.unique(torch.cat([wn.nodes[wn.nodes_mask], wn.seeds]))
    assert torch.equal(present[assoc[present]], present)       # assoc round-trip
    assert torch.equal(assoc[wn.seeds].ge(0), torch.ones_like(wn.seeds, dtype=torch.bool))  # seeds present


def test_empty_batch():
    wn = WalkNodes(seeds=torch.empty(0, dtype=torch.long),
                   nodes=torch.empty(0, 6, 5, dtype=torch.long),
                   nodes_mask=torch.empty(0, 6, 5, dtype=torch.bool))
    _, ne = NodeEncoding(NUM_NODES, 8, 3)(wn)
    assert ne.shape == (0, 4 * 8)


def test_seed_only_walks_are_isolated():
    # max_walk_len=1 -> each walk is just the seed, no edges -> every node's diffusion blocks are 0.
    np.random.seed(8)
    seeds = np.random.choice(NUM_NODES, 5, replace=False)
    wn = real_walk_nodes(seeds, n_walks=4, max_len=1)
    dim, n_hops = 8, 2
    _, ne = NodeEncoding(NUM_NODES, dim, n_hops)(wn)
    assert torch.count_nonzero(ne[:, dim:]) == 0               # all diffused blocks zero
    assert torch.isfinite(ne).all()
