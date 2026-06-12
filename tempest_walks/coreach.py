"""Walk-derived time-decayed co-reachability (TPNet pair-feature #3, exact).

For a query pair (u, v), co-reachability = the weighted count of third-party nodes w
that appear in BOTH u's and v's backward walks — i.e. the shared recent neighbourhood.
This is the genuine, causal analog of TPNet's ``⟨A^(li)_u, A^(lj)_v⟩`` Gram block
(``pair-feature-integration.md`` #3), computed EXACTLY from the walks Tempest already
samples for every unique node in the batch (no JL random-feature sketch). It closes
the new-edge slice that exact recurrence (#1) leaves untouched: a never-before-seen
pair still scores high if u and v share active neighbours.

Implementation: build a sparse membership matrix ``M[node, w]`` = (weighted) count of
w in that node's walk context, then ``coreach[b,c] = ⟨M[u_b], M[v_bc]⟩`` as one
vectorized elementwise sparse multiply + row-sum. Memory is O(#walk slots), never
O(N²). Strict-causal: the walks reflect only pre-query edges, so does this.

Walk contract (backward): context positions are ``p < lens-1`` (the seed slot lens-1
is the node itself and is excluded — we want shared THIRD parties, not the pair
itself); padding is ``-1``. See CLAUDE.md "Tempest walk contract".
"""
import numpy as np
import scipy.sparse as sp
import torch


def _membership(nodes: np.ndarray, lens: np.ndarray, num_nodes: int,
                weights: np.ndarray = None) -> sp.csr_matrix:
    """nodes [M,K,L] int, lens [M,K] int -> csr [M, num_nodes] of context counts."""
    M, K, L = nodes.shape
    pos = np.arange(L)[None, None, :]
    is_ctx = (pos < (lens[..., None] - 1)) & (nodes >= 0)     # [M,K,L]
    rows = np.broadcast_to(np.arange(M)[:, None, None], (M, K, L))[is_ctx]
    cols = nodes[is_ctx]
    data = (np.ones(rows.shape, np.float32) if weights is None
            else weights[is_ctx].astype(np.float32))
    memb = sp.csr_matrix((data, (rows, cols)), shape=(M, num_nodes))
    memb.sum_duplicates()
    return memb


def coreach_feature(nodes: torch.Tensor, lens: torch.Tensor, num_nodes: int,
                    u_pos: torch.Tensor, v_pos: torch.Tensor,
                    weights: np.ndarray = None) -> torch.Tensor:
    """nodes [M,K,L], lens [M,K] (cpu); u_pos [B], v_pos [B,C] index into M.
    Returns coreach_log [B, C] float32 (log1p of the shared-neighbour weight)."""
    nd = nodes.detach().cpu().numpy().astype(np.int64)
    ln = lens.detach().cpu().numpy().astype(np.int64)
    up = u_pos.detach().cpu().numpy().astype(np.int64)
    vp = v_pos.detach().cpu().numpy().astype(np.int64)
    B, C = vp.shape

    memb = _membership(nd, ln, num_nodes, weights)               # [M, N] csr
    # Dense source rows (B is small: B<=batch_size) then a sparse matvec of the
    # C candidate rows against each source — 5x faster than a [B*C, N] elementwise
    # multiply, and the dense block is only [B, N] (~MBs).
    su_dense = np.asarray(memb[up].todense())                    # [B, N]
    co = np.empty((B, C), np.float32)
    for b in range(B):
        co[b] = memb[vp[b]].dot(su_dense[b])                     # [C,N]·[N] -> [C]
    return torch.from_numpy(np.log1p(co))
