"""Walk-neighbourhood CSR — two-level packed (seed → walk → position), COUNT-FREE.

This replaces the earlier dedup-to-distinct-node CSR. The old layout collapsed each
seed's walk-neighbours into distinct nodes carrying occurrence ages + an explicit COUNT,
then re-weighted μ / co-reach by log(1+k). The new layout keeps every reached POSITION as
its own token (no dedup, no count): a node that recurs k times simply appears as k tokens,
so multiplicity is carried implicitly by token repetition and the explicit count terms
disappear from both the data and the head's equations.

  WalkBatch — fully packed, NO padding:
    seeds           [G]          int64    deduped unique walked seed nodes
    walk_nodes      [P]          int32    reached-node id at each position
    walk_pos_ts     [P]          int64    raw edge timestamp t_edge (age = t_query − t_edge
                                          is formed downstream, per the walk contract)
    walk_edge_feats [P, d_ef]    float32  per-position edge feature, or None
    walk_csr        [W+1]        int64    walk  → its slice of positions  (W = G·K walks)
    seed_csr        [G+1]        int64    seed  → its slice of walks       (uniform stride K)

  Positions are packed walk-major, position-ascending. Walk w owns
  walk_nodes[walk_csr[w] : walk_csr[w+1]] (and the parallel ts / edge-feat slices); seed g
  owns walks [seed_csr[g] : seed_csr[g+1]). Because positions are contiguous within a walk
  and walks are contiguous within a seed, seed g's positions are ALSO one contiguous slice
  [walk_csr[seed_csr[g]] : walk_csr[seed_csr[g+1]]) — `walk_batch_to_dense` exploits this.

Three things are excluded before packing (same contract as before):
  • the seed SLOT (position lens−1, the INT64_MAX sentinel) and padding (p ≥ lens),
  • the walk's OWN ORIGIN node-id wherever it recurs at a context position — a backward
    walk can re-enter its origin as a "neighbour"; since Log_{E[seed]}(E[seed]) = 0 it would
    pull μ toward the zero vector and let the connector witness itself in co-reach. Each side
    passes its own seeds, so this drops u from the source batch and v from the candidate
    batch while keeping u among v's connectors (u is not the candidate walk's origin).

The two-level structure preserves walk identity (which positions form one temporal path),
which the current μ / co-reach head does not need but a future path-count co-reachability
signal will. The current head consumes the per-seed dense view from `walk_batch_to_dense`.
Strict-causal: walks reflect the pre-ingest graph snapshot.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

DenseTokens = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # node_ids, node_mask, pos_ts


@dataclass
class WalkBatch:
    seeds: torch.Tensor                       # [G]      int64
    walk_nodes: torch.Tensor                  # [P]      int32
    walk_pos_ts: torch.Tensor                 # [P]      int64  raw t_edge
    walk_edge_feats: Optional[torch.Tensor]   # [P, d_ef] float32, or None
    walk_csr: torch.Tensor                    # [W+1]    int64  walk → positions
    seed_csr: torch.Tensor                    # [G+1]    int64  seed → walks


def build_walk_batch(walk_gen, device: torch.device, seeds_unique: torch.Tensor,
                     *, max_walk_len: Optional[int] = None,
                     num_walks_per_node: Optional[int] = None,
                     start_bias: Optional[str] = None,
                     walk_bias: Optional[str] = None) -> WalkBatch:
    """Walk for `seeds_unique`, pack the real context tokens into a two-level CSR WalkBatch.

       walk_gen        a WalkGenerator (supports per-call walk-param overrides)
       device          torch device for the packed tensors
       seeds_unique    [G] long   unique seed node ids (sources OR candidates)
       -> WalkBatch

    Excludes the seed slot, padding, and the walk's own origin node-id (see module doc).
    NO dedup, NO counts — every surviving (node, t_edge, edge-feat) position is a token."""
    G = int(seeds_unique.shape[0])
    wd = walk_gen.walks_for_nodes(
        seeds_unique.cpu().numpy(),
        max_walk_len=max_walk_len,
        num_walks_per_node=num_walks_per_node,
        start_bias=start_bias,
        walk_bias=walk_bias)
    K, L = int(wd.K), int(wd.nodes.shape[1])
    W = int(wd.nodes.shape[0])                                   # G·K walk rows
    nodes = wd.nodes.to(device)                                 # [W, L] int64 (−1 pad)
    ts = wd.timestamps.to(device)                               # [W, L] int64 edge times
    lens = wd.lens.to(device)                                   # [W]

    pos = torch.arange(L, device=device).view(1, L)
    is_ctx = pos < (lens.view(W, 1) - 1)                        # context: excl seed slot + pad
    # rows [g·K, (g+1)·K) belong to seed g (shuffle_walk_order=False) → origin per walk row.
    origin = wd.seeds.to(device).repeat_interleave(K)          # [W]
    valid = is_ctx & (nodes != origin.view(W, 1))              # [W, L] real, non-origin tokens

    flat_valid = valid.reshape(-1)                             # [W·L]
    walk_nodes = nodes.reshape(-1)[flat_valid].to(torch.int32)         # [P]
    walk_pos_ts = ts.reshape(-1)[flat_valid].to(torch.int64)          # [P]
    walk_edge_feats = None
    if wd.edge_feats is not None:
        d_ef = wd.edge_feats.shape[-1]
        walk_edge_feats = wd.edge_feats.to(device).reshape(-1, d_ef)[flat_valid]   # [P, d_ef]

    cnt_walk = valid.sum(dim=1)                                # [W] tokens per walk
    walk_csr = torch.zeros(W + 1, dtype=torch.int64, device=device)
    walk_csr[1:] = torch.cumsum(cnt_walk, dim=0)
    # every seed contributes exactly K walk rows (Tempest pads short/empty walks as rows).
    seed_csr = torch.arange(G + 1, device=device, dtype=torch.int64) * K

    return WalkBatch(seeds=wd.seeds.to(device), walk_nodes=walk_nodes,
                     walk_pos_ts=walk_pos_ts, walk_edge_feats=walk_edge_feats,
                     walk_csr=walk_csr, seed_csr=seed_csr)


def walk_batch_to_dense(wb: WalkBatch) -> DenseTokens:
    """Collapse the walk level — give each seed ONE dense row of its tokens (for the μ /
    co-reach head, which scores a per-seed token bag and does not need walk identity).

       -> node_ids [G, U] (−1 pad), node_mask [G, U], pos_ts [G, U]   (U = max tokens/seed)

    Seed g's tokens are the contiguous packed slice [seed_start[g], seed_end[g]); we scatter
    each token to (its seed, its within-seed slot). Cold seeds (no tokens) → all-False row."""
    device = wb.walk_nodes.device
    G = int(wb.seeds.shape[0])
    seed_start = wb.walk_csr[wb.seed_csr[:-1]]                 # [G] first token of each seed
    seed_end = wb.walk_csr[wb.seed_csr[1:]]                    # [G] one-past-last token
    cnt_seed = seed_end - seed_start                          # [G] tokens per seed
    U = max(int(cnt_seed.max().item()) if G > 0 else 1, 1)
    P = int(wb.walk_nodes.shape[0])

    node_ids = torch.full((G, U), -1, device=device, dtype=wb.walk_nodes.dtype)
    node_mask = torch.zeros((G, U), device=device, dtype=torch.bool)
    pos_ts = torch.zeros((G, U), device=device, dtype=wb.walk_pos_ts.dtype)
    if P > 0:
        ar = torch.arange(P, device=device)
        seed_of = torch.searchsorted(seed_start, ar, right=True) - 1   # [P] owning seed
        slot_of = ar - seed_start[seed_of]                            # [P] within-seed slot
        node_ids[seed_of, slot_of] = wb.walk_nodes
        node_mask[seed_of, slot_of] = True
        pos_ts[seed_of, slot_of] = wb.walk_pos_ts
    return node_ids, node_mask, pos_ts


def gather_dense(dense: DenseTokens, index: torch.Tensor, out_shape) -> DenseTokens:
    """Index a per-unique-seed dense view [G, U] onto the batch grid via `index` (long [P]),
    reshaping the leading axis to `out_shape` (e.g. [B] for sources, [B,C] for candidates).
    Each seed's tokens are query-independent, so gathering replicates them to every cell
    naming that seed (scatter-add adjoint on backward)."""
    node_ids, node_mask, pos_ts = dense
    U = node_ids.shape[-1]
    return (node_ids[index].view(*out_shape, U),
            node_mask[index].view(*out_shape, U),
            pos_ts[index].view(*out_shape, U))
