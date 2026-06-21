"""Symmetric walk-neighbourhood CSR — shared token prep for both scoring sides.

The source side (u → μ) and the candidate side (v → connectors) of the geometric link
head consume the SAME per-seed deduplicated representation. This module holds that prep:

  walk_csr(...)   — walk the unique seeds, flatten the (K,L) axes, dedup per seed → CSR.
  dedup_to_csr()  — collapse a flat per-seed token set (with repeats) into the compact CSR.
  gather_csr()    — index a per-unique-seed CSR onto the batch grid ([B] or [B,C]).

The CSR bundle (WalkCSR) is:
  node_ids  [G, U]        distinct neighbour node ids per seed (−1 at padded node slots)
  node_mask [G, U]        True at a real distinct node
  ages      [G, U, kmax]  each distinct node's OCCURRENCE ages (raw; 0 at padded slots)
  age_mask  [G, U, kmax]  True at a real occurrence (count_node = age_mask.sum(-1))

Distinct nodes carry ALL their occurrence ages (recency mean stays exact) and, implicitly,
their COUNT. Both scoring sides emit this identical layout; they differ only in which seeds
(sources vs candidates) and which walk params. The dedup is EXACT — the per-node CSR is
query-independent given a single pre-ingest snapshot and shift-invariant recency weighting —
so a per-unique-seed CSR can be gathered/scattered to every cell naming that seed.
"""
from typing import Optional, Tuple

import torch

WalkCSR = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def walk_csr(walk_gen, device: torch.device,
             seeds_unique: torch.Tensor, t_query_per_seed: torch.Tensor,
             *, max_walk_len: Optional[int] = None,
             num_walks_per_node: Optional[int] = None,
             start_bias: Optional[str] = None,
             walk_bias: Optional[str] = None) -> WalkCSR:
    """Walk for `seeds_unique`, then DEDUPLICATE each seed's walk-neighbours into a compact
    per-node CSR. Symmetric: the source and candidate sides call this with the same contract,
    differing only in seeds + walk params.

       walk_gen          a WalkGenerator (supports per-call walk-param overrides)
       device            torch device for the CSR tensors
       seeds_unique      [G] long      unique seed node ids (sources OR candidates)
       t_query_per_seed  [G] long      the query time to age each seed's tokens against
       -> WalkCSR (node_ids [G,U], node_mask [G,U], ages [G,U,kmax], age_mask [G,U,kmax])

    Dedup is per seed-row: occurrences of the same neighbour node collapse to ONE node slot
    carrying ALL its occurrence ages (recency mean stays exact) and its count
    (= #occurrences = age_mask.sum(-1)). Seed (= the node itself) and walk padding are
    excluded before dedup. Strict-causal: pre-ingest graph snapshot."""
    G = int(seeds_unique.shape[0])
    wd = walk_gen.walks_for_nodes(
        seeds_unique.cpu().numpy(),
        max_walk_len=max_walk_len,
        num_walks_per_node=num_walks_per_node,
        start_bias=start_bias,
        walk_bias=walk_bias)
    K, L = wd.K, wd.nodes.shape[1]
    nodes = wd.nodes.to(device).view(G, K, L)               # [G, K, L]
    ts = wd.timestamps.to(device).view(G, K, L)             # [G, K, L] int64 edge times
    lens = wd.lens.to(device).view(G, K)                    # [G, K]
    # Context = real walk-neighbours: exclude the seed slot (lens-1) and padding.
    is_ctx = torch.arange(L, device=device).view(1, 1, L) < (lens - 1).unsqueeze(-1)

    n = K * L
    flat_ids = nodes.reshape(G, n)                          # [G, n] node ids (−1 pad)
    flat_ts = ts.reshape(G, n)                              # [G, n] raw edge times
    flat_mask = is_ctx.reshape(G, n)                        # [G, n] True at real token
    # DROP THE WALK'S OWN ORIGIN node-id (not just its seed SLOT): a backward walk that
    # revisits its origin re-enters the seed as a "neighbour" at a context position. Since
    # the seed is the most-revisited node, the logsumexp-over-occurrences hands it dominant
    # softmax mass in μ — but Log_base(E[base]) = 0, so μ is pulled toward a ZERO vector
    # (≈27% of μ's mass on wiki, ≈50% on cold/low-degree sources), diluting the genuine
    # neighbours; on the candidate side it also makes the connector v witness ITSELF in
    # co-reach (the witness collapses to the identity distance). Each side passes its OWN
    # seeds, so this removes u from the source CSR and v from the candidate CSR while keeping
    # u-among-v's-connectors (u is not the candidate walk's origin) — the real pair signal.
    flat_mask = flat_mask & (flat_ids != seeds_unique.view(G, 1).to(flat_ids.dtype))
    # Per-token RAW age = t_query − t_edge (≥0), masked. clamp_min neutralises the seed
    # sentinel; the mask zeroes padding's bogus value — finite everywhere.
    flat_age = ((t_query_per_seed.view(G, 1).to(torch.int64) - flat_ts).clamp_min(0)
                .to(torch.float32)) * flat_mask.to(torch.float32)       # [G, n]

    return dedup_to_csr(flat_ids, flat_age, flat_mask)


def dedup_to_csr(flat_ids: torch.Tensor, flat_age: torch.Tensor,
                 flat_mask: torch.Tensor) -> WalkCSR:
    """Collapse a flat per-seed token set [G, n] (with repeated nodes) into the compact CSR
    (node_ids [G,U], node_mask [G,U], ages [G,U,kmax], age_mask [G,U,kmax]) by grouping
    occurrences of the same node id within each row. Vectorised, dense-padded.

    U = max distinct nodes over rows; kmax = max occurrences of any one node in a row.
    All occurrence ages are kept (recency stays exact); count = age_mask.sum(-1)."""
    device = flat_ids.device
    G, n = flat_ids.shape

    # Sort each row by node id so equal ids are contiguous; invalids (mask False) pushed to
    # the end via a large sentinel key. Stable so occurrence order within a node holds.
    BIG = torch.iinfo(torch.int64).max
    sort_key = torch.where(flat_mask, flat_ids, torch.full_like(flat_ids, BIG))
    order = torch.argsort(sort_key, dim=1, stable=True)              # [G, n]
    ids_s = torch.gather(flat_ids, 1, order)                        # [G, n] sorted ids
    age_s = torch.gather(flat_age, 1, order)                        # [G, n]
    msk_s = torch.gather(flat_mask, 1, order)                       # [G, n]

    # "new distinct node" boundary within a row: first valid slot, or id changes.
    prev_id = torch.cat([torch.full((G, 1), -2, device=device, dtype=ids_s.dtype),
                         ids_s[:, :-1]], dim=1)
    prev_msk = torch.cat([torch.zeros((G, 1), device=device, dtype=torch.bool),
                          msk_s[:, :-1]], dim=1)
    is_new = msk_s & (~prev_msk | (ids_s != prev_id))               # [G, n] start of a node-run
    # distinct-node index per valid slot (cumsum of run-starts − 1); invalids → −1.
    node_idx = torch.cumsum(is_new.to(torch.int64), dim=1) - 1      # [G, n]
    node_idx = torch.where(msk_s, node_idx, torch.full_like(node_idx, -1))
    U = int(node_idx.max().item()) + 1 if msk_s.any() else 1

    # occurrence index WITHIN each distinct node: position minus the run-start position.
    ar = torch.arange(n, device=device).view(1, n).expand(G, n)     # [G, n] col positions
    run_start_pos = torch.where(is_new, ar, torch.zeros_like(ar))
    # cummax of run_start_pos gives, at each slot, the start position of its current run.
    run_start = torch.cummax(run_start_pos, dim=1).values            # [G, n]
    occ_idx = torch.where(msk_s, ar - run_start, torch.zeros_like(ar))   # [G, n]
    kmax = int(occ_idx[msk_s].max().item()) + 1 if msk_s.any() else 1

    # Scatter sorted (id, age) into [G, U, kmax] by (node_idx, occ_idx).
    node_ids = torch.full((G, U), -1, device=device, dtype=flat_ids.dtype)
    node_mask = torch.zeros((G, U), device=device, dtype=torch.bool)
    ages = torch.zeros((G, U, kmax), device=device, dtype=flat_age.dtype)
    age_mask = torch.zeros((G, U, kmax), device=device, dtype=torch.bool)

    valid = msk_s
    rows = torch.arange(G, device=device).view(G, 1).expand(G, n)[valid]
    u_at = node_idx[valid]
    k_at = occ_idx[valid]
    node_ids[rows, u_at] = ids_s[valid]
    node_mask[rows, u_at] = True
    ages[rows, u_at, k_at] = age_s[valid]
    age_mask[rows, u_at, k_at] = True
    return node_ids, node_mask, ages, age_mask


def gather_csr(csr: WalkCSR, index: torch.Tensor, out_shape) -> WalkCSR:
    """Index a per-unique-seed CSR [G,…] onto the batch grid via `index` (long [P]),
    reshaping the leading axis to `out_shape` (e.g. [B] for sources, [B,C] for candidates).
    The dedup is exact: each seed's CSR is query-independent, so gathering replicates it to
    every cell naming that seed (scatter-add adjoint on backward)."""
    node_ids, node_mask, ages, age_mask = csr
    U = node_ids.shape[-1]
    kmax = ages.shape[-1]
    ni = node_ids[index].view(*out_shape, U)
    nm = node_mask[index].view(*out_shape, U)
    ag = ages[index].view(*out_shape, U, kmax)
    am = age_mask[index].view(*out_shape, U, kmax)
    return ni, nm, ag, am
