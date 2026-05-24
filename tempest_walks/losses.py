"""Loss functions.

One pure function, no shared state, no classes:

alignment_loss(E, P_target, P_context, walks, t_now, T_train, β, τ,
               ..., chunk_size) -> scalar
  - InfoNCE contrastive loss over batched walks.
  - For each seed s_i with positive contexts {n_p^+ : p in walk i},
    and all batch contexts collected into a flat pool of size M=NK*L:

        L_i = - (Σ_p w[i,p] · log p(n_p^+ | s_i)) / (Σ_p w[i,p])

        log p(n_p^+ | s_i) =  -||P_target(E(s_i)) - P_context(E(n_p^+))||² / τ
                            - log Σ_j exp(-||P_target(E(s_i)) - P_context(E(n_j))||² / τ)

    where j ranges over ALL VALID batch contexts (every other walk's
    positions as in-batch negatives).
  - Hop/time weights w(K, Δt) = 1/K + (1 + Δt/T_train)^(-β) survive
    as multipliers on each positive's log-prob. Padding and seed
    slots are masked out via a very-negative similarity (-1e9).
  - The softmax denominator does what Wang-Isola uniformity did
    (push seed projections away from non-positive contexts), but
    with task-relevant negatives instead of random pairs.

  - chunk_size > 0: process seeds in chunks of that size. Each
    chunk's [chunk, M] sim matrix is built independently; each
    seed's log Z_i is still computed over the FULL pool, so the
    chunked result is algebraically identical to the non-chunked
    result. Used to fit InfoNCE on memory-constrained GPUs.
  - chunk_size == 0: no chunking (full [NK, M] matrix at once).

Link BCE is one line at the call site
(F.binary_cross_entropy_with_logits(...)), not factored out.
"""

from typing import Optional

import torch

from .walks import WalkData


def alignment_loss(
    embedding_table,                              # EmbeddingTable
    p_target,                                     # ProjectionHead — seed/downstream
    p_context,                                    # ProjectionHead — walk-internal/upstream
    walks: WalkData,
    t_now: int,
    T_train: float,
    beta: float = 1.0,
    tau: float = 0.5,
    node_feat: Optional[torch.Tensor] = None,     # [num_nodes, d_nf] or None
    chunk_size: int = 0,                          # Task 16: 0 = no chunking
) -> torch.Tensor:
    """InfoNCE contrastive alignment over batched walks. Returns
    scalar mean loss over seeds-with-positives.

    chunk_size > 0 processes seeds in chunks; loss is exact regardless.
    """
    device = embedding_table.E.weight.device
    nodes = walks.nodes.to(device).long()                         # [NK, L]
    timestamps = walks.timestamps.to(device).long()               # [NK, L]
    lens = walks.lens.to(device).long()                           # [NK]
    seeds_t = walks.seeds.to(device).long()                       # [N]
    K = walks.K
    NK, L = nodes.shape
    M = NK * L                                                    # pool size

    # Valid context positions: p ∈ [0, lens - 2]. Position lens-1 is
    # the seed (excluded), positions >= lens are padding.
    positions = torch.arange(L, device=device).unsqueeze(0)       # [1, L]
    seed_pos = (lens - 1).unsqueeze(1)                            # [NK, 1]
    is_context = positions < seed_pos                             # [NK, L]
    ctx_valid_mask = is_context.reshape(M)                        # [M]

    # Projections.
    seed_per_row = seeds_t.repeat_interleave(K)                   # [NK]
    e_seed = embedding_table(seed_per_row)                        # [NK, d_emb]
    nodes_safe = nodes.clamp_min(0)                               # [NK, L]
    e_ctx_flat = embedding_table(nodes_safe.reshape(-1))          # [M, d_emb]

    if node_feat is not None:
        nf_seed = node_feat[seed_per_row]                         # [NK, d_nf]
        nf_ctx_flat = node_feat[nodes_safe.reshape(-1)]           # [M, d_nf]
        p_seed_all = p_target(e_seed, node_feat=nf_seed)          # [NK, d_proj]
        p_ctx_pool = p_context(e_ctx_flat, node_feat=nf_ctx_flat) # [M, d_proj]
    else:
        p_seed_all = p_target(e_seed)                             # [NK, d_proj]
        p_ctx_pool = p_context(e_ctx_flat)                        # [M, d_proj]

    # Positive-mask helper: pool entry j is a positive of seed i iff
    # j // L == i (same walk) AND ctx_valid_mask[j]. The mask itself
    # is built per-chunk inside the loop below — at full [NK, M] bool
    # it is the dominant memory term (5+ GB on comment-scale batches)
    # and defeats the point of chunking.
    pool_walk_idx = torch.arange(M, device=device) // L           # [M]

    # Hop/time weights (computed once; broadcast across pool entries).
    hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()  # [NK, L]
    dt = (float(t_now) - timestamps.float()).clamp_min(0.0)       # [NK, L]
    dt_norm = dt / max(T_train, 1.0)
    w_hop = 1.0 / hop_dist                                        # [NK, L]
    w_time = (1.0 + dt_norm).pow(-beta)                           # [NK, L]
    w = w_hop + w_time                                            # [NK, L]
    w_flat = w.reshape(M)                                         # [M]

    # Determine chunk boundaries. chunk_size <= 0 OR >= NK means
    # no chunking — process all seeds in one pass.
    if chunk_size <= 0 or chunk_size >= NK:
        chunk_bounds = [(0, NK)]
    else:
        chunk_bounds = [
            (s, min(s + chunk_size, NK))
            for s in range(0, NK, chunk_size)
        ]

    # Accumulate loss across chunks. Each chunk's per-seed loss
    # contributes independently; the final mean is over all seeds
    # with positives.
    INVALID_MASK = -1e9
    total_loss_sum = torch.zeros((), device=device)
    total_valid_seeds = torch.zeros((), device=device)

    for start, end in chunk_bounds:
        chunk_n = end - start
        p_seed_chunk = p_seed_all[start:end]                      # [chunk, d_proj]

        # Sim against the FULL pool — exact log Z_i for each seed.
        sim_dot = p_seed_chunk @ p_ctx_pool.T                     # [chunk, M]
        sq_dist = 2.0 - 2.0 * sim_dot                             # [chunk, M]
        sim = -sq_dist / tau                                      # [chunk, M]
        sim = sim.masked_fill(~ctx_valid_mask.unsqueeze(0), INVALID_MASK)

        log_Z = torch.logsumexp(sim, dim=1)                       # [chunk]
        log_p = sim - log_Z.unsqueeze(1)                          # [chunk, M]

        # Per-chunk positive mask — sized [chunk, M], not [NK, M].
        seed_indices_chunk = torch.arange(
            start, end, device=device).unsqueeze(1)               # [chunk, 1]
        pos_mask_chunk = (pool_walk_idx.unsqueeze(0) == seed_indices_chunk) \
                         & ctx_valid_mask.unsqueeze(0)            # [chunk, M]
        w_pos = w_flat.unsqueeze(0).expand(chunk_n, -1) \
                * pos_mask_chunk.float()                          # [chunk, M]

        numerator = (w_pos * log_p).sum(dim=1)                    # [chunk]
        denominator = w_pos.sum(dim=1).clamp_min(1e-6)            # [chunk]
        loss_per_seed = -numerator / denominator                  # [chunk]

        valid = (w_pos.sum(dim=1) > 1e-9).float()                 # [chunk]
        total_loss_sum = total_loss_sum + (loss_per_seed * valid).sum()
        total_valid_seeds = total_valid_seeds + valid.sum()

    return total_loss_sum / total_valid_seeds.clamp_min(1.0)
