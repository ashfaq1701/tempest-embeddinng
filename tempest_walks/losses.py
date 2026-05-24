"""Loss functions.

One pure function, no shared state, no classes:

alignment_loss(E, P_target, P_context, walks, t_now, T_train, β, τ, ...) -> scalar
  - InfoNCE contrastive loss over batched walks.
  - For each seed s_i with positive contexts {n_p^+ : p in walk i},
    and all batch contexts collected into a flat pool of size NK*L:

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
) -> torch.Tensor:
    """InfoNCE contrastive alignment over batched walks. Returns
    scalar mean loss over seeds-with-positives.
    """
    device = embedding_table.E.weight.device
    nodes = walks.nodes.to(device).long()                         # [NK, L]
    timestamps = walks.timestamps.to(device).long()               # [NK, L]
    lens = walks.lens.to(device).long()                           # [NK]
    seeds_t = walks.seeds.to(device).long()                       # [N]
    K = walks.K
    NK, L = nodes.shape

    # Valid context positions: p ∈ [0, lens - 2]. Position lens-1 is
    # the seed (excluded), positions >= lens are padding.
    positions = torch.arange(L, device=device).unsqueeze(0)       # [1, L]
    seed_pos = (lens - 1).unsqueeze(1)                            # [NK, 1]
    is_context = positions < seed_pos                             # [NK, L]

    # Projections.
    seed_per_row = seeds_t.repeat_interleave(K)                   # [NK]
    e_seed = embedding_table(seed_per_row)                        # [NK, d_emb]
    nodes_safe = nodes.clamp_min(0)                               # [NK, L]
    e_ctx_flat = embedding_table(nodes_safe.reshape(-1))          # [NK*L, d_emb]

    if node_feat is not None:
        nf_seed = node_feat[seed_per_row]                         # [NK, d_nf]
        nf_ctx_flat = node_feat[nodes_safe.reshape(-1)]           # [NK*L, d_nf]
        p_seed = p_target(e_seed, node_feat=nf_seed)              # [NK, d_proj]
        p_ctx_flat = p_context(e_ctx_flat, node_feat=nf_ctx_flat) # [NK*L, d_proj]
    else:
        p_seed = p_target(e_seed)                                 # [NK, d_proj]
        p_ctx_flat = p_context(e_ctx_flat)                        # [NK*L, d_proj]

    # Similarity matrix: every seed vs every batch context.
    # Both p_seed and p_ctx_flat are L2-normalised, so
    # ||a - b||² = 2 - 2·a·b.
    sim_dot = p_seed @ p_ctx_flat.T                               # [NK, NK*L]
    sq_dist_full = 2.0 - 2.0 * sim_dot                            # [NK, NK*L]
    sim = -sq_dist_full / tau                                     # [NK, NK*L]

    # Mask invalid pool entries (padding + seed slots) with very
    # negative similarity so exp(sim) ≈ 0.
    ctx_valid_mask = is_context.reshape(NK * L)                   # [NK*L]
    INVALID_MASK = -1e9
    sim = sim.masked_fill(~ctx_valid_mask.unsqueeze(0), INVALID_MASK)

    # Positive mask: pool entry j is a positive of seed i iff
    # j // L == i (same walk) AND ctx_valid_mask[j] (valid context).
    pool_walk_idx = torch.arange(NK * L, device=device) // L      # [NK*L]
    seed_indices = torch.arange(NK, device=device).unsqueeze(1)   # [NK, 1]
    same_walk = pool_walk_idx.unsqueeze(0) == seed_indices        # [NK, NK*L]
    pos_mask = same_walk & ctx_valid_mask.unsqueeze(0)            # [NK, NK*L]

    # Hop/time weights for positives.
    hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()   # [NK, L]
    dt = (float(t_now) - timestamps.float()).clamp_min(0.0)       # [NK, L]
    dt_norm = dt / max(T_train, 1.0)
    w_hop = 1.0 / hop_dist                                        # [NK, L]
    w_time = (1.0 + dt_norm).pow(-beta)                           # [NK, L]
    w = w_hop + w_time                                            # [NK, L]
    w_flat = w.reshape(NK * L)                                    # [NK*L]
    w_pos = w_flat.unsqueeze(0).expand(NK, -1) * pos_mask.float() # [NK, NK*L]

    # Per-seed log-partition over all batch contexts.
    log_Z = torch.logsumexp(sim, dim=1)                           # [NK]
    log_p = sim - log_Z.unsqueeze(1)                              # [NK, NK*L]

    # Weighted mean of positive log-probs per seed.
    weighted_log_p = w_pos * log_p                                # [NK, NK*L]
    sum_weighted_log_p = weighted_log_p.sum(dim=1)                # [NK]
    sum_weights = w_pos.sum(dim=1).clamp_min(1e-6)                # [NK]
    loss_per_seed = -sum_weighted_log_p / sum_weights             # [NK]

    # Mean over seeds that have positives (cold-start walks with
    # lens<2 have zero weight and contribute zero to the average).
    has_positives = (w_pos.sum(dim=1) > 1e-9)                     # [NK]
    n_valid_seeds = has_positives.float().sum().clamp_min(1.0)
    return (loss_per_seed * has_positives.float()).sum() / n_valid_seeds
