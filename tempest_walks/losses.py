"""Loss functions.

InfoNCE contrastive alignment for walk-supervised temporal node
embeddings.

alignment_loss(E, P_target, P_context, walks, ...)
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

Returns a standard graph-attached scalar tensor. The trainer
combines it with the BCE term and calls .backward() once.

Link BCE remains a separate term computed in the trainer using a
detached embedding (stop-grad on E for BCE).
"""

from typing import Optional

import torch

from .walks import WalkData


# Sim value for invalid (padding/seed-slot) pool entries. Chosen so
# exp(_INVALID_SIM) ≈ 0 in float32 but not -inf — prevents NaN when
# every entry in a row is invalid (e.g. a walk with zero contexts).
_INVALID_SIM = -1e9

# Threshold for treating a seed as having no valid positives. Seeds
# whose weight sum is below this are excluded from the loss mean.
# The same threshold is used in the denominator clamp for numerical
# safety, so the "valid" mask and the clamp are consistent: a seed
# excluded from the mean is also exactly the seed whose denominator
# would otherwise be artificially inflated by the clamp.
_SEED_VALID_THRESHOLD = 1e-9


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
    """InfoNCE contrastive alignment over batched walks.

    Returns the scalar mean loss over seeds-with-positives as a
    graph-attached tensor. The caller is responsible for backward.
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
    ctx_valid_mask = (positions < seed_pos).reshape(M)            # [M]

    # Positive mask: pool entry j is a positive of seed i iff
    # j // L == i (same walk) AND ctx_valid_mask[j].
    pool_walk_idx = torch.arange(M, device=device) // L           # [M]
    seed_indices = torch.arange(NK, device=device).unsqueeze(1)   # [NK, 1]
    pos_mask = (pool_walk_idx.unsqueeze(0) == seed_indices) \
               & ctx_valid_mask.unsqueeze(0)                      # [NK, M]

    # Hop/time weights (positive everywhere since w_hop > 0 always).
    hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()  # [NK, L]
    dt = (float(t_now) - timestamps.float()).clamp_min(0.0)       # [NK, L]
    dt_norm = dt / max(T_train, 1.0)
    w = 1.0 / hop_dist + (1.0 + dt_norm).pow(-beta)               # [NK, L]
    w_flat = w.reshape(M)                                         # [M]

    # Projections.
    seed_per_row = seeds_t.repeat_interleave(K)                   # [NK]
    e_seed = embedding_table(seed_per_row)                        # [NK, d_emb]
    nodes_safe = nodes.clamp_min(0)                               # [NK, L]
    e_ctx_flat = embedding_table(nodes_safe.reshape(-1))          # [M, d_emb]

    if node_feat is not None:
        nf_seed = node_feat[seed_per_row]                         # [NK, d_nf]
        nf_ctx_flat = node_feat[nodes_safe.reshape(-1)]           # [M, d_nf]
        p_seed = p_target(e_seed, node_feat=nf_seed)              # [NK, d_proj]
        p_ctx = p_context(e_ctx_flat, node_feat=nf_ctx_flat)      # [M, d_proj]
    else:
        p_seed = p_target(e_seed)                                 # [NK, d_proj]
        p_ctx = p_context(e_ctx_flat)                             # [M, d_proj]

    # Sim against the FULL pool — exact log Z_i for each seed.
    sim_dot = p_seed @ p_ctx.T                                    # [NK, M]
    sim = -(2.0 - 2.0 * sim_dot) / tau                            # [NK, M]
    sim = sim.masked_fill(~ctx_valid_mask.unsqueeze(0), _INVALID_SIM)

    log_Z = torch.logsumexp(sim, dim=1)                           # [NK]
    log_p = sim - log_Z.unsqueeze(1)                              # [NK, M]

    w_pos = w_flat.unsqueeze(0) * pos_mask.float()                # [NK, M]
    w_pos_sum = w_pos.sum(dim=1)                                  # [NK]
    numerator = (w_pos * log_p).sum(dim=1)                        # [NK]
    loss_per_seed = -numerator / w_pos_sum.clamp_min(_SEED_VALID_THRESHOLD)
    valid = (w_pos_sum > _SEED_VALID_THRESHOLD).float()           # [NK]

    return (loss_per_seed * valid).sum() / valid.sum().clamp_min(1.0)
