"""Loss functions.

InfoNCE contrastive alignment for walk-supervised temporal node
embeddings, with sampled-negative partition function.

alignment_loss(E, P_target, P_context, walks, ...,
               num_align_negatives)
  - InfoNCE contrastive loss over batched walks.
  - For each seed s_i with positive contexts {n_p^+ : p in walk i}:

        L_i = - (Σ_p w[i,p] · log p(n_p^+ | s_i)) / (Σ_p w[i,p])

        log p(n_p^+ | s_i) =  -‖P_target(E(s_i)) - P_context(E(n_p^+))‖² / τ
                            - log Σ_j exp(-‖P_target(E(s_i)) - P_context(E(n_j))‖² / τ)

    where j ranges over (positives of seed i) ∪ (per-seed sampled
    negatives drawn from the pool's unique-node frequency
    distribution^0.75). Word2Vec convention.
  - Hop/time weights w(K_hop, Δt) = 1/K_hop + (1 + Δt/T_train)^(-β)
    multiply each positive's log-prob. Padding and seed slots are
    masked out via a very-negative similarity (-1e9).
  - The softmax denominator does what Wang-Isola uniformity did
    (push seed projections away from non-positive contexts), but
    with task-relevant negatives sampled by frequency.

False negatives (sampled nodes that happen to be positives of the
same seed) are accepted. Per-sample bias is small (~3%) and
matches standard SimCLR/CLIP practice. No exclusion masking.

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

# Frequency-weighting exponent for negative sampling, applied to
# unique-node counts. Word2Vec convention; balances sampling
# popular and rare nodes.
_SAMPLING_EXPONENT = 0.75


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
    num_align_negatives: int = 64,
) -> torch.Tensor:
    """InfoNCE contrastive alignment over batched walks.

    Partition function runs over (per-seed positives ∪ per-seed
    sampled negatives). Negatives are drawn from the pool's unique-
    node distribution weighted by count^0.75 (Word2Vec convention).
    False-negative bias is accepted (small, ~3% per sample) — matches
    standard SimCLR/CLIP practice.

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
    is_context = positions < seed_pos                             # [NK, L]
    ctx_valid_mask = is_context.reshape(M)                        # [M]

    # Degenerate batch: no walk has any valid context position. Happens
    # on the first training batch when Tempest's graph is empty. Return
    # zero loss; no gradient signal until the graph fills.
    if not bool(ctx_valid_mask.any()):
        return torch.zeros((), device=device, requires_grad=True)

    # Hop/time weights (positive everywhere since w_hop > 0 always).
    hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()  # [NK, L]
    dt = (float(t_now) - timestamps.float()).clamp_min(0.0)       # [NK, L]
    dt_norm = dt / max(T_train, 1.0)
    w = 1.0 / hop_dist + (1.0 + dt_norm).pow(-beta)               # [NK, L]

    # ── Projections ──────────────────────────────────────────────
    # Seeds and per-walk contexts are computed via the projection
    # heads. The pool projection p_ctx_pool is then reshaped to
    # [NK, L, d_proj] so per-seed-own-walk positives can be sliced
    # by indexing — no [NK, M] sim matrix is built.
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

    d_proj = p_seed.shape[-1]
    p_ctx_own_walks = p_ctx.view(NK, L, d_proj)                   # [NK, L, d_proj]

    # ── Sim to positives: own walk only ──────────────────────────
    sq_dist_pos = ((p_seed.unsqueeze(1) - p_ctx_own_walks) ** 2).sum(dim=-1)
    sim_pos = -sq_dist_pos / tau                                  # [NK, L]
    sim_pos = sim_pos.masked_fill(~is_context, _INVALID_SIM)

    # ── Sampled negatives ────────────────────────────────────────
    # Build the unique-node frequency distribution over the VALID
    # context positions in the pool (excludes padding and seed slots).
    valid_nodes = nodes.reshape(M)[ctx_valid_mask]                # [V_total]
    unique_nodes, counts = torch.unique(valid_nodes, return_counts=True)
    sampling_weights = counts.float().pow(_SAMPLING_EXPONENT)     # [V_unique]

    # Sample NK × num_align_negatives integer indices into unique_nodes
    # via inverse-CDF sampling. torch.multinomial on CUDA fails at
    # large num_samples (kernel-launch limits); searchsorted on a
    # normalised cumulative distribution has the same distributional
    # semantics with no launch-size cap.
    cum_weights = sampling_weights.cumsum(0)                      # [V_unique]
    cum_weights = cum_weights / cum_weights[-1]                   # normalised to [0, 1]
    total_samples = NK * num_align_negatives
    u = torch.rand(total_samples, device=device)
    flat_neg_idx = torch.searchsorted(cum_weights, u).clamp(
        max=unique_nodes.shape[0] - 1,
    )                                                             # [NK * R]
    sampled_neg_node_ids = unique_nodes[flat_neg_idx].view(
        NK, num_align_negatives,
    )                                                             # [NK, R]

    # Project the sampled negatives through P_context.
    e_neg = embedding_table(sampled_neg_node_ids.reshape(-1))     # [NK·R, d_emb]
    if node_feat is not None:
        nf_neg = node_feat[sampled_neg_node_ids.reshape(-1)]
        p_neg = p_context(e_neg, node_feat=nf_neg)
    else:
        p_neg = p_context(e_neg)
    p_neg = p_neg.view(NK, num_align_negatives, d_proj)           # [NK, R, d_proj]

    # ── Sim to sampled negatives ─────────────────────────────────
    sq_dist_neg = ((p_seed.unsqueeze(1) - p_neg) ** 2).sum(dim=-1)
    sim_neg = -sq_dist_neg / tau                                  # [NK, R]

    # ── Partition function: log Z_i over positives ∪ negatives ───
    sim_combined = torch.cat([sim_pos, sim_neg], dim=1)           # [NK, L + R]
    log_Z = torch.logsumexp(sim_combined, dim=1)                  # [NK]
    log_p_pos = sim_pos - log_Z.unsqueeze(1)                      # [NK, L]

    # ── Per-seed weighted-average cross-entropy ──────────────────
    w_pos = w * is_context.float()                                # [NK, L]
    w_pos_sum = w_pos.sum(dim=1)                                  # [NK]
    numerator = (w_pos * log_p_pos).sum(dim=1)                    # [NK]
    loss_per_seed = -numerator / w_pos_sum.clamp_min(_SEED_VALID_THRESHOLD)
    valid = (w_pos_sum > _SEED_VALID_THRESHOLD).float()           # [NK]

    return (loss_per_seed * valid).sum() / valid.sum().clamp_min(1.0)
