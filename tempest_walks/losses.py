"""Loss functions.

InfoNCE contrastive alignment for walk-supervised temporal node
embeddings, with FULL-pool partition function over the unique
nodes appearing in the batch's walks.

alignment_loss(E, P_target, P_context, walks, ...)
  - InfoNCE contrastive loss over batched walks.
  - For each seed s_i with positive contexts {n_p^+ : p in walk i}:

        L_i = - (Σ_p w[i,p] · log p(n_p^+ | s_i)) / (Σ_p w[i,p])

        log p(n_p^+ | s_i) =  -‖P_target(E(s_i)) - P_context(E(n_p^+))‖² / tau_align
                            - log Σ_{n ∈ pool} count(n) · exp(-‖P_target(E(s_i)) - P_context(E(n))‖² / tau_align)

    The pool is the set of UNIQUE node ids appearing at any valid
    context position across the batch (excludes padding sentinels
    and the seed slot at lens-1). Each pool entry is weighted by
    its in-batch occurrence count (linear in frequency — closed-
    form equivalent of multinomial sampling under the same
    distribution, with zero sampling variance). False-negative
    bias from the seed's own positives lying in the pool is
    accepted (standard SimCLR/CLIP convention).

  - Hop/time weights w(K_hop, t_edge) = 1/K_hop + \\tilde t_e ** β
    where \\tilde t_e = (t_edge - t_min) / T_train ∈ [0, 1] is the
    edge's absolute position in the training span. The weight is
    FIXED per edge — the same (seed, context) pair gets the same
    gradient weight in every batch (no t_now drift).

Returns a standard graph-attached scalar tensor. The trainer
combines it with the per-query ranking link loss and calls
.backward() once.
"""

from typing import Optional

import torch

from .walks import WalkData


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
    t_min: int,
    T_train: float,
    beta: float = 1.0,
    tau_align: float = 0.5,
    node_feat: Optional[torch.Tensor] = None,     # [num_nodes, d_nf] or None
) -> torch.Tensor:
    """InfoNCE contrastive alignment over batched walks with a
    full unique-batch-node partition.

    The partition function for each seed runs over every unique node
    id that appears at any valid (non-padding, non-seed-slot) walk
    position in the batch. Each pool node is projected once via
    P_context; the [NK, V_unique] sim matrix is then materialised
    via the polar-decomposition identity
        ‖a - b‖² = ‖a‖² + ‖b‖² - 2⟨a, b⟩
    to avoid the [NK, V_unique, d_proj] broadcast intermediate.

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
    M = NK * L

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

    # Hop/time weights. Time component is a FIXED per-edge weight
    # tied to the edge's absolute position in the training span,
    # so the same (seed, context) pair gets the same gradient
    # weight every time it's drawn (no t_now drift across batches).
    # The clamp_min(0) covers padding (-1) and clamp_max(1) covers
    # the seed-slot INT64_MAX sentinel; both positions are masked
    # out downstream via w_pos = w * is_context.
    hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()  # [NK, L]
    edge_t_norm = (
        (timestamps - t_min).clamp_min(0).float() / max(T_train, 1.0)
    ).clamp(max=1.0)                                              # [NK, L]
    w_time = edge_t_norm.pow(beta)
    w = 1.0 / hop_dist + w_time                                   # [NK, L]

    # ── Unique node pool from VALID context positions ────────────
    # Excludes padding (-1, clamped to 0 above) and the seed slot.
    # Each unique node is projected through P_context exactly once;
    # the sim matrix scores every seed against every pool node.
    # Counts of in-batch occurrences weight the partition linearly
    # (closed-form equivalent of multinomial sampling under the
    # count distribution).
    nodes_safe = nodes.clamp_min(0)                               # [NK, L]
    valid_nodes = nodes_safe.reshape(M)[ctx_valid_mask]           # [V_total]
    unique_nodes, inverse_idx_valid, counts = torch.unique(
        valid_nodes, return_inverse=True, return_counts=True,
    )                                                              # [V_unique]
    log_w_pool = torch.log(counts.float().clamp_min(1.0))         # [V_unique]

    # pool_idx[i, p] = column index into unique_nodes for the node
    # at walk position (i, p). Invalid positions get 0 here but are
    # masked out of the loss via w_pos.
    pool_idx_flat = torch.zeros(M, dtype=torch.long, device=device)
    pool_idx_flat[ctx_valid_mask] = inverse_idx_valid
    pool_idx = pool_idx_flat.view(NK, L)                          # [NK, L]

    # ── Projections ──────────────────────────────────────────────
    seed_per_row = seeds_t.repeat_interleave(K)                   # [NK]
    e_seed = embedding_table(seed_per_row)                        # [NK, d_emb]
    e_pool = embedding_table(unique_nodes)                        # [V_unique, d_emb]

    if node_feat is not None:
        nf_seed = node_feat[seed_per_row]
        nf_pool = node_feat[unique_nodes]
        p_seed = p_target(e_seed, node_feat=nf_seed)              # [NK, d_proj]
        p_pool = p_context(e_pool, node_feat=nf_pool)             # [V_unique, d_proj]
    else:
        p_seed = p_target(e_seed)                                 # [NK, d_proj]
        p_pool = p_context(e_pool)                                # [V_unique, d_proj]

    # ── Sim matrix seed × pool, polar-decomposition form ─────────
    # ‖p_seed - p_pool‖² = ‖p_seed‖² + ‖p_pool‖² - 2 ⟨p_seed, p_pool⟩
    # Materialises [NK, V_unique] but never the [NK, V_unique, d_proj]
    # broadcast intermediate.
    sq_seed = (p_seed * p_seed).sum(dim=-1, keepdim=True)         # [NK, 1]
    sq_pool = (p_pool * p_pool).sum(dim=-1).unsqueeze(0)          # [1, V_unique]
    inner = p_seed @ p_pool.t()                                   # [NK, V_unique]
    sq_dist = (sq_seed + sq_pool - 2.0 * inner).clamp_min(0.0)
    sim_pool = -sq_dist / tau_align                               # [NK, V_unique]

    # ── Partition: log Z_i over the full pool, count-weighted ───
    # Adds log(count_j) to each per-node sim before logsumexp.
    # Numerator is left UNWEIGHTED (raw sim_pool gathered for the
    # positive) — the positive is already counted once per occurrence
    # in the loss via w_pos, no need to double-amplify it via the
    # partition weight too.
    log_Z = torch.logsumexp(
        sim_pool + log_w_pool.unsqueeze(0), dim=1,
    )                                                              # [NK]

    # ── Per-position positive sim, gathered from sim_pool ───────
    # For each (seed i, walk position p), the positive's sim is
    # sim_pool[i, pool_idx[i, p]]. Invalid positions get pool_idx=0,
    # whose sim is real but is zero-weighted by w_pos below.
    sim_pos = sim_pool.gather(dim=1, index=pool_idx)              # [NK, L]
    log_p_pos = sim_pos - log_Z.unsqueeze(1)                      # [NK, L]

    # ── Per-seed weighted-average cross-entropy ──────────────────
    w_pos = w * is_context.float()                                # [NK, L]
    w_pos_sum = w_pos.sum(dim=1)                                  # [NK]
    numerator = (w_pos * log_p_pos).sum(dim=1)                    # [NK]
    loss_per_seed = -numerator / w_pos_sum.clamp_min(_SEED_VALID_THRESHOLD)
    valid = (w_pos_sum > _SEED_VALID_THRESHOLD).float()           # [NK]

    return (loss_per_seed * valid).sum() / valid.sum().clamp_min(1.0)
