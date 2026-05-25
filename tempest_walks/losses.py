"""Loss functions.

InfoNCE contrastive alignment for walk-supervised temporal node
embeddings.

alignment_loss(E, P_target, P_context, walks, ..., chunk_size)
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

API contract — this function performs backward INTERNALLY when
called with grad enabled and returns a DETACHED scalar suitable
for logging only. Callers MUST NOT call .backward() on the return
value (it has no graph).

Why internal backward — chunking memory model.
  Without internal backward, accumulating `total_loss_sum` across
  chunks pins every chunk's autograd graph until the outer
  .backward() runs. Peak memory then scales with NK·M regardless of
  chunk_size — the chunking flag is a no-op for memory.

  With per-chunk backward inside the loop, each chunk's autograd
  graph is freed by Python refcounting between iterations once its
  .backward() completes. The shared upstream graph (p_target(e_seed)
  and p_context(e_ctx_flat)) is kept alive across chunks via
  retain_graph=True for chunks 0..N-2; the last chunk uses the
  default retain_graph=False to free everything at the end.

  Peak memory becomes:
      fixed (model + Adam + retained projection graphs)
    + max over chunks of (one chunk's sim/log_p/w_pos intermediates)
  which is bounded by chunk_size and lets memory actually scale
  with the knob.

No-grad path. If called under torch.no_grad() or with grad disabled,
returns the loss value as a normal tensor (no backward). The
trainer's eval path doesn't currently invoke alignment_loss; this
is here for diagnostic flexibility.

Link BCE remains a separate term computed in the trainer using a
detached embedding. The trainer calls l_bce.backward() after
alignment_loss returns; the two backward calls share the optimizer's
.grad accumulators on disjoint subsets of parameters.
"""

from typing import Optional, Tuple

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
    chunk_size: int = 0,                          # 0 = no chunking (single chunk)
) -> torch.Tensor:
    """InfoNCE contrastive alignment over batched walks.

    With grad enabled (training path): performs per-chunk forward +
    backward internally and returns a detached scalar (loss value).
    Gradients have been accumulated on embedding_table, p_target,
    and p_context parameters.

    With grad disabled (eval / diagnostic path): performs single-pass
    forward and returns a regular tensor (no backward).

    chunk_size > 0 caps the per-chunk pool slice size — peak memory
    becomes proportional to chunk_size, not to NK. chunk_size <= 0
    or >= NK gives a single chunk (no memory bounding).
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

    # Positive-mask helper: pool entry j is a positive of seed i iff
    # j // L == i AND ctx_valid_mask[j]. Used per chunk; the mask
    # itself is built inside the chunk loop sized [chunk, M].
    pool_walk_idx = torch.arange(M, device=device) // L           # [M]

    # Hop/time weights (positive everywhere since w_hop > 0 always).
    hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()  # [NK, L]
    dt = (float(t_now) - timestamps.float()).clamp_min(0.0)       # [NK, L]
    dt_norm = dt / max(T_train, 1.0)
    w = 1.0 / hop_dist + (1.0 + dt_norm).pow(-beta)               # [NK, L]
    w_flat = w.reshape(M)                                         # [M]

    # Chunk boundaries.
    if chunk_size <= 0 or chunk_size >= NK:
        chunk_bounds = [(0, NK)]
    else:
        chunk_bounds = [
            (s, min(s + chunk_size, NK))
            for s in range(0, NK, chunk_size)
        ]

    # Shared upstream projection graph. Built once; with retain_graph=True
    # on chunks 0..N-2, this graph stays alive across all chunks' backwards.
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

    # ── No-backward path: forward only, single-pass accumulate ───
    if not torch.is_grad_enabled():
        total_loss_sum = torch.zeros((), device=device)
        total_valid_seeds = torch.zeros((), device=device)
        for start, end in chunk_bounds:
            chunk_sum, chunk_valid = _compute_chunk(
                p_seed_all[start:end], p_ctx_pool, ctx_valid_mask,
                pool_walk_idx, w_flat, start, end, tau,
            )
            total_loss_sum = total_loss_sum + chunk_sum
            total_valid_seeds = total_valid_seeds + chunk_valid
        return total_loss_sum / total_valid_seeds.clamp_min(1.0)

    # ── Backward path: per-chunk forward + backward ──────────────
    # n_valid_total is the divisor for chunk_mean. Since w > 0
    # everywhere, "seed has a non-zero positive weight sum" is
    # equivalent to "walk has at least one valid context position",
    # which is exactly (lens >= 2). Single tensor op, no chunking
    # needed — avoids the two-pass anti-pattern.
    n_valid_total = int((lens >= 2).sum())
    if n_valid_total == 0:
        return torch.zeros((), device=device)

    last_idx = len(chunk_bounds) - 1
    total_loss_value = 0.0

    for i, (start, end) in enumerate(chunk_bounds):
        chunk_sum, _ = _compute_chunk(
            p_seed_all[start:end], p_ctx_pool, ctx_valid_mask,
            pool_walk_idx, w_flat, start, end, tau,
        )
        chunk_mean = chunk_sum / n_valid_total

        # retain_graph=True for all but the last chunk: keeps the
        # shared upstream projection saved tensors alive so the next
        # chunk can backward through them too. The chunk-local
        # intermediates released between iterations by refcounting.
        chunk_mean.backward(retain_graph=(i < last_idx))
        total_loss_value += float(chunk_mean.detach())

    return torch.tensor(total_loss_value, device=device)


def _compute_chunk(
    p_seed_chunk: torch.Tensor,                   # [chunk, d_proj]
    p_ctx_pool: torch.Tensor,                     # [M, d_proj]
    ctx_valid_mask: torch.Tensor,                 # [M] bool
    pool_walk_idx: torch.Tensor,                  # [M] long
    w_flat: torch.Tensor,                         # [M] float
    start: int,
    end: int,
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One chunk's forward pass.

    Returns:
        chunk_sum: scalar sum of per-seed losses in this chunk
                   (un-normalised; caller divides by n_valid_total).
        chunk_valid: scalar count of seeds with valid positives in
                     this chunk. Only used by the no-grad path; the
                     backward path computes n_valid_total separately.
    """
    device = p_seed_chunk.device

    # Sim against the FULL pool — exact log Z_i for each seed.
    sim_dot = p_seed_chunk @ p_ctx_pool.T                         # [chunk, M]
    sim = -(2.0 - 2.0 * sim_dot) / tau                            # [chunk, M]
    sim = sim.masked_fill(~ctx_valid_mask.unsqueeze(0), _INVALID_SIM)

    log_Z = torch.logsumexp(sim, dim=1)                           # [chunk]
    log_p = sim - log_Z.unsqueeze(1)                              # [chunk, M]

    seed_indices_chunk = torch.arange(start, end, device=device).unsqueeze(1)
    pos_mask_chunk = (pool_walk_idx.unsqueeze(0) == seed_indices_chunk) \
                     & ctx_valid_mask.unsqueeze(0)                # [chunk, M]
    w_pos = w_flat.unsqueeze(0) * pos_mask_chunk.float()          # [chunk, M]

    w_pos_sum = w_pos.sum(dim=1)                                  # [chunk]
    numerator = (w_pos * log_p).sum(dim=1)                        # [chunk]
    loss_per_seed = -numerator / w_pos_sum.clamp_min(_SEED_VALID_THRESHOLD)
    valid = (w_pos_sum > _SEED_VALID_THRESHOLD).float()           # [chunk]

    return (loss_per_seed * valid).sum(), valid.sum()
