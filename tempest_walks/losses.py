"""Loss functions.

Two pure functions, no shared state, no classes:

alignment_loss(E, P_target, P_context, walks, t_now, T_train, β, ...) -> scalar
  - For each seed s and each valid context position p of each walk:
      ℓ_{s,p} = w(K_p, Δt_p) * || P_target(E(s)) - P_context(E(n_p)) ||²
    where K_p = lens - 1 - p (hop distance to seed),
          Δt_p = t_now - t_{p+1} (timestamps[p] under Tempest's convention:
                                  edge between nodes[p] and nodes[p+1]),
          w(K, Δt) = 1/K + (1 + Δt / T_train)^(-β).
  - Reduced as weighted-mean over all valid (seed, position) pairs.
  - Padding positions and the seed position itself are masked out.
  - Edge features are NOT plumbed through this initial version
    (convention β attachment to context-side is left as future work).

uniformity_loss(E, P_target, sample_idx_a, sample_idx_b, t, ...) -> scalar
  - Wang-Isola form on L2-normalised P_target outputs:
      L = log E_{x,y ~ q⊗q} [ exp(-t ||P(E(x)) - P(E(y))||²) ]
  - Caller supplies the M independent index pairs.
  - Numerically stabilised: L = logsumexp(-t * sq_dist) - log(M).
  - Operates on P_target only; if context-projection uniformity is
    needed later, call the function a second time with P_context.

Link BCE is one line at the call site
(F.binary_cross_entropy_with_logits(...)), not factored out.
"""

import math
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
    node_feat: Optional[torch.Tensor] = None,     # [num_nodes, d_nf] or None
) -> torch.Tensor:
    """Multi-positive alignment with hop / time-weighted contexts.

    Pulls P_target(E(seed)) toward P_context(E(walk-internal)) in
    squared distance on the unit sphere, weighted by
        w(K, Δt) = 1/K + (1 + Δt / T_train)^(-β).
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

    # Hop distance K_p = lens - 1 - p. clamp_min(1) keeps 1/K finite at
    # padding rows (where is_context is False; values irrelevant after masking).
    hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()   # [NK, L]

    # Δt_p = t_now - timestamps[p]. Tempest fills the seed slot with
    # INT64_MAX and padding with -1; both produce non-positive Δt after
    # the subtraction. clamp_min(0) keeps them inert.
    dt = (float(t_now) - timestamps.float()).clamp_min(0.0)       # [NK, L]
    dt_norm = dt / max(T_train, 1.0)                              # dimensionless

    w_hop = 1.0 / hop_dist                                        # [NK, L]
    w_time = (1.0 + dt_norm).pow(-beta)                           # [NK, L]
    w = w_hop + w_time                                            # [NK, L]

    # Seeds: walks.seeds is the unique seed array; row i belongs to
    # seeds[i // K] under the K-contiguous grouping contract from walks.py.
    seed_per_row = seeds_t.repeat_interleave(K)                   # [NK]
    e_seed = embedding_table(seed_per_row)                        # [NK, d_emb]

    nodes_safe = nodes.clamp_min(0)                               # [NK, L]
    e_ctx = embedding_table(nodes_safe)                           # [NK, L, d_emb]

    if node_feat is not None:
        nf_seed = node_feat[seed_per_row]                         # [NK, d_nf]
        nf_ctx = node_feat[nodes_safe]                            # [NK, L, d_nf]
        p_seed = p_target(e_seed, node_feat=nf_seed)              # [NK, d_proj]
        p_ctx = p_context(e_ctx, node_feat=nf_ctx)                # [NK, L, d_proj]
    else:
        p_seed = p_target(e_seed)                                 # [NK, d_proj]
        p_ctx = p_context(e_ctx)                                  # [NK, L, d_proj]

    sq_dist = (p_seed.unsqueeze(1) - p_ctx).pow(2).sum(dim=-1)    # [NK, L]

    mask = is_context.float()                                     # [NK, L]
    weighted_dist = w * sq_dist * mask                            # [NK, L]

    numerator = weighted_dist.sum()
    denominator = (w * mask).sum().clamp_min(1e-6)
    return numerator / denominator


def uniformity_loss(
    embedding_table,                              # EmbeddingTable
    p_target,                                     # ProjectionHead — same as alignment's seed-side
    sample_idx_a: torch.Tensor,                   # [M] long
    sample_idx_b: torch.Tensor,                   # [M] long
    t: float = 2.0,
    node_feat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wang-Isola uniformity on L2-normalised P_target outputs.

    Estimator over M caller-supplied independent pairs:
        L = log (1/M) Σ_i exp(-t || P(E(a_i)) - P(E(b_i)) ||²)
          = logsumexp(-t * sq_dist) - log(M).
    """
    device = embedding_table.E.weight.device
    a = sample_idx_a.to(device).long()
    b = sample_idx_b.to(device).long()
    M = a.shape[0]
    if M == 0:
        return torch.tensor(0.0, device=device)

    e_a = embedding_table(a)                                      # [M, d_emb]
    e_b = embedding_table(b)                                      # [M, d_emb]

    if node_feat is not None:
        p_a = p_target(e_a, node_feat=node_feat[a])
        p_b = p_target(e_b, node_feat=node_feat[b])
    else:
        p_a = p_target(e_a)
        p_b = p_target(e_b)

    sq_dist = (p_a - p_b).pow(2).sum(dim=-1)                      # [M]
    return torch.logsumexp(-t * sq_dist, dim=0) - math.log(M)
