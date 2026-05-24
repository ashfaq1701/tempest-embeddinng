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
  - Edge features (when present) are passed to p_context under
    convention β: walks.edge_feats[p] is the edge OUT of context p.
    p_target never sees edge features.

uniformity_loss(E, head, sample_idx_a, sample_idx_b, t, ..., bypass_ef) -> scalar
  - Wang-Isola form on L2-normalised head outputs.
  - Caller supplies the M independent index pairs.
  - Numerically stabilised: L = logsumexp(-t * sq_dist) - log(M).
  - Called once per head (target, context); trainer SUMS them.
  - bypass_ef=head.has_ef skips the EF branch on EF-bearing heads.

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
    edge_feat: Optional[torch.Tensor] = None,     # [NK, L-1, d_ef] or None
    ef_weight_mod=None,                           # Task 14a: EFWeightModulator or None
) -> torch.Tensor:
    """Multi-positive alignment with hop / time-weighted contexts.

    Pulls P_target(E(seed)) toward P_context(E(walk-internal)) in
    squared distance on the unit sphere, weighted by
        w(K, Δt) = 1/K + (1 + Δt / T_train)^(-β).

    edge_feat: Per-walk edge features attached to context positions
        under convention β (edge OUT of context p lives at index p).
        Shape [NK, L_max-1, d_edge_feat] from Tempest. Pass None
        only when the dataset has no edge features. The function
        pads with one zero row so the projection sees [NK, L, d_ef];
        the appended slot lines up with the seed position (lens-1),
        which is masked out of the loss anyway.
    """
    device = embedding_table.E.weight.device
    nodes = walks.nodes.to(device).long()                         # [NK, L]
    timestamps = walks.timestamps.to(device).long()               # [NK, L]
    lens = walks.lens.to(device).long()                           # [NK]
    seeds_t = walks.seeds.to(device).long()                       # [N]
    K = walks.K
    NK, L = nodes.shape

    # Pad edge features to align with e_ctx's [NK, L, d_edge_feat] shape.
    # Tempest stores ef[p] for the edge between nodes[p] and nodes[p+1]
    # in slots [0, L-1); position L-1 (seed slot) is the appended zero
    # row and is masked out of the loss.
    if edge_feat is not None:
        ef_dev = edge_feat.to(device)
        assert ef_dev.shape[:2] == (NK, L - 1), (
            f"edge_feat shape {tuple(ef_dev.shape)} doesn't match walks of "
            f"shape {(NK, L)} (expected [NK, L-1, d_ef])"
        )
        d_ef = ef_dev.shape[-1]
        ef_padded = torch.cat(
            [
                ef_dev,
                torch.zeros(NK, 1, d_ef, device=device, dtype=ef_dev.dtype),
            ],
            dim=1,
        )                                                          # [NK, L, d_ef]
    else:
        ef_padded = None

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

    # Task 14a: optional EF modulation of the alignment weight.
    # mod ∈ (-1, +1) via tanh; (1 + mod) gives a ±100% scale on
    # the base weight per (seed, context) position. EF is read from
    # ef_padded — same per-position EF that p_context would see in
    # the original Task 12 master architecture. Position L-1 has
    # zero EF padding → tanh(linear(0)) ≈ 0 (modulo bias), but
    # that position is masked out of the loss anyway.
    if ef_weight_mod is not None and ef_padded is not None:
        mod = ef_weight_mod(ef_padded)                            # [NK, L]
        w = w * (1.0 + mod)

    # Seeds: walks.seeds is the unique seed array; row i belongs to
    # seeds[i // K] under the K-contiguous grouping contract from walks.py.
    seed_per_row = seeds_t.repeat_interleave(K)                   # [NK]
    e_seed = embedding_table(seed_per_row)                        # [NK, d_emb]

    nodes_safe = nodes.clamp_min(0)                               # [NK, L]
    e_ctx = embedding_table(nodes_safe)                           # [NK, L, d_emb]

    # Build per-head kwargs. p_target never takes edge_feat (seeds
    # have no edge channel under convention β). p_context takes
    # edge_feat ONLY if its has_ef is True (Task 14 decouples
    # ef_in_p_context from d_edge_feat being present).
    target_kwargs = {}
    context_kwargs = {}
    if node_feat is not None:
        nf_seed = node_feat[seed_per_row]                         # [NK, d_nf]
        nf_ctx = node_feat[nodes_safe]                            # [NK, L, d_nf]
        target_kwargs['node_feat'] = nf_seed
        context_kwargs['node_feat'] = nf_ctx
    if ef_padded is not None and p_context.has_ef:
        context_kwargs['edge_feat'] = ef_padded
    p_seed = p_target(e_seed, **target_kwargs)                    # [NK, d_proj]
    p_ctx = p_context(e_ctx, **context_kwargs)                    # [NK, L, d_proj]

    sq_dist = (p_seed.unsqueeze(1) - p_ctx).pow(2).sum(dim=-1)    # [NK, L]

    mask = is_context.float()                                     # [NK, L]
    weighted_dist = w * sq_dist * mask                            # [NK, L]

    numerator = weighted_dist.sum()
    denominator = (w * mask).sum().clamp_min(1e-6)
    return numerator / denominator


def uniformity_loss(
    embedding_table,                              # EmbeddingTable
    head,                                         # ProjectionHead (target or context)
    sample_idx_a: torch.Tensor,                   # [M] long
    sample_idx_b: torch.Tensor,                   # [M] long
    t: float = 2.0,
    node_feat: Optional[torch.Tensor] = None,
    bypass_ef: bool = False,
) -> torch.Tensor:
    """Wang-Isola uniformity on L2-normalised projection outputs.

    Estimator over M caller-supplied independent pairs:
        L = log (1/M) Σ_i exp(-t || P(E(a_i)) - P(E(b_i)) ||²)
          = logsumexp(-t * sq_dist) - log(M).

    Called once per head. The caller SUMS the two losses (target +
    context). Pass bypass_ef=head.has_ef so heads with EF channels
    skip the EF branch during uniformity (Task 12 Option γ).
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
        p_a = head(e_a, node_feat=node_feat[a], bypass_ef=bypass_ef)
        p_b = head(e_b, node_feat=node_feat[b], bypass_ef=bypass_ef)
    else:
        p_a = head(e_a, bypass_ef=bypass_ef)
        p_b = head(e_b, bypass_ef=bypass_ef)

    sq_dist = (p_a - p_b).pow(2).sum(dim=-1)                      # [M]
    return torch.logsumexp(-t * sq_dist, dim=0) - math.log(M)
