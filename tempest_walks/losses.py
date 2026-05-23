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
  - Wang-Isola form on L2-normalised head outputs:
      L = log E_{x,y ~ q⊗q} [ exp(-t ||P(E(x)) - P(E(y))||²) ]
  - Caller supplies the M independent index pairs.
  - Numerically stabilised: L = logsumexp(-t * sq_dist) - log(M).
  - Called once per head (target, context). Caller SUMS the two
    losses (not averages): heads have disjoint parameter sets, so
    averaging would halve each head's anti-collapse gradient and
    cause collapse. Empirically verified on C1 smoke test.
  - bypass_ef=head.has_ef skips the EF branch on EF-bearing heads
    (Task 12 Option γ — zeros at the merge concat boundary, NOT
    at the ef_mlp input).

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
    ef_target_per_position: bool = False,         # Task 12 C5
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
    # Task 12: also build seed_ef under convention B-target — the
    # seed's "outgoing" edge does not exist in the walk, so we use the
    # walk's LAST edge (index lens-2), which is the edge INTO the seed
    # from its immediate context. For walks with lens < 2 (no edges),
    # zero-fill.
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
        seed_ef_idx = (lens - 2).clamp(min=0)                      # [NK]
        batch_idx = torch.arange(NK, device=device)
        seed_ef_raw = ef_dev[batch_idx, seed_ef_idx]               # [NK, d_ef]
        has_valid_edge = (lens >= 2).unsqueeze(-1).float()         # [NK, 1]
        seed_ef = seed_ef_raw * has_valid_edge                     # [NK, d_ef]
    else:
        ef_padded = None
        seed_ef = None

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

    # Task 12: p_target may or may not have an EF channel (controlled
    # by ef_on_target flag). Build kwargs per-head based on the head's
    # advertised has_ef and the loss-time availability of features.
    # C5: if ef_target_per_position is set, p_target evaluates
    # per-position using the same ef_padded that p_context sees.
    context_kwargs = {}
    if node_feat is not None:
        nf_seed = node_feat[seed_per_row]                         # [NK, d_nf]
        nf_ctx = node_feat[nodes_safe]                            # [NK, L, d_nf]
        context_kwargs['node_feat'] = nf_ctx
    if ef_padded is not None and p_context.has_ef:
        context_kwargs['edge_feat'] = ef_padded
    p_ctx = p_context(e_ctx, **context_kwargs)                    # [NK, L, d_proj]

    if ef_target_per_position and p_target.has_ef and ef_padded is not None:
        # C5: p_target evaluates per-position so target's EF at (i, p)
        # matches context's ef_padded[i, p] exactly. Same physical edge
        # on both sides; loss becomes a per-position symmetric pull.
        e_seed_pp = e_seed.unsqueeze(1).expand(NK, L, -1)         # [NK, L, d_emb]
        target_kwargs_pp = {'edge_feat': ef_padded}
        if node_feat is not None:
            target_kwargs_pp['node_feat'] = nf_seed.unsqueeze(1).expand(NK, L, -1)
        p_seed_pp = p_target(e_seed_pp, **target_kwargs_pp)       # [NK, L, d_proj]
        sq_dist = (p_seed_pp - p_ctx).pow(2).sum(dim=-1)          # [NK, L]
    else:
        target_kwargs = {}
        if node_feat is not None:
            target_kwargs['node_feat'] = nf_seed
        if seed_ef is not None and p_target.has_ef:
            target_kwargs['edge_feat'] = seed_ef                  # convention B-target
        p_seed = p_target(e_seed, **target_kwargs)                # [NK, d_proj]
        sq_dist = (p_seed.unsqueeze(1) - p_ctx).pow(2).sum(dim=-1)# [NK, L]

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

    The function is called once per head. The caller SUMS the two
    losses (target + context); averaging halved each head's
    anti-collapse gradient and caused collapse (see trainer comment).
    Pass bypass_ef=head.has_ef so heads with EF channels skip the EF
    branch during uniformity (no edge to feed). Implements Task 12
    Option γ: zeros are injected at the merge-concat boundary, not
    at the ef_mlp input.
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
