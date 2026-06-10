r"""Source-alignment loss (per-source InfoNCE over designated negatives).

Each walk seed is a SOURCE node `u`. The loss pulls `E[u]` toward `u`'s
backward-walk context nodes (its recent predecessors), weighted by a
hop/recency profile, and pushes it away from a per-source DESIGNATED
negative set `N_u` (historical reservoir + uniform mix, drawn by the
Trainer). The partition is over `N_u` ALONE — there is no shared batch
pool and no count^0.75 proposal correction: the sampler IS the prior.

Indices:
    i ∈ [NK]            row (walk); rows are grouped K-per-seed contiguously
    p ∈ [L]             walk position
    k ∈ [n_neg]         designated negative of the row's seed

Per row / seed / length:
    s_i := walks.seeds[i // K]              (seed = source of row i)
    ℓ_i := walks.lens[i]
    valid(i, p):  BACKWARD → 0 ≤ p ≤ ℓ_i − 2   (seed at p = ℓ_i − 1)
                  FORWARD  → 1 ≤ p ≤ ℓ_i − 1   (seed at p = 0)
    n_{i,p} := walks.nodes[i, p]            (a positive context node)
    N_{s_i} := neg_nodes[i // K]            (the seed's designated negatives)

Energy on the unit sphere (E rows are unit by manifold construction, so
‖E_u − E_v‖² = 2 − 2⟨E_u, E_v⟩; the additive constant cancels in every
per-seed softmax). Logit:

    ψ(u, v) := (2 / τ) · ⟨E_u, E_v⟩            ∈ [−2/τ, +2/τ]

Per-seed negative log-partition (computed once per row from N_{s_i}):

    Z⁻_i := logsumexp_k ψ(s_i, N_{s_i,k})

Per-positive term (positive in the numerator, the seed's negatives in the
denominator — 1-positive InfoNCE):

    ℓ_{i,p} := softplus( Z⁻_i − ψ(s_i, n_{i,p}) )
             = log( 1 + exp(Z⁻_i − ψ(s_i, n_{i,p})) )
             = −log p(n_{i,p} | {n_{i,p}} ∪ N_{s_i})

No dedup against positives: a node that is both a positive context and a
designated negative contributes opposing gradients on the same row, which
is permitted (the contradiction is small and the speed is worth it).

Per-position weight (convex mix of hop and stationary recency, unchanged):
    K_hop(i, p) := max(1, ℓ_i − 1 − p)               (backward)
    gap(i, p)   := |t_seed_edge(i) − t_{i, p}|        (stationary anchor)
    h_p         := normalize(1 / K_hop)               sums to 1 over valid p
    r_p         := normalize(exp(−gap / scale))       sums to 1 over valid p
    w̃(i, p)     := (1 − γ) · h_p + γ · r_p

Per-row and batch reduction:
    W_i := Σ_p 𝟙[valid] · w̃(i, p)
    ℒ_i := (1 / W_i) · Σ_p 𝟙[valid] · w̃(i, p) · ℓ_{i,p}
    I⁺  := { i : W_i > ε }
    ℒ   := (Σ_{i∈I⁺} β_i ℒ_i) / (Σ_{i∈I⁺} β_i)        (β = seed_weights, def 1)

ψ uses the embedding rows DIRECTLY (no projection MLP); gradients flow
straight to `embedding_table.E.weight`. The caller owns `.backward()`.
"""

import torch
import torch.nn.functional as F

from .walks import WalkData


_EPS = 1e-9     # W_i validity threshold and denominator clamp


def source_alignment_loss(
    embedding_table,
    walks: WalkData,
    neg_nodes: torch.Tensor,
    recency_scale: float,
    gamma_recency: float = 0.4,
    tau_align: float = 0.5,
    direction: str = "backward",
    seed_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Returns ℒ as a scalar graph-attached tensor. See module docstring.

    `neg_nodes` is the per-seed designated negative set, shape
    `[N_seeds, n_neg]` (int), ALIGNED ROW-FOR-ROW to `walks.seeds`
    (row j holds the negatives for `walks.seeds[j]`). The Trainer draws
    it (historical reservoir + uniform) and shares the SAME array with
    the link-prediction candidate matrix.

    `recency_scale` is a frozen Python float (the train split's mean
    inter-arrival, owned by the Trainer).

    The partition is O(n_neg) per row — a logsumexp over the designated
    negatives only — so there is no pool chunking or gradient
    checkpointing here (unlike the old batch-pool InfoNCE).
    """
    device = embedding_table.E.weight.device
    nodes = walks.nodes.to(device).long()                         # [NK, L]
    timestamps = walks.timestamps.to(device).long()               # [NK, L]
    lens = walks.lens.to(device).long()                           # [NK]
    seeds_t = walks.seeds.to(device).long()                       # [N]
    K = walks.K
    NK, L = nodes.shape

    # Direction-aware valid-context mask.
    #   BACKWARD: seed at p = ℓ_i − 1, contexts p ∈ [0, ℓ_i − 2]
    #   FORWARD : seed at p = 0,       contexts p ∈ [1, ℓ_i − 1]
    positions = torch.arange(L, device=device).unsqueeze(0)       # [1, L]
    if direction == "backward":
        is_context = positions < (lens - 1).unsqueeze(1)          # [NK, L]
    elif direction == "forward":
        is_context = (positions >= 1) & (positions < lens.unsqueeze(1))
    else:
        raise ValueError(
            f"direction must be 'backward' or 'forward', got {direction!r}")

    # Empty batch (Tempest graph not yet populated): ℒ = 0 with grad.
    if not bool(is_context.any()):
        return torch.zeros((), device=device, requires_grad=True)

    mask_f = is_context.float()                                   # [NK, L]

    # ── Per-position weight w̃ = (1−γ)·hop + γ·recency ────────────────
    # Hop profile: 1/K_hop, masked, normalised per row to sum 1.
    if direction == "backward":
        hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()
    else:
        hop_dist = positions.expand(NK, L).clamp_min(1).float()
    hop_w = (1.0 / hop_dist) * mask_f
    hop_p = hop_w / hop_w.sum(dim=1, keepdim=True).clamp_min(_EPS)

    # Stationary recency: gap anchored at the SEED-ADJACENT edge time
    # (latest valid t for backward, first valid t for forward), so a
    # week-1 seed and a week-4 seed with the same local history get the
    # same recency profile. Sentinels on the wrong side are masked to
    # ±inf so they never win the min/max.
    ts_f = timestamps.float()
    if direction == "backward":
        t_seed_edge = ts_f.masked_fill(~is_context, float("-inf")).max(
            dim=1, keepdim=True).values
    else:
        t_seed_edge = ts_f.masked_fill(~is_context, float("inf")).min(
            dim=1, keepdim=True).values
    gap = (ts_f - t_seed_edge).abs().clamp_min(0.0)
    scale = max(recency_scale, 1.0)
    rec_w = torch.exp(-gap / scale) * mask_f
    rec_p = rec_w / rec_w.sum(dim=1, keepdim=True).clamp_min(_EPS)

    w_tilde = (1.0 - gamma_recency) * hop_p + gamma_recency * rec_p
    w_valid = w_tilde * mask_f                                    # [NK, L]
    W = w_valid.sum(dim=1)                                        # [NK]

    # ── Embedding lookups (direct on E; no projection) ───────────────
    inv_tau2 = 2.0 / tau_align
    seed_per_row = seeds_t.repeat_interleave(K)                   # [NK]
    a = embedding_table(seed_per_row)                            # [NK, d]

    # Positives: ψ_pos[i, p] = (2/τ)·⟨E[s_i], E[n_{i,p}]⟩. Invalid
    # positions read E[0] but are masked out by w_valid downstream.
    ctx_nodes = nodes.clamp_min(0)                               # [NK, L]
    b_ctx = embedding_table(ctx_nodes.reshape(-1)).view(NK, L, -1)  # [NK, L, d]
    psi_pos = inv_tau2 * (a.unsqueeze(1) * b_ctx).sum(dim=-1)    # [NK, L]

    # Negatives: the partition depends only on the seed, so compute it
    # once per seed ([N, n_neg]) and broadcast to the seed's K rows —
    # avoids the K×-duplicated [NK, n_neg, d] negative-embedding tensor.
    neg_nodes = neg_nodes.to(device).long().clamp_min(0)         # [N, n_neg]
    n_neg = neg_nodes.shape[1]
    a_seed = embedding_table(seeds_t)                            # [N, d]
    b_neg = embedding_table(neg_nodes.reshape(-1)).view(
        seeds_t.shape[0], n_neg, -1)                             # [N, n_neg, d]
    psi_neg = inv_tau2 * torch.einsum("nd,nkd->nk", a_seed, b_neg)  # [N, n_neg]
    Z_neg = torch.logsumexp(psi_neg, dim=1).repeat_interleave(K)  # [NK]

    # ℓ_{i,p} = softplus(Z⁻_i − ψ_pos[i,p]); positive sits in its own
    # denominator (1-positive InfoNCE over {pos} ∪ N_seed).
    ell = F.softplus(Z_neg.unsqueeze(1) - psi_pos)              # [NK, L]

    # Per-row weighted mean over valid positions, then batch reduction.
    L_per_row = (w_valid * ell).sum(dim=1) / W.clamp_min(_EPS)   # [NK]
    valid = (W > _EPS).float()                                  # [NK]
    if seed_weights is None:
        row_w = valid
    else:
        row_w = seed_weights.to(device).float().repeat_interleave(K) * valid
    return (L_per_row * row_w).sum() / row_w.sum().clamp_min(_EPS)
