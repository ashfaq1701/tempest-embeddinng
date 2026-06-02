r"""Walks-supervised InfoNCE alignment loss.

Indices:
    i ∈ [NK]            row (walk); walks are grouped K-per-seed contiguously
    p ∈ [L]             walk position
    v ∈ U               unique node in the batch's valid-context pool

Per row, seed and walk-length:
    s_i := walks.seeds[i // K]          (seed of row i)
    ℓ_i := walks.lens[i]                (effective walk length)
    valid(i, p) := 0 ≤ p ≤ ℓ_i - 2      (seed lives at p = ℓ_i - 1)
    n_{i, p}   := walks.nodes[i, p]
    t_{i, p}   := walks.timestamps[i, p]    (= edge time for valid p)

Candidate distribution over the batch pool U
(word2vec NEG convention, α = 0.75):
    c(v)     := #{ (i, p) : valid(i, p) ∧ n_{i, p} = v }
    μ(v)     := c(v)^α / Σ_u c(u)^α                        (proposal)

Energy-based conditional with μ as base measure:
    φ(s, v)  := -‖P_t(E[s]) - P_c(E[v])‖² / τ
    Z(s)     := Σ_{v ∈ U} c(v)^α · exp(φ(s, v))
    p(v | s) := c(v)^α · exp(φ(s, v)) / Z(s)
    log p(v | s) = α · log c(v) + φ(s, v) - log Z(s)

Per-position weight (convex combination of hop and stationary recency):
    K_hop(i, p)  := max(1, ℓ_i - 1 - p)
    gap(i, p)    := max(0, t_seed_edge(i) - t_{i, p})            stationary
                    where t_seed_edge(i) = max_{valid p} t_{i, p}
                    — the timestamp of the seed's own most-recent edge,
                    so gap is elapsed time between context p and the
                    seed's interaction (not absolute calendar position).
                    Window-position-invariant: a week-1 seed and a
                    week-4 seed with the same local history get the
                    same recency profile.
    h_p          := normalize(1/K_hop)        — hop profile, sums to 1
    r_p          := normalize(exp(-gap/scale)) — recency profile, sums to 1
                    where scale = recency_scale, in raw timestamp units
                    (data-driven: median inter-arrival of train edges).
    w̃(i, p)      := (1 - γ) · h_p(i, p) + γ · r_p(i, p)
                    Convex mix; γ ∈ [0, 1]. γ=0 is hop-only, γ=1 is
                    recency-only. Per-row sum is 1 by construction.

Per-row weighted negative log-likelihood:
    W_i      := Σ_p 𝟙[valid(i, p)] · w̃(i, p)
    ℒ_i      := -(1 / W_i) · Σ_p 𝟙[valid(i, p)] · w̃(i, p) · log p(n_{i, p} | s_i)

Batch loss:
    I⁺       := { i : W_i > ε }                            (rows with positives)
    ℒ        := (1 / |I⁺|) · Σ_{i ∈ I⁺} ℒ_i

Notes:
    Positives lying in U are not removed from the partition (SimCLR/CLIP).
    The caller is responsible for calling .backward() on ℒ.
"""

from typing import Optional

import torch

from .walks import WalkData


_ALPHA = 0.75   # μ(v) ∝ c(v)^α — softens partition emphasis on popular v
_EPS = 1e-9     # W_i validity threshold and denominator clamp


def alignment_loss(
    embedding_table,
    p_target,
    p_context,
    walks: WalkData,
    recency_scale: torch.Tensor,
    gamma_recency: float = 0.4,
    tau_align: float = 0.5,
    node_feat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns ℒ as a scalar graph-attached tensor. See module docstring.

    `recency_scale` is a frozen scalar tensor (owned by the Trainer,
    not a Parameter). Its value is the recency time-constant in raw
    timestamp units, set once from `TrainStats.mean_inter_arrival`
    at Trainer-init. Previously this was a learnable parameter under
    a softplus reparameterisation, but the scalar was observed to
    collapse toward zero under longer runs without improving val MRR,
    so it's now frozen.
    """
    device = embedding_table.E.weight.device
    nodes = walks.nodes.to(device).long()                         # [NK, L]
    timestamps = walks.timestamps.to(device).long()               # [NK, L]
    lens = walks.lens.to(device).long()                           # [NK]
    seeds_t = walks.seeds.to(device).long()                       # [N]
    K = walks.K
    NK, L = nodes.shape
    M = NK * L

    # valid(i, p) := 0 ≤ p ≤ ℓ_i - 2  (seed at ℓ_i - 1, padding at p ≥ ℓ_i)
    positions = torch.arange(L, device=device).unsqueeze(0)       # [1, L]
    is_context = positions < (lens - 1).unsqueeze(1)              # [NK, L]
    ctx_valid_mask = is_context.reshape(M)                        # [M]

    # Empty batch (Tempest graph not yet populated): ℒ = 0 with grad.
    if not bool(ctx_valid_mask.any()):
        return torch.zeros((), device=device, requires_grad=True)

    # K_hop(i, p) = max(1, ℓ_i - 1 - p)
    hop_dist = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()
    mask_f = is_context.float()                                   # [NK, L]

    # Hop profile: 1/K_hop, masked, normalised per row so it sums to 1.
    hop_w = (1.0 / hop_dist) * mask_f                             # [NK, L]
    hop_p = hop_w / hop_w.sum(dim=1, keepdim=True).clamp_min(_EPS)

    # Stationary recency: gap = t_seed_edge - t_{i, p}, ≥ 0.
    # t_seed_edge = max of valid timestamps in the row. Masking
    # invalid positions to -inf before .max() ensures sentinels
    # (INT64_MAX at seed slot, -1 padding) cannot win the max.
    ts_f = timestamps.float()
    ts_masked = ts_f.masked_fill(~is_context, float("-inf"))
    t_seed_edge = ts_masked.max(dim=1, keepdim=True).values       # [NK, 1]
    # clamp_min(0) absorbs every sentinel: invalid rows get -inf
    # subtracted to -inf, then clamped to 0; mask zeros their weight.
    gap = (t_seed_edge - ts_f).clamp_min(0.0)                     # [NK, L]
    # `recency_scale` is a frozen scalar (mean inter-arrival of the
    # train split). clamp_min(1.0) is a numerical guard; for any real
    # dataset the configured value sits well above the floor.
    scale = recency_scale.clamp_min(1.0)
    recency_w = torch.exp(-gap / scale) * mask_f                  # [NK, L]
    rec_p = recency_w / recency_w.sum(dim=1, keepdim=True).clamp_min(_EPS)

    # Convex combination: per-row profile, sums to 1 by construction.
    w_tilde = (1.0 - gamma_recency) * hop_p + gamma_recency * rec_p

    # U   = unique({ n_{i, p} : valid(i, p) })
    # c   = ( c(v) )_{v ∈ U}
    nodes_safe = nodes.clamp_min(0)
    valid_nodes = nodes_safe.reshape(M)[ctx_valid_mask]
    unique_nodes, inverse_idx_valid, counts = torch.unique(
        valid_nodes, return_inverse=True, return_counts=True,
    )                                                              # [V]

    # log_c_pool[v] = α · log c(v)
    log_c_pool = _ALPHA * torch.log(counts.float().clamp_min(1.0))  # [V]

    # pool_idx(i, p) = index in U of n_{i, p}  (arbitrary at invalid p; masked)
    pool_idx_flat = torch.zeros(M, dtype=torch.long, device=device)
    pool_idx_flat[ctx_valid_mask] = inverse_idx_valid
    pool_idx = pool_idx_flat.view(NK, L)                          # [NK, L]

    # E[s_i], E[v]  (embedding lookups)
    seed_per_row = seeds_t.repeat_interleave(K)                   # [NK]
    e_seed = embedding_table(seed_per_row)                        # [NK, d]
    e_pool = embedding_table(unique_nodes)                        # [V, d]

    # a_i := P_t(E[s_i]),   b_v := P_c(E[v])    (unit-norm on the sphere)
    if node_feat is not None:
        nf_seed = node_feat[seed_per_row]
        nf_pool = node_feat[unique_nodes]
        a = p_target(e_seed, node_feat=nf_seed)                   # [NK, d]
        b = p_context(e_pool, node_feat=nf_pool)                  # [V, d]
    else:
        a = p_target(e_seed)                                      # [NK, d]
        b = p_context(e_pool)                                     # [V, d]

    # φ(s_i, v) = -‖a_i - b_v‖² / τ,  with ‖a-b‖² = ‖a‖² + ‖b‖² - 2 ⟨a, b⟩
    sq_a = (a * a).sum(dim=-1, keepdim=True)                      # [NK, 1]
    sq_b = (b * b).sum(dim=-1).unsqueeze(0)                       # [1, V]
    inner = a @ b.t()                                             # [NK, V]
    phi_pool = -(sq_a + sq_b - 2.0 * inner).clamp_min(0.0) / tau_align

    # log Z(s_i) = logsumexp_{v ∈ U} ( α log c(v) + φ(s_i, v) )
    log_Z = torch.logsumexp(
        phi_pool + log_c_pool.unsqueeze(0), dim=1,
    )                                                              # [NK]

    # φ(s_i, n_{i, p})  and  α · log c(n_{i, p})
    phi_pos = phi_pool.gather(dim=1, index=pool_idx)              # [NK, L]
    log_c_pos = log_c_pool[pool_idx]                              # [NK, L]

    # log p(n_{i, p} | s_i) = α · log c(n_{i, p}) + φ(s_i, n_{i, p}) - log Z(s_i)
    log_p_pos = log_c_pos + phi_pos - log_Z.unsqueeze(1)          # [NK, L]

    # w_valid(i, p) = 𝟙[valid(i, p)] · w̃(i, p);   W_i = Σ_p w_valid(i, p)
    w_valid = w_tilde * is_context.float()                        # [NK, L]
    W = w_valid.sum(dim=1)                                        # [NK]

    # ℒ_i = -(1 / W_i) · Σ_p w_valid(i, p) · log p(n_{i, p} | s_i)
    L_per_row = -(w_valid * log_p_pos).sum(dim=1) / W.clamp_min(_EPS)

    # ℒ = (1 / |I⁺|) · Σ_{i ∈ I⁺} ℒ_i,    I⁺ = { i : W_i > ε }
    valid = (W > _EPS).float()                                    # [NK]
    return (L_per_row * valid).sum() / valid.sum().clamp_min(1.0)
