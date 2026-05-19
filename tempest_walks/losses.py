"""Alignment + uniformity (Wang & Isola 2020 pattern) + link BCE.

Plus the v2.3 §4.7 loss-family search primaries (InfoNCE, Triplet,
SGNS) and the §4.4 norm-brake regularizer.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .walks import WalkData


def alignment_loss(
    e_target_seed: torch.Tensor,    # [N, d] — E_target[seeds]
    e_context_all: torch.Tensor,    # [N*K, L, d] — E_context lookup on all walk positions
    walks: WalkData,                # carries walk_lens, timestamps for masking & weighting
    t_query: torch.Tensor,          # [N] int64 — query timestamp per seed (= batch t_max)
    beta: float,
    time_scale: float,
    weighting: str = "A",            # Phase 1 ablation: "A" | "B" | "C"
) -> torch.Tensor:
    """Pull E_target[seed] toward E_context[walk-neighbours]. Three loss-weighting
    variants for the Phase 1 ablation:

      Variant A (control, default):
          α(c) = (1 / depth(c)) · (1 + Δt(c) / time_scale)^(−β)
          — both positional decay AND temporal decay applied.

      Variant B (distance only — drop time decay; sampler does it):
          α(c) = 1 / depth(c)

      Variant C (uniform over valid positions — sampler does everything):
          α(c) = 1

    All three share the same valid mask (excludes padding and the seed
    position itself) and the same masked-mean reduction.

    Per (walk w, context position c):
        depth(c) = lens_w − 1 − c     (steps from c to the seed)
        Δt(c)    = t_query − timestamps[w, c]
        loss contribution = α(c) · (1 − cos(E_target[seed_w], E_context[walk_w, c]))

    Vectorised: one cosine per (w, c) cell, masked-mean reduced.
    """
    if weighting not in ("A", "B", "C"):
        raise ValueError(f"weighting must be 'A', 'B', or 'C', got {weighting!r}")
    device = e_target_seed.device
    K = walks.K
    NK, L, _ = e_context_all.shape
    lens = walks.lens.to(device)                            # [N*K]
    ts = walks.timestamps.to(device)                        # [N*K, L]

    positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
    valid = positions < lens.unsqueeze(1)                    # [N*K, L]
    # Exclude the seed position (the last valid one in each walk) — alignment
    # is a SELF-cosine there and contributes nothing.
    seed_pos = (lens - 1).clamp_min(0).unsqueeze(1)          # [N*K, 1]
    not_seed = positions != seed_pos
    use = valid & not_seed                                   # [N*K, L]

    # Per-walk seed embedding broadcast to per-position: E_target[seed_w]
    # broadcast across L positions of that walk. seeds are (walk → seed) =
    # repeat each of N seed embeddings K times.
    e_target_per_walk = e_target_seed.repeat_interleave(K, dim=0)  # [N*K, d]
    e_target_bc = e_target_per_walk.unsqueeze(1).expand_as(e_context_all)

    cos = F.cosine_similarity(e_target_bc, e_context_all, dim=-1)  # [N*K, L]

    # Build α(c) per the weighting variant.
    if weighting == "A":
        # Positional × temporal — the original alignment loss.
        depth = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()
        pos_w = 1.0 / depth
        t_query_per_walk = t_query.repeat_interleave(K).unsqueeze(1)
        dt = (t_query_per_walk - ts).clamp_min(0).float()
        time_w = (1.0 + dt / max(time_scale, 1e-6)) ** (-beta)
        w = pos_w * time_w
    elif weighting == "B":
        # Positional only — drop time decay (sampler bias already encodes recency).
        depth = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()
        w = 1.0 / depth
    else:  # weighting == "C"
        # Uniform — sampler handles both positional and temporal weighting.
        w = torch.ones_like(cos)

    use_f = use.float()
    contrib = use_f * w * (1.0 - cos)
    denom = (use_f * w).sum().clamp_min(1e-6)
    return contrib.sum() / denom


def uniformity_loss(
    e_target: torch.Tensor,    # [B, d] — embeddings of unique batch nodes
    temperature: float = 2.0,
    cap: int = 20_000,
) -> torch.Tensor:
    """Wang & Isola uniformity: log E_{x,y} [exp(−t · ||x − y||²)].

    Encourages embeddings to spread uniformly on the unit hypersphere.
    All-pairs O(B²); we cap B at `cap` by random subsampling — at d=128
    this keeps it well under 1 GB peak.
    """
    if e_target.size(0) > cap:
        idx = torch.randperm(e_target.size(0), device=e_target.device)[:cap]
        e_target = e_target[idx]
    e = F.normalize(e_target, dim=-1)
    sq_dist = torch.cdist(e, e, p=2.0).pow(2)   # [B, B]
    # Mask the diagonal: distance(x, x) = 0 contributes exp(0) = 1 spuriously.
    n = e.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=e.device)
    exp_neg_d = torch.exp(-temperature * sq_dist)[mask]
    return exp_neg_d.mean().clamp_min(1e-12).log()


def link_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits on positives (label=1) + negatives (label=0)."""
    return F.binary_cross_entropy_with_logits(logits, labels)


# ====================================================================== #
#  v2.3 §4.7 — Loss-family search                                        #
# ====================================================================== #

def _positional_weights(walks: WalkData, t_query: torch.Tensor, beta: float,
                        time_scale: float, weighting: str, device) -> tuple:
    """Shared weight + valid-mask computation used by all primary losses.

    Returns (w, valid_use) — each shape [N*K, L]:
      w          = α(c) — positional / temporal weight per position
      valid_use  = bool mask: valid positions, excluding the seed slot
    """
    K = walks.K
    L = walks.timestamps.shape[1]
    lens = walks.lens.to(device)
    ts = walks.timestamps.to(device)

    positions = torch.arange(L, device=device).unsqueeze(0)
    valid = positions < lens.unsqueeze(1)
    seed_pos = (lens - 1).clamp_min(0).unsqueeze(1)
    not_seed = positions != seed_pos
    valid_use = valid & not_seed

    if weighting == "A":
        depth = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()
        pos_w = 1.0 / depth
        t_query_per_walk = t_query.repeat_interleave(K).unsqueeze(1)
        dt = (t_query_per_walk - ts).clamp_min(0).float()
        time_w = (1.0 + dt / max(time_scale, 1e-6)) ** (-beta)
        w = pos_w * time_w
    elif weighting == "B":
        depth = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()
        w = 1.0 / depth
    else:  # "C"
        w = torch.ones_like(ts, dtype=torch.float32)
    return w, valid_use


def triplet_loss(
    e_target_seed: torch.Tensor,    # [N, d]  — E_target[seeds]
    e_context_all: torch.Tensor,    # [N*K, L, d] — E_context on walk positions
    e_context_neg: torch.Tensor,    # [N*K, d]    — context lookups for uniform-random negatives
    walks: WalkData,
    t_query: torch.Tensor,
    beta: float,
    time_scale: float,
    weighting: str = "A",
    margin: float = 0.5,
) -> torch.Tensor:
    """A3.2 — Triplet/margin loss with semi-hard mining on cosine similarity.

    Δt enters as a per-pair LOSS WEIGHT, NOT as a sampling probability.
    This makes triplet consistent with InfoNCE/SGNS (all three use
    `w(p)` as a multiplicative weight on the per-pair loss term); see
    v2.3 §4.7.1 audit and amendment v1.3 §4.2 for rationale.

    One triplet per walk:
      - anchor   = target(seed_walk)
      - positive = context(position p), where p ~ Uniform over valid positions
      - negative = e_context_neg[walk]  (pre-sampled in the caller from
                   uniform-random destinations, decoupled from walks for speed)

    Loss = Σ_walks (keep · w(p) · hinge)  /  Σ_walks (keep · w(p))
       where hinge = ReLU(margin - cos(a, p) + cos(a, n)),
             keep  = semi-hard mining mask.

    Semi-hard mask: cos(a,p) - cos(a,n) < margin  AND  cos(a,n) < cos(a,p).
    Excludes "easy" (already-separated) and "hard" (negative-closer-than-positive,
    which collapses bipartite-flavored graphs).

    Cosine normalisation puts both similarities in [-1, 1], so `margin=0.5` is
    a literature-default that does not need per-dataset retuning.
    """
    device = e_target_seed.device
    K = walks.K
    NK, L, d = e_context_all.shape

    w, valid_use = _positional_weights(walks, t_query, beta, time_scale, weighting, device)
    # UNIFORM sample over valid walk positions — Δt is *not* in the sampling
    # distribution; it enters the loss via the w(p) multiplier on the hinge.
    valid_f = valid_use.float()
    row_sum = valid_f.sum(dim=1, keepdim=True)
    has_any = (row_sum.squeeze(1) > 0)
    if not has_any.any():
        # Degenerate batch (all walks empty/seed-only); return zero loss.
        return e_target_seed.sum() * 0.0
    # Replace empty rows with uniform-over-row to make multinomial well-defined;
    # we mask these walks out of the final loss anyway.
    safe_probs = torch.where(
        row_sum > 0, valid_f, torch.ones_like(valid_f) / L,
    )
    pos_idx = torch.multinomial(safe_probs, num_samples=1).squeeze(1)  # [N*K]
    row_arange = torch.arange(NK, device=device)
    pos_emb = e_context_all[row_arange, pos_idx]                      # [N*K, d]
    # Gather w(p) at the sampled positions — this is the Δt-decay
    # multiplicative weight that goes on the hinge.
    w_at_pos = w[row_arange, pos_idx]                                 # [N*K]

    # Anchor: target(seed) broadcast per walk (each seed has K walks).
    a = e_target_seed.repeat_interleave(K, dim=0)                     # [N*K, d]

    # Cosine similarities (no eps needed; F.cosine_similarity handles zero-norm).
    cos_pos = F.cosine_similarity(a, pos_emb, dim=-1)                 # [N*K]
    cos_neg = F.cosine_similarity(a, e_context_neg, dim=-1)           # [N*K]

    # Semi-hard mask
    semi_hard = (cos_pos - cos_neg < margin) & (cos_neg < cos_pos)
    keep = has_any & semi_hard
    if not keep.any():
        return e_target_seed.sum() * 0.0
    hinge = F.relu(margin - cos_pos + cos_neg)
    keep_f = keep.float()
    return (hinge * w_at_pos * keep_f).sum() / (w_at_pos * keep_f).sum().clamp_min(1e-6)


def infonce_loss(
    e_target_seed: torch.Tensor,    # [N, d]
    e_context_all: torch.Tensor,    # [N*K, L, d]
    e_context_unif_neg: torch.Tensor,  # [num_neg_unif, d]  uniform-random destination contexts
    walks: WalkData,
    t_query: torch.Tensor,
    beta: float,
    time_scale: float,
    weighting: str = "A",
    tau: float = 0.1,
    num_neg_in_batch: int = 256,
) -> torch.Tensor:
    """A3.1 — Multi-positive InfoNCE with positional weighting.

    For each anchor (walk seed), every earlier walk position is a positive
    weighted by w(i). Negatives = `num_neg_in_batch` random contexts pooled
    from the *flat* set of all walk positions across the batch (collision
    rate with anchor's own positives is small at typical NK·L sizes; the
    temperature softens it) + `num_neg_unif` uniform-random destination
    contexts (provided by the caller).

    Loss per (walk w, position i):
        score_pos = (target(u_w) · context(w, i)) / τ
        score_neg = (target(u_w) · context_neg) / τ
        L_{w,i}   = -log( exp(score_pos) / (exp(score_pos) + Σ_j exp(score_neg_j)) )
        weighted by w(i).
    """
    device = e_target_seed.device
    K = walks.K
    NK, L, d = e_context_all.shape

    w_pos, valid_use = _positional_weights(walks, t_query, beta, time_scale, weighting, device)

    a = e_target_seed.repeat_interleave(K, dim=0)                     # [N*K, d]

    # Positive scores: dot(a_w, context[w, i]) / τ
    score_pos = (a.unsqueeze(1) * e_context_all).sum(dim=-1) / tau    # [N*K, L]

    # In-batch negatives: sample num_neg_in_batch random (w', i') flat indices
    flat = e_context_all.reshape(NK * L, d)
    if num_neg_in_batch > 0:
        neg_idx = torch.randint(0, NK * L, (num_neg_in_batch,), device=device)
        in_batch_neg = flat[neg_idx]                                  # [Nb, d]
    else:
        in_batch_neg = torch.empty(0, d, device=device)

    # Uniform-random destination negatives — provided by caller
    all_neg = torch.cat([in_batch_neg, e_context_unif_neg], dim=0)    # [Nb + Nu, d]
    score_neg = (a @ all_neg.t()) / tau                               # [N*K, Nb+Nu]

    # Multi-positive InfoNCE — numerically stable log-softmax via logsumexp.
    # Per (walk w, position i): -log(softmax) = -(s_pos[w,i] - logsumexp([s_pos[w,i], s_neg[w,:]])).
    # The earlier `exp(s)/(exp(s)+Σexp(neg))` form overflows at τ=0.1 once
    # scores exceed ~30 — this rewrite preserves the same math but stays
    # finite under any score magnitude.
    Nn = score_neg.shape[1]
    score_neg_bc = score_neg.unsqueeze(1).expand(-1, L, -1)           # [N*K, L, Nn]
    all_scores = torch.cat(
        [score_pos.unsqueeze(-1), score_neg_bc], dim=-1,
    )                                                                  # [N*K, L, 1+Nn]
    denom_log = torch.logsumexp(all_scores, dim=-1)                   # [N*K, L]
    L_per_ic = -(score_pos - denom_log)                                # [N*K, L]

    weight = w_pos * valid_use.float()
    contrib = L_per_ic * weight
    denom_w = weight.sum().clamp_min(1e-6)
    return contrib.sum() / denom_w


def sgns_loss(
    e_target_seed: torch.Tensor,    # [N, d]
    e_context_all: torch.Tensor,    # [N*K, L, d]
    e_context_neg: torch.Tensor,    # [N*K, L, k_neg, d] — pre-sampled unigram^0.75 contexts
    walks: WalkData,
    t_query: torch.Tensor,
    beta: float,
    time_scale: float,
    weighting: str = "A",
    subsample_keep_prob_per_node: Optional[torch.Tensor] = None,  # [num_nodes] or None
) -> torch.Tensor:
    """A3.3 — Skip-gram with negative sampling (Mikolov 2013).

    For each (anchor u, walk position v with weight w(i)):
        L = -log σ(target(u) · context(v))
            - Σ_{v_-} log σ(-target(u) · context(v_-))
        weighted by w(i).

    Negatives `e_context_neg` are pre-sampled by the caller from the unigram^0.75
    distribution over training destinations (Mikolov default). `k_neg=5` is the
    classical default; the trainer's CLI exposes it.

    Per amendment §4.7.1: η_uniform = 0 (Trainer enforces).
    """
    device = e_target_seed.device
    K = walks.K
    NK, L, d = e_context_all.shape

    w_pos, valid_use = _positional_weights(walks, t_query, beta, time_scale, weighting, device)

    a = e_target_seed.repeat_interleave(K, dim=0)                     # [N*K, d]
    a_bc = a.unsqueeze(1)                                             # [N*K, 1, d]

    # Positive score: target · context_pos, per (w, i)
    score_pos = (a_bc * e_context_all).sum(dim=-1)                    # [N*K, L]

    # Negative score: target · context_neg, per (w, i, k_neg)
    # e_context_neg: [N*K, L, k_neg, d]
    score_neg = (a_bc.unsqueeze(2) * e_context_neg).sum(dim=-1)       # [N*K, L, k_neg]

    # BCE-with-logits, summed over k_neg.
    # -log σ(score_pos) = softplus(-score_pos)
    # -log σ(-score_neg) = softplus(score_neg)
    pos_term = F.softplus(-score_pos)                                 # [N*K, L]
    neg_term = F.softplus(score_neg).sum(dim=-1)                      # [N*K, L]

    weight = w_pos * valid_use.float()
    # Mikolov subsampling of frequent positives (optional). Discards each
    # (anchor, context_v) pair with prob (1 - keep_prob(v)) where
    # keep_prob(v) = min(1, sqrt(t / f(v))). Implemented as a Bernoulli
    # mask on the weight tensor; t=1e-5 is the Mikolov literature default.
    if subsample_keep_prob_per_node is not None:
        # walks.nodes: [N*K, L] int. Gather per-position keep_prob.
        nodes = walks.nodes.to(device).long().clamp_min(0)            # [N*K, L]
        # Use keep_prob[node] only where valid_use; else weight is already 0.
        per_pos_keep = subsample_keep_prob_per_node[nodes]              # [N*K, L]
        bern = torch.bernoulli(per_pos_keep)                            # [N*K, L]
        weight = weight * bern
    contrib = (pos_term + neg_term) * weight
    return contrib.sum() / weight.sum().clamp_min(1e-6)


def normbrake_loss(
    E_target: torch.Tensor,         # [N, d]
    E_context: torch.Tensor,        # [N, d]
    threshold: float,
) -> torch.Tensor:
    """A3.x_normbrake — diagnostic-derived per-column L2 hinge.

    Penalises column norms past `threshold`. Self-limiting: zero gradient
    below threshold. Composes with any primary loss.

    L = mean_j (max(0, ||E[:, j]||_2 - threshold))^2  summed over the two tables.

    `threshold` is calibrated once per dataset from the anchor's best-val
    checkpoint (`1.5 × joint_mean_col_norm`); see v2.3 §4.7.4 for the procedure.
    """
    col_norms_t = E_target.norm(dim=0)        # [d]
    col_norms_c = E_context.norm(dim=0)       # [d]
    excess_t = F.relu(col_norms_t - threshold)
    excess_c = F.relu(col_norms_c - threshold)
    return (excess_t.pow(2).mean() + excess_c.pow(2).mean())
