"""Minimal production losses for tempest-walks-v3.

Four losses:
  1. alignment_loss — pulls E_target[seed] toward E_context[walk_neighbor]
     with positional × temporal weighting.
  2. uniformity_loss — Wang & Isola hypersphere spread (regularizes
     embedding directions).
  3. normbrake_loss — per-column L2 hinge; clamps embedding magnitudes
     above a calibrated threshold. The locked cliff fix (with
     weight_decay_link applied to the link MLP).
  4. link_bce — BCE-with-logits for the link prediction head.
"""

import torch
import torch.nn.functional as F

from .walks import WalkData


def alignment_loss(
    e_target_seed: torch.Tensor,    # [N, d] — E_target[seeds]
    e_context_all: torch.Tensor,    # [N*K, L, d] — E_context lookup at all walk positions
    walks: WalkData,                # carries walk_lens, timestamps for masking + weighting
    t_query: torch.Tensor,          # [N] int64 — query timestamp per seed (= batch t_max)
    beta: float,
    time_scale: float,
) -> torch.Tensor:
    """Pull E_target[seed] toward E_context[walk-neighbour] with position-
    and time-weighted cosine similarity.

    Per (walk w, context position c):
        w(c)  = (1 / depth(c)) · (1 + Δt(c) / time_scale)^(−β)
                depth(c) = lens_w − 1 − c    (steps from c to the seed)
                Δt(c)    = t_query − timestamps[w, c]
        loss += w(c) · (1 − cos(E_target[seed_w], E_context[walk_w, c]))

    Reduced with a masked mean (denominator = Σ valid weights). Padding
    positions and the seed position itself are masked OUT.
    """
    device = e_target_seed.device
    K = walks.K
    NK, L, _ = e_context_all.shape
    lens = walks.lens.to(device)                            # [N*K]
    ts = walks.timestamps.to(device)                        # [N*K, L]

    positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
    valid = positions < lens.unsqueeze(1)                    # [N*K, L]
    # Exclude the seed (last valid position) — self-cosine = 1, contributes nothing.
    seed_pos = (lens - 1).clamp_min(0).unsqueeze(1)
    not_seed = positions != seed_pos
    use = valid & not_seed                                   # [N*K, L]

    # Broadcast E_target[seed] across the K walks of that seed × L positions.
    e_target_per_walk = e_target_seed.repeat_interleave(K, dim=0)
    e_target_bc = e_target_per_walk.unsqueeze(1).expand_as(e_context_all)
    cos = F.cosine_similarity(e_target_bc, e_context_all, dim=-1)  # [N*K, L]

    depth = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()
    pos_w = 1.0 / depth
    t_query_per_walk = t_query.repeat_interleave(K).unsqueeze(1)
    dt = (t_query_per_walk - ts).clamp_min(0).float()
    time_w = (1.0 + dt / max(time_scale, 1e-6)) ** (-beta)
    w = pos_w * time_w

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
    All-pairs O(B²); cap subsamples B at random when it exceeds the limit.
    """
    if e_target.size(0) > cap:
        idx = torch.randperm(e_target.size(0), device=e_target.device)[:cap]
        e_target = e_target[idx]
    e = F.normalize(e_target, dim=-1)
    sq_dist = torch.cdist(e, e, p=2.0).pow(2)
    n = e.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=e.device)
    exp_neg_d = torch.exp(-temperature * sq_dist)[mask]
    return exp_neg_d.mean().clamp_min(1e-12).log()


def link_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits on positives (label=1) + negatives (label=0)."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def normbrake_loss(
    E_target: torch.Tensor,         # [N, d]
    E_context: torch.Tensor,        # [N, d]
    threshold: float,
) -> torch.Tensor:
    """Per-column L2 hinge — saturates embedding magnitudes above threshold.

        L = mean_j (max(0, ||E[:, j]||₂ - threshold))²  summed over both tables.

    Self-limiting: zero gradient below threshold, quadratic above. Composes
    additively with the primary loss. Threshold calibrated per dataset
    (1.5× col_norm at ep 1–2).
    """
    col_norms_t = E_target.norm(dim=0)
    col_norms_c = E_context.norm(dim=0)
    excess_t = F.relu(col_norms_t - threshold)
    excess_c = F.relu(col_norms_c - threshold)
    return excess_t.pow(2).mean() + excess_c.pow(2).mean()
