"""Alignment + uniformity (Wang & Isola 2020 pattern) + link BCE."""

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
) -> torch.Tensor:
    """Pull E_target[seed] toward E_context[walk-neighbours], position- and
    time-weighted. Seed and padding positions are masked OUT.

    Per (walk w, context position c):
        w(c) = (1 / depth(c)) · (1 + Δt(c) / time_scale)^(−β)
        where depth(c) = lens_w − 1 − c     (steps from c to the seed)
              Δt(c)    = t_query − timestamps[w, c]
        loss contribution = w(c) · (1 − cos(E_target[seed_w], E_context[walk_w, c]))

    Vectorised: one cosine per (w, c) cell, masked-mean reduced.
    """
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

    # Positional weight: 1 / depth(c), depth(c) = lens − 1 − c (number of
    # walk steps from context c to the seed). depth 1 (last context before
    # seed) gets weight 1; deepest past gets weight 1/(lens-1).
    depth = (lens.unsqueeze(1) - 1 - positions).clamp_min(1).float()  # [N*K, L]
    pos_w = 1.0 / depth

    # Temporal weight: (1 + Δt / time_scale)^(−β). Δt = t_query − timestamps.
    t_query_per_walk = t_query.repeat_interleave(K).unsqueeze(1)  # [N*K, 1]
    dt = (t_query_per_walk - ts).clamp_min(0).float()             # [N*K, L]
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
