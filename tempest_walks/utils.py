"""Pure helper utilities shared across trainer and CLI.

Functions in this module have no module-level state and don't depend
on any class instance. They are pulled here so trainer.py and
scripts/train.py stay focused on orchestration, not boilerplate.

Contents:
  Determinism:
    - seed_all(seed)              — seed Python/numpy/torch RNGs.
  Dataset derivation:
    - derive_t_train(train_ts)    — span of training timestamps.
    - detect_bipartite(train_split) — src/dst disjointness check.
  Training-side memory + schedule:
    - compute_auto_chunk_size(walks, chunk_size_override, device,
                              overhead_bytes, intermediates_kept,
                              safety_factor)
                                  — pick an InfoNCE seed-chunk size
                                    that fits available GPU memory.
    - make_lr_lambda(warmup_steps, decay_steps, lr_min_ratio)
                                  — closure for LambdaLR that does
                                    linear warmup then cosine decay
                                    to lr_min_ratio.
"""

import math
import random
from typing import Callable

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────


def seed_all(seed: int) -> None:
    """Seed every standard RNG from one root seed.

    Sampler-internal RNGs (negative samplers) are seeded via
    TrainerConfig.seed downstream. Tempest's walk RNG is NOT
    controlled here — Tempest CPU mode uses its own internal RNG
    and may exhibit small run-to-run drift even with the same Python
    seed. Multi-seed anchoring is the correct way to measure this.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────
# Dataset derivation
# ──────────────────────────────────────────────────────────────────────


def derive_t_train(train_ts: np.ndarray) -> float:
    """T_train: training-span (max - min). Required > 0 — used as a
    denominator in alignment_loss's time weighting."""
    if train_ts.size == 0:
        raise ValueError("Empty training timestamps; cannot derive T_train.")
    span = float(train_ts.max() - train_ts.min())
    if span <= 0:
        raise ValueError(f"Non-positive T_train: {span}")
    return span


def detect_bipartite(train_split) -> bool:
    """A graph is bipartite (under the link-pred convention) iff the
    set of source IDs and the set of destination IDs are disjoint.
    Holds for tgbl-wiki (users→pages), tgbl-review (users→items),
    tgbl-subreddit (users→subreddits). Fails for tgbl-coin / tgbl-flight
    / tgbl-comment where any node can be either endpoint."""
    src_set = set(np.unique(train_split.sources).tolist())
    dst_set = set(np.unique(train_split.destinations).tolist())
    return src_set.isdisjoint(dst_set)


# ──────────────────────────────────────────────────────────────────────
# Training-side memory + schedule
# ──────────────────────────────────────────────────────────────────────


def compute_auto_chunk_size(
    walks,
    chunk_size_override: int,
    device: torch.device,
    overhead_bytes: int = 1500 * 1024 * 1024,
    intermediates_kept: int = 6,
    bytes_per_intermediate: int = 4,
    safety_factor: float = 0.7,
    projection_d_hidden: int = 128,
    projection_saved_tensors: int = 7,
) -> int:
    """Auto-size InfoNCE seed-chunk based on available GPU memory.

    Memory model (Option B with per-chunk backward).

    With per-chunk backward (chunk_mean.backward(retain_graph=...) inside
    alignment_loss), peak memory is bounded by:

        fixed_overhead = optimizer Adam state + the projection-graph
                         saved activations retained across chunks by
                         retain_graph=True

        per_chunk      = intermediates_kept × chunk_size × M
                         × bytes_per_intermediate
                         (sim_dot / sim / log_p / w_pos / etc — only
                         ONE chunk's worth alive at a time, freed by
                         Python refcounting between iterations)

    The projection retention term scales with M (the pool size), so a
    fixed `overhead_bytes` constant isn't enough for high-vocab datasets.
    We add an explicit `projection_retention` estimate sized to M.

    Defaults:
      overhead_bytes = 1.5 GB
        Covers model parameters + Adam state + small safety margin.
        Bigger than the old 500 MB default which only accounted for
        the model.
      projection_d_hidden = 128
        ProjectionHead.d_hidden default. Override if model is wider.
      projection_saved_tensors = 7
        Each ProjectionHead saves ~7 activations per forward (3 in
        e_mlp Linear→GELU→Linear, 3 in merge Linear→GELU→Linear, 1
        in F.normalize). Two heads call into the same upstream:
        p_target on [NK] inputs, p_context on [M] inputs. We size
        the term by M since M >> NK in realistic batches.

    Returns:
        - chunk_size_override if > 0 (manual override).
        - 0 if running on CPU (no chunking needed; chunked path
          gives no benefit without GPU memory pressure).
        - Auto-computed positive integer on GPU, in [1, NK].
    """
    # User override takes priority.
    if chunk_size_override > 0:
        return chunk_size_override

    # On CPU, no memory pressure; let alignment_loss use the
    # no-chunking branch (chunk_size=0).
    if not torch.cuda.is_available() or device.type != "cuda":
        return 0

    NK, L = walks.nodes.shape
    M = NK * L
    if M == 0:
        return 0

    free_bytes, _ = torch.cuda.mem_get_info(device)

    # Projection-graph saved activations retained across the whole
    # chunk loop (because retain_graph=True keeps them alive). The
    # term scales linearly with M.
    projection_retention = (
        projection_saved_tensors
        * (NK + M)
        * projection_d_hidden
        * bytes_per_intermediate
    )

    available = max(free_bytes - overhead_bytes - projection_retention, 0)

    bytes_per_seed = M * bytes_per_intermediate * intermediates_kept
    if bytes_per_seed == 0:
        return 0

    raw_chunk = available // bytes_per_seed
    # Safety factor for PyTorch allocator fragmentation.
    safe_chunk = int(raw_chunk * safety_factor)
    # Cap at NK — no point chunking larger than the batch itself.
    safe_chunk = min(safe_chunk, NK)
    # Floor at 1 (no kernel-launch amortisation; correctness over
    # efficiency under tight memory — better to be slow than OOM).
    safe_chunk = max(safe_chunk, 1)
    return safe_chunk


def make_lr_lambda(
    warmup_steps: int,
    decay_steps: int,
    lr_min_ratio: float,
) -> Callable[[int], float]:
    """Build a LambdaLR lambda for linear warmup + cosine decay.

    Shape:
      step 0..warmup_steps    linear ramp from 0 (at step 0) to 1.0
      step warmup..decay      cosine from 1.0 to lr_min_ratio
      step > decay_steps      stay at lr_min_ratio

    lr_min_ratio is lr_min / peak_lr. The lambda returns a scale
    factor that LambdaLR multiplies by the optimizer's initial_lr.
    """

    def lr_lambda(step: int) -> float:
        # step is 0-indexed (PyTorch LambdaLR convention).
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_progress = step - warmup_steps
        decay_total = decay_steps - warmup_steps
        if decay_total <= 0:
            return lr_min_ratio

        progress = float(decay_progress) / float(decay_total)
        if progress >= 1.0:
            return lr_min_ratio

        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cos_factor

    return lr_lambda
