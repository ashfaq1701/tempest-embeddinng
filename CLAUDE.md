# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. Currently being
rebuilt from first principles. The prior architecture and its 26+ lessons
are preserved on branch `backup/important-walk-embedding` for reference.
This file will be repopulated as the new design stabilises.

---

# Task 13 — Variant 4 EF-as-separate-head architectures

Branch: `experiments/ef-separate-head` (from master)
Started: 2026-05-23

Three Variant 4 architectures tested:
  V4-asym       — edge head on target side only
  V4-sym-shared — one edge head shared across both sides
  V4-sym-two    — two separate edge heads (one per side)

All variants use a learnable α (zero-init) to scale the
edge-head contribution, with re-normalisation to maintain
unit-sphere geometry.

Two uniformity fixes from `experiments/ef-symmetry` are ported
forward (without per-head EF gating or C5 machinery):
  - `bypass_ef` parameter in `ProjectionHead.forward` (Option γ).
  - Two-head uniformity with SUM (not /2 average).

## V4-asym — Edge head on target side only

Flag: `--ef-variant asym`. p_target: 66,048 params. p_context: 66,048
params. EdgeHead on target side: ~55,040 params. α_t learnable scalar
(zero-init).

Per-seed val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 28): val 0.3740 / test 0.3518
  seed 123 (best ep 26): val 0.3221 / test 0.2851
  seed 7   (best ep 29): val 0.3678 / test 0.3410

Mean ± std:
  val  0.3546 ± 0.023
  test 0.3260 ± 0.029

Δ vs C1 (no-EF anchor from Task 12 post-fix, val 0.3966 ± 0.014):
  Δval  = -0.042 (-11% relative — V4-asym slightly worse than C1)
  Δtest = -0.053 (-14%)

Loss components at ep 30 (mean across seeds):
  align ≈ 0.40   unif ≈ -7.69   bce ≈ 0.15

Trajectory notes:
  - Despite α=0 init, V4-asym does not match C1. The extra
    edge_head parameters perturb the random init enough to slow
    training by ~11% on average.
  - α grows during training (would need explicit reporting; not
    instrumented this run), so the edge head's EF contribution
    becomes non-zero by mid-training. But the contribution doesn't
    seem to help — it adds variance without improving the mean.
  - Seed 123 was notably worse (val 0.32 vs 0.37-0.37 for the
    other seeds) — std doubled vs C1 (0.023 vs 0.014).

## V4-sym-shared — One shared edge head feeds both sides

Flag: `--ef-variant sym_shared`. p_target: 66,048 params. p_context:
66,048 params. Shared EdgeHead: ~55,040 params. α_t and α_c
learnable scalars (both zero-init), but both reference the SAME
underlying EdgeHead module.

Per-seed val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 2): val 0.1296 / test 0.1227
  seed 123 (best ep 5): val 0.1348 / test 0.1258
  seed 7   (best ep 6): val 0.1677 / test 0.1573

Mean ± std:
  val  0.1440 ± 0.017
  test 0.1353 ± 0.016

Δ vs C1 (val 0.3966 ± 0.014):
  Δval  = -0.253 (-64% relative — CATASTROPHIC)
  Δtest = -0.244 (-64%)

Loss components at ep 30 (mean across seeds):
  align ≈ 0.01   unif ≈ -7.77   bce ≈ 0.29

Trajectory notes:
  - **All three seeds peak by ep 2-6 and degrade thereafter.**
    Each ends with align ≈ 0.01 (collapsed) and bce rising from
    ~0.24 → 0.29.
  - Mechanism: α grows during training, the shared EdgeHead
    output projects EF to a unit vector that's identical on both
    sides (same module), and the addition+renormalise gives both
    p_seed and p_ctx outputs that are dominated by the shared EF
    direction. Alignment becomes trivial because both sides see
    the same EF→direction contribution. This is the same C4
    degeneracy from Task 12, with a different mechanism (here via
    shared EdgeHead instead of identical EF inputs).
  - The spec flagged this as the structural risk for sym_shared.
    Empirically confirmed.
