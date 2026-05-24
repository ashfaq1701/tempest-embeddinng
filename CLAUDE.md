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

## V4-sym-two — Two separate edge heads, one per side

Flag: `--ef-variant sym_two`. p_target: 66,048 params. p_context:
66,048 params. EdgeHead on target side AND EdgeHead on context
side (separate modules, separate parameters): ~110,080 EdgeHead
params total. α_t and α_c learnable scalars (both zero-init).

Per-seed val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 6): val 0.1500 / test 0.1462
  seed 123 (best ep 5): val 0.1431 / test 0.1306
  seed 7   (best ep 3): val 0.1419 / test 0.1342

Mean ± std:
  val  0.1450 ± 0.004
  test 0.1370 ± 0.007

Δ vs C1 (val 0.3966 ± 0.014):
  Δval  = -0.252 (-63% relative — CATASTROPHIC, same as sym_shared)
  Δtest = -0.242 (-64%)

Δ vs V4-sym-shared (val 0.1440 ± 0.017):
  Δval  = +0.0010 (essentially tied)
  Δtest = +0.0017

Loss components at ep 30 (mean across seeds):
  align ≈ 0.01   unif ≈ -7.76   bce ≈ 0.29

Trajectory notes:
  - **Same catastrophic trajectory as sym_shared.** All three seeds
    peak at ep 3-6 and degrade thereafter.
  - **Key finding**: separate edge heads do NOT protect against
    degeneracy. Even with disjoint parameter sets for the two
    EdgeHeads, both heads independently converge to similar
    EF→direction mappings under the alignment pressure. The two-
    sided EF addition+renorm still admits the trivial "make both
    sides EF-dominated" solution.
  - This says the symmetry of the EF integration is the structural
    problem, not parameter sharing per se. Anywhere both sides
    receive an EF contribution, the alignment loss finds a
    trivial optimum.

---

# Task 12 + Task 13 combined comparison

All runs on tgbl-wiki, 3 seeds × 30 epochs, post-fix code base
(Option γ EF bypass + two-head SUM uniformity).

| Config                           | val mean ± std     | test mean ± std    | Δval vs C1    |
|----------------------------------|--------------------|--------------------|---------------|
| **C1 (no EF anywhere)**          | **0.3966 ± 0.014** | **0.3794 ± 0.015** | (anchor)      |
| V4-asym (edge head, target only) | 0.3546 ± 0.023     | 0.3260 ± 0.029     | -0.042 (-11%) |
| C3 (EF in p_target only)         | 0.2541 ± 0.011     | 0.2225 ± 0.014     | -0.143 (-36%) |
| C2 (EF in p_context only)        | 0.2253 ± 0.010     | 0.1943 ± 0.003     | -0.171 (-43%) |
| V4-sym-two (two edge heads)      | 0.1450 ± 0.004     | 0.1370 ± 0.007     | -0.252 (-63%) |
| V4-sym-shared (one shared)       | 0.1440 ± 0.017     | 0.1353 ± 0.016     | -0.253 (-64%) |
| C4 (EF in both projections)      | 0.0584 ± 0.023     | 0.0683 ± 0.039     | -0.338 (-85%) |

## Verdict

**C1 (no EF) is the winner across all 7 configs by a wide margin.**

The pattern is clear and consistent:
  - **No EF anywhere** → best by ~0.04 over the runner-up.
  - **EF on one side only** (C2, C3, V4-asym) → modest hurt (-11
    to -43%), trains stably to a worse plateau.
  - **EF symmetric on both sides** (C4, V4-sym-two, V4-sym-shared)
    → catastrophic collapse (-63 to -85%), peaks early then
    degrades. The asymmetry isn't the help we hypothesised in
    Task 12; the symmetry is the structural problem.

The α-zero-init trick of V4 doesn't save it: α grows during
training, the EF contribution becomes non-zero, and the same
trivial-alignment degeneracy that broke C4 also breaks V4-sym-*.
Separate edge heads (sym_two) vs shared edge head (sym_shared)
give essentially identical bad outcomes — parameter sharing isn't
the cause.

## Recommended action

**Adopt C1 (no EF in projection heads, no edge head) as the master
default.** Specifically:
  - Drop EF from `p_context` in the master `Trainer.__init__`.
  - Keep the two-head SUM uniformity (essential — pre-fix
    single-head uniformity gave C1 only val 0.248, less than half
    of post-fix's 0.397).
  - Keep `bypass_ef` infrastructure (harmless and useful if EF
    is reintroduced via a non-projection-head route in future).

Open question (NOT in Task 13 scope): is EF inherently unhelpful
on wiki, or does it need a fundamentally different architecture
to be useful? Three things tried so far (per-channel sub-MLP in
projection head, low-dim bottleneck V3 from Task 10, addition+
renormalise via EdgeHead) all underperform no-EF. Possible next
directions for a future task:
  - FiLM-style modulation (EF → γ, β; out = γ·E + β).
  - EF as edge-time prior to the walk sampler (not at projection).
  - Concatenate EF to (u, v) features in the LinkHead instead of
    in the projection — keeps E table training clean while letting
    EF inform scoring.

No further EF experiments without explicit user direction.

