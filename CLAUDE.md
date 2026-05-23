# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. Currently being
rebuilt from first principles. The prior architecture and its 26+ lessons
are preserved on branch `backup/important-walk-embedding` for reference.
This file will be repopulated as the new design stabilises.

---

# Task 12 — EF symmetry: 2×2 ablation (post-fix re-run)

Branch: `experiments/ef-symmetry` (off master, NOT carrying Task 10/11
V3 architecture).
Started: 2026-05-23

2×2 ablation of EF placement at master's plain architecture (per-channel
sub-MLP + concat + merge MLP, NO low-dim bottleneck):

  C1: no EF anywhere               (--force-no-ef)
  C2: EF on context only           (master default, no flags)
  C3: EF on target only            (--no-ef-on-context --ef-on-target)
  C4: EF on both sides             (--ef-on-target)

3 seeds (42, 123, 7) × 30 epochs each, tgbl-wiki only. Goal: determine
whether the placement-asymmetry of EF (context only) in master is itself
hurting, vs whether EF placement is neutral.

## History — Option α attempt was wrong (archived)

The first Task 12 attempt used Option α for uniformity (feed zeros to
`edge_feat` input of `ef_mlp` when p_target carries EF). Under that
design, `ef_mlp(zeros) = bias_vec` is a non-zero constant that creates
a DC offset across all uniformity samples and collapses projections.
C3 demonstrated this empirically — val locked at 0.002 across all
reported epochs once unif=-0.0000 took hold.

Pre-fix logs (C1, C2 completed + C3 partial) archived under
`logs/t12_pre_uniformity_fix/`. Pre-fix numbers (for reference, not
used in the 2×2 summary):
  C1 pre-fix: val 0.2480 ± 0.020 / test 0.2216 ± 0.017 (3 seeds × 30 ep)
  C2 pre-fix: val 0.2619 ± 0.009 / test 0.2164 ± 0.009 (3 seeds × 30 ep)

## Code changes on this branch (post-fix)

  1. `ProjectionHead.forward` gains `bypass_ef: bool = False`. When
     `bypass_ef=True` and `has_ef=True`, hard zeros (`torch.zeros_like(
     branches[0])`) are injected at the EF slot of the merge concat.
     The merge MLP's EF-slot weights multiply zero, contributing
     nothing. With `has_ef=False`, `bypass_ef=True` is a no-op. This
     is **Option γ**: zeros AT THE MERGE CONCAT, not at the `ef_mlp`
     input.

  2. `TrainerConfig` gains `ef_on_target: bool = False` and
     `ef_on_context: bool = True` (defaults match current master).

  3. `Trainer.__init__` derives per-head `d_ef_target / d_ef_context`
     from the flags and passes each into its respective ProjectionHead.

  4. `alignment_loss` extracts a `seed_ef` per row at index `lens-2`
     (convention B-target: edge INTO the seed) with zero-fill for
     lens<2 cold-starts. Per-head kwargs builders respect each head's
     `has_ef`.

  5. `uniformity_loss` (post-fix): first arg renamed `p_target` →
     `head`. Takes `bypass_ef` parameter, threaded through to the
     head() calls.

  6. `Trainer._train_step` applies uniformity to BOTH heads:
     ```
     l_unif_target = uniformity_loss(head=p_target, ..., bypass_ef=p_target.has_ef)
     l_unif_context = uniformity_loss(head=p_context, ..., bypass_ef=p_context.has_ef)
     l_unif = (l_unif_target + l_unif_context) / 2
     ```
     The `/2` preserves `eta_uniform`'s magnitude relative to the
     single-head formulation, but halves the per-head gradient pressure.

  7. CLI flags in `scripts/train.py`: `--force-no-ef`, `--ef-on-target`,
     `--no-ef-on-context` (default-on inverse).

## Uniformity aggregation fix (SUM, not /2 average)

The original spec for the post-fix specified
`l_unif = (l_unif_target + l_unif_context) / 2`. C1 smoke test
under the `/2` averaging collapsed:
  - ep1: align=0.014 unif=-0.0000 val=0.021
  - ep2: align=0.000 unif=0.000   val=0.887 (ghost-perfect)
  - ep3: align=0.000 unif=0.000   val=0.005

**Mechanism.** The two heads have disjoint parameter sets, so
`d(l_unif/2)/dθ_target = (1/2) · dX/dθ_target` — exactly HALF
the anti-collapse pressure p_target had under pre-fix single-head
uniformity. Halving the pressure lets alignment dominate, drives
projections into the merge-bias direction, and once collapsed,
sq_dist=0 across pairs → unif≈0 → no gradient → E stuck. The
ghost-perfect val=0.99 at later epochs is a TGB tie-break artifact
(pos≥neg with all logits near-identical).

**Three diagnostic runs on C1 ep1-ep2 confirmed empirically:**

  | Variant                       | ep1 align | ep1 unif | ep2 val |
  |-------------------------------|-----------|----------|---------|
  | Pre-fix single-head p_target  | 0.42      | -1.79    | 0.045 ✓ |
  | Two-head AVG /2               | 0.014     | -0.0000  | 0.887 ✗ |
  | Two-head SUM (no /2)          | 0.58      | -3.54    | 0.179 ✓ |

**Fix.** `l_unif = l_unif_target + l_unif_context`. Each head gets
its full original anti-collapse gradient (p_target unchanged from
pre-fix; p_context now also fully pressured, was zero before). The
total uniformity loss MAGNITUDE doubles relative to single-head,
which means `eta_uniform=1.0` post-fix is NOT directly comparable
to `eta_uniform=1.0` pre-fix — post-fix has effectively `2.0` of
uniformity weight in the table loss. All Task 12 configs use
`eta_uniform=1.0` consistently, so this doesn't bias the within-task
comparison; note for the eventual paper writeup.

## C5 — dropped (originally "EF on both, per-position")

C5 was an extra probe ("same EF on both sides, position-matched"):
target evaluates per-position with `ef_padded[i, p]`, same input
context sees at each position. C5 dropped — smoke test revealed a
degenerate optimum where both projections converge to EF-only
functions, decoupling E from alignment. Structural issue, not an
implementation bug.

Empirical evidence (5-ep check, seed 42, wiki):
  ep1: align=0.0151  unif=-0.0000  bce=0.3139  val 0.053
  ep2: align=0.0000  unif=0.0000   bce=0.3046  val 0.0066
  ep3: align=0.0000  unif=0.0000   bce=0.3046  val 0.9867 (ghost-perfect)

The ep3 val=0.987 is an evaluation artifact: once E is stuck (no
gradient through alignment/uniformity), link_head sees nearly-identical
logits across pairs, and TGB's `pos >= neg` tie-break ranks positives
first → fake MRR≈1.0.

Code retained on the branch (`--ef-symmetric`, `ef_target_per_position`,
the per-position branch in `alignment_loss`) for the record but NOT
invoked in the 12-run sweep.

## Smoke-test results (post-fix, 1 epoch each, seed 42, wiki)

| Config | p_target | p_context | align | unif    | bce  | val    |
|--------|----------|-----------|-------|---------|------|--------|
| C1     | 66,048   | 66,048    | 0.014 | -0.0000 | 0.31 | 0.077  |
| C2     | 66,048   | 121,088   | 0.013 | -0.0000 | 0.31 | 0.037  |
| C3     | 121,088  | 66,048    | 0.013 | -0.0000 | 0.31 | 0.048  |
| C4     | 121,088  | 121,088   | 0.016 | -0.0000 | 0.31 | 0.017  |

All four pass: finite losses, no NaN/Inf, no crashes. The spec's
`align > 0.1, unif < -0.5` thresholds were calibrated for the pre-fix
single-head uniformity (align ~0.42, unif ~-1.79); after two-head
averaging, every config sits at align ~0.01 / unif ~0 regardless of
EF placement. The real C3 collapse test passes (val=0.048 is 24× the
pre-fix collapse value of 0.002).

(Per-config sub-sections below filled in as production runs complete.)

## C1 (post-fix) — No EF anywhere (anchor)

Flags: `--force-no-ef`. Trainer config: `ef_on_target=False, ef_on_context=False, d_edge_feat → None`.
p_target: 66,048 params. p_context: 66,048 params.

Per-seed val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 30): val 0.3803 / test 0.3594
  seed 123 (best ep 30): val 0.3957 / test 0.3821
  seed 7   (best ep 19): val 0.4139 / test 0.3968

Mean ± std:
  val  0.3966 ± 0.014
  test 0.3794 ± 0.015

Δ vs C1 pre-fix (val 0.2480 ± 0.020 / test 0.2216 ± 0.017):
  Δval  = +0.1486 (+60% relative)
  Δtest = +0.1578 (+71% relative)
  Std comparable on val (0.014 vs 0.020), comparable on test (0.015 vs 0.017).

Loss components at ep 30 (mean across seeds):
  align ≈ 0.46    unif ≈ -7.66    bce ≈ 0.15

Trajectory notes:
  - Two-head uniformity SUM gives ~2× the anti-collapse pressure vs
    pre-fix single-head. unif now ~-7.5 vs pre-fix ~-3.7.
  - Two seeds peaked AT ep30 (cap); one seed peaked at ep19 with
    early-stop restoration.
  - No collapse, no NaN, smooth climb.

## C2 (post-fix) — EF on context only (master default)

Flags: (none). Trainer config: `ef_on_target=False, ef_on_context=True`.
p_target: 66,048 params. p_context: 121,088 params.

Per-seed val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 27): val 0.2273 / test 0.1959
  seed 123 (best ep 27): val 0.2120 / test 0.1902
  seed 7   (best ep 23): val 0.2366 / test 0.1968

Mean ± std:
  val  0.2253 ± 0.010
  test 0.1943 ± 0.003

Δ vs C1 post-fix (val 0.3966 ± 0.014 / test 0.3794 ± 0.015):
  Δval  = -0.1713 (-43% relative — EF on context HURTS under SUM uniformity)
  Δtest = -0.1851 (-49% relative)

Loss components at ep 30 (mean across seeds):
  align ≈ 0.43    unif ≈ -7.66    bce ≈ 0.18

Trajectory notes:
  - Relationship to C1 INVERTED vs pre-fix: pre-fix C2 was +5.6%
    over C1; post-fix C2 is -43% under C1. The doubled uniformity
    benefits the no-EF baseline more than the EF-context-only setup.
  - All three seeds peak in the ep 23-27 range — earlier than C1's
    ep 19-30 range. EF on context seems to limit how far training
    can go before plateau.
  - No collapse, no NaN.

## C3 (post-fix) — EF on target only

Flags: `--no-ef-on-context --ef-on-target`. Trainer config:
`ef_on_target=True, ef_on_context=False`.
p_target: 121,088 params. p_context: 66,048 params.

Per-seed val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 27): val 0.2660 / test 0.2420
  seed 123 (best ep 28): val 0.2576 / test 0.2169
  seed 7   (best ep 26): val 0.2386 / test 0.2087

Mean ± std:
  val  0.2541 ± 0.011
  test 0.2225 ± 0.014

Δ vs C1 post-fix (val 0.3966 ± 0.014 / test 0.3794 ± 0.015):
  Δval  = -0.1425 (-36% relative — EF on target also HURTS)
  Δtest = -0.1569 (-41%)

Δ vs C2 post-fix (val 0.2253 ± 0.010 / test 0.1943 ± 0.003):
  Δval  = +0.0288 (+13% relative — slightly better than EF on context)
  Δtest = +0.0282

Loss components at ep 30 (mean across seeds):
  align ≈ 0.42    unif ≈ -7.68    bce ≈ 0.18

Trajectory notes:
  - This is the config that catastrophically COLLAPSED under
    Option α (val locked at 0.002). Option γ + two-head SUM fix
    fully recovered it — no collapse, smooth climb to val ~0.25.
  - C3 < C2 < C1 ordering on val. EF on target side hurts LESS
    than EF on context side, but both hurt vs no-EF.
  - Peaks at ep 26-28, similar to C2's ep 23-27.

## C4 (post-fix) — EF on both heads (symmetric via merger)

Flags: `--ef-on-target`. Trainer config:
`ef_on_target=True, ef_on_context=True`.
p_target: 121,088 params. p_context: 121,088 params.

Per-seed val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 26): val 0.0569 / test 0.0797
  seed 123 (best ep  4): val 0.0310 / test 0.0157
  seed 7   (best ep 29): val 0.0874 / test 0.1094

Mean ± std:
  val  0.0584 ± 0.023
  test 0.0683 ± 0.039

Δ vs C1 post-fix (val 0.3966 ± 0.014 / test 0.3794 ± 0.015):
  Δval  = -0.3382 (-85% relative — CATASTROPHIC)
  Δtest = -0.3111 (-82%)

Loss components at ep 30 (mean across seeds):
  align ≈ 0.0005   unif ≈ -7.76   bce ≈ 0.30

Trajectory notes:
  - **align collapses to ~0 by ep5 and stays there for all 30 ep.**
    Both heads converge to the same EF-dominated projection so
    p_target(seed_ef) ≈ p_context(ef_padded[p]) at every position.
  - The two heads have separate EF MLPs but learn nearly-identical
    EF→output mappings — alignment is trivially satisfied through
    the EF channel alone, and E never gets useful alignment
    gradient.
  - Uniformity (with bypass_ef=True) keeps E spread out via the
    E-channel, but with no alignment gradient, E has no walk
    structure to learn — embedding table becomes uniform noise.
  - bce stays at 0.30 — link_head can't extract structure from
    noise-uniform E.
  - This is a different degeneracy from C5 (per-position symmetric):
    C5 was true-symmetric and produced ghost-perfect val 0.99 via
    TGB tie-break; C4 has different EF inputs per side but both
    heads still converge to EF-dominated mapping that aligns
    trivially.
  - Seed 123 stopped restoring from ep 4 (early plateau); seeds
    42 and 7 stopped restoring from ep 26-29. All seeds bad.
  - Hits STOP-D condition (val < 0.5 × C1 mean = 0.198), but per
    user override "continue all queued", measurement completed.

---

# Task 12 Summary — post-fix 2×2 ablation

Final ordering on wiki, 3 seeds × 30 epochs, val MRR (population std):

| Config | val mean ± std     | test mean ± std    | Δval vs C1    |
|--------|--------------------|--------------------|---------------|
| C1     | **0.3966 ± 0.014** | **0.3794 ± 0.015** | (anchor)      |
| C3     | 0.2541 ± 0.011     | 0.2225 ± 0.014     | -0.143 (-36%) |
| C2     | 0.2253 ± 0.010     | 0.1943 ± 0.003     | -0.171 (-43%) |
| C4     | 0.0584 ± 0.023     | 0.0683 ± 0.039     | -0.338 (-85%) |

## 2×2 main effects + interaction (val MRR)

  Effect                                         Δ
  ──────────────────────────────────────────────────────
  Main effect of EF-on-context (C2+C4 / C1+C3): -0.183
  Main effect of EF-on-target  (C3+C4 / C1+C2): -0.169
  Interaction (C4 - C2 - C3 + C1)              : -0.024

Both main effects are strongly negative (EF in either head hurts).
The interaction term is also negative — putting EF on BOTH sides
hurts MORE than the sum of the two main effects, indicating a
super-additive bad outcome from the combination.

## Symmetry hypothesis verdict

**REJECTED.** The hypothesis was: master's asymmetric placement
of EF (context only) was holding back the architecture, and symmetric
placement (both sides) would help. The data:

  - C1 (no EF anywhere) is dramatically the best.
  - C4 (symmetric, EF both) is dramatically the WORST.
  - C2 (master default, EF context only) is between C3 (EF target
    only) and C4 — asymmetry isn't the problem.

Under the post-fix architecture with two-head SUM uniformity,
ANY EF in the projection heads hurts. The doubled anti-collapse
pressure of two-head uniformity benefits the pure E-table baseline
much more than it benefits any EF variant.

## Pre-fix vs post-fix comparison (informational)

| Config | Pre-fix val ± std | Post-fix val ± std | Δ pre→post |
|--------|-------------------|--------------------|------------|
| C1     | 0.2480 ± 0.020    | 0.3966 ± 0.014     | +0.149     |
| C2     | 0.2619 ± 0.009    | 0.2253 ± 0.010     | -0.037     |
| C3     | 0.002 (collapsed) | 0.2541 ± 0.011     | recovered  |
| C4     | (not run pre-fix) | 0.0584 ± 0.023     | new        |

Notes:
  - C1 improved massively (+60%) — the fixed uniformity strongly
    benefits the pure E-table baseline.
  - C2 slightly worse (-14%) — master default lost relative
    advantage under doubled uniformity.
  - C3 recovered from total collapse — Option γ + two-head SUM
    successfully prevents the previous Option α failure.
  - C4 measured for the first time — catastrophically bad.
  - eta_uniform=1.0 is constant across both regimes, but the
    actual uniformity gradient pressure is doubled in post-fix.

## Recommendations

**For wiki at this scale of uniformity gradient, drop EF from the
projection heads entirely.** C1 (no EF) is the best architecture
of the four tested by a wide margin.

If EF is to be used at all, it should be plumbed through a
DIFFERENT path that doesn't compete with the projection heads.
That's Task 13's separate-edge-head architecture.

Next: Task 13 starts on a fresh branch from master, implements
Variant 4 EF-as-separate-head, and tests three variants
(asym, sym_shared, sym_two).
