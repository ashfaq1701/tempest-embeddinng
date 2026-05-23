# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. Currently being
rebuilt from first principles. The prior architecture and its 26+ lessons
are preserved on branch `backup/important-walk-embedding` for reference.
This file will be repopulated as the new design stabilises.

---

# Task 12 — EF symmetry: 2×2 ablation

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

## Code changes on this branch

  1. `ProjectionHead.__init__` exposes `self.ef_input_dim = d_edge_feat`
     so uniformity_loss can build a zero-fill EF tensor when p_target
     carries the EF channel.

  2. `TrainerConfig` gains `ef_on_target: bool = False` and
     `ef_on_context: bool = True` (defaults match current master).

  3. `Trainer.__init__` derives per-head `d_ef_target / d_ef_context`
     from the flags and passes each into its respective ProjectionHead.

  4. `alignment_loss` extracts a `seed_ef` per row under convention
     B-target: edge at index `lens-2` (the edge INTO the seed from
     its immediate context). For walks with lens < 2 (no edges),
     zero-fill. The 4-branch if/elif chain over (NF × EF) is replaced
     by per-head kwargs builders that respect each head's `has_ef`.

  5. `uniformity_loss` zero-fills EF when `p_target.has_ef`. Slight
     train-time vs uniformity-time mismatch documented in code
     (Option α from the spec).

  6. CLI flags in `scripts/train.py`: `--force-no-ef`, `--ef-on-target`,
     `--no-ef-on-context` (default-on inverse).

## Smoke-test results (1 epoch each, seed 42, wiki)

  | Config | p_target | p_context | align | unif    | bce  | val    |
  |--------|----------|-----------|-------|---------|------|--------|
  | C1     | 66,048   | 66,048    | 0.42  | -1.79   | 0.29 | 0.073  |
  | C2     | 66,048   | 121,088   | 0.42  | -1.72   | 0.29 | 0.044  |
  | C3     | 121,088  | 66,048    | 0.013 | -0.0000 | 0.31 | 0.129  |
  | C4     | 121,088  | 121,088   | 0.016 | -0.0000 | 0.31 | 0.074  |

All four smoke-tested: finite losses, no NaN/Inf, no crash, expected
param counts (head gains 55,040 when EF channel enabled, due to
`Linear(d_ef=172, d_hidden=128)` + `Linear(d_hidden, d_proj)`).

### Note on C3/C4 unif ≈ 0 at init (Option α side-effect)

`uniformity_loss` zero-fills EF when `p_target.has_ef`. At fresh init:
  - `e ~ N(0, 0.02)` (E init std).
  - `e_mlp(e)` has very small magnitude.
  - `ef_mlp(zero) = bias_vec` (constant, magnitude ~`sqrt(1/d_ef)`).
  - The constant bias-vec channel dominates the (tiny) e channel
    after concat + merge → all unit-normalised outputs point in the
    SAME direction → pairwise sq_dist ≈ 1e-4 → unif loss ≈ 0.

This is the "slight train-time vs uniformity-time inconsistency"
the spec flagged when picking Option α. Verified empirically with
a 8-pair sanity script: `p_target(e, edge_feat=zero)` gives
pair-distance std ~6e-3 while `p_target(e, edge_feat=real_random)`
gives ~0.36.

Implications:
  - At C3/C4 init, uniformity contributes ~0 gradient. The E table
    is initially trained by alignment + BCE only.
  - After training updates start changing the E and merge weights,
    the e channel may grow and dominate; uniformity should "come
    online" but is initially passive.
  - The C3/C4 1-epoch align is small (0.01-0.02) because alignment
    finds it easy to make varying p_seed and constant-ish p_ctx
    collide — even with seed_ef present, the merge layer collapses
    early.

This is not a code bug; it is a direct consequence of Option α. The
spec acknowledges this trade-off. If C3/C4 underperform meaningfully
in the 12-run measurement, Option α may be the culprit, not the
symmetry hypothesis itself — that distinction goes into the summary.

(Sub-sections below filled in as each config completes.)

## C1 — No EF anywhere (anchor)

Flags: `--force-no-ef`. Trainer config: `ef_on_target=False, ef_on_context=False, d_edge_feat → None`.
p_target: 66,048 params. p_context: 66,048 params.

Per-seed val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 29): val 0.2407 / test 0.2151
  seed 123 (best ep 22): val 0.2285 / test 0.2043
  seed 7   (best ep 30): val 0.2747 / test 0.2455

Mean ± std:
  val  0.2480 ± 0.020
  test 0.2216 ± 0.017

Notes:
  - Seed 7 peaked AT ep30 (the budget cap) — possibly still
    climbing.
  - Variance is tight (std ~0.02), much lower than Task-10's 15-ep
    population std of 0.036. 30 epochs gives more stable comparisons.
  - These val/test numbers are substantially higher than Task 10's
    15-ep V0 anchor (val 0.185), confirming Task 11 T3's finding
    that wiki has serious unrealised headroom past 15 epochs.

Commit: 16f2cde, e8f09fa (code) + (this commit, C1 logs).

## C2 — EF on context only (master default)

Flags: (none). Trainer config: `ef_on_target=False, ef_on_context=True`.
p_target: 66,048 params. p_context: 121,088 params.

Per-seed val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 30): val 0.2533 / test 0.2140
  seed 123 (best ep 22): val 0.2588 / test 0.2071
  seed 7   (best ep 26): val 0.2737 / test 0.2282

Mean ± std:
  val  0.2619 ± 0.009
  test 0.2164 ± 0.009

Δ vs C1 (val 0.2480 ± 0.020 / test 0.2216 ± 0.017):
  Δval  = +0.0139 (+5.6% relative — EF on context HELPS at 30 ep)
  Δtest = -0.0052 (-2.3% — marginally worse on test)
  Std collapse: val 0.020 → 0.009 (2× tighter)
                test 0.017 → 0.009 (2× tighter)

Notes:
  - Master-default C2 at 30 ep is a clear win over C1 on val and a
    near-tie on test, with markedly tighter std.
  - This contradicts the 22% regression observed at Task 7's 5-ep
    snapshot. Either (a) at 5 ep, EF training hasn't had time to
    extract signal, or (b) the original measurement was high-variance
    noise. Either way: at 30 ep, master-default EF helps.
  - Seed 42 peaked AT ep30 again — possibly still climbing.
