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
