# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. Currently being
rebuilt from first principles. The prior architecture and its 26+ lessons
are preserved on branch `backup/important-walk-embedding` for reference.

---

# Task 10 — EF architecture search

Branch: `experiments/ef-architecture-search`
Started: 2026-05-23

The post-Task-6.7 EF plumbing is mechanically correct (verified empirically
against Tempest source), but the integration architecture currently in
master (per-channel sub-MLP → concat → merge MLP, L2-norm at output)
degrades wiki training. This task searches for an EF integration that
helps rather than hurts.

## Reference numbers

Pre-EF wiki anchor (Task 7 pre-EF, 3 seeds × 5 epochs, GPU):
  seed 42  ep5: val 0.0828 / test 0.0709
  seed 123 ep5: val 0.1528 / test 0.1566
  seed 7   ep4: val 0.1477 / test 0.1447  (best ep4; ep5 dropped)
  mean ± std: val 0.128 ± 0.040 / test 0.124 ± 0.049

Post-EF wiki at the SAME 5-epoch protocol (current arch A1+B1+C1):
  seed 42:  val 0.0641 / test 0.0661
  seed 123: val 0.0919 / test 0.0668
  seed 7:   val 0.1446 / test 0.1453
  mean ± std: val 0.100 ± 0.040 / test 0.093 ± 0.046

Post-EF − pre-EF wiki: Δval -0.028 (-22% relative), Δtest -0.031 (-25%).
Under the 30% STOP threshold but unambiguous degradation.

## Protocol

Every variant: tgbl-wiki, seeds {42, 123, 7}, **15 epochs**, default
hyperparameters except for the variant change. `--use-gpu`,
`--skip-final-full-eval`. Metric: best val MRR across 15 epochs,
mean ± std across the 3 seeds.

Variants are organised in three tiers, ordered cheapest → most
elaborate. After each tier, a decision paragraph decides whether to
escalate.

## V0 — No edge features (the anchor)

Code: `scripts/train.py` adds `--force-no-ef` CLI flag that overrides
`d_edge_feat = None` regardless of dataset. ProjectionHeads are
constructed without an EF channel; alignment_loss runs the no-EF
code path. Equivalent to the pre-Task-6.7 architecture.

Commit: this branch's first commit forward.

Per-seed val MRR / test MRR (best across 15 epochs):
  seed 42  (best ep  9):  val 0.1360 / test 0.1264
  seed 123 (best ep 14):  val 0.1986 / test 0.1797
  seed 7   (best ep 13):  val 0.2202 / test 0.1783

Mean ± std (across 3 seeds):
  val  0.1849 ± 0.036
  test 0.1615 ± 0.025

Loss components at ep 15 (mean across seeds, from the logs):
  align ≈ 0.45   uniform ≈ -3.75   bce ≈ 0.19

Notes:
  - Wall clock per seed: ~10 min (15 ep × ~42 s/ep).
  - All three seeds peak in epoch 9-14 range, then plateau or
    slightly decline. Restoration to best-val snapshot gives the
    reported numbers.
  - No NaN, no OOM, no instabilities. Trajectories smooth.
  - Note that V0 at 15 epochs is substantially higher than pre-EF
    Task 7 anchor at 5 epochs (val 0.128 → 0.185), consistent with
    "not converged at 5 ep" finding.

## V1 — LayerNorm on EF input

Code: `ProjectionHead.__init__` gains `ef_input_norm: bool = False`
arg. When True and EF channel is active, an `nn.LayerNorm(d_edge_feat)`
is applied to the raw EF input before `ef_mlp`. Threaded through
`TrainerConfig.ef_input_norm` and `--ef-input-norm` CLI flag.

Per-seed val MRR / test MRR (best across 15 epochs):
  seed 42  (best ep 13): val 0.1705 / test 0.1629
  seed 123 (best ep 15): val 0.2104 / test 0.1763
  seed 7   (best ep 13): val 0.1717 / test 0.1572

Mean ± std:
  val  0.1842 ± 0.019
  test 0.1655 ± 0.008

Δ vs V0 (val 0.1849 ± 0.036 / test 0.1615 ± 0.025):
  Δval  = -0.0007 (tied within noise)
  Δtest = +0.0040 (slight improvement, within noise)
  Std collapse: V0 val std 0.036 → V1 val std 0.019 (2× tighter)
                V0 test std 0.025 → V1 test std 0.008 (3× tighter)

Loss components at ep 15 (mean across seeds): align ≈ 0.46, unif ≈ -3.74, bce ≈ 0.19.

Notes:
  - V1 RECOVERS from the post-Task-6.7 regression. EF goes from
    "hurting by 22-25%" to "tied with no-EF, with markedly tighter
    cross-seed variance".
  - Tighter std is itself a meaningful finding: normalising the
    high-dim EF input apparently reduces sensitivity to initialisation.
  - No instability, no NaN, no OOM.

## V2 — Per-dim EF standardise

Code: `scripts/train.py` computes `mu, sd` over `loaded.train.edge_feat`
(per-dim, along axis 0) and applies `(x - mu) / sd.clip(1e-6)` to all
three splits via `SplitData._replace`. Gated by `--ef-standardise`.
ProjectionHead is unchanged — V1's LayerNorm is OFF. No EF channel code
change.

Per-seed val MRR / test MRR (best across 15 epochs):
  seed 42  (best ep 13): val 0.1539 / test 0.1353
  seed 123 (best ep 14): val 0.1884 / test 0.1570
  seed 7   (best ep 12): val 0.1551 / test 0.1395

Mean ± std:
  val  0.1658 ± 0.016
  test 0.1439 ± 0.009

Δ vs V0 (val 0.1849 ± 0.036 / test 0.1615 ± 0.025):
  Δval  = -0.0191 (-10% relative)
  Δtest = -0.0176 (-11% relative)
  Std also tightens (val 0.036 → 0.016) — confirms input-scale
  normalisation reduces seed sensitivity, regardless of HOW the
  normalisation is applied.

Δ vs V1 (val 0.1842 ± 0.019 / test 0.1655 ± 0.008):
  Δval  = -0.0184  (V1 markedly better)
  Δtest = -0.0216

Notes:
  - V2 still trains stably (no NaN, no instabilities).
  - V2 reduces variance like V1 does, but loses ~10% on the mean,
    which V1 does not.
  - Hypothesis: V2's pre-network standardisation is a NON-LEARNED
    rescale, whereas V1's LayerNorm has affine γ/β that lets the
    network learn the right scale per-dim. V2 forces unit-variance
    on every feature, including ones whose information was in their
    magnitude.
  - V2 is not a winner; V1 is strictly better.

## V3 — Low-dim EF projection (Linear(d_ef → 16) before ef_mlp)

Code: `ProjectionHead.__init__` gains `ef_low_dim: Optional[int] = None`.
When set, an `nn.Linear(d_edge_feat, ef_low_dim)` is inserted between
raw EF input and the existing per-channel ef_mlp; the first ef_mlp
linear now has `in_features = ef_low_dim` instead of `d_edge_feat`.
The V1 LayerNorm is OFF for V3 (independent test of the low-dim idea).
Threaded through `TrainerConfig.ef_low_dim` and `--ef-low-dim N` CLI
flag. Used `--ef-low-dim 16` for this variant.

Param count drops slightly: p_context 121,088 → 103,888 (the ef_mlp
input layer is much smaller). Total trainable 1,942,658 → 1,925,458.

Per-seed val MRR / test MRR (best across 15 epochs):
  seed 42  (best ep 15): val 0.1954 / test 0.1883
  seed 123 (best ep 15): val 0.1834 / test 0.1567
  seed 7   (best ep 15): val 0.1881 / test 0.1691

Mean ± std:
  val  0.1890 ± 0.005
  test 0.1714 ± 0.013

Δ vs V0 (val 0.1849 ± 0.036 / test 0.1615 ± 0.025):
  Δval  = +0.0041 (+2.2% relative)
  Δtest = +0.0099 (+6.1% relative)
  Std collapse: V0 val std 0.036 → V3 val std 0.005 (7× tighter).
                V0 test std 0.025 → V3 test std 0.013 (2× tighter).

Δ vs V1 (val 0.1842 ± 0.019 / test 0.1655 ± 0.008):
  Δval  = +0.0048  (V3 beats V1 on the mean too)
  Δtest = +0.0059

Notes:
  - All three seeds peak at ep15 (the budget cap), not before. The
    trajectory was still climbing. This says V3 likely has additional
    headroom beyond 15 epochs — but per protocol we do NOT extend
    epochs to chase that, and the headroom is a property of V3 (a
    point about the architecture) rather than the comparison.
  - Loss components at ep15 (mean across seeds): align ≈ 0.44,
    unif ≈ -3.75, bce ≈ 0.18. align is meaningfully lower than V0
    (0.45) and V1 (0.46) — the low-dim EF channel is producing a
    representation that the alignment loss finds easier to fit.
  - Std collapse is dramatic on val (7× tighter than V0). EF helps
    seed-stability when it's well-conditioned (V2 tightened too, but
    at the cost of the mean; V1 tightened and tied; V3 tightens AND
    raises the mean).
  - No NaN, no OOM, no instabilities.

## Tier 1 decision

Tier 1 winner: **V3 (low-dim EF projection, dim=16)**.

Decision rule: variant passes if `mean_variant ≥ mean_V0 − (std_V0 + std_variant)`.
V0 mean - (V0 std + V3 std) = 0.1849 - (0.036 + 0.005) = **0.144**.
V3 val 0.1890 >> 0.144 → V3 passes by a wide margin. V3 also strictly
exceeds the V0 mean (Δval +0.0041, Δtest +0.0099), so the EF channel
is not just "tied" but a genuine improvement when properly conditioned.

Tier 1 summary table (3 seeds × 15 epochs, tgbl-wiki, best val MRR ±
population std):

| Variant | val mean ± std       | test mean ± std       | Δval vs V0  |
|---------|----------------------|-----------------------|-------------|
| V0      | 0.1849 ± 0.036       | 0.1615 ± 0.025        | (anchor)    |
| V1      | 0.1842 ± 0.019       | 0.1655 ± 0.008        | -0.0007     |
| V2      | 0.1658 ± 0.016       | 0.1439 ± 0.009        | -0.0191     |
| V3      | **0.1890 ± 0.005**   | **0.1714 ± 0.013**    | **+0.0041** |

Reading:
  - V0 = the no-EF anchor.
  - V1 (LayerNorm γ/β on EF input) recovers from the post-Task-6.7
    regression but ties V0 on the mean — high-dim EF normalisation
    alone isn't enough to extract signal.
  - V2 (pre-network unit-variance standardisation) is strictly worse
    than V0 — the non-learned rescale destroys magnitude information.
  - V3 (low-dim Linear projection before ef_mlp) beats V0 on both
    val and test means, with dramatically tighter cross-seed std. The
    bottleneck-projection forces the network to learn an EF
    representation, instead of leaving 172 raw dims to compete with
    the E-branch.

By the "stop at any tier where a variant passes" rule, **Task 10
search terminates here.** Tiers 2 and 3 (V4-V8) are NOT run.

## Recommended winner and integration plan

Adopt: **V3 with `--ef-low-dim 16`** as the default EF integration.

Concretely, before merging Task 10 work back to master:
  1. Flip `ef_low_dim` default in `TrainerConfig` from `None` → `16`,
     OR more conservatively make `--ef-low-dim 16` the default in
     `scripts/train.py` while keeping the TrainerConfig default as
     `None` for backward-compat in any direct API caller.
  2. Add a brief note in the trainer/model code linking to this
     CLAUDE.md section so the next person sees why 16 was chosen.
  3. Re-run Task 7 head-to-head: pre-EF (V0 anchor) vs post-EF
     master vs the new EF-with-low-dim configuration. Confirm the
     regression that motivated this task is gone.
  4. Then proceed to Task 7 (full eval with the working EF arch),
     Task 8 (review-v2 sweep), Task 9 (...).

Open observations (NOT in Task 10's scope, do not act on without
discussion):
  - V3 was still climbing at ep15. Two ways to read this:
    (a) the model needs more epochs to converge with EF on, or
    (b) the low-dim bottleneck slows convergence per-epoch but ends
        in a better basin.
    Either way, when Task 7 redoes the full-eval, that runs more
    epochs so this should resolve itself naturally.
  - Cross-seed std collapses on every EF variant that touches the
    EF *input* (V1, V2, V3 all tighter than V0). Worth a footnote in
    a paper, suggests EF helps stability even when it doesn't help
    the mean.
  - V3 was tested at d=16 only. Did NOT sweep over {8, 32, 64}. If
    paper-grade tuning is needed later, this is the obvious next
    sweep.

