# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. Currently being
rebuilt from first principles. The prior architecture and its 26+ lessons
are preserved on branch `backup/important-walk-embedding` for reference.
This file will be repopulated as the new design stabilises.

---

# Loss variation — InfoNCE contrastive alignment

Branch: `loss_variation/infonce` (from master, clean C1)
Started: 2026-05-24

Replaces regression alignment + Wang-Isola uniformity with single
InfoNCE contrastive alignment. Drops uniformity entirely; InfoNCE's
softmax denominator does the anti-collapse work using task-relevant
in-batch negatives.

This branch contains the architectural change only. Experiments run
on `feature/infonce-experiments` which branches from this.

---

# InfoNCE experiments

Branch: `feature/infonce-experiments` (from loss_variation/infonce)
Started: 2026-05-24

Two experiments:
  Stage A — Wiki 30-ep × 3 seeds at tau=0.5. Does wiki perform at
            C1's best or better under InfoNCE? (C1 reference val
            0.3966 ± 0.014.)
  Stage B — Comment 5-ep × 3 seeds at tau=0.5. Does the model
            train stably on a 44M-edge dataset where the old
            architecture collapsed (review was 25K batches/ep;
            comment is ~120K batches/ep)?

If Stage A degrades wiki by more than 20%, or Stage B shows the
same collapse pattern, the architecture needs rethinking before
committing to InfoNCE as the default.

## Stage A — Wiki 3 seeds × 30 ep, tau=0.5

Per-seed peak val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 22): val 0.4924 / test 0.4651
  seed 123 (best ep 25): val 0.4934 / test 0.4722
  seed 7   (best ep 23): val 0.4852 / test 0.4636

Mean ± std:
  val  0.4903 ± 0.004
  test 0.4670 ± 0.004

Compare to Task 12 post-fix C1 (regression alignment + uniformity):
  C1 val 0.3966 ± 0.014 / test 0.3794 ± 0.015 (30 ep, 3 seeds).

Δ vs C1: Δval = +0.094 (+24% relative), Δtest = +0.088 (+23%).

Std collapse: val 0.014 → 0.004 (3.5× tighter). InfoNCE training
is dramatically more reproducible across seeds than C1.

Trajectory shape: all three seeds peak in ep22-25 range, then
val noise/overfit drifts the value down slightly. Best-val
snapshotting captures the peak. Loss decomposition (from
verification): pull (= -mean sim_pos) and push (= mean log_Z)
both decrease monotonically from ep1 to ep30; align went from
7.98 → 6.08 across the trajectory.

Stage A decision: **PASS** (val 0.49 ≫ 0.32 threshold). InfoNCE
wins wiki by +24% on val and +23% on test with 3.5× tighter std.
Proceeding to Stage B.

## Stage B — Review 3 seeds × 5 ep, tau=0.5

Dataset swapped from comment to review for Stage B. Review (~5M
edges, ~25K batches/epoch) is the dataset where C1 + two-head SUM
uniformity demonstrably collapsed (Task 15 Stage 7 review anchor
locked at val 0.0196 across all 5 epochs with align=unif=0).

Expectation: InfoNCE should NOT collapse on review like the old
alignment+uniformity formulation did. The softmax denominator
provides anti-collapse via task-relevant in-batch negatives at
every batch — it doesn't depend on the seed-vs-context bias
asymmetry that the old uniformity term needed to stabilise.

