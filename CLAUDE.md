# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. Currently being
rebuilt from first principles. The prior architecture and its 26+ lessons
are preserved on branch `backup/important-walk-embedding` for reference.
This file will be repopulated as the new design stabilises.

---

# Task 15 — Post-cleanup baseline characterisation

Branch: `experiments/post-cleanup-baseline` (from master, clean C1)
Started: 2026-05-24

Resumes the original Task 7 → 8 → 9 pipeline on the post-cleanup
master. Tasks 10-14 (EF integration exploration) are complete and
closed; master no longer contains EF in the loss or scoring path.

Three stages, gated sequentially:
  Stage 7  — anchor baseline (3 seeds × 5 ep, wiki + review)
  Stage 8  — convergence + hyperparameter sweep
  Stage 9  — walk-bias sweep on best stack

Target reference: leaderboard wiki MRR is 0.82.

User override: GPU for model (--use-gpu), CPU for Tempest (no
--use-gpu-tempest). Time budget waived — run stages 7-9 in sequence
unless catastrophic.

## Stage 7 — Anchor baseline

### Wiki anchor (3 seeds × 5 ep)

Per-seed val MRR / test MRR (best across 5 epochs):
  seed 42  (best ep 5): val 0.2861 / test 0.2458
  seed 123 (best ep 5): val 0.2474 / test 0.1934
  seed 7   (best ep 4): val 0.2535 / test 0.2183

Mean ± std:
  val  0.2623 ± 0.017
  test 0.2192 ± 0.021

Loss components at ep 5 (mean across seeds):
  align ≈ 0.56   unif ≈ -7.40   bce ≈ 0.18

Notes:
  - Wiki anchor numbers are much higher than the original pre-fix
    expectation of 0.05-0.10 at ep5. Consistent with Task 12 post-fix
    C1 trajectory (seed 42 was val ~0.29 at ep5 in the C1 SUM run).
  - Two-head SUM uniformity gives unif ~-7.4 (vs pre-fix single-head
    ~-3.7). E table spreads faster, MRR climbs faster.
  - No NaN, no collapse, smooth climb across all 3 seeds.

### Review anchor (3 seeds × 5 ep) — in progress
