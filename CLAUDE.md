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

_(forthcoming — first run)_

## V1, V2, V3 — Tier 1 (preprocessing only)

_(forthcoming)_

## Tier 1 decision point

_(forthcoming)_
