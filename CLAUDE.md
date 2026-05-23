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
