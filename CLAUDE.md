# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. Currently being
rebuilt from first principles. The prior architecture and its 26+ lessons
are preserved on branch `backup/important-walk-embedding` for reference.
This file will be repopulated as the new design stabilises.

---

# Task 14 — EF use patterns without scoring-path leakage

Branch: `experiments/ef-auxiliary` (from master)
Started: 2026-05-24

Two EF integration patterns tested. Both put EF in the training
signal only; neither lets EF reach LinkHead. Eval scoring uses
E alone, so pos/neg asymmetry doesn't bite.

  14a — EF modulates alignment weight w(K, Δt) via tanh modulator.
  14b — EF as auxiliary regularisation loss on E (predict EF from
        endpoint embeddings, MSE).

Anchor: C1 (no EF anywhere) from Task 12 post-fix, val 0.3966 ± 0.014.

Two uniformity fixes from `experiments/ef-symmetry` are ported
forward (without per-head EF gating, C5, or any Variant 4 code):
  - `bypass_ef` parameter in `ProjectionHead.forward` (Option γ).
  - Two-head uniformity with SUM (not /2 average).
