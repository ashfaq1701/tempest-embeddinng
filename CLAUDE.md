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
