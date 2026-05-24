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

