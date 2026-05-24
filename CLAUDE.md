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
