# Iter 2 design — inverse-degree seed weighting

## Hypothesis

Phase-6 analysis showed the both-active hard cohort (~15% of test) has
median u-degree=1 vs easy=65. These are low-activity training-time nodes
that DID train (so E is updated) but received very few gradient updates
(because they appeared in very few walks).

InfoNCE per-row loss currently aggregates as a uniform mean over walk
rows. Each walk row contributes equally regardless of which seed it
samples. Popular seeds appear in many rows; rare seeds in few. Over the
epoch, this means popular seeds receive O(deg) gradient updates while
rare seeds receive O(1).

## Fix

Weight each walk row's loss contribution by `1 / log1p(deg(seed_of_row))`.
This makes the per-epoch gradient mass on each seed roughly:

    gradient_mass(seed) ∝ (# rows with that seed) / log1p(deg(seed))
                       ≈ K * count(seed in batch.tgt) / log1p(deg(seed))

If we sample seeds proportional to batch incidence and weight by inverse
degree, rare seeds with frequent batch incidence get amplified, and
common seeds with frequent batch incidence get damped toward parity.

Math: the gradient of L_align w.r.t. E[seed] is roughly:

    ∇_{E[seed]} L = α(seed) · sum_over_walk_positions of (positive_pull - negative_push)

Currently α(seed) = 1/n_rows (uniform). Proposed: α(seed) = 1/log1p(deg) /
sum_seeds 1/log1p(deg(seed)). This makes E[low-degree-seed] receive
gradient at a level matching E[high-degree-seed] in absolute terms.

## Implementation

`alignment_loss(... seed_degrees=None)`:
- If `seed_degrees` provided (one int per row), compute `w_seed(i) = 1 / log1p(seed_degrees[i])`.
- Final aggregation: `Σ_i w_seed(i) · L_i · valid(i) / Σ_i w_seed(i) · valid(i)` (weighted mean).

`trainer.py`:
- Maintain a frozen `train_deg` array (per node).
- In `_train_step`, for each walk row, look up the seed's degree.
- Pass `seed_degrees` to alignment_loss.

`train_deg`: count of how many training edges incident on each node, computed once at trainer init from the train split.

## Combinations to test (after iter 1 lands)

- iter 2a: inverse-degree only (no forward-walks)
- iter 2b: forward-walks + inverse-degree (combined)
- pick whichever wins val on first epoch, run to 50ep

## Expected outcome

The both-active hard cohort has E that is currently undertrained because
rare seeds got few updates. Inverse-degree weighting boosts them. If iter
2 lifts the hard cohort's MRR from current ~0.10 to even ~0.30, that's
+0.045 to total test MRR.
