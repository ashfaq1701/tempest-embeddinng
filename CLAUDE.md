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

## InfoNCE implementation verification

Verified `tempest_walks/losses.py` and `tempest_walks/trainer.py`
against the mathematical specification. All 15 static checks pass.

### Static checks

Check 1 (squared distance via dot product): **PASS**
  `sim_dot = p_seed @ p_ctx_flat.T` (losses.py:81)
  `sq_dist_full = 2.0 - 2.0 * sim_dot` (losses.py:82)
  Constant 2.0 and negative sign correct for unit-sphere outputs.

Check 2 (sim sign and τ position): **PASS**
  `sim = -sq_dist_full / tau` (losses.py:83)
  Negative sign present; τ is divisor.

Check 3 (log Z over ENTIRE pool): **PASS**
  `log_Z = torch.logsumexp(sim, dim=1)  # [NK]` (losses.py:109)
  sim has shape [NK, NK*L]; dim=1 sums over the full pool.

Check 4 (invalid pool entries masked before logsumexp): **PASS**
  `sim = sim.masked_fill(~ctx_valid_mask.unsqueeze(0), INVALID_MASK)`
  with `INVALID_MASK = -1e9` (losses.py:88-89). Path (a).
  Masking happens BEFORE logsumexp.

Check 5 (positive mask = walk_idx AND valid context): **PASS**
  `same_walk = pool_walk_idx.unsqueeze(0) == seed_indices` (losses.py:95)
  `pos_mask = same_walk & ctx_valid_mask.unsqueeze(0)` (losses.py:96)
  Both conditions present.

Check 6 (weights for positives only): **PASS**
  `w_pos = w_flat.unsqueeze(0).expand(NK, -1) * pos_mask.float()`
  (losses.py:106). Non-positives multiplied by 0.

Check 7 (log_p broadcast): **PASS**
  `log_p = sim - log_Z.unsqueeze(1)  # [NK, NK*L]` (losses.py:110)
  log_Z [NK] → [NK, 1] broadcast across pool.

Check 8 (per-seed weighted average + negative sign): **PASS**
  `sum_weighted_log_p = weighted_log_p.sum(dim=1)  # [NK]` (losses.py:114)
  `sum_weights = w_pos.sum(dim=1).clamp_min(1e-6)  # [NK]` (losses.py:115)
  `loss_per_seed = -sum_weighted_log_p / sum_weights` (losses.py:116)
  Per-seed division, negative sign at end.

Check 9 (division-by-zero protection): **PASS, both (a) and (b)**
  (a) `clamp_min(1e-6)` on denominator (losses.py:115).
  (b) Separate mask `has_positives = (w_pos.sum(dim=1) > 1e-9)`
      (losses.py:120) used to exclude 0-positive seeds from the
      final mean.

Check 10 (batch mean over valid seeds only): **PASS**
  `n_valid_seeds = has_positives.float().sum().clamp_min(1.0)`
  (losses.py:121)
  `return (loss_per_seed * has_positives.float()).sum() / n_valid_seeds`
  (losses.py:122). 0-positive seeds contribute 0 to both
  numerator and denominator.

Check 11 (no uniformity in trainer): **PASS**
  Only match in trainer.py is a comment at L224 ("anti-collapse
  mechanism (replaces Wang-Isola uniformity)") — explanatory, not
  a code path. No `uniform_temperature`, `eta_uniform`, `unif_a`,
  `unif_b`, `_sample_uniformity_pairs`, or `uniformity_loss`
  references remain.

Check 12 (TrainerConfig clean): **PASS**
  No `eta_uniform`, `uniform_temperature`, `uniform_pairs`, or
  `uniform_sample_size` fields. Present: `tau: float = 0.5`
  (trainer.py:78).

Check 13 (CLI flags clean): **PASS**
  No `--eta-uniform`, `--uniform-temperature`, or `--uniform-pairs`
  CLI flags in scripts/train.py. Present: `--tau` at line 85.

Check 14 (metrics dict has no "uniform" key): **PASS**
  `return {"align": ..., "bce": ..., "total": ...}` (trainer.py:270-274).
  No "uniform" key.

Check 15 (l_total construction): **PASS**
  `l_total = l_align + l_bce` (trainer.py:258). No `l_table`
  wrapper, no uniformity term.

### Empirical sanity

Mock batch on N=20 seeds × K=5 walks × L=20 (NK=100, pool=2000,
valid contexts=964). Random init with `torch.manual_seed(42)`.

  sim stats (before mask): min=-3.884, max=-3.856, mean=-3.870.
  sq_dist mean: 1.935 (random unit vectors → expected ~2.0).
  log_Z per-seed: mean=3.001, min=2.996, max=3.007.
  log(valid pool) = log(964) = 6.871.

Sanity 1 (log_Z magnitude): the spec expected `log_Z ~ 7-9 at init`.
Observed `log_Z = 3.0`. The spec's expectation assumed `sim ≈ 0`
(cross-head projections aligned). In reality, at random init the
two heads have DIFFERENT bias directions; their cross-head sim
is `~ -d²/τ ~ -2/0.5 = -4`. Then `log_Z ≈ log(M) + mean_sim ≈
6.87 + (-3.87) = 3.0`. The math is internally consistent; the
spec's hand-wavy 7-9 was based on a different init assumption.

This is consistent with the observed wiki ep1 align = 7.98:
  - Wiki pool ≈ 3000 valid → log(3000) ≈ 8.0.
  - sim_pos ≈ -4 → push = log_Z ≈ 8.0 - 4 = 4.0.
  - pull = -sim_pos ≈ +4.0.
  - align ≈ pull + push ≈ 8.0. ✓

Sanity 2 (pull/push decomposition): **EXACT MATCH**
  pull  = 3.8704  (avg of -sim_pos weighted by w)
  push  = 3.0005  (avg of log_Z)
  pull + push = 6.8710
  L_align (from function) = 6.8710
  Match within float epsilon.

This confirms the loss decomposition `L_align = pull + push` is
algebraically exact, and the function output matches the manual
computation.

### Issues found

None. Implementation matches specification on all 15 static checks
and 2 empirical sanity tests.

### Verdict

**Implementation matches specification.** The empirical sanity-check
log_Z magnitude is lower than the spec's hand-wavy expectation
because the spec assumed cross-head bias alignment at init
(unrealistic for fresh random init); the actual magnitude is
algebraically correct and consistent with the observed wiki ep1
loss. Proceeding with Stage A as originally specified.
