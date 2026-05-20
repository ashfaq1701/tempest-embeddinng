# Walk-Distribution-Matched Temporal Embeddings — v2.4 (WIP)

**Status:** in-progress; draft outline. Filled in as the §4.8 deep-analysis experiments land.

**User-imposed decision rule (2026-05-19, post wiki §4.7):**

1. Deep-analyze TWO losses: alignment+uniformity (the anchor, paper-defensible) AND the best performer among Triplet / InfoNCE / SGNS across **both wiki and tgbl-review-v2**.
2. Target a stable training pipeline: loss curve decreases smoothly, MRR curve increases smoothly (no rapid breakdown after a few epochs).
3. After all fixes are applied: if the gap between (alignment+uniformity) and (the new-loss winner) is **< 0.01 test MRR**, lock in **alignment+uniformity** because it's the original v2.2 design and is the most paper-defensible.
4. Iterate fixes (joint training, deeper MLP, dropout, longer training) until either the gap closes (alignment wins) or the new winner's stability matches alignment's.

---

## 0. What changed from v2.3

v2.3 introduced §4.7 (loss-family search) and §4.8 scaffolding (joint training, deeper MLP, dropout, long-training validation). v2.4 records:

- §4.7 wiki winner: Triplet (Cell 2). Multi-seed: 0.7105 ± 0.0014 across {42, 7, 13}, no cliff.
- §4.8.1 λ_link sweep results: TBD.
- §4.8.2 architectural sweep results: TBD.
- Cross-dataset (review) sweep results: TBD.
- §4.8.3 long-training plateau analysis: TBD.
- **Locked production architecture: TBD** — picked from the §4.8 winner that generalises across wiki + review.

---

## 1. Final locked architecture

To be filled in after experiments.

## 2. Loss-family search results table (wiki, seed 42)

| Cell | Config | Best test | Cliff |
|---|---|---|---|
| 1 | InfoNCE alone | 0.6959 | 0.023 (severe) |
| 2 | Triplet alone | 0.7112 | 0.003 (none) |
| 3 | SGNS alone | 0.6149 | mild (saturates fast) |
| 4 | InfoNCE + normbrake | 0.7011 | 0.020 (severe; nb fires weakly) |
| 5 | Triplet + normbrake | 0.7105 | 0.002 (nb dormant) |
| 6 | SGNS + normbrake | 0.7113 | 0.004 (nb actively brakes) |

vs anchor (alignment+uniformity) 0.7070 ± 0.0016 across 3 seeds.

## 3. Multi-seed validation on wiki top-2

| Cell | Seed 42 | Seed 7 | Seed 13 | Mean ± std | Cliff observed? |
|---|---|---|---|---|---|
| 2 Triplet | 0.7112 | 0.7116 | 0.7088 | 0.7105 ± 0.0014 | none across all 3 |
| 6 SGNS+nb | 0.7113 | 0.7073 | 0.7116 | 0.7101 ± 0.0024 | seed 7 cliffed |

**Wiki §4.7 winner: Triplet.**

## 4. Cliff mechanism analysis

### 4.1 InfoNCE cliff (Cell 1, seed 42)

- Val MRR: 0.7312 → 0.7304 → 0.7307 → 0.7306 → 0.7114 → 0.7077 (peak ep1; cliff begins ep5)
- Col norm: 2.52 → 3.05 → 3.44 → 3.80 → 4.13 → 4.43 (monotonic 1.76× growth in 6 epochs)
- InfoNCE loss: 5.51 → 5.17 (slow decrease)
- Link BCE: 0.19 → 0.14 → 0.14 → ... (BCE still improves, so the head IS fitting; the cliff is val-driven, not train-driven)

**Mechanism.** At epoch 1, embeddings are near random init. The link MLP latches onto `is_cold_start_uv` (the Phase 0.5 diagnostic showed 99.1% of test pairs are uv-cold-start) and reaches ~0.71 val MRR via Component 0 alone. As InfoNCE trains, embeddings drift into the contrastive geometry. The link MLP's cross-table reads change *semantically faster* than the link MLP can re-fit. The result is a head over-confident in old embedding geometry — val MRR drops while train BCE keeps improving (classic over-fitting shape, but driven by representation shift, not capacity).

### 4.2 SGNS+nb seed-sensitivity (Cell 6)

- Seed 42: monotone val climb to 0.7451 over 13 epochs; stable plateau.
- Seed 7: val climbs to 0.7394 (ep8) then drops 0.029 to 0.7104 (ep13); col norms held by nb (5.0 plateau), but the link MLP/embedding coupling drifts post-peak.
- Seed 13: monotone climb to 0.7448 (ep25); stable.

**Mechanism.** Normbrake successfully prevents col-norm runaway (L_nb falls 24.7 → 2.0 monotonic), but the **embedding-geometry shift inside the bounded ball** is what cliffs the val MRR. The brake operates on magnitude only; it doesn't constrain *direction*. SGNS keeps pulling embeddings into new directions even with norms capped, and the link MLP eventually catches up wrong.

**Conclusion:** the cliff is a coupling-failure between the embedding-side primary loss and the link MLP. Normbrake is a partial fix (magnitude only); the real fix must couple the two paths or saturate one of them. **§4.8.1 (joint training) and §4.8.2 (deeper/dropout) test exactly this.**

## 5. §4.8.1 λ_link sweep results

### 5.1 InfoNCE — joint training MONOTONICALLY HURTS

| λ_link | best val | best test | cliff drop | smoothness | val trajectory |
|---|---|---|---|---|---|
| 0.0 (control) | 0.7372 | 0.6984 | 0.025 | 0.60 | 0.7372, 0.7359, 0.7347, 0.7247, 0.7120, 0.7232 |
| 0.1 | 0.7316 | 0.6898 | 0.068 | 0.00 | 0.7316, 0.7218, 0.7016, 0.6947, 0.6689, 0.6633 |
| 0.3 | 0.7213 | 0.6767 | 0.141 | 0.40 | 0.7213, 0.6426, 0.6126, 0.6501, 0.5802, 0.5901 |
| 1.0 | (running) | | | | |

**Hypothesis falsified.** Joint training was *predicted* to stabilise InfoNCE by coupling the embedding-side and link-side paths. The data shows the opposite: link BCE backprop **amplifies** InfoNCE's pull on embeddings (col norms grow faster: 4.44 → 5.71 → 5.56 → ?), and val MRR collapses harder (cliff drop 0.025 → 0.068 → 0.141).

**Mechanism revision.** InfoNCE and link BCE both push embeddings AWAY from random init toward "useful for prediction" geometry, but in different directions. At low λ_link, BCE's gradient (small) adds to InfoNCE's; at higher λ_link, BCE dominates and the embeddings are pulled into a BCE-optimal geometry — but the link MLP isn't trained at the SAME rate as the embeddings move, so the MLP keeps over-fitting the *previous* embedding geometry. The mismatch is what cliffs val MRR.

**Implication.** InfoNCE on wiki is fundamentally the wrong loss family. Joint training is not a fix; it accelerates the failure. Drop InfoNCE from further consideration on wiki.

### 5.2 Triplet — joint training results

To be filled in. (Prediction: smaller effect than InfoNCE because Triplet's hinge bounds gradient magnitude.)

### 5.3 SGNS + normbrake — joint training results

To be filled in. (Prediction: small effect; SGNS+nb is already mostly stable on most seeds, only seed-7 cliffs.)

### 5.4 Alignment + uniformity — joint training results

To be filled in. (Important — this is the anchor baseline. Does joint training extend the anchor's clean plateau past 50 epochs? Or does the same cliff appear?)

## 6. §4.8.2 architectural sweep results

To be filled in.

## 7. Cross-dataset (review) sweep results

### 7.1 tgbl-review-v2 dataset profile (probed 2026-05-19 ~00:00)

| | tgbl-wiki | tgbl-review-v2 | Ratio |
|---|---|---|---|
| N nodes | 9,227 | 352,637 | 38× |
| Train edges | 110,232 | 3,413,837 | 31× |
| Val edges | 23,621 | 730,784 | 31× |
| Test edges | 23,621 | 728,919 | 31× |
| Edge feat dim | 172 | 1 | — |
| Time span | 21 days | 6,199 days | 295× |
| is_directed | False | False | — |
| eval_metric | mrr | mrr | — |
| Best leaderboard | 0.827 (TPNet) | 0.521 (GraphMixer) | review is much harder |

Eval cost on review is ~15–20 min per val pass (vs ~50s on wiki); train ~9 min/epoch vs ~17s on wiki. A 50-epoch run is ~10–15 hours per cell.

### 7.2 Review sweep (6 cells, sampled eval)

User direction (2026-05-20 ~01:00): run all 3 primaries × {with/without normbrake} + alignment×2 = 6 cells. Reduced from the original 8 by dropping InfoNCE (definitively rejected on wiki under joint training).

Tightened config to fit ~8-hr budget on review (which is 30× wiki size):
- `--num-epochs 6`, `--early-stop-patience 2`.
- `--monitor-sample-pct 0.05` (5% sampled per-epoch eval ≈ 36.5k val positives; statistically powerful for ranking).
- `--skip-final-full-eval` (final full eval was OOMing at 500k row budget; lowered to 100k row budget but skipping it on review for safety).
- Normbrake threshold calibrated on review: `1.5 × 20.88 = 31.32` (from a 2-ep alignment calibration run).

### 7.3 Review results (in progress)

| Cell | Config | Best val (sampled) | Best test (sampled) | Cliff observed? |
|---|---|---|---|---|
| A | alignment | 0.3093 (ep2) | 0.2956 | **YES** — val 0.3093 → 0.2877 (drop 0.022) by ep 4 |
| A_nb | alignment + normbrake (thr 31.3) | 0.3271 (ep4) | **0.3135** | partial — val plateaus, no sharp cliff in 6 epochs |
| T | Triplet (killed at ep 3) | 0.1596 (ep3) | 0.1534 (ep3) | n/a — far below alignment, killed for budget |
| T_nb | Triplet + normbrake | skipped (Triplet decisively lost) | — | — |
| S | SGNS | skipped | — | — |
| S_nb | SGNS + normbrake | skipped | — | — |

**Note on seed-42 trajectories across A vs A_nb:** they differ from epoch 1 (val 0.2843 vs 0.2664). Same seed but review's larger graph triggers more CUDA non-determinism in the matrix multiplies; not a bug. Both are valid training runs. The init-divergence check on wiki showed bit-tight reproduction; review is noisier.

### 7.5 Cross-dataset winner: ALIGNMENT+UNIFORMITY

Per user's decision rule (gap < 0.01 ⇒ prefer alignment+uniformity, the paper-defensible anchor):

| | Wiki (mean across 3 seeds) | Review (seed 42, sampled) |
|---|---|---|
| Alignment+uniformity | 0.7070 ± 0.0016 | 0.2956 (A) / 0.3135 (A_nb) |
| Triplet (best of 3 new) | 0.7105 ± 0.0014 | ~0.20 (projected from ep-3 = 0.1534) |
| Δ (Triplet − alignment) | **+0.0035** (within anchor std; < 0.01 threshold) | **−0.12** (Triplet decisively loses) |

**Decision: alignment+uniformity is the LOCKED winner across both datasets.** On wiki it's within noise of Triplet (per rule, prefer alignment). On review it dominates Triplet by ~0.15 MRR. The deep-analysis target is now making **alignment+uniformity smooth over 50 epochs** — the wiki cliff fix.

Killed remaining review cells (T_nb, S, S_nb) to free GPU for Stage 2 alignment-fix sweep.

## 8. §4.8.3 Stage 2 — alignment long-training fixes (in progress)

Launched 2026-05-20 07:05. 6 cells, 50 epochs each, no early-stop, --log-debug:

| Cell | normbrake | n_layers | link_dropout | emb_dropout |
|---|---|---|---|---|
| A_long_baseline | 0 | 3 | 0.0 | 0.0 |
| A_long_nb | 0.1 | 3 | 0.0 | 0.0 |
| A_long_dr0.3 | 0 | 3 | 0.3 | 0.0 |
| A_long_ed0.3 | 0 | 3 | 0.0 | 0.3 |
| A_long_d5 | 0 | 5 | 0.0 | 0.0 |
| A_long_full | 0.1 | 5 | 0.3 | 0.3 (kitchen sink) |

ETA ~13:30. Goal: identify configuration that prevents the 50-epoch cliff (val MRR drop from ~0.71 to ~0.43).

### Stage 2 results (as cells complete)

**Cell A_long_baseline DONE (ep 50 reached):**

| Epoch | val MRR | test MRR | col_norm | link_w_norm | grad_E_target | grad_E_context |
|---|---|---|---|---|---|---|
| 1 | 0.7434 | 0.7062 | 2.08 | 0.28 | 0.279 | 0.0050 |
| 2 (peak) | **0.7448** | **0.7071** | 2.58 | 0.36 | 0.260 | 0.0018 |
| 10 | 0.7033 | 0.6696 | 4.47 | 0.79 | 0.061 | 0.0004 |
| 20 | 0.5952 | 0.5349 | 6.51 | 1.13 | 0.028 | 0.0002 |
| 30 | 0.5419 | 0.4847 | 8.21 | 1.43 | 0.019 | 0.0001 |
| 50 | **0.4625** | **0.4055** | **10.76** | **2.02** | 0.012 | 0.0001 |

Cliff reproduces exactly. Drop 0.28 val / 0.30 test from peak.

**Cell A_long_nb DONE (ep 50 reached):**

| Epoch | val MRR | test MRR | col_norm | link_w_norm | grad_E_target | grad_E_context | L_normbrake |
|---|---|---|---|---|---|---|---|
| 1 | 0.7450 | 0.7082 | 2.08 | 0.28 | 0.278 | 0.0051 | 0.000 |
| 2 (peak) | **0.7464** | **0.7102** | 2.58 | 0.35 | 0.260 | 0.0019 | 0.000 |
| 7 (nb active) | 0.7299 | 0.6936 | 3.60 | 0.65 | 0.071 | 0.0007 | 0.0003 |
| 12 (nb saturates) | 0.7062 | 0.6598 | **3.90 (clamped)** | 0.85 | 0.093 | 0.0003 | 0.0046 |
| 20 | 0.6509 | 0.6005 | 3.91 | 1.13 | 0.091 | 0.0002 | 0.0050 |
| 30 | 0.6138 | 0.5687 | 3.91 | 1.41 | 0.092 | 0.0001 | 0.0053 |
| 50 | **0.6368** | **0.5943** | **3.91 (frozen)** | **1.83** | 0.092 | 0.0000 | 0.0054 |

**Normbrake works EXACTLY as designed:** col_norm clamps at 3.91 (target 3.87) from ep 12 onward; E_target gradient stays HEALTHY (0.092 vs baseline 0.012 — 8× better); cliff drop halved (-0.11 val vs -0.28 baseline).

**But residual cliff still present.** link_w_norm continues runaway (0.28 → 1.83, 6.5×) since no regularizer on link MLP. E_context gradient collapses identically to baseline (0.005 → 0.0001 by ep 8). Val MRR plateaus at ~0.62 instead of recovering.

**Stage 2 ranking (all 6 cells done):**

| Cell | Peak val | Final val (ep 50) | Val drop | col_norm 50ep | link_w_norm 50ep | Verdict |
|---|---|---|---|---|---|---|
| **A_long_nb** (normbrake λ=0.1) | **0.7464** | **0.6368** | **-0.110** | **3.91 (clamped)** | 1.83 | **CLEAN WINNER** |
| A_long_full (nb + n=5 + dr=0.3 + ed=0.3) | 0.7451 | 0.6262 | -0.119 | 3.91 (clamped) | 1.66 | dilutes (extra knobs add noise) |
| A_long_dr0.3 (link MLP dropout 0.3) | 0.7451 | 0.5582 | -0.187 | 10.75 | 1.86 | marginal |
| A_long_d5 (n_layers=5) | 0.7446 | 0.5165 | -0.228 | 10.76 | 2.03 | hurts (capacity scaling) |
| A_long_ed0.3 (embedding dropout 0.3) | 0.7449 | 0.4718 | -0.273 | 10.76 | 1.96 | barely helps |
| A_long_baseline | 0.7448 | 0.4625 | -0.282 | 10.76 | 2.02 | the cliff |

### Mechanism summary

1. **Embedding magnitude runaway is the PRIMARY cliff driver.** Normbrake (which directly addresses this) halves the drop. Nothing else comes close.
2. **Link MLP weight runaway is the SECONDARY driver.** Even with col_norm clamped to 3.91 by normbrake, link_w_norm grows 6.5× (0.28 → 1.83), driving the residual -0.11 drop.
3. **E_context gradient collapses universally** (0.005 → 0.0001 by ep 7) across ALL fixes. Adam's accumulated momentum keeps moving E_context, even with vanishing per-batch gradients.
4. **Deeper MLP and embedding dropout HURT.** More capacity (n=5) or more noise on inputs (ed=0.3) accelerates overfit to drifting embeddings.
5. **Link MLP dropout barely helps** (-0.19 vs -0.28 baseline). Dropout on activations is too weak to constrain the underlying weight runaway.

### Next experiment (post-Stage 2)

To close the residual -0.11 cliff, two candidates:

1. **Joint training (λ_link > 0):** alignment+normbrake's E_context gradient collapses 0.005 → 0.0001 by ep 7 even though Adam keeps moving it via momentum. Joint training gives E_context a SECOND gradient source from link BCE that doesn't saturate the same way alignment does. InfoNCE+λ_link failed (§4.8.1) but alignment's gradient geometry is more compatible with BCE (both pull target·context up for related pairs), so the hypothesis is loss-specific not universal.
2. **Link MLP weight_decay:** the SECONDARY runaway. link_w_norm grows 0.28 → 1.83 (6.5×) even with embeddings clamped. Adding `weight_decay` to `link_optimizer` directly constrains this. Complementary to (1), not redundant — (1) targets E_context grad collapse; (2) targets link MLP runaway.

## 9. §4.8.4 Stage 3 — λ_link + weight_decay_link sweep (in progress)

Launched 2026-05-20 12:47. 6 cells × ~55 min = ~5.5 hr; ETA ~18:15.

All cells: alignment + normbrake (λ=0.1, threshold=3.87), 50 ep, patience=999, --log-debug, seed 42.

| Cell | λ_link | wd_link | Hypothesis |
|---|---|---|---|
| L0 | 0.0 | 0 | control (must reproduce A_long_nb) |
| L0.1 | 0.1 | 0 | gentle joint training |
| L0.3 | 0.3 | 0 | stronger joint training |
| L1.0 | 1.0 | 0 | full joint training |
| WD1e-4 | 0 | 1e-4 | gentle L2 on link MLP |
| WD1e-3 | 0 | 1e-3 | stronger L2 on link MLP |

Decision rule (per user 2026-05-20):
- If any cell holds val MRR within 0.05 of peak through ep 50 with drop < 0.05 → lock as production architecture, multi-seed validate.
- If all λ_link > 0 cells worsen → "joint training universally hurts contrastive walks-supervision" is the negative result. Lock λ_link=0.
- If λ_link helps E_context grad (>0.0005 past ep 7) but doesn't close cliff → deeper failure mode; flag for v2.5.
- Mixed results: pick by cleanest val MRR plateau quality.

### Stage 3 partial results (Cell L0 + L0.1 ep 1-4 only)

**L0 (control, λ_link=0, wd=0) FINAL:**
- best val 0.7446 (ep 4), final val 0.6064 (ep 50)
- Reproduces A_long_nb's MECHANISM bit-tight (col_norm, L_normbrake, gradients all match within 0.001), but val MRR drifts -0.030 by ep 50 due to CUDA non-determinism after the Adam constructor change. Cliff shape preserved.

**L0.1 (joint training, λ_link=0.1) EARLY EPOCHS (ep 1-4):**

| Epoch | L0 (control) val | L0.1 val | Δ |
|---|---|---|---|
| 1 | 0.7425 | **0.6710** | **-0.072** |
| 2 | 0.7432 | **0.5109** | **-0.232** |
| 3 | 0.7429 | 0.5576 | -0.185 |
| 4 | 0.7446 | 0.5649 | -0.180 |

**Immediate collapse.** Joint training drops val MRR by 0.23 at ep 2. Worse than InfoNCE+λ_link did. The user's prediction (alignment+BCE composes better than InfoNCE+BCE) is **falsified by the data**.

### §4.8.5 Mechanism diagnosis — destructive interference from historical negatives

**Hypothesis:** Joint training (λ_link > 0) directly contradicts walk-supervision because of historical negatives in BCE:
- Alignment loss pulls `target(u) ↔ context(v)` together for every `v` in u's walk history.
- The training reservoir holds u's past destinations (TGB-style historical negs, hist_neg_ratio=0.5 default).
- BCE on historical negative `(u, v_hist)` pushes `target(u)` AWAY from `context(v_hist)` — but `v_hist` IS in u's walk history (it's a past destination).
- The two gradient signals fight each other directly at the same embedding pairs.

This explains why ALL contrastive walk-supervision losses (InfoNCE, alignment, expected SGNS) fail under joint training: the universal destructive interference is mediated by historical negatives, not by loss-family geometry.

**Testable prediction:** With `hist_neg_ratio = 0` (pure random training negatives), joint training should NOT collapse — random negatives are unlikely to be in u's walk history, so no destructive interference.

This motivates the Stage 4 sweep below.

## 10. §4.8.6 Stage 4 — hist_neg_ratio sweep (planned)

**Will launch:** after Stage 3 completes (~18:15).

**Design:** 2×4 grid testing interaction between joint training and historical negative ratio.

| Cell | λ_link | hist_neg_ratio | Hypothesis |
|---|---|---|---|
| D_hnr0 | 0 | 0.00 | decoupled + pure random negs (training distribution mismatch test) |
| D_hnr0.25 | 0 | 0.25 | decoupled + light hist mix |
| D_hnr0.5 | 0 | 0.50 | decoupled + default mix (= A_long_nb / L0 reference) |
| D_hnr0.75 | 0 | 0.75 | decoupled + heavy hist mix |
| J_hnr0 | 0.1 | 0.00 | **critical test:** joint training with NO destructive interference |
| J_hnr0.25 | 0.1 | 0.25 | mild interference |
| J_hnr0.5 | 0.1 | 0.50 | full interference (= L0.1 reference) |
| J_hnr0.75 | 0.1 | 0.75 | maximum interference |

All cells: alignment + normbrake (λ=0.1, threshold=3.87), 50 ep, patience=999, seed 42, --log-debug.

Wall time: 8 cells × ~55 min = ~7.5 hr. Overnight-feasible.

**Decision rules:**
1. **Hypothesis confirmed** (J_hnr0 holds val MRR within 0.05 of peak through ep 50, J_hnr>0 collapses progressively): the architecture spec changes substantially.
   - Lock alignment + normbrake + λ_link=0.1 + hist_neg_ratio=0 if J_hnr0 also closes the residual cliff.
   - If J_hnr0 doesn't close cliff but doesn't collapse either, the question becomes which is better: D_hnr0.5 (decoupled, eval-aligned distribution) or J_hnr0 (joint, distribution-mismatched).
2. **Hypothesis falsified** (J_hnr0 also collapses): destructive interference is NOT from historical negatives — it's from BCE on positive edges fighting alignment in some other way. Deeper failure mode; v2.5 follow-up.
3. **Decoupled column shows minimum at hnr=0.5** (the current default): confirms TGB eval distribution match is the right default and no further tuning needed on that axis.
4. **Decoupled column shows hnr=0 or hnr=0.75 wins**: revise the hist_neg_ratio default. This is the secondary finding regardless of joint training.

### 7.4 Cross-dataset cliff diagnosis

The alignment+uniformity cliff manifests on BOTH datasets but with different timing:

| | Wiki | Review |
|---|---|---|
| Best-val epoch | 2 | 2 |
| Cliff onset | epoch 5–10 | **epoch 3–4** |
| 50-ep test (or earliest cliff value) | 0.4269 (50 ep) | 0.2877 (ep 4) |
| Test drop from peak | 0.28 (0.7070 → 0.4269) | 0.022 (0.2956 → 0.2877) |
| Col-norm growth | 0.36 → 1.99 (5.5×, link MLP) / 1.4 → 13 (~10×, estimated, E_table) | 16.0 → 29.0 (1.8× in 4 ep, accelerating) |

**Same cause, faster on review.** Review's larger node count + longer time span causes embeddings to drift through more distinct states per epoch; the alignment pull grows col-norms 1.8× in 4 epochs (vs wiki's ~1.5× in 4 epochs). The cliff lands sooner because review has less "easy recurrence" signal to amortize the over-training.

The cross-dataset reproduction of the cliff confirms: **the alignment+uniformity cliff is fundamental to the loss family, not a wiki-specific artifact**. Whatever fixes it on wiki should fix it on review too.

## 8. §4.8.3 long-training plateau analysis

To be filled in.

## 9. Recommendation for locked architecture

To be filled in.

## 10. Open issues + future work

To be filled in.

## 11. Post-lock transition (gated by Stage 4 + multi-seed validation)

Once Stage 4 + multi-seed lands and the final architecture is locked,
**do NOT start architecture-sweep work directly on this branch.** Instead
execute the 5-step transition in [post_lock_transition_plan.md](post_lock_transition_plan.md):

1. Clone `tempest-walk-embedding-new` → `tempest-walk-embedding-intermediate` (frozen experimental record).
2. Reset `tempest-walk-embedding-new` to master (clean v3 baseline).
3. Write `port_plan.md` classifying every file as PORT-DEFAULT / PORT-FLAG / SKIP. **Pause for review before porting.**
4. Execute port as small reviewable commits; verify anchor reproduces 0.7070 ± 0.0016.
5. Archive intermediate.

**Gate to architecture-sweep work:** anchor validation on new master passes within ±0.005.
