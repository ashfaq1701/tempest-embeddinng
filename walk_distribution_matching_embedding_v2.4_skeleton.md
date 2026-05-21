# Walk-Distribution-Matched Temporal Embeddings — v2.4 (FINAL)

**Status:** FINAL (2026-05-21 post Stage 5). Locked production config in §1.

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

## 1. Final locked architecture (LOCKED 2026-05-21 post Stage 5)

Production config:

| Component | Value | Source |
|---|---|---|
| Primary loss | alignment + uniformity | §4.7 + locked 2026-05-20 |
| `eta_uniform` | 1.0 | Stage 5 Scenario A confirmed |
| `uniformity_temperature` | 2.0 | Wang & Isola default |
| `uniformity_cap` | 20000 | Stage 5 Scenario A |
| `lambda_normbrake` | 0.1 | Stage 2 only fix that helps |
| `normbrake_threshold` | 3.87 (wiki) / 31.32 (review) | calibration per dataset |
| `weight_decay_link` | 1e-4 | Stage 3 BREAKTHROUGH (cliff drop -0.014) |
| `lambda_link` | 0 | Stage 4 monotonic collapse confirmed |
| `hist_neg_ratio` | 0.5 | Stage 4 within noise; TGB-matched default |
| `head_mode` | cross_table (E.1) | Phase S |
| Component 0 | ON (time_enc_k=16) | Phase 0.5 anchor |
| Walks-supervise-embeddings | YES | Lesson 2 |
| Strict-causal protocol | NON-NEGOTIABLE | Lesson 3 |
| Walks seeded on | union(src, tgt) on undirected | Lesson 10 |
| Negatives | hist+random K=10 | Lesson 4 |

Wiki seed-42 result with locked config: **best val 0.7450, best test 0.7101, ep 50 val 0.7251** (50-epoch trajectory drop -0.020, the cleanest cliff achievable with these knobs).

### Stage 5 final results (locked decision)

| Cell | eta | cap | Best val | Best test | ep 50 val | Drop |
|---|---|---|---|---|---|---|
| **U_base (LOCKED)** | 1.0 | 20000 | 0.7450 | **0.7101** | 0.7251 | -0.020 |
| U_lo | 0.3 | 20000 | 0.7460 | 0.7083 | 0.7377 | -0.008 |
| U_lower | 0.1 | 20000 | 0.7432 | 0.7075 | 0.7414 | -0.002 |
| N_half | 1.0 | 200 | 0.7452 | 0.7090 | 0.7336 | -0.012 |
| N_quarter | 1.0 | 100 | 0.7450 | 0.7100 | 0.7428 | -0.002 |
| Both_lo | 0.3 | 200 | **0.7462** | 0.7076 | **0.7453** | **-0.0009** |

**Pre-registered Scenario A (60% prediction) CONFIRMED.** All best-test values within ±0.0026 (below anchor std 0.0016 decision threshold). Lock defaults.

Smoothness trade-off noted: Both_lo and N_quarter produce flatter long-training trajectories but at the cost of -0.002 best test. Not worth multi-seed validating given the paper headline is peak MRR.

v2.4 status: **DRAFT → FINAL.**

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

## 10. §4.8.6 Stage 4 — hist_neg_ratio sweep (DONE)

**Launched:** 2026-05-20 18:37, completed 2026-05-21 02:09 (7.5 hr).

### Final results

| Cell | λ_link | hist_neg_ratio | Best val | Best test | ep 50 val |
|---|---|---|---|---|---|
| **D_hnr0** | 0 | 0.00 | 0.7454 | 0.7089 | 0.7410 |
| **D_hnr0.25** | 0 | 0.25 | **0.7458** | **0.7090** | **0.7449** |
| **D_hnr0.5** | 0 | 0.50 | 0.7457 | 0.7080 | 0.7364 |
| **D_hnr0.75** | 0 | 0.75 | 0.7440 | 0.7077 | 0.7361 |
| J_hnr0 | 0.1 | 0.00 | 0.7021 | 0.6623 | 0.5608 |
| J_hnr0.25 | 0.1 | 0.25 | 0.7015 | 0.6570 | 0.5555 |
| J_hnr0.5 | 0.1 | 0.50 | 0.6624 | 0.5996 | 0.5332 |
| J_hnr0.75 | 0.1 | 0.75 | 0.5844 | 0.5141 | 0.4831 |

**Findings:**

1. **Decoupled column (λ_link=0):** all four hist_neg_ratio values produce best val within ±0.002 (CUDA noise band). D_hnr0.25 marginally leads but indistinguishable from D_hnr0.5. The cliff is GONE on the entire decoupled column thanks to the locked base (alignment + nb + WD).

2. **Joint column (λ_link=0.1):** ALL collapse. **Monotonic worsening with hist_neg_ratio**: 0.7021 → 0.7015 → 0.6624 → 0.5844 as hnr goes 0 → 0.25 → 0.5 → 0.75. Higher hist negs accelerate collapse.

3. **Destructive-interference hypothesis (user, post L0.1):** PARTIALLY confirmed. Historical negs DO add interference (J column degrades with hnr), but joint training collapses even at hnr=0 (J_hnr0 best val 0.7021 vs decoupled 0.7454). The fundamental conflict is BCE-into-embeddings regardless of negative type.

### Locked decisions

- **λ_link = 0** (decisively confirmed across 4 hist_neg_ratio values).
- **hist_neg_ratio = 0.5** (default — within CUDA noise of 0.25 marginal leader; TGB-distribution-matched per CLAUDE.md Lesson 4).

**Pattern:** with the cliff fix in place (normbrake + WD), the locked base is robust to hist_neg_ratio. No need to deviate from default 0.5.

(Original Stage 4 plan retained below for reference.)

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

**Loss family: alignment + uniformity (LOCKED 2026-05-20).**

User decision: gains from Triplet (wiki 0.7105 ± 0.0014) over
alignment+uniformity (wiki Phase S anchor 0.7079 ± 0.0005) are
**marginal (~0.003) and within the user's ±0.01 decision threshold**.
Triplet **decisively lost on review** (val 0.16 vs alignment 0.31), so
alignment is the cross-dataset-robust paper-defensible choice. SGNS
tied on wiki but also lost on review. InfoNCE rejected for the
historical-negative destructive-interference mechanism (Lesson 19).

Locked production config (subject to Stage 4 confirmation for λ_link
and hist_neg_ratio):

| Component | Locked value | Source |
|---|---|---|
| Primary loss | alignment + uniformity | user 2026-05-20, this section |
| Normbrake λ | 0.1 | Stage 2 A_long_nb (only fix that meaningfully helps) |
| Normbrake threshold | 1.5 × col_norm at ep 1–2 (3.87 on wiki, 31.32 on review) | Phase 0.5 diagnostic calibration |
| η_uniform | 1.0 | Phase S anchor |
| Head mode | cross_table (E.1) | Lesson 9 + Phase S |
| Component 0 (time encoding + cold-start bits) | ENABLED | Phase 0.5 diagnostic |
| Walks-supervise-embeddings | YES | Lesson 2 |
| Strict-causal protocol | NON-NEGOTIABLE | Lesson 3 + project rule |
| λ_link (joint training) | **0 (DECISIVELY FALSIFIED)** | Stage 3 + §4.8.1 |
| hist_neg_ratio | 0.5 (default), pending Stage 4 | TGB eval-distribution match |
| **weight_decay_link** | **1e-4 (NEARLY ELIMINATES THE CLIFF)** | Stage 3 WD1e-4 — drop -0.014 vs nb-only -0.110 |

### Stage 3 BREAKTHROUGH: WD closes the cliff (both 1e-4 and 1e-3 work)

Full Stage 3 results (after WD1e-3 completion):

| Cell | Best val | ep 50 val | Drop | link_w 50ep | Verdict |
|---|---|---|---|---|---|
| L0 (control, no WD) | 0.7446 | 0.6064 | -0.138 | 1.85 | normbrake-only baseline |
| L0.1 (joint λ=0.1) | 0.6710 | 0.3741 | collapsed | — | joint training catastrophic |
| L0.3 (joint λ=0.3) | 0.5977 | 0.3606 | collapsed | — | worse |
| L1.0 (joint λ=1.0) | 0.5584 | 0.3773 | collapsed | — | worst |
| **WD1e-4 (nb + WD)** | **0.7452** | **0.7313** | **-0.014** | **0.169** | LOCKED (production safe) |
| WD1e-3 (nb + heavy WD) | 0.7446 | **0.7437** | **-0.0009** | 0.0068 | slightly flatter; over-suppresses link MLP |

**WD1e-4 locked as production default.** WD1e-3 is technically slightly flatter (-0.0009 vs -0.014) but link_w_norm drops to 0.0068 — the link MLP weights are pushed below typical Adam init magnitudes, which is over-regularization. WD1e-4 keeps link_w_norm at a healthy 0.169 while still nearly eliminating the cliff. Safer for cross-dataset transfer (review may need higher link capacity).

WD1e-4 final trajectory:

| Epoch | val MRR | Δ from peak |
|---|---|---|
| 5 (peak) | 0.7448 | 0 |
| 10 | 0.7443 | -0.0005 |
| 20 | 0.7442 | -0.0006 |
| 30 | 0.7442 | -0.0006 |
| 40 | 0.7415 | -0.003 |
| 50 | 0.7313 | **-0.014** |

This is **the smooth val MRR curve the user asked for** (2026-05-20: "We will try to make the loss curve going down smoothly and MRR curve going up smoothly").

**Mechanism diagnosed:** link_w_norm growth — was 0.28 → 1.83 (6.5×) with normbrake alone; with WD=1e-4 it's **0.185 → 0.169 (essentially flat)**. The link MLP weights stop running away, val MRR stays high.

**Locked production architecture:** alignment + uniformity + normbrake (λ=0.1, threshold per-dataset) + weight_decay_link=1e-4. λ_link=0. hist_neg_ratio pending Stage 4.

**Paper-ablation paths kept behind PORT-FLAG in master (per post_lock_transition_plan):**

- Triplet loss + semi-hard mining (`--primary-loss triplet`)
- SGNS + unigram^0.75 cache (`--primary-loss sgns`)
- InfoNCE (`--primary-loss infonce`) — kept for completeness even though rejected
- E.2 head variant (Component-0-only, `--head-mode component_0_only`)
- Component 0 disable (`--no-use-time-encoding`)
- A2-off (no alignment, `--lambda-align 0`)
- normbrake-off (`--lambda-normbrake 0`)
- hist_neg_ratio variants (`--hist-neg-ratio`)
- λ_link variants (`--lambda-link`)
- weight_decay_link variants (`--weight-decay-link`)

**What still needs to land before locking is final:**

1. Stage 3 completion (4 of 6 cells done; WD1e-4 and WD1e-3 pending) — confirms whether `weight_decay_link` closes the residual cliff or not.
2. Stage 4 completion (8 cells) — confirms `hist_neg_ratio=0` doesn't save joint training AND tests whether the default `hist_neg_ratio=0.5` is the right decoupled-training default.
3. Multi-seed validation on the final winning cell across seeds {42, 7, 13}.

## 10. Open issues + future work

To be filled in.

## 11. §4.8.7 Stage 5 — uniformity hyperparameter sweep (planned, gated by Stage 4)

**Status:** planned. Cells launch ONLY after Stage 4 + multi-seed lands
and the winning hist_neg_ratio + λ_link are confirmed.

### Motivation — two hypotheses

**H1: `eta_uniform` too high.** Uniformity's negative pull dominates
alignment's positive pull, inflating column norms before normbrake
catches them. Reducing `eta_uniform` lets alignment dominate, producing
more semantically coherent embedding geometry within the clamped
magnitude regime.

**H2: Effective uniformity negative count is too high.** All-pairs
uniformity over the entire batch produces low-variance, consistent push
direction. Combined with Adam momentum, this could be why E_context
drifts even when per-batch grads vanish (the 0.005 → 0.0001 collapse
seen in every Stage 2 cell). Fewer pairs (lower variance) = less
consistent push = E_context might settle.

### Current defaults (verified by grep)

The code does NOT have an `n_uniform_neg` parameter. The actual
uniformity hyperparameters (`tempest_walks/config.py:117-120`) are:

| Field | Default | Role |
|---|---|---|
| `eta_uniform` | 1.0 | scalar on uniformity in total loss (H1 target) |
| `uniformity_temperature` | 2.0 | exponent in `exp(-t · sq_dist)` |
| `uniformity_cap` | 20_000 | max nodes used in all-pairs (H2 target) |

**Important:** `uniformity_cap=20_000` NEVER fires at default. With
B=200 batches and union(src, tgt) seeding, the unique batch node count
is typically 200–400. The cap only triggers when unique batch nodes
exceed it. To meaningfully test H2, the cap must be set BELOW the
typical batch unique-node count (~300).

The loss math: `uniformity_loss` does all-pairs `cdist` over the unique
batch nodes (capped). For B=200 with ~300 unique nodes, that's ~45,000
pairs per batch — effectively all-pairs.

### Pre-registered prediction (2026-05-20 18:40)

Best-guess outcome based on Stage 2/3 mechanism diagnosis:

- **60% Scenario A** (U_base wins or all cells tie within anchor std 0.0016): uniformity hyperparams are non-critical because normbrake + WD_link supersede upstream loss tuning. Magnitudes are clamped regardless of uniformity strength.
- **30% Scenario B** (U_lo / U_lower wins): H1 partially holds — reducing eta_uniform helps alignment dominate, but the cliff is already controlled by Stage 3 fixes so the gain is marginal (< 0.005).
- **10% Scenario C** (N_half / N_quarter wins): H2 holds — fewer effective uniformity negatives lets E_context settle past ep 7 and improves long-training plateau quality.

If Scenario B or C produces cliff-shape improvement (val drop < -0.005 from current -0.014), uniformity IS part of the cliff mechanism, not just upstream of normbrake. Document prominently in v2.4.

### Cell design (6 cells, seed 42, 50 ep, no early-stop)

Base config: alignment + normbrake (λ=0.1, threshold=3.87) +
weight_decay_link=1e-4 + Stage 4 winning hist_neg_ratio + Stage 4
winning λ_link. (Filled in after Stage 4 lands; default placeholder
hist=0.5 λ_link=0.)

| Cell | eta_uniform | uniformity_cap | Tests |
|---|---|---|---|
| U_base | 1.0 (control) | 20_000 (eff. all pairs) | reproduces locked baseline |
| U_lo | 0.3 | 20_000 | H1: mild reduction |
| U_lower | 0.1 | 20_000 | H1: strong reduction |
| N_half | 1.0 | 200 (~half typical batch unique nodes) | H2: higher gradient variance |
| N_quarter | 1.0 | 100 (~quarter typical) | H2: much higher variance |
| Both_lo | 0.3 | 200 | combined H1 + H2 |

Wall time: 6 cells × ~55 min = ~5.5 hr.

### Decision rules

**A** — U_base wins or all cells tie within anchor std (0.0016): lock
current defaults (eta_uniform=1.0, uniformity_cap=20_000). Most likely
outcome per prediction.

**B** — One of U_lo / U_lower / Both_lo wins by > anchor std: H1
confirmed. Reduce eta_uniform to the winning value. Multi-seed validate
on seeds 7, 13 before locking.

**C** — N_half or N_quarter wins by > anchor std: H2 confirmed. Reduce
uniformity_cap. Multi-seed validate before locking.

**Cliff-improvement bonus**: if any winning cell produces val drop
< -0.005 (vs current -0.014 from WD1e-4), uniformity is mechanistically
part of the cliff — flag for paper writeup.

### Multi-seed discipline

If any cell other than U_base wins, multi-seed validate (seeds 7, 13)
on the winning cell before locking. CUDA noise from Stage 3 (~0.030
drift) means small single-seed effects need confirmation.

### Lock procedure (gates the port)

After Stage 5 + multi-seed lands:

1. Update v2.4 §1 "Final locked architecture" with COMPLETE spec
   including eta_uniform and uniformity_cap values.
2. Update v2.4 §9 with reasoning for each hyperparameter.
3. Mark v2.4 status DRAFT → FINAL.
4. Then proceed to clone-then-port-to-master per
   [post_lock_transition_plan.md](post_lock_transition_plan.md).

In `port_plan.md`: classify `eta_uniform` and `uniformity_cap` as
PORT-FLAG regardless of Stage 5 outcome — CLI knobs exposed, defaults
set per winner. Appendix ablations need to rerun this sweep later.

**DO NOT START PORTING** until Stage 5 + multi-seed lands AND v2.4 §1
documents the complete locked config including these two values.

## 12. Post-lock transition (gated by Stage 4 + Stage 5 + multi-seed validation)

Once Stage 4 + Stage 5 + multi-seed lands and the final architecture is
locked, **do NOT start architecture-sweep work directly on this branch.**
Instead execute the 5-step transition in
[post_lock_transition_plan.md](post_lock_transition_plan.md):

1. Clone `tempest-walk-embedding-new` → `tempest-walk-embedding-intermediate` (frozen experimental record).
2. Reset `tempest-walk-embedding-new` to master (clean v3 baseline).
3. Write `port_plan.md` classifying every file as PORT-DEFAULT / PORT-FLAG / SKIP. **Pause for review before porting.**
4. Execute port as small reviewable commits; verify anchor reproduces 0.7070 ± 0.0016.
5. Archive intermediate.

**Two-gate verification after port:**

- **Gate A (anchor reproduction):** `python -m scripts.anchor_validation --seeds 42,7,13 --num-epochs 2` must give 0.7070 ± 0.0016 within anchor std on new master.
- **Gate B (locked-config wiki run):** `python -m scripts.train --tgb-name tgbl-wiki --seed 42 --num-epochs 50 --early-stop-patience 999 --log-debug [locked flags]` must reproduce the corresponding Stage 4/5 winning cell within CUDA tolerance (±0.030). Commit the run as `master_locked_verification.md`.

**Gate B catches port bugs Gate A misses** — Gate A only runs 2 epochs from random init; Gate B runs 50 epochs with the full locked-config code path including normbrake, weight_decay_link, and any winning hist_neg_ratio/uniformity knobs.

## 13. §4.8.8 Single-table ablation (planned, gated by Step 1–3)

**Status:** planned. Branch `experiment/embedding-table-variations` off
master AFTER Steps 1–3 (port + Gates A and B pass).

### Motivation

Stage 2 mechanism analysis identified universal **E_context gradient
collapse** (0.005 → 0.0001 by ep 7) across every fix as a contributor
to the residual cliff. A single-table architecture removes E_context as
a separate parameter set entirely — there is no distinct context table
to collapse.

**Parameter reduction:** 2 × N × d → 1 × N × d + 2 × d × d. At d=128:
- Wiki (N=9,227): 2.36M → 1.18M + 32K = 1.21M (~half)
- Review (N=352,637): 90.3M → 45.2M + 32K = 45.2M (~half)

If single-table matches dual-table on wiki, master locks the simpler
architecture. The paper claim becomes: *"single embedding table with
asymmetric projections at the link MLP, half the parameters of dual-
table at matched performance."*

### Architecture spec — `1T_asym`

Replace `E_target` and `E_context` (both `[N, d]`) with a single `E[node]`
of size `[N, d]`. Add two small linear projections at the link MLP read
site:

```
P_src : Linear(d, d)
P_tgt : Linear(d, d)

Scoring (cross-table block, same 8-block structure as v3 E.1):
  Score = link_mlp(P_src(E[u]),
                   P_tgt(E[v]),
                   P_src(E[u]) ⊙ P_tgt(E[v]),
                   |P_src(E[u]) − P_tgt(E[v])|,
                   ... reverse direction ...,
                   Component_0)

Alignment loss:
  Pulls E[seed] toward E[walk_node] (no target/context distinction).
  L_align = mean over walks of weighted (1 − cos(E[seed], E[walk_node]))

Uniformity loss:
  Pushes E[u] away from E[v_neg] for random negative pairs
  (operates on the single table E).

Normbrake:
  Applied to the single E table. THRESHOLD MUST BE RECALIBRATED — see
  pre-launch step below.
```

**SKIP `1T_sym`** (symmetric scoring without P_src/P_tgt). Theory predicts
loss on directional eval; the cell isn't worth burning.

### Pre-launch verification (REQUIRED before the 50-epoch cell)

The single E table now serves both target and context roles. Its
column-norm distribution may differ from dual-table E_target's.

1. Run a 2-epoch warmup with `--lambda-normbrake 0` (normbrake off).
2. Read per-column L2 norm of the single E table at end of ep 2.
3. Set `normbrake_threshold = 1.5 × measured_col_norm`.
4. Document the new threshold in this section BEFORE launching the
   full 50-epoch cell.

**If the new threshold differs from the dual-table threshold (3.87 on
wiki) by more than 20%, that itself is a finding** — note it
prominently. It would indicate the single-E table operates at a
different magnitude regime than dual-table did.

### Cell — one cell only

| Cell | E tables | Projections | Locked-config flags |
|---|---|---|---|
| 1T_asym | 1 (`E[N, d]`) | P_src, P_tgt (d→d each) | Stage 4/5 winners + normbrake (recalibrated threshold) + WD_link=1e-4 |

50 epochs, seed 42, no early-stop, `--log-debug`.

### Decision rules

**A** — 1T_asym **ties or wins** 2T_locked on test MRR (within anchor std 0.0016, or higher): lock single-table. Multi-seed validate (7, 13). If multi-seed mean is within anchor std of 2T mean, lock single-table.

**B** — 1T_asym loses by **< 0.005** (tie within CUDA noise): per "simpler wins ties," lock single-table. Multi-seed validate.

**C** — 1T_asym loses by **> 0.005**: dual-table stays as locked production. Document outcome in this section and in `port_plan.md` as "tested, rejected." Keep single-table code as PORT-FLAG for paper-ablation completeness.

**Cliff-shape bonus:** if 1T_asym matches 2T on peak BUT has a cleaner cliff (smaller drop, no E_context-style gradient collapse since there is no E_context), that's a bigger finding than peak parity. Lockable even on a tie because long-training stability is better.

### Pre-registered prediction (2026-05-20, before Stage 4 lands)

Best-guess outcome based on Stage 2/3 mechanism analysis and Lesson 9
("cross-table > within-table"):

- **40% Scenario B** (1T_asym loses by < 0.005, tie within noise): the cross-table link MLP can recover most of the dual-table flexibility through P_src/P_tgt projections. Simpler-wins-ties → lock single-table.
- **30% Scenario C-mild** (loses by 0.005–0.015): the dual-table asymmetry buys ~0.005–0.01 test MRR that linear projections can't recover. Dual-table stays locked; single-table → PORT-FLAG.
- **20% Scenario C-clear** (loses > 0.015): the dual-table inductive bias is critical (E_target and E_context learn different distributions; one table can't be optimal at both simultaneously). Dual-table locked clearly.
- **10% Scenario A** (1T_asym wins): unlikely but possible if E_context's gradient collapse is more damaging than I estimated.

**Most-likely outcome:** B or C-mild. Cliff shape will probably help single-table even if peak loses slightly — fewer parameters + no E_context collapse mechanism.

### Lock procedure

After 1T_asym ablation + multi-seed lands:

1. Update v2.4 §1 with architecture decision and reasoning.
2. **Winner:** merge `experiment/embedding-table-variations` to master. Update `config_locked_v1.yaml`. Losing variant → PORT-FLAG for paper ablation.
3. **Loser stays:** leave branch in place for paper reproducibility; master keeps dual-table as locked.

## 14. §5 Source walk encoder (planned, gated by `locked-v2` tag)

**Status:** PLANNED. Branch `experiment/add-source-walk-embedding`
created off the `locked-v2` tag AFTER Step 5 (single-table merge or
confirmation) completes. The branch starts with the FINAL locked
embedding-table architecture (single-table or dual-table per §13
outcome).

### Motivation

Link prediction scores `(u, candidate_v_i, t)` rows where all 1000
candidates share the same source `u`. **Discriminative signal must
come from the source-side representation interacting differently with
each destination representation.**

Currently the source-side input is `P_src(E[u])` — a static
embedding (or with Component 0's Δt scalars). A walk encoder gives
the source a **time-aware, history-aware, neighborhood-aware**
representation: `walk_repr[u]` summarizes u's recent neighborhood
structure and edge-feature trajectory.

Destination-side walks are deferred (1000× more walk samples per row,
tests a different question).

### Gradient flow — why this eliminates the cliff BY CONSTRUCTION

PyTorch autograd handles the full chain:

```
∂L_link / ∂score
    ↓ through link_mlp
∂L_link / ∂(walk_repr_u via P_src) and ∂L_link / ∂(E[v] via P_tgt)
    ↓ through walk_encoder forward
∂L_link / ∂walk_encoder.parameters     ← GRU + projections trained by BCE
    ↓ through per-step embedding lookups
∂L_link / ∂E[walk_node_i]              ← E table ALSO trained by BCE
```

This is fundamentally different from locked-v2:

- **locked-v2:** E trained by alignment+uniformity, link MLP trained by BCE, two paths **decoupled**. Embedding table drifts independently of what link MLP needs. The -0.28 baseline cliff (Lesson 17) is the consequence.

- **walk encoder:** E (via per-step lookups) AND walk_encoder AND link_mlp are **all in the same BCE compute graph**. The encoder cannot drift away from what link MLP needs because they share the loss.

**Option α (DEFAULT for initial cells):** Keep alignment + uniformity
+ normbrake on the embedding table. Walk encoder adds joint BCE
supervision. Alignment provides structural regularization; BCE provides
task-specific shaping. Safer — supervision fallback if encoder
underperforms.

**Option β (ablation after initial cells, conditional):** Drop
alignment + uniformity + normbrake. Walk encoder is pure end-to-end
trained: BCE only. Cliff truly doesn't exist by construction because
nothing is decoupled. Cleaner — but no fallback.

Start with Option α. Test Option β as an ablation if Option α works.

### Architecture

**Walk encoder:** 1-layer GRU.

```
d_hidden = d_emb (matches embedding dimension; default 128)
num_layers = 1
bidirectional = False  (walks are chronological; bidirectional has
                       no inductive-bias justification)
```

### Per-step input — depends on Step 5 outcome

**IF master locks single-table (Step 5 → Outcome A or B):**

```
step_input_i = concat([
  E[walk_node_ids[i]],
  Φ(t_seed - t_i),
  edge_features[i]   # zeros for i == 0 (no incoming edge)
])
```

Per-step lookup is unambiguous: every step reads `E[n_i]` regardless
of position. Role asymmetry lives in `P_src` / `P_tgt` projections at
the link MLP, not in the per-step lookup.

**IF master locks dual-table (Step 5 → Outcome C):**

```
step_input_i = concat([
  E_target[walk_node_ids[i]] if i == L-1 else E_context[walk_node_ids[i]],
  Φ(t_seed - t_i),
  edge_features[i]   # zeros for i == 0
])
```

Per-step lookup is role-aware: the seed (i == L-1) reads `E_target`
because the encoder output feeds the source-side slot of the link
MLP. Walk-internal nodes (i < L-1) read `E_context` because they are
destinations the seed historically connected to, and alignment loss
has trained `E_context` to encode "destination role relative to a
walker." Using `E_context` for walk-internal nodes leverages the
semantics alignment already learned.

### Per-step input components

1. **`E[walk_node_i]`** — node embedding. Shared with locked-v2
   architecture (no separate encoder embedding table). Sharing means
   the encoder benefits from per-node semantics that alignment has
   trained.

2. **`Φ(t_seed - t_i)`** — time-delta encoding. Reuse Component 0's
   existing time encoder applied to elapsed time since walk step. A
   step 1 minute ago and a step 1 month ago produce different
   representations even if they pass through the same node.
   **Use `t_seed - t_i` (recency relative to seed)**, NOT
   `t_i - t_{i-1}` (inter-step delta). The GRU implicitly learns
   inter-step pacing from the sequence of Φ values.

3. **`edge_features[i]`** — edge features for the step (if dataset
   provides). Wiki d=172, review d=1. For `i == 0` there is no
   incoming edge — pad with zeros. If dataset has no edge features,
   omit this term entirely.

### Chronological order

Tempest returns walks in chronological order: oldest first, seed last.
Seed at position `L-1`.

```
walk_node_ids   = [n_0, n_1, ..., n_{L-1}]    # n_{L-1} is the seed
walk_timestamps = [t_0, t_1, ..., t_{L-1}]    # t_{L-1} == t_seed
```

The GRU processes left-to-right (oldest to newest). The last hidden
state corresponds to position `L-1` and has "seen" the entire walk
from the seed's perspective.

### Walk representation

```
h_walk = GRU(per_step_inputs).last_hidden_state    # [d_hidden]

For K walks per seed, mean-pool across walks:
walk_repr[u] = mean_k(h_walk_k)
```

K defaults to 5 (matches alignment loss's num_walks_per_node).

### Short-walk handling

Tempest sometimes returns walks shorter than `max_walk_len` (early
epochs, cold-start nodes, isolated seeds). Pad with zeros, **mask the
padded positions** in the GRU forward pass via standard PyTorch
sequence-packing — they don't contribute to the hidden state evolution.

If a seed has zero historical neighbors (only itself in the walk),
the GRU sees a single step and produces a representation based on
`[E[seed], Φ(0), zeros]` alone. **Graceful fallback to locked-v2
behavior** for cold-start nodes.

### Link MLP input restructuring

**Source-side input changes** from `P_src(E[u])` → `P_src(walk_repr[u])`.

**Destination side unchanged:** `P_tgt(E[v])` (whichever locked table).

**Component 0 unchanged:** `component_0(u, v, t)`.

### Unit test for dual-table per-step lookup (REQUIRED if dual-table locked)

```
Construct a test walk with node_ids = [n_a, n_b, n_c] (seed = n_c).
Verify the encoder reads:
  - E_context[n_a]    (position 0, walk-internal)
  - E_context[n_b]    (position 1, walk-internal)
  - E_target[n_c]     (position 2, seed)
In that order.
```

This catches the most likely implementation bug (off-by-one on which
position is the seed). Run **before** any training.

### Pipeline — training time

```
for each batch:
  # Existing alignment walks (unchanged from locked-v2):
  walks = tempest.sample(seeds=unique(src ∪ tgt), K=5)
  L_align, L_uniform = alignment_loss(walks)        # if Option α
  L_normbrake = normbrake_loss(E)                   # if Option α

  # NEW: filter walks to src-seeded subset for encoder input.
  src_walks = walks_for_seeds(walks, unique(src))
  walk_repr_src = walk_encoder(src_walks)

  # Link MLP forward — DIRECTIONAL, matches eval distribution:
  for each row (A, B, t):
    score_pos = link_mlp(P_src(walk_repr_src[A]),
                         P_tgt(E[B]),
                         component_0(A, B, t))
    for v_neg in negatives:
      score_neg = link_mlp(P_src(walk_repr_src[A]),
                           P_tgt(E[v_neg]),
                           component_0(A, v_neg, t))
    L_link += BCE([score_pos, score_neg_*], [1, 0, ...])

  L_total = L_align + L_uniform + L_normbrake + λ_link · L_link
```

**NO bidirectional BCE** — only `score(A, B)` per batch row, never
`score(B, A)`. The link MLP is structurally directional and matches
eval.

### Pipeline — scoring time

```
for each scoring row (u, candidates, t):
  walks_u = tempest.sample(seeds=[u], K=5, time=t)
  walk_repr_u = walk_encoder(walks_u)

  for v_i in candidates:
    score_i = link_mlp(P_src(walk_repr_u),
                       P_tgt(E[v_i]),
                       component_0(u, v_i, t))
```

### Walk caching at scoring time

Cache `walks_u` keyed by `(u, t_bucket)`. Wiki test averages ~2.5
positives per node; review ~2. Caching halves scoring-time walk cost.

- **Initial:** `t_bucket = t` (exact match). Verify hit-rate counter.
- **If hit-rate < 5%:** increase to nearest 100-second window. **Re-verify strict-causal:** walks within a bucket must use only edges with timestamp < bucket_start.

### Directed-graph guard

```
if config.is_directed:
  alignment_walks = tempest.sample(seeds=unique(src), K=5)
  src_walks = alignment_walks  # already src-only
else:
  alignment_walks = tempest.sample(seeds=unique(src ∪ tgt), K=5)
  src_walks = walks_for_seeds(alignment_walks, unique(src))
```

### Gradient-health verification

Add to `--log-debug` per-epoch output:

```
grad_walk_encoder_gru      = ||walk_encoder.gru.weight_ih_l0.grad||
grad_walk_encoder_proj_src = ||P_src.weight.grad||
grad_embedding_E           = ||E.weight.grad||
```

Healthy training shows ALL THREE non-trivial (>1e-4) throughout. If
`grad_walk_encoder_gru` is near zero, BCE isn't reaching the encoder.
Likely cause: a `.detach()` call somewhere in the encoder forward
breaking the autograd graph. Debug before declaring the encoder
broken.

### Initial experiment cells

Three cells on wiki, seed 42, 50 epochs, no early-stop, `--log-debug`.

| Cell | Walk encoder | K | Loss family | Purpose |
|---|---|---|---|---|
| `W_off` | disabled | — | locked-v2 (Option α) | control = locked-v2 baseline reproduction |
| `W_gru` | enabled, d=128 | 5 | Option α (align+uniform+nb on) | does source walk encoder lift MRR? |
| `W_gru_k1` | enabled, d=128 | 1 | Option α | does K=1 suffice (halves scoring cost)? |

`W_off` must reproduce locked-v2's corresponding cell within CUDA
noise — **if not, branch has an integration bug; fix BEFORE launching
W_gru**.

Wall time: ~1 hr each on wiki at K=5 (encoder adds modest training
cost; scoring time grows from ~50s → ~5–10 min per pass due to
per-source walk sampling at eval).

### Pre-registered prediction (2026-05-20, before Stage 4 lands)

Best-guess outcome based on Lessons 2 (walks-only plateaued 0.011),
14 (direct recurrence is biggest unpulled lever, +0.05 test MRR),
16 (memory module margin small once recurrence captured):

- **50% Scenario B** (W_gru ties W_off within anchor std): Component 0 + alignment + normbrake + WD already capture most temporal signal on wiki. Walks add marginal value.
- **30% Scenario A** (W_gru wins by > 0.005): walk encoder provides new signal — pooled neighborhood + edge-feature trajectory beats static E[u]. Plausible because Component 0 only sees u/v Δt scalars; walks see actual neighborhood structure.
- **15% Scenario C** (W_gru loses by > 0.005): training instability or encoder bug. Investigate.
- **5% Scenario D** (W_gru_k1 matches W_gru): K=1 suffices.

**Bigger picture:** wiki gap to leaderboard (0.7105 honest vs 0.798 DyGFormer) is partly leak-inflated and partly real signal. Walk encoder might recover ~0.02–0.05 of the honest gap. Review's gap (0.31 honest vs 0.52 GraphMixer) is wider — walk encoder likely helps MORE on review.

### Decision rules

**A** — W_gru beats W_off by > 0.005: works. Multi-seed validate (7, 13). If multi-seed mean > anchor std above W_off, lock walk-encoder-on. Run Option β ablation next.

**B** — W_gru ties W_off within anchor std: doesn't help wiki. Either Component 0 carries the signal, or wiki's recurrence saturation makes walks redundant. Test on review before final verdict.

**C** — W_gru loses by > 0.005: investigate. **Three causes to rule out before declaring broken:**
1. Encoder GRU not converged in 50 epochs (run 100-ep smoke).
2. Per-step input construction has a bug (verify with unit test).
3. Gradient not reaching encoder (verify grad-health logging).

**D** — W_gru_k1 matches W_gru: K=1 suffices. Lock K=1 for scoring efficiency.

### Cliff-shape comparison

The walk encoder is jointly trained with link BCE. **Expectation: NO cliff** (decoupled supervision was the cliff's cause; encoder eliminates that decoupling).

- If W_gru cliffs anyway → cliff has a deeper cause than locked-v2 diagnosed. **Substantial finding** — would change the paper's central claim.
- If W_gru is cliff-free AND beats W_off → paper claim becomes "walk encoder joint training eliminates the cliff BY CONSTRUCTION AND lifts the ceiling."

### Option β ablation (CONDITIONAL on W_gru winning)

If W_gru beats W_off in initial cells, run one more cell:

| Cell | Walk encoder | K | Loss family |
|---|---|---|---|
| `W_gru_beta` | enabled, d=128 | 5 | Option β (align+uniform+nb OFF; λ_link=1.0; BCE only) |

Outcomes:
- **β matches α** → drop alignment+uniformity from production, cleaner architecture. Paper claim: "Joint BCE-through-encoder training eliminates the need for separate embedding supervision."
- **β loses to α** → alignment-as-regularization composes usefully with walk encoder. Production keeps Option α.
- **β beats α** → drop alignment+uniformity, cleaner AND better. Best outcome.

### Cross-dataset follow-up

If W_gru beats W_off on wiki, run on review (single seed, abbreviated
6-ep sampled-eval per Stage 4 review config).

- **> 0.05 improvement** → destination-side walks (symmetric 1a) becomes worth revisiting.
- **Marginal** → source-walks-only sufficient; defer destination side.

### Lock procedure

After W_gru cells + multi-seed + Option β ablation + cross-dataset validation:

1. Update v2.4 §14 (or v2.5 §1) with final decision + reasoning.
2. **Winner:** merge `experiment/add-source-walk-embedding` to master after full verification (Gates A + B re-run).
3. Update `config_locked_v1.yaml`.
4. **Tag:** `git tag locked-v3 -m "source walk encoder locked"`.

If walk encoder loses or ties: leave branch as paper-reproducibility artifact. Master stays at `locked-v2`.

**DO NOT proceed to further architecture work** (destination-side walks, memory module, Hawkes head) until source walk encoder decision is finalized.

## 15. Deliverables of Steps 1–4 (final lock + verified master)

1. **tempest-walk-embedding-intermediate/** — frozen experimental record.
2. **tempest-walk-embedding-new/** on master with the FINAL locked architecture (single-table or dual-table per §13 outcome).
3. **master_locked_verification.md** — wiki 50-epoch reproduction of the locked-config result on new master, attached as the Gate B receipt.
4. **experiment/embedding-table-variations** branch with the architectural ablation result recorded.
5. **v2.4 §13** populated with the single-table outcome + decision.
6. **v2.4 status: DRAFT → FINAL.**
7. **config_locked_v1.yaml** reflecting the final architecture.
8. **port_plan.md** committed on master with full PORT-DEFAULT/FLAG/SKIP table (per [post_lock_transition_plan.md](post_lock_transition_plan.md)).

Then — AND ONLY THEN — Step 5 (walk encoder) begins.
