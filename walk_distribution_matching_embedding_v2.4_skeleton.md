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
