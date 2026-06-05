# Overnight transcript — embedding analysis & iteration

Started 2026-06-05 ~00:10. Working directory `analysis/`.
Baseline E: `logs/embeddings/tgbl-wiki_seed42_demb128_ep33.npy`
(from master `7f87f01..6b795bc`-era 50-epoch wiki run).

## Baseline reference numbers (model that produced E)

- best_val_mrr: **0.5418** @ epoch 33
- best_test_mrr: **0.4709** @ epoch 33

## Phase 1 — basic geometry  (analysis/phase1/stats.json)

```
shape (9227, 128) float32

L2 norms
    all      mean 1.280  std 0.562  range [0.180, 2.822]
    active   mean 1.528  std 0.260      (7475 / 9227 nodes)
    inactive mean 0.225  std 0.014      (1752 / 9227 nodes — 19% never trained)

Centered SVD on active rows
    D_eff @ 90% var:  69
    D_eff @ 95% var:  86
    D_eff @ 99% var: 113
    participation_ratio  : 54.11
    stable_rank          : 18.80      (heavy-head, long-tail spectrum)
    anisotropy (var[0]/mean(var)): 6.81
    top-1 var fraction   : 0.0532
    top-10 cumulative    : 0.3193

Pairwise cosine (200k random active pairs)
    mean   0.0015     median  −0.0167
    P10   −0.1466     P90      0.1616
    P(cos > 0.5) 0.84%      P(cos > 0.8) 0.19%      P(cos < 0) 55.7%

Per-anchor sample (10 random active u):
    max cos to any other active node = [0.95, 0.97, 0.996, 0.996,
                                        0.80, 0.99, 0.97, 0.998,
                                        0.998, 0.24]
```

**Reads:**
1. **19% dead nodes.** ~1750 nodes never train; their E sits near the init magnitude (norm ~0.22). Every val/test edge touching one of them is, by construction, an inductive case.
2. **E uses dimensions broadly.** Soft effective dim ~54, hard cutoff at 86 / 128 for 95% of variance. No catastrophic anisotropic collapse but a modest 6.8× top-direction prominence.
3. **Active subset is isotropic on the sphere.** Random pair cos ≈ 0; tight tails (±0.15 at the 10/90 percentiles). Pure white-noise embeddings would also do this; the *interesting* structure comes from local clustering — 9 of 10 sampled anchors have a near-duplicate in the active set (cos > 0.94, several at 0.998). E has discovered "topic groups."

## Phase 2 — standard-metric separability  (analysis/phase2/summary.json)

| Metric | val MRR | test MRR | hits@1 (test) | hits@10 (test) |
|---|---|---|---|---|
| `cos(E[u], E[v])` | 0.5115 | **0.4709** | 0.389 | 0.599 |
| `-L2(E[u], E[v])` | 0.5201 | 0.4743 | 0.389 | 0.608 |
| `dot(E[u], E[v])` | 0.4890 | 0.4470 | 0.356 | 0.595 |
| **Model (bilinear + 574k-param pair_mlp)** | 0.5418 | **0.4709** | n/a | n/a |

Margin (val): pos_cos mean 0.616, neg_cos mean 0.005, pos−neg mean 0.611 (std 0.436).
- P10 of pos−neg = −0.035 — bottom 10% of positives are LESS similar to v+ than to mean v-.
- P90 = 0.998 — top 10% are near-perfect separation.

**Reads:**
1. **`cos(E[u], E[v])` alone gets exactly the model's test MRR (0.4709).** The 574k pair-MLP head contributes literally nothing on test. On val it adds +0.022, but that's likely val-side overfitting.
2. **`-L2(u, v)` beats both cos and dot.** Magnitude is informative — the link signal lives in both direction AND norm.
3. There is a hard 10–15% tail where positives are no closer to u than negatives — these dominate the MRR shortfall.

## Phase 3 — does time help?  (analysis/phase3/scorer_sweep.json)

Spearman correlation of `log1p(Δ)` against `pos − neg cos margin` on test:

| feature | ρ |
|---|---|
| log1p(Δ_u) | **−0.41** |
| log1p(Δ_v) | −0.31 |
| log1p(Δ_u + Δ_v) | **−0.54** |
| \|log1p(Δ_u) − log1p(Δ_v)\| | −0.52 |

τ-swept multiplicative modulator `cos × exp(−Δ/τ)`:

| form | best τ | val MRR | test MRR | Δval vs static |
|---|---|---|---|---|
| static cos | — | 0.5115 | 0.4709 | — |
| u_recency | 3.0e3 | 0.5109 | 0.4693 | **−0.001** |
| v_recency | 2.1e6 | 0.5200 | 0.4759 | +0.008 |
| joint | 2.1e6 | 0.5194 | 0.4751 | +0.008 |
| **synchrony** `|Δ_u − Δ_v|` | 4.4e6 | **0.5201** | **0.4782** | **+0.009** |

**Reads:**
1. **Time strongly predicts hardness** (ρ ≈ −0.54 on joint Δ) but only weakly *recovers* it via simple multiplicative modulation (+0.007 test best).
2. **Synchrony — being active at similar times — is the winning form.** Beats u-side or v-side individually.
3. **Synchrony-modulated cos (test 0.4782) actually beats the model (test 0.4709) by +0.007.** The link head has thrown away a free temporal lift.

## Phase 4 — does u's forward walk neighborhood predict v?  (analysis/phase4/summary.json)

Sampled 2000 test edges. Forward walks from u with strict-causal train state.
n_walks=20, max_walk_len=20, walk_bias=ExpW, start_bias=Uniform.
Mean |W_u| (unique walk nodes) = 23.

| Scorer | MRR (sample) |
|---|---|
| baseline cos | 0.4644 |
| hit(v ∈ W_u) | 0.2264 |
| max_cos(E[v], E[W_u]) | 0.2664 |
| mean_cos | 0.3999 |
| top3_mean_cos | 0.3428 |
| -min_l2 | 0.2330 |
| cos + max_cos (sum) | **0.4676** |

**Conditional split — the biggest finding of the night:**

| Condition | Fraction | MRR_cos |
|---|---|---|
| v+ IS in W_u (u's forward walks reach v+) | **59.2%** | **0.7598** |
| v+ NOT in W_u | 40.8% | 0.0358 |

Test MRR decomposes:  0.592 · 0.760 + 0.408 · 0.036 = **0.465** ≈ observed **0.4644** ✓

**Reads:**
1. **Almost all of the test-MRR signal is "v+ has been a forward-walk-neighbor of u before."** When the walks reach v+, cos retrieves it brilliantly. When they don't, cos is essentially random.
2. **Standalone walk-mediated scorers underperform cos** — the walks add coverage but the scoring still flows through cos. The combined `cos + max_cos_to_walks` only edges baseline by +0.003.
3. **The 40% "v not in walks" cohort is the ceiling.** Pure walk-and-similarity machinery can't rescue them without new information — they're either (a) inductive (u or v unseen) or (b) genuine novel pairs.

## Phase 5 — parametric discriminators  (analysis/phase5/summary.json)

Fit on val (positives + per-edge TGB negs), evaluated on test (no leakage):

| Method | val MRR | test MRR | Δtest vs cos |
|---|---|---|---|
| cos baseline | 0.5115 | 0.4709 | 0 |
| LIN (logit on `[E_u*E_v, (E_u-E_v)², |E_u-E_v|]`) | 0.5181 | 0.4740 | +0.003 |
| BIL rank 16 | 0.2462 | 0.2155 | −0.255 |
| BIL rank 32 | 0.4199 | 0.3858 | −0.085 |
| BIL rank 64 | 0.4712 | 0.4211 | −0.050 |
| **MAH** (Mahalanobis with pos-diff cov) | **0.5496** | **0.4765** | **+0.006** |
| PROJ k=16 | 0.4988 | 0.4607 | −0.010 |
| PROJ k=32 | 0.5109 | 0.4685 | −0.002 |
| PROJ k=64 | 0.5118 | 0.4709 | 0 |
| PROJ k=86 | 0.5122 | 0.4714 | +0.001 |
| PROJ k=128 | 0.5115 | 0.4709 | 0 |

**Reads:**
1. **Mahalanobis is the best parametric metric**: +0.038 val / +0.006 test. The pos-diff covariance defines an "edge manifold"; directions positives share get downweighted in the distance, so deviations hurt more on the off-manifold axes.
2. **Low-rank bilinear UNDER-performs** — it strips structure E uses. Full rank or learned re-weighting wins.
3. **PCA k=64 already saturates** — the bottom ~64 components carry no link-prediction signal. Confirms phase 1's D_eff_95=86.
4. The full Hadamard / squared-diff / absolute-diff feature stack (LIN) gets +0.007 val / +0.003 test — modest, but encouraging that *combinations* of per-dim primitives carry signal beyond a single scalar.

## Phase 6 — hardness stratification  (analysis/phase6/hardness_summary.json)

Activity-stratified test (n=23,621):

| Group | Count | Fraction | MRR_cos | Margin mean |
|---|---|---|---|---|
| both endpoints active in train | 17,889 | **75.7%** | **0.620** | 0.757 |
| u inactive only | 2,657 | 11.2% | 0.008 | 0.005 |
| v inactive only | 1,842 | 7.8% | 0.004 | 0.016 |
| both inactive | 1,233 | 5.2% | 0.006 | 0.000 |

Margin-stratified (margin = pos_cos − mean_neg_cos):

| Group | Count | Fraction | MRR_cos | u_active | v_active | u_deg med | v_deg med |
|---|---|---|---|---|---|---|---|
| hard (margin < 0) | 3,765 | 15.9% | **0.001** | 51% | 63% | **1** | 28 |
| mid (0 ≤ m ≤ 0.5) | 5,496 | 23.3% | 0.018 | 63% | 70% | 7 | 22 |
| easy (m > 0.5) | 14,360 | 60.8% | **0.768** | 100% | 100% | **65** | 84 |

**Reads:**
1. **24% of test edges are structurally inductive** (at least one endpoint unseen in train). Their cos MRR is essentially 0 → they contribute 0.001 to the overall test MRR.
2. **The both-active hard group (≈15% of total) is the bullseye for embedding improvement.** These are low-degree nodes (u-median 1, v-median 28) that DID train but couldn't get enough gradient because they appear in few walks. Improving their geometry from MRR 0 to MRR 0.5 would lift test by ~+0.075.
3. **Test MRR = 0.757 × 0.620 + 0.243 × 0.006 = 0.470 ≈ observed 0.4709.** Algebraically exact: the model's score is "both-active cos performance," nothing else.

## Quantified improvement headroom

| Scenario | Hard-cohort MRR | Inductive-cohort MRR | Predicted total test MRR |
|---|---|---|---|
| current | 0.10 (mid+hard avg) | 0.006 | 0.471 |
| fix hard-cohort to easy level (0.62) | 0.62 | 0.006 | **0.562** (+0.09) |
| fix inductive-cohort to 0.30 | 0.10 | 0.30 | **0.546** (+0.08) |
| both | 0.62 | 0.30 | **0.679** (+0.21) |

This sets concrete targets for the iterations below.

---

## Iteration plan

The two distinct populations need different mechanisms:

### Hard-but-both-active (15% of test)
- Diagnosis: low-degree nodes get few backward-walk positives → undertrained.
- Fix candidates:
  1. **Forward-walks supervision on E**: also train E to be close to its *successors*, not only its predecessors. Doubles the gradient density on source-side embeddings. Implemented as: add `L_align(walks_fwd_from_sources)` to total loss.
  2. **Inverse-degree seed weighting** in alignment loss.
  3. **Hard-negative mining** in the InfoNCE pool — push E[low-degree] away from look-alike-but-not-real negatives.

### Inductive (24% of test)
- Diagnosis: u or v never seen in train → E[unseen] sits at init scale.
- Fix candidates:
  1. **Inductive bootstrap**: at val/test time, ingest the first edge involving the unseen node into a quick E update before scoring. Architecturally invasive.
  2. **Mean-of-neighbors init**: when a new node appears, set E[new] ← mean(E[its first-edge-partners]). Cheap.
  3. **Don't try**: accept the ceiling at ~0.56 (no inductive lift) and ship.

Iter 1 picks Forward-walks supervision — the cheapest principled change with biggest expected lift on the hard-but-both-active population.

---

## Iter 1 — forward-walks alignment supervision

**Change:** Add `L_align_fwd` over forward walks from batch sources to the existing
`L_align_bwd` (backward walks from batch targets). Total alignment loss = sum.

**Implementation:**
- `losses.py:alignment_loss(direction="backward"|"forward")` — direction-aware
  is_context mask, K_hop (lens-1-p vs p), and t_seed_edge (max vs min of valid).
  Backward and forward sentinels differ (INT64_MAX at lens-1 vs INT64_MIN at 0).
- `trainer.py`: when `enable_forward_alignment`, sample fwd walks from
  `unique(batch.src)` via `walks_for_nodes_link_pred`; add forward-direction
  loss to total.
- `--enable-forward-alignment` CLI flag.

**Result:**

| | Baseline | Iter 1 |
|---|---|---|
| Model val MRR | 0.5418 @ ep33 | 0.5347 @ ep34 |
| Model test MRR | 0.4709 @ ep33 | 0.4676 @ ep34 |
| Cos val (re-analysis on saved E) | 0.5115 | 0.5148 |
| Cos test (re-analysis on saved E) | 0.4709 | 0.4679 |
| both_active cos test | 0.620 | 0.616 |

Training trajectory:
- ep1: val 0.4388 (vs baseline 0.3811, +0.058) — fast warm start
- ep9: val 0.5318 / test **0.4866** — best-test peak (+0.016 over baseline final)
- ep20: val 0.5346 / test 0.4683 — best-val moves up, test drops
- ep34: val 0.5347 / test 0.4676 (saved snapshot)

**Read:** Forward-walks accelerates convergence (test peak at ep9) but
also accelerates val/test divergence. The best-val snapshot at ep34
misses the ep9 test peak. Net: E slightly worse on cos test (−0.003)
and both-active cos test (−0.004). **Not adopted.**

## Iter 2 — inverse-degree seed weighting

**Change:** Weight each walk row's contribution in `alignment_loss` by
`1 / log1p(deg(seed))`. Rare seeds (median 1 train-edge incidence in the
hard cohort) get gradient parity with popular seeds (median 65 in easy cohort).

**Implementation:**
- `losses.py:alignment_loss(seed_weights=None)` — optional `[N]` tensor;
  per-row weights broadcast across K walks/seed; final aggregation is a
  weighted mean over valid rows.
- `trainer.py`: precompute `train_deg` from `loaded.train` (undirected
  incidence count). Per-batch: `seed_weights = 1/log1p(train_deg[seeds_np])`.
- `--inverse-degree-seed-weighting` CLI flag.

**Result:**

| | Baseline | Iter 1 | **Iter 2** |
|---|---|---|---|
| Model val MRR | 0.5418 | 0.5347 | **0.5449** |
| Model test MRR | 0.4709 | 0.4676 | **0.4757** |
| Δval vs baseline | — | −0.007 | **+0.003** |
| Δtest vs baseline | — | −0.003 | **+0.005** |
| Cos val (saved E) | 0.5115 | 0.5148 | 0.5132 |
| Cos test (saved E) | **0.4709** | 0.4679 | **0.4705** |
| both_active cos test | 0.620 | 0.616 | 0.619 |

Training trajectory:
- ep26: val 0.5436 / test **0.4793** (test peak)
- ep33: val **0.5449** / test 0.4757 (saved best-val snapshot)

**Read — the surprising structural finding:**
- Iter 2's E and baseline's E give nearly identical raw cosine MRR
  on test (0.4705 vs 0.4709 — within noise).
- But Iter 2's E + bilinear-pair-MLP link head delivers +0.005 test MRR
  over baseline (0.4757 vs 0.4709).
- The link head extracted +0.005 from Iter 2's E but **0** from baseline's E.

→ Inverse-degree weighting produced an E with cleaner *per-dim* / Hadamard-
   space structure that the pair_mlp can decode, even though scalar cosine
   doesn't see the lift. The 574k pair_mlp finally earns its keep on Iter 2's E.

→ This complicates Phase 2's "raw cos == model test MRR" conclusion: that
   identity held for baseline's E but does NOT generalise — it's a property
   of how cleanly the gradient mass was distributed during training.

**Adopted.** Iter 2 is the new best E and the baseline for subsequent iterations.

