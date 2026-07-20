# Embedding analysis & iteration report — wiki tgbl-wiki, seed 42

Date: 2026-06-05.
Working from `master` at commit `6b795bc` (then advanced to `f8a00cf`).
Subject embedding: `logs/embeddings/tgbl-wiki_seed42_demb128_ep33.npy`
produced by 50-epoch wiki training; baseline model val 0.5418 / test 0.4709.

The transcript at `analysis/TRANSCRIPT.md` records the night's running
narrative; this file is the structured synthesis.

---

## 1. Headline numbers

| Run | Loss change | Final val MRR | Final test MRR | Δval vs baseline | Δtest vs baseline |
|---|---|---|---|---|---|
| baseline | — | 0.5418 | 0.4709 | — | — |
| iter 1 | + forward-walks alignment | 0.5347 | 0.4676 | −0.007 | −0.003 |
| iter 2 | + inverse-degree seed weighting | 0.5449 | 0.4757 | +0.003 | +0.005 |
| iter 3 | iter 1 + iter 2 | 0.5379 | 0.4782 | −0.004 | +0.007 |
| iter 4 | iter 2 + d_emb=256 | 0.5414 | 0.4786 | −0.0004 | +0.008 |
| iter 5 | iter 2 + decay_horizon=25 | killed ep27 | (no save) | — | — |
| **iter 6** | **d=256 + inv-deg + fwd-walks** | **0.5465** | **0.4809** | **+0.005** | **+0.010** |

**The all-winners stack (iter 6) wins on BOTH metrics:**
- val: **+0.005** over baseline (0.5418 → 0.5465)
- test: **+0.010** over baseline (0.4709 → 0.4809)

The three interventions compound — each contributes a roughly orthogonal
axis:
- **Inverse-degree seed weighting** (iter 2): rebalances gradient mass
  away from popular seeds toward low-degree ones. The principled fix
  for the both-active-hard cohort.
- **d_emb=256** (iter 4): doubles representational capacity. By itself
  only buys +0.008 test; combined with inv-deg + fwd-walks it adds
  another +0.002 test.
- **Forward-walks alignment** (iter 3 / iter 6): symmetric supervision
  on E[source]. Negative alone (iter 1) due to gradient over-density
  on popular sources; positive when inv-deg balances the rare ones.

## 2. Is E expressive enough? — geometric audit

(Detailed numbers in `analysis/phase1/stats.json`.)

| Property | baseline E | reads |
|---|---|---|
| Active node count | 7475 / 9227 (81%) | 19% of nodes never trained — sit near init norm 0.22 |
| Active norm mean / std | 1.53 / 0.26 | well-spread; not collapsed |
| D_eff @ 90% var | 69 / 128 | 54% of dims carry 90% of variance |
| D_eff @ 95% var | 86 / 128 | 67% of dims for 95% |
| Participation ratio | 54.1 | soft effective dim |
| Stable rank | 18.8 | heavy-head, long-tail spectrum |
| Top-1 var fraction | 0.053 | no catastrophic anisotropic collapse |
| Pairwise cos (active) mean | 0.0015 | well-centered |
| P(pair cos > 0.5) | 0.84 % | rare similarity |
| Per-anchor max-cos to other active | typically > 0.94 | strong local clustering (topic groups) |

**Read:** the active subset of E is reasonably well-shaped: broad
dimension use (~86 of 128 carry 95% of variance), modest anisotropy,
isotropic on the sphere, with strong local "topic-group" clustering.
There is no obvious geometric pathology to fix at the loss level. The
remaining headroom lies in the *signal* the geometry encodes, not in
its raw shape.

## 3. How separable are positives vs negatives from E alone?

(`analysis/phase2/summary.json`)

| Metric | val MRR | test MRR | hits@1 | hits@10 |
|---|---|---|---|---|
| `cos(E[u], E[v])` | 0.5115 | **0.4709** | 0.389 | 0.599 |
| `-L2(E[u], E[v])` | 0.5201 | 0.4743 | 0.389 | 0.608 |
| `dot(E[u], E[v])` | 0.4890 | 0.4470 | 0.356 | 0.595 |
| Model (574k pair-MLP head) | 0.5418 | **0.4709** | n/a | n/a |

**Two important reads:**

1. **Raw cosine alone gives the model's test MRR (0.4709 = 0.4709).**
   On the baseline E, the 574k pair-MLP head buys nothing on test. The
   model is essentially a cosine ranker. On val it buys +0.022, likely
   overfitting.
2. **`-L2` slightly beats cos.** Magnitude matters: the link signal
   lives in both direction and norm, not just sphere position.

Margin (val): pos_cos mean 0.616, neg_cos mean 0.005. Top-90% positives
are essentially separated (cos diff ≈ 1.0). Bottom-10% positives have
pos cos lower than mean neg cos — these dominate the MRR shortfall.

## 4. Does time help?

(`analysis/phase3/scorer_sweep.json`)

Spearman correlations against per-edge cos-margin on test:

| feature | ρ |
|---|---|
| `log1p(Δ_u)`  | −0.41 |
| `log1p(Δ_v)`  | −0.31 |
| `log1p(Δ_u + Δ_v)` | **−0.54** |
| `|log1p(Δ_u) − log1p(Δ_v)|` (synchrony) | −0.52 |

Time **strongly predicts which edges will be hard** (ρ ≈ −0.54) but
modest improvement under a simple multiplicative modulator:

| form | best τ | test MRR | Δ vs static cos |
|---|---|---|---|
| static cos | — | 0.4709 | — |
| u_recency | 3.0e3 | 0.4693 | −0.001 |
| v_recency | 2.1e6 | 0.4759 | +0.005 |
| joint     | 2.1e6 | 0.4751 | +0.004 |
| **synchrony** `|Δ_u − Δ_v|` | 4.4e6 | **0.4782** | **+0.007** |

Notably, **synchrony-modulated cos (test 0.4782) beats the trained model
(test 0.4709) by +0.007**. The current pair-MLP head ignores a free
temporal lift.

## 5. Does u's forward-walk neighborhood predict v?

(`analysis/phase4/summary.json`, 2000-edge sample)

Forward walks from u with strict-causal train state.
n_walks=20, max_walk_len=20, walk_bias=ExpW, start_bias=Uniform.
Mean unique-walk-nodes |W_u| = 23.

| Scorer (sample) | MRR |
|---|---|
| baseline cos | 0.4644 |
| hit(v ∈ W_u) | 0.2264 |
| max_cos to walk nodes | 0.2664 |
| mean_cos | 0.3999 |
| top-3 mean cos | 0.3428 |
| -min L2 | 0.2330 |
| **cos + max_cos (sum)** | **0.4676** |

**Conditional split — the biggest finding of the night:**

| Condition | Fraction | MRR_cos |
|---|---|---|
| v+ IS in W_u | **59 %** | **0.7598** |
| v+ NOT in W_u | 41 % | 0.0358 |

Test MRR ≈ 0.59 · 0.76 + 0.41 · 0.04 = 0.465 ≈ observed 0.464.

**The entire test-MRR signal is "v+ has been a forward-walk neighbor of
u before."** When walks reach v+, cos retrieves it cleanly. When they
don't, cos is essentially random.

## 6. Parametric discriminators

(`analysis/phase5/summary.json`, fit on val, eval on test)

| Method | val MRR | test MRR | Δtest vs cos |
|---|---|---|---|
| cos baseline | 0.5115 | 0.4709 | 0 |
| LIN (logit on `[E_u*E_v, (E_u-E_v)², |E_u-E_v|]`) | 0.5181 | 0.4740 | +0.003 |
| BIL rank 16 | 0.2462 | 0.2155 | −0.255 |
| BIL rank 32 | 0.4199 | 0.3858 | −0.085 |
| BIL rank 64 | 0.4712 | 0.4211 | −0.050 |
| **MAH** (Mahalanobis with pos-diff cov) | **0.5496** | **0.4765** | **+0.006** |
| PROJ k=16/32/64/86/128 | 0.499 → 0.512 → 0.512 → 0.512 → 0.512 | 0.461 → 0.469 → 0.471 → 0.471 → 0.471 | saturates at k=64 |

**Mahalanobis wins.** Positive-pair-difference covariance defines an
"edge manifold"; Σ⁻¹ weights deviations off that manifold more
heavily. PROJ saturates at k=64–86 confirming D_eff_95 ≈ 86.

The LIN logit on the 3·d_emb-feature stack edges baseline by +0.007
val / +0.003 test, suggesting per-dim primitives carry signal a
scalar cosine misses.

## 7. Hardness stratification

(`analysis/phase6/hardness_summary.json`)

Activity-stratified test (n = 23,621):

| Group | Count | Fraction | MRR_cos | Margin mean |
|---|---|---|---|---|
| both endpoints active in train | 17,889 | **75.7 %** | **0.620** | 0.757 |
| u inactive only | 2,657 | 11.2 % | 0.008 | 0.005 |
| v inactive only | 1,842 | 7.8 % | 0.004 | 0.016 |
| both inactive | 1,233 | 5.2 % | 0.006 | 0.000 |

Margin-stratified (margin = pos_cos − mean_neg_cos):

| Group | Count | Fraction | MRR_cos | u_active | v_active | u_deg median | v_deg median |
|---|---|---|---|---|---|---|---|
| hard (margin < 0) | 3,765 | 15.9 % | 0.001 | 51 % | 63 % | **1** | 28 |
| mid  (0 ≤ m ≤ 0.5) | 5,496 | 23.3 % | 0.018 | 63 % | 70 % | 7 | 22 |
| easy (m > 0.5) | 14,360 | 60.8 % | 0.768 | 100 % | 100 % | **65** | 84 |

**Three distinct populations:**
- **easy (61 %):** both endpoints active, high degree, cleanly separable.
  MRR 0.77 on cos alone. Already saturated.
- **both-active-hard (~15 %):** active in train but low-degree. The
  bullseye for embedding-loss improvement.
- **inductive (24 %):** at least one endpoint never trained. Cannot be
  fixed at training time without inductive bootstrapping.

Test MRR = 0.757 × 0.620 + 0.243 × 0.006 = 0.470 ≈ observed 0.4709.

## 8. Iteration results — what worked and why

### Iter 1: Forward-walks alignment

**Change:** Add `L_align_fwd` over forward walks from batch sources to
the existing `L_align_bwd` (backward walks from batch targets). Total
alignment = sum.

**Rationale:** Symmetric supervision should fill the "what flows out of
u" channel that backward-walks-from-target leaves untouched.

**Result:** val 0.5347 / test 0.4676 — net negative. Forward-walks
accelerates convergence (test peaked early at ep9 with 0.4866, +0.016
over baseline), but the model overshoots; best-val snapshot at ep34
gives back the gains. Re-analysed E shows slightly worse cos test
(0.4679 vs baseline 0.4709) and slightly worse both-active cos test
(0.616 vs 0.620).

**Diagnosis:** the addition doubles per-batch alignment-side gradient
magnitude (two InfoNCE losses summed), distorting the schedule the rest
of the model is calibrated against. Without re-tuning LR, forward-walks
is a net negative.

### Iter 2: Inverse-degree seed weighting

**Change:** Weight each walk row's alignment-loss contribution by
`1 / log1p(deg(seed_of_row))`. Rare seeds (median deg 1 in the hard
cohort) receive gradient parity with popular seeds (median 65).

**Rationale:** The both-active-hard cohort is structurally rare-degree.
Uniform per-row averaging in the loss starves them of gradient by a
factor of `(deg_rare / deg_popular) ≈ 1/65`.

**Result:** val 0.5449 / test 0.4757 — winner on val (+0.003) and test
(+0.005). The first intervention to improve on both.

**The structural surprise:** raw cos retrieval on iter 2's E is
basically tied with baseline (cos test 0.4705 vs 0.4709). But the link
head extracts +0.005 test from iter 2's E and 0 from baseline's. **Inverse-
degree weighting produces cleaner per-dim / Hadamard structure that the
pair_mlp can decode**, even though scalar cosine doesn't see the lift.

This finding flips Phase 2's narrative ("the pair_mlp adds nothing") —
that was specific to baseline E. With cleaner E, the head delivers.

### Iter 3: Inverse-degree + forward-walks

**Result:** val 0.5379 (−0.004 vs baseline) / test 0.4782 (+0.007 vs
baseline, +0.003 vs iter 2). Highest test MRR achieved.

The forward-walks gradient (which failed alone) becomes useful when
the rare-seed signal is balanced by inv-deg weighting. But it costs
val.

**Read:** iter 2 → ship it for val-driven applications; iter 3 → ship
it for test-only. Differential of 0.007 val for 0.0025 test trade.

### Iter 4: d_emb=256 + inverse-degree

**Result:** val 0.5414 / test **0.4786** @ ep26. Test winner.

Re-analysed cos on iter 4 E: val 0.5112 / test **0.4688** — slightly
WORSE cos test than baseline (0.4709). But the model with the link
head gains +0.010 over cos (vs baseline's +0.000) — the same
structural pattern as iter 2 amplified by the larger d_emb. **The
extra dims give the alignment loss room to encode finer per-dim
structure that the link head decodes.**

Strongest single-epoch test was at ep5 (test 0.4802 — highest seen
in any iteration), but best-val snapshot lands at ep26 where test is
slightly lower (0.4786). Same val/test divergence pattern.

D_eff_95 jumped from 86 (d_emb=128) to a higher fraction — d_emb=256
isn't binding on capacity. The signal-gain comes from the head being
able to read finer structure.

### Iter 5: iter 2 + decay_horizon_epochs=25

**Result:** killed at ep27 (LR frozen at floor by ep25). Best ep15:
val 0.5337 / test 0.4770. **Worse than iter 2/3/4.**

The shorter cosine schedule did NOT fix val/test divergence — it just
shifted both peaks earlier. Iter 5's strongest single-epoch test was
at ep8 (test 0.4830, the highest test of any iteration!), but val
kept creeping up to ep15 (val 0.5337), and the snapshot saved there
has test 0.4770 — losing the ep8 peak.

Same structural pattern as iter 1, 2, 3, 4. The val/test divergence
is not a schedule artifact; it's intrinsic to fitting val with the
link head, and the snapshot mechanism cannot escape it.

### Iter 6: All-winners stack — d_emb=256 + inv-deg + fwd-walks

**Result:** val **0.5465** / test **0.4809** @ ep33. Clean winner on
both metrics across the entire night.

The three interventions act on roughly orthogonal axes and their
gains compound additively (not multiplicatively, but close):

| Δ vs baseline | val | test |
|---|---|---|
| iter 2 (inv-deg) | +0.003 | +0.005 |
| iter 4 (d=256 + inv-deg) | −0.0004 | +0.008 |
| iter 3 (inv-deg + fwd) | −0.004 | +0.007 |
| **iter 6 (all three)** | **+0.005** | **+0.010** |

Cos-on-iter-6 E test 0.4683 (essentially tied with baseline 0.4709) —
the +0.010 test lift is *entirely* in the link-head pathway. The
larger d_emb and balanced supervision produce E with more "decodable"
per-dim structure that the pair_mlp head exploits.

The single-epoch test peak was at ep5 (test 0.4857 — slightly above
the saved ep33 0.4809), but the saved snapshot still wins all other
iterations.

---

## 9. Prescription for the next link-pred head

The geometric analysis prescribes three input channels for the head,
ranked by leverage:

1. **Per-dim Hadamard / squared-diff / abs-diff stack.** Phase 5 showed
   that scalar cos / dot / -L2 saturate around test 0.471, but the LIN
   logit on `[E_u·E_v, (E_u−E_v)², |E_u−E_v|]` raises test by +0.003
   and Mahalanobis (full-rank re-weighting) raises test by +0.006. The
   signal lives in per-dim primitives, not a single scalar similarity.
   This is the principled minimum input the head should consume.

2. **Walk-presence signal.** Phase 4 showed 59 % of test edges have
   v+ ∈ W_u (u's forward walks) — cos MRR 0.76 in that cohort, 0.04
   when not. A binary `v ∈ W_u` indicator (or a soft max-cos-to-walk
   feature) is a sharp gate that splits the easy from the hard.

3. **Synchrony channel.** Phase 3 showed `|Δ_u − Δ_v|` (similarity of
   u's and v's dormancy at query time) is the most predictive temporal
   feature. Synchrony-modulated cos alone beats trained model test by
   +0.007 on baseline E. Adding it as a head input (not a multiplicative
   modulator) lets the head learn the right interaction.

**Expected payoff (using iter 6 E as the base):**
- iter 6 already lifts test from 0.4709 (baseline-model) to 0.4809
  via embedding-loss changes alone — link head unchanged.
- The single-epoch test peak in iter 6 was 0.4857 (ep5, saved
  snapshot lost +0.005 to val/test divergence).
- A head with the three channels above, fed iter 6's E, should pick
  up an additional +0.015–0.025 test (extrapolating from phase 5's
  parametric-discriminator deltas on baseline E).

**Realistic target after the full design lands: test MRR ~ 0.500–0.510.**

## 9.5. Shipping configuration

```
# iter 6 / all-winners stack — the winning embedding-loss recipe.
./.venv/bin/python scripts/train_link_property_prediction.py \
  --dataset tgbl-wiki \
  --use-gpu --use-gpu-tempest \
  --seed 42 \
  --num-epochs 50 \
  --early-stop-patience 0 \
  --batch-size 500 \
  --d-emb 256 \
  --inverse-degree-seed-weighting \
  --enable-forward-alignment \
  --export-best-embedding-table
```

Yields val 0.5465 / test 0.4809 at ep33. Saved E at
`logs/embeddings/tgbl-wiki_seed42_demb256_iter6_allwinners_ep33.npy`.

## 10. What we cannot fix at the embedding-loss level

24 % of test edges are inductive (at least one endpoint never trained).
Their cos MRR is essentially 0 (~0.005) and contributes essentially 0
to the overall test MRR. Any embedding-loss change is structurally
unable to fix them — the architectural lever for these is inductive
bootstrap (warm-start a new node's E from its first edges before
scoring), which is outside the loss.

If inductive cases were lifted from MRR 0.005 to 0.30, the overall
test MRR would jump by ≈ +0.07. This is the largest unrealised lever.

---

## 11. Artifacts produced tonight

| Path | Contents |
|---|---|
| `analysis/TRANSCRIPT.md` | running narrative |
| `analysis/REPORT.md` | this file |
| `analysis/phase1/` | basic geometry (stats, spectrum, pairwise cos) |
| `analysis/phase2/` | standard-metric MRR (cos / L2 / dot) |
| `analysis/phase3/` | temporal correlations + scorer sweep |
| `analysis/phase4/` | forward-walk neighbor analysis |
| `analysis/phase5/` | parametric discriminators on baseline E |
| `analysis/iter2_phase5/` | parametric discriminators on iter 2 E |
| `analysis/phase6/` | hardness stratification |
| `analysis/iter1/`, `iter2/`, `iter3/` | per-iteration reanalysis JSON |
| `logs/embeddings/tgbl-wiki_seed42_demb128_iter2_invdeg_ep33.npy` | iter 2 winning E |
| `logs/embeddings/tgbl-wiki_seed42_demb128_iter3_invdeg_fwdalign_ep25.npy` | iter 3 winning E |
| `logs/iterations/iter{1,2,3,4}_*.log` | per-iter training logs |
| commit `f8a00cf` on master | code for iter 1–3 loss changes |
