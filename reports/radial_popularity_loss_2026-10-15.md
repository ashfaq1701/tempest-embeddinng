# Radial popularity loss: popularity into the score through geometry only

**Branch:** `feature/radial-popularity-loss` (branched from `master` = `02f4ce4`)
**Status:** design + measurements complete, **no code written**. This document is the spec.
**Date:** 2026-10-15
**Author of the measurements:** server lane (artemis, RTX PRO 6000 Blackwell Server Edition)

---

## 0. One-paragraph summary

Popularity is the largest single untapped signal in this model: adding a causal
popularity term to master's frozen scores post-hoc is worth **+0.1529 test MRR on
tgbl-review** and **+0.0617 on YouTube**. Four previous attempts to put popularity or
radius into the *scorer* all lost (net −0.041 on the most careful one). The design
constraint adopted here is that popularity must reach the score **only by changing where
nodes sit on the manifold** — no scorer column, no feature column, no per-batch
standardisation. We tested whether that is even possible by repositioning the radii of a
frozen, converged checkpoint and re-scoring: on YouTube it is worth **+0.0442 test**, of
which **+0.0320 is the popularity ordering** and +0.0122 is the incidental radial
expansion (isolated by a scale-matched control). On tgbl-review the same operation is
worth **−0.0265** — it actively hurts. The proposal is therefore a **one-dataset
(YouTube) auxiliary loss** that pulls each node's hyperbolic radius toward a target set by
its **causal in-degree**, leaving the scorer `w·(−d_uv)` untouched and reducing to master
bitwise at `λ = 0`.

---

## 1. Symbols

Every symbol used in this document, in one place.

### Geometry

| symbol | meaning |
|---|---|
| `E` | the node embedding table, `nn.Embedding(num_nodes, d_emb)`, a `geoopt.ManifoldParameter` on `LorentzManifold`. `d_emb = 64`. This is the only large parameter block (22.6M on review). |
| `E[v]` | row `v` of the table: node `v`'s point on the manifold, in **intrinsic** coordinates (the `d`-vector `x'`, not the ambient `(d+1)`-vector). |
| `r(x)` or `dist0(x)` | hyperbolic distance from the origin to `x`. **In this chart `dist0(x) = asinh(‖x‖)`** — verified: `lorentz.py:249` computes `2·asinh(√(w/2))` with `w = ‖x‖²/(x0+1)`, `x0 = √(1+‖x‖²)`, which equals `asinh(‖x‖)` exactly (at `‖x‖ = 1` both give 0.8814). |
| `r_v` | shorthand for `r(E[v])`, node `v`'s radius. **Static**: one number per node, cannot vary with query time. |
| `p_u`, `p_v` | the **pooled point** for the source `u` / candidate `v`: `geom.weighted_midpoint` of the node's walk-token bag under the pooler's softmax weights (`model.py:70`). Returned by `BagWeights.pool`. |
| `r(p_v)` | radius of the pooled point. **Dynamic**: the walk bag is cut off at the query's timestamp, so this varies with query time for the same `v`. |
| `d_uv` | `geom.dist(p_u, p_v)`, the geodesic distance between the two pooled points. The scorer's only input. |
| `r_mean`, `r̄` | mean radius of the whole table, `geom.dist0(E.weight.detach()).mean()`, computed once per forward (`model.py:101-103`). Used as the `d_0` feature's scale. |
| `r_max` | max radius over the table. Logged per epoch as a boundary watch. |
| `θ` | angle at the origin between two points. The law of cosines in this chart is `cosh d(u,v) = cosh r_u · cosh r_v − sinh r_u · sinh r_v · cos θ`. **Consequence: `d_uv` is already a function of `r_v`** — the metric carries radius without any extra term. |
| `w` | the scorer's weight, `nn.Parameter(torch.tensor([1.0]))` on master. Score is `w · (−d_uv)` (`model.py:99, 113`). On the rejected scorer arms `w` was a 2-vector. |
| `INIT_RADIUS` | `1e-3` (`model.py:79`). **Every node starts at radius ≈ 1e-3**, so any quantity defined from the table's radial spread is degenerate at step 0. |

### Popularity

| symbol | meaning |
|---|---|
| `deg_t(v)` | **causal degree**: the number of edges incident to `v` with timestamp **strictly before** `t`, counted over the full stream (train ∪ val ∪ test). This is exactly the model's own information set: the trainer ingests the full graph once and enforces causality by a per-query exclusive cutoff (`trainer.py:111-118`). |
| `indeg_t(v)` | the same, counting only edges where `v` is the **destination**. **The chosen signal.** |
| `deg_train(v)` | static degree over the train split only, no time cutoff. Causal for val/test queries (all train edges precede them) but **NOT** causal for a training query at time `t` inside the train split. |
| `pop` | generic placeholder for whichever causal count is used. In the final spec `pop = indeg_t(v)`. |
| `π` (pi) | normalised log-popularity, `π(v,t) = clamp(log1p(pop) / C, 0, 1) ∈ [0,1]`. `π = 1` is the most popular node, `π = 0` a node with no prior in-edge. |
| `C` | the fixed popularity normaliser, `C = log1p(max in-degree over the train split)`. A single global scalar, train-derived, detached. **YouTube: `C = log1p(43860) = 10.6888`.** Same status as master's existing `log1p(age)/log_T_train` normaliser. |
| `T_train` | **NOT a timestamp and NOT a split boundary.** It is a *span* used to normalise ages (`model.log_T_train = log1p(T_train)`, `model.py:19`). Review's `T_train = 535,593,600` while its train split's max timestamp is `1,464,825,600`. Measured: using `T_train` as a cutoff returns **zero** edges on both datasets. See §3.5. |

### The loss

| symbol | meaning |
|---|---|
| `ρ` (rho) | normalised radius, `ρ(v) = (r_v − r_lo) / (r_hi − r_lo)`. `ρ = 0` innermost, `ρ = 1` at the rim. **Normalised globally over the node table, never per candidate list** — see §5.3. |
| `r_lo`, `r_hi` | the radial band the target maps into. **The 1st and 99th percentiles of `r(E)` over train-active nodes**, detached. Not min/max: YouTube's `r_max = 2.7308` against `p99 = 1.0079`, so min/max would stretch the band 2.7× on one node. **YouTube: `r_lo = 0.5426`, `r_hi = 1.0079`.** |
| `r̂(v)` | **target radius** for node `v`: `r̂ = r_lo + (r_hi − r_lo)·(1 − π)`. Popular → inner, obscure → rim. Detached (a label, not a parameter). |
| `λ` (lambda) | the loss weight. `λ = 0` must reproduce master **bitwise**. |
| `L_link` | master's existing objective: `F.cross_entropy(logits, zeros)` over `[B, 1+K]` logits (`trainer.py:169`). |
| `L_rad` | the new term, `mean[(ρ + π − 1)²]` over the batch's candidate nodes. |
| `α` (alpha) | **diagnostic only, not in the loss.** The blend used in test 7 to interpolate a frozen table toward the target: `r_new = (1−α)·r_old + α·r̂`. `α = 0` is master, `α = 1` full replacement. |
| `A` | soft-XOR agreement, `A = ρ + π − 2ρπ ∈ [0,1]`. **Logged, never optimised.** `A = 1` at the two self-consistent corners (inner+popular, outer+obscure), `A = 0` at the two inconsistent ones. |

### Metric

| symbol | meaning |
|---|---|
| MRR | mean reciprocal rank. **Both suites use the mid-rank rule**: `rank = 0.5·(#{neg > pos} + #{neg ≥ pos}) + 1`, then `1/float32(rank)`. Ties count half. See §3.1 — using the optimistic rank alone inflates MRR by 29% on tie-heavy data. |
| val / test | TGB-Seq: **val** negatives come from our sampler, **test** uses the shipped `test_ns` verbatim (`tgb_seq_eval.py:91-108`). TGB: pre-generated negatives for **both**. `--k-eval` is ignored on the TGB path. |
| "bar" | master `02f4ce4` at seed 5, the number any arm must beat. |

---

## 2. The bars this arm is measured against

Master `02f4ce4`, seed 5, `d64 K5 wpn5 mwl5 lr1e-3 pat10 h32`, same GPU model, all `rc=0`.
`02f4ce4` is **verified equivalent to `815a938`** (which produced the original bars) both by
a forward probe (`max|diff| = 0.000e+00`, shared `state_dict` tensors bitwise equal) and
**end-to-end through three full re-runs** (§3.0).

| dataset | nl | best val | ep | **test (the bar)** | ran | drift |
|---|---|---|---|---|---|---|
| YouTube | 2 | 0.6423 | 63 | **0.5512** | 73 | 0 |
| GoogleLocal | 2 | 0.6790 | 49 | **0.6473** | 59 | 0 |
| tgbl-review | 1 | 0.1300 | 11 | **0.1470** | 21 | 0 |

Drift is zero on all three (max test == val-selected test), so a single number per cell is
safe here. The external bar on YouTube is **SGNN-HN 0.5964** (CRAFT 0.5892), so YouTube's
deficit is **0.0452**.

Checkpoints (`state_dict` + `args` + `T_train` + `best_epoch` + `best_val_mrr` + `r_mean` +
`r_max`, saved **after** `trainer._restore(best_snap)` so they are the best-val state):

```
ckpt/master02f4ce4_YouTube_nl2_seed5.pt
ckpt/master02f4ce4_GoogleLocal_nl2_seed5.pt
ckpt/master02f4ce4_tgbl-review_nl1_seed5.pt
```

---

## 3. Measurements

All measurement scripts are on NFS in `scripts/` (sbatch compute nodes cannot see node-local
`/tmp`). All outputs in `logs/analysis/`.

### 3.0 Wave 1 — the checkpoints, and `02f4ce4 ≡ 815a938`

Jobs `11600693` (YouTube), `11600694` (GoogleLocal), `11600695` (review). Driver
`scripts/run_mckpt.sh`, logs `logs/mckpt/`. All three reproduced their bars **exactly** at
the same epochs: review ep11 0.1300/0.1470, GoogleLocal ep49 0.6790/0.6473, YouTube ep63
0.6423/0.5512. So every delta in this document rests on a reproduced baseline.

### 3.1 A metric bug found and fixed (affects any future rerank harness)

The first vectorised MRR used `rank = 1 + #{neg > pos}` — the **optimistic** rank. Both
suites actually use the **mid** rank:

```python
# tgb_seq/LinkPred/evaluator.py  and  tgb/linkproppred/evaluate.py::_eval_hits_and_mrr
optimistic  = (y_pred_neg >  y_pred_pos).sum(axis=1)
pessimistic = (y_pred_neg >= y_pred_pos).sum(axis=1)
ranking     = 0.5 * (optimistic + pessimistic) + 1
mrr         = 1.0 / ranking.astype(np.float32)
```

Character-for-character the same in both files. Both also cast inputs to **float64** before
comparing and the rank to **float32** before the reciprocal.

On a tie-heavy synthetic (2,000 rows × 101, 18,780 injected ties) the optimistic rule gives
MRR **0.0634** against the evaluators' **0.0492** — a 29% inflation — and differs on
**1,974 of 2,000 rows**. The corrected form is exact on all rows against *both* evaluators.

**Two lessons for any future harness.** (i) Reproduce the mid-rank rule, the float64
compare and the float32 reciprocal. (ii) Validate **per query**, not on the mean: a mean
over 10⁵–10⁶ reciprocals hides offsetting errors and is polluted by summation order, so a
tolerance on the mean is the wrong instrument at any threshold. `scripts/test23_rerank.py`
and `scripts/test5_geom_popularity.py` now assert elementwise equality against the suite's
own evaluator on every split's baseline before any sweep runs, and abort naming the first
disagreeing queries.

GoogleLocal was unaffected (its assertion passed even under the old rule, which can only
happen if no val or test query there has a tied score) and its numbers reproduced to 4
decimals under the corrected rule.

### 3.2 Test 1 — what the radius already encodes, and the correction

`scripts/test1_radius_vs_degree.py`, job `11600705`, output
`logs/analysis/test1-11600705.out`. Spearman over **train-active nodes only** (including
inactive nodes manufactures a correlation out of "was this row ever touched by a gradient").

| dataset | table | ρ(r,deg) | **ρ(r,in_deg)** | ρ(r,out_deg) | shape |
|---|---|---|---|---|---|
| tgbl-review | ep1 | −0.767 | **−0.875** | −0.522 | monotone ↓ |
| GoogleLocal | ep52 | −0.362 | −0.054 | −0.155 | monotone ↓ |
| YouTube | ep16 | −0.268 | −0.552 | +0.099 | non-monotone (min at decile 9) |

Review's deciles span in-degree **71.48 → 0.12** (a 600× spread) over radius 0.0366 → 0.3724.

**This measurement overstates the case and must not be quoted as evidence that the geometry
encodes popularity usefully.** It weights all 345k nodes equally. The population that
governs MRR is the *candidate* population, which is degree-weighted (popular nodes dominate
candidate lists), and there the explained variance is far lower — `R² = 0.063` on review,
`0.227` on YouTube (§3.4). A strong node-level rank correlation and a weak pairwise `R²`
are perfectly consistent, and **the `R²` is the one that matters.**

Caveat on provenance: these tables came from the `score-rv-perquery` runs (radius term
live), not from master, and the three datasets are at three different table ages (ep1 /
ep16 / ep52), so dataset and epoch are confounded.

### 3.3 Tests 2 & 3 — popularity and radius as post-hoc rerank

`scripts/test23_rerank.py`. Protocol: `new_score = master_score + a · signal(candidate)`,
with `a` swept on **val only** over 63 grid points, then the **test** MRR reported at the
val-selected `a`. The test-optimal `a` is also printed, labelled ORACLE and not reportable.

Jobs `11601214` (review), `11601215` (YouTube), `11601216` (GoogleLocal).

| | GoogleLocal | YouTube | tgbl-review |
|---|---|---|---|
| baseline test | 0.6467 | 0.5508 | 0.1470 |
| **Test 2** `log1p(deg_t(v))`, val `a*` | +0.178 | +1 | +5.62 |
| val Δ | +0.0003 | +0.0493 | +0.1338 |
| **test Δ** | **+0.0007** | **+0.0617** | **+0.1529** |
| oracle | +0.0007 | +0.0659 | +0.1537 |
| **Test 3** `−r(E[v])`, val `a*` | **−1** | **+10** | −5.62 |
| val Δ | +0.0001 | +0.0178 | +0.0228 |
| **test Δ** | **+0.0003** | **+0.0148** | **−0.0015** |
| oracle | +0.0007 | +0.0148 | +0.0014 |

**Readings.**

1. **Popularity is the big signal and it is complementary to master's score** — +0.1529 on
   review more than doubles that cell, and the resulting 0.2999 beats both prior reference
   points there (a popularity-only ranker at 0.2660, and the leaky `[-geo, pop]` scorer at
   0.2867, §4.2).
2. **Radius in the score is not.** Across three datasets test 3 sums to **+0.0136** against
   test 2's **+0.2153** — radius recovers **6%** of the available signal. On review it is
   *negative* and its oracle ceiling is +0.0014, i.e. no coefficient works.
3. **The sign of test 3 flips with the coupling.** GoogleLocal picks `a = −1` (mildly
   prefers *high* radius) and has ρ(r,in_deg) = −0.054; YouTube picks `a = +10` (prefers
   low radius = popular) and has ρ = −0.552. GoogleLocal's negative sign independently
   matches `w[1] → −3.71` on that dataset in the `score-rv-perquery` arm.
4. **YouTube's optimal coefficient equals the measured Krioukov slope.** From test 1's
   deciles, `log1p(in_deg)` falls 3.49 → 0.95 over radius 0.394 → 0.647, slope **−10.1**;
   test 3's val-selected `a` is **+10**. So the functional form `a·(−r)` is already correct
   — test 3's weakness is not a form error. (Review's slope is −12.4 and ±{10, 17.8} were
   both on the grid and both lost.)
5. **Review's val cannot select a radius coefficient.** Val said +0.0228, test said
   −0.0015. This is the third time review's val has pointed the wrong way on a radius-ish
   quantity. For test 2 val behaves fine (val-selected 0.2999 vs oracle 0.3007), so the
   defect is specific to the radius term there.

### 3.4 Test 5 — static vs dynamic, and the pooled radius

`scripts/test5_geom_popularity.py`, jobs `11601239` (review), `11601240` (YouTube).
Same rerank protocol. Signals, all fit on **val** and applied unchanged to **test**:

| id | signal | kind |
|---|---|---|
| D1 | `log1p(deg_t(v))` | dynamic (= test 2) |
| D2 | `−r(p_v)` pooled radius | dynamic |
| D3 | `Ê[log1p deg_t \| r(p_v)]`, binned conditional mean | dynamic |
| S1 | `log1p(deg_train(v))` | static |
| S2 | `−r(E[v])` | static (= test 3) |
| S3 | `Ê[log1p deg_t \| r(E[v])]`, binned conditional mean | static |
| **S4** | `Ê_val[log1p deg_t(v)]`, per-node mean | **static — THE CEILING** |

D3/S3/S4 are binned conditional means with **no monotonicity imposed**, so they are the
best *arbitrary* function of their input at bin resolution (256 quantile bins) — a strictly
stronger ceiling than any parametric or monotone form.

**Why S4 is the gate.** `r(E[v])` is a property of the node alone: it cannot differ between
two queries for the same candidate at different times. So **every static geometric feature
— table radius, local density, k-NN distance, any function whatever of a frozen embedding
table — shares one ceiling**, namely the best achievable by some `g(v)` with no `t`
dependence. S4 measures that ceiling non-parametrically.

| signal | **review** test Δ | **YouTube** test Δ |
|---|---|---|
| D1 `log1p(deg_t)` | **+0.1538** | **+0.0617** |
| D2 `−r(p_v)` | +0.0063 | +0.0238 |
| D3 best fn of `r(p_v)` | +0.0168 | +0.0302 |
| S1 `log1p(deg_train)` | +0.1101 | +0.0324 |
| S2 `−r(E[v])` | −0.0003 | +0.0148 |
| S3 best fn of `r(E[v])` | +0.0056 | +0.0261 |
| **S4 static ceiling** | **+0.1205** | **+0.0394** |
| **S4 / D1** | **78%** | **64%** |

Diagnostics over all val (query, candidate) pairs:

| | review | YouTube |
|---|---|---|
| R² of `log1p(deg_t)` on `r(E[v])` | **0.0631** | 0.2268 |
| R² of `log1p(deg_t)` on `r(p_v)` | 0.1215 | 0.2724 |
| R² of `log1p(deg_t)` on `deg_train` | **0.9469** | **0.9718** |
| **corr(score, `−r(E[v])`)** | **−0.0789** | **+0.5481** |
| corr(score, `−r(p_v)`) | **+0.6847** | +0.5734 |

**Readings.**

1. **The signal is mostly static** (78% / 64%), so requiring popularity to pass through a
   static coordinate costs about a third of it at most, not all of it.
2. **Popularity barely moves within the eval window**: `R²(deg_train, deg_t) = 0.947 /
   0.972`. A static target is therefore a good approximation of the causal one on these two
   datasets — which is what makes the per-step causal target in §5 land near test 7's
   static arrangement.
3. **Radius is an encoding failure, not an information bound.** On review the best
   arbitrary function of radius buys +0.0056 against a static ceiling of +0.1205: radius
   discards **95% of a signal fully available to it**. That is a training problem, not a
   scorer problem — which is why this proposal is a loss term.
4. **The pooled radius is strictly better than the table radius on every measure** and is
   *dynamic*, so it escapes S4's ceiling. It is also far better coupled to the score on
   review (+0.685 vs −0.079). **It is the obvious next target if the `E` version works, and
   it is untested** — see §7.2.

### 3.5 Test 6 — the popularity source, and two traps

`scripts/test6_popularity_source.py`, jobs `11601267` (review), `11601268` (YouTube).

**Trap 1 — `T_train` is a span, not a timestamp.** Cutting Tempest's causal count at
`T_train` returns **sum 0** on both datasets: review's `T_train = 535,593,600` (≈17 years in
seconds) against absolute timestamps ≈`1.46e9`. Anyone building a static train-split cutoff
must use `max(train.timestamps) + 1`, not `T_train`. Under the per-query design in §5 this
never arises, because there is no static cutoff anywhere.

**Trap 2 — Tempest is undirected.** `WalkGenerator` builds `Tempest(is_directed=False, …)`
(`walks.py:49`), so `get_node_popularity` (`walks.py:94`, via
`get_node_participation_counts(direction="Backward_In_Time")`) returns **total
participation**. In-degree and out-degree are **not obtainable** from it. `get_node_degrees`
must never be used — it takes no cutoff, and since the trainer ingests the full graph it
would count val and test edges.

**Which definition carries the signal** (static train-split counts, rerank protocol):

| definition | R² vs `deg_t` | **review** test Δ | **YouTube** test Δ |
|---|---|---|---|
| total participation | 0.947 / 0.972 | +0.1101 | +0.0324 |
| **in-degree** | 0.794 / 0.611 | +0.0818 | **+0.0422** |
| **out-degree** | 0.650 / 0.638 | **+0.1570** | +0.0126 |

The best definition is **opposite on the two datasets**, and `R²` against total degree is
*anti*-correlated with the MRR gain — so fidelity to total degree is not the thing that
matters. Note also that a *static* out-degree on review (+0.1570) beats even the *dynamic*
total degree (+0.1538).

### 3.6 Test 7 — THE DECISIVE MEASUREMENT: radial repositioning

`scripts/test7_radial_reposition.py`, jobs `11601328` (YouTube), `11601329` (review).
Outputs `logs/analysis/test7-11601328.out`, `…-11601329.out`. (The first pass without the
scale control is kept at `logs/analysis/superseded/test7-116013{07,08}.no-scale-control.out`.)

**What it does.** On the frozen checkpoint, move every node radially to `r̂` keeping its
**angle** untouched, then re-pool and re-score:

```
r_new = (1−α)·r_old + α·r̂            α ∈ {0.25, 0.5, 0.75, 1.0}
E_new = E · sinh(r_new) / ‖E‖         (exact, since dist0(x) = asinh(‖x‖))
```

`r̂ = r_lo + (r_hi − r_lo)·(1 − s)`, `s = log1p(pop)/max log1p(pop)`, `r_lo/r_hi` = p1/p99 of
`r(E)` over train-active nodes. Train-**inactive** nodes keep their trained radius (they
carry no popularity signal; pulling them to the rim would move `r_mean` for a reason
unrelated to the hypothesis).

**Three design points that make it trustworthy.**

- **Walks are built once per batch and reused by all 25 tables.** Tempest's RNG advances per
  call, so re-walking per arm would confound the comparison with walk noise. All arms are
  paired on identical bags.
- **α = 0 is master bitwise** and serves as the in-run control (its MRR matches the
  checkpoint: YouTube val 0.6443 on the 65k subsample vs 0.6423 full-split).
- **Every arm has a scale-matched control**: all radii multiplied by one constant to reach
  the *same* `r_mean`, carrying no popularity ordering at all. `arm − control` is the part
  attributable to the ordering. This control is essential because radial scale is
  independently potent on this suite (INIT_STD 0.5 is +0.098 on review and −0.53 on
  GoogleLocal).

Queries are subsampled by stride (deterministic, identical across arms); the evaluator's
negative cursor is advanced on **every** batch so the sampling does not desynchronise it.

#### YouTube — it works (test split, 78,000 queries, baseline 0.5508)

| arm | r_mean | arm Δ | scale-ctrl Δ | **ordering alone** |
|---|---|---|---|---|
| master (α=0) | 0.7123 | — | — | — |
| total α=0.25 | 0.7647 | +0.0133 | +0.0027 | +0.0106 |
| total α=0.50 | 0.8170 | +0.0222 | +0.0056 | +0.0166 |
| total α=0.75 | 0.8694 | +0.0264 | +0.0085 | +0.0179 |
| total α=1.00 | 0.9218 | +0.0256 | +0.0111 | +0.0145 |
| in α=0.25 | 0.7700 | +0.0180 | +0.0030 | +0.0150 |
| in α=0.50 | 0.8278 | +0.0316 | +0.0062 | +0.0254 |
| **in α=0.75** | 0.8855 | +0.0406 | +0.0093 | **+0.0313** |
| **in α=1.00** | 0.9433 | **+0.0442** | +0.0122 | **+0.0320** |
| out α=0.25 | 0.7733 | +0.0055 | +0.0032 | +0.0023 |
| out α=0.50 | 0.8343 | +0.0051 | +0.0064 | −0.0013 |
| out α=0.75 | 0.8954 | −0.0024 | +0.0099 | −0.0123 |
| out α=1.00 | 0.9564 | −0.0183 | +0.0129 | **−0.0312** |

Val agrees on the ordering (in α=1.00: val +0.0330, control +0.0104), so **val-selection
picks in-degree/α=1 honestly** and lands on test +0.0442, i.e. **0.5508 → 0.5950** against
the SGNN-HN bar of 0.5964.

Three things to take from this table:

- **The ordering, not the scale, is the mechanism**: +0.0320 of +0.0442. Uniform expansion
  to the same `r_mean` buys only +0.0122.
- **The control cleanly separates the definitions**: in-degree **+0.0320**, total +0.0145,
  out-degree **−0.0312**. Out-degree's raw numbers look merely flat; net of scale it is
  actively harmful. Using total because Tempest offers it costs more than half the effect.
- **It exceeds the additive-rerank ceiling.** S4 capped a static additive channel at
  +0.0394 and S3 (best function of existing radius) at +0.0261; repositioning gets +0.0442.
  That is not a contradiction: repositioning changes every pairwise distance *and* what the
  pooler sees, because a node appears in other nodes' walk bags too. It is not bounded by
  what an additive offset on 101 scores can do.

#### tgbl-review — it fails (test split, 61,000 queries, baseline 0.1468)

Every arm is negative and falls monotonically with α; every scale control is flat
(+0.0001 … +0.0010).

| arm | arm Δ | scale-ctrl Δ | **ordering alone** |
|---|---|---|---|
| total α=1.00 | −0.0235 | +0.0006 | −0.0241 |
| in α=0.75 | −0.0164 | +0.0008 | −0.0172 |
| in α=1.00 | −0.0255 | +0.0010 | **−0.0265** |
| out α=1.00 | −0.0261 | +0.0004 | −0.0265 |

Monotone decline was pre-registered as the signature for "the metric cannot exploit radius
here", and it matches `corr(score, −r(E[v])) = −0.0789` on review (against +0.5481 on
YouTube): review's pooling already discards E's radius almost entirely.

**So this arm is YouTube-only.** See §6 for why review is likely different and what would
be needed there.

### 3.7 The constants

`scripts/radius_target_constants.py`, job `11601365`, output `logs/analysis/const-11601365.out`.

| | **YouTube** | tgbl-review |
|---|---|---|
| `r(E)` p1 = **`r_lo`** | **0.5426** | 0.1898 |
| `r(E)` p50 | 0.6998 | 0.4592 |
| `r(E)` p99 = **`r_hi`** | **1.0079** | 0.6855 |
| `r(E)` mean | 0.7123 | 0.4534 |
| `r(E)` max | 2.7308 | 1.1632 |
| in-degree mean / p50 / p90 / p99 / max | 6.78 / 3 / 9 / 49 / **43,860** | 9.89 / 2 / 17 / 142 / 5,408 |
| **`C` = log1p(max in-deg)** | **10.6888** | 8.5958 |
| log1p(p99 in-deg), for reference | 3.9120 | 4.9628 |

`π` distribution with `C = log1p(max)` over train-active nodes:

| | mean π | median π | p90 π | frac π>0.5 | frac π=0 |
|---|---|---|---|---|---|
| YouTube | 0.1388 | 0.1297 | 0.2154 | **0.16%** | 3.9% |
| tgbl-review | 0.1453 | 0.1278 | 0.3363 | 2.3% | **26.9%** |

**The validated arrangement is strongly asymmetric, and the implementer should expect
that.** In-degree's tail is so heavy that `π` is small for nearly everyone: a median
YouTube node gets `r̂ = 0.5426 + 0.465·(1 − 0.130) = 0.947` against its actual median radius
of 0.700 — a substantial *outward* push. Only ~0.2% of nodes have `π > 0.5` and are pulled
inward. That is exactly why test 7's `r_mean` rose 0.7123 → 0.9433 at α=1, and the control
says only +0.0122 of the gain came from that rise. **`C = log1p(p99)` with clipping would
use the band far more evenly (median π 0.354, 16% above 0.5) but is UNTESTED** — do not
substitute it without running it.

Note in passing: **26.9% of review's train-active nodes never receive an edge** against 3.9%
on YouTube. A quarter of review's nodes have `π ≡ 0` and are indistinguishable to this term.

---

## 4. Why a loss term and not a scorer term

This is the central design decision, and it rests on four independent arguments.

### 4.1 It is not expressible as a logit term

Test 7's +0.0320 comes from **where the nodes are**. Moving `E[v]` changes every pairwise
distance that node participates in, and it changes what the pooler sees — `v` appears in
*other* nodes' walk bags, so its radius propagates into their `p_u` as well. A logit term
adds a number to 101 candidate scores *after* the arrangement is fixed. It cannot produce an
arrangement. The thing that was measured to work is a property of `E`, and only a loss on
`E` can produce it.

### 4.2 In the logit it competes with geometry; in the loss it shapes it

A logit channel is an **alternative ranker**: if popularity separates candidates cheaply,
the model satisfies cross-entropy without improving the embedding. This is measured in this
project, not speculative — CLAUDE.md on the `[-d_H, cos]` family: *"`cos` is exactly
scale-invariant (measured `d(cos)/d(radius) = 3.7e-09`), so it gives a cheap angular way to
separate candidates and the embedding never needs to spread — r_mean 0.049 at ep1 against
the baseline's 0.225, r_max collapsing to 1.038 while the baseline climbed to 3.116. The
objective gets solved without geometry, and then there is nothing left to generalise with."*

A loss term cannot be used to rank at eval, because it is not in the forward score. It can
only move points. The pressure is therefore forced **through** the geometry rather than
**around** it.

### 4.3 The logit route has been tried four times and lost, with a known mechanism

| # | branch | commit | change | result |
|---|---|---|---|---|
| 1 | `feature/score-d0v` | `366e905` | shared weight, pooled-point `d_0` | YouTube 0.4554 (**−0.096**); train loss 7.6× master's **and rising** |
| 2 | `feature/score-d0v-norm` | `c2cd6bc` | global `/r_mean` | YouTube 0.4419 (**−0.109**) |
| 3 | `tmp/duv-over-rmean` | `2e681cd` | distance-only control `w·−(d_uv/r_mean)` | net ≈ 0 (faster convergence, same MRR) |
| 4 | `feature/score-rv-perquery` | `c2ba9d6` | `w·[−d_uv, −r_v·s]`, per-query scale-matched | YouTube −0.0135, GoogleLocal −0.0237, review −0.0039; **summed −0.041** |

Attempt #4 is the careful one and it closes the family. Its pre-launch probe passed five
checks: the arm started at master **bitwise** (forward `0.000e+00`, params 1254 → 1255,
exactly +1), the channel was live (`w[1]=1` moved scores 1.405e-03 against a score scale of
4.55e-04), the per-query scale was matched to **8.24e-08 relative**, the clamp fired on 0/4
queries, and no new cross-query coupling was introduced. So the scale-mismatch explanation
for #2 is **ruled out** — a correctly scaled, bitwise-master-initialised radius channel gets
the identical rejection curve.

**The mechanism, measured.** `w[1]` rises, peaks, then unwinds monotonically, while `w[0]`
runs away:

| dataset | `w[1]` peak | `w[1]` final | `w[0]` final | ratio collapse |
|---|---|---|---|---|
| tgbl-review | +8.77 @ep4 | +0.73 @ep11 | 34.7 | 0.81 → 0.021 (**39×**) |
| GoogleLocal | +3.62 @ep3 | **−3.71** (crossed 0 at ep26) | 58.4 | — |
| YouTube | +6.58 @ep6 | +2.53 | 36.6 | — |

An auxiliary logit channel must share the softmax's scale with `−d_uv`, and cross-entropy
drives that scale up without bound, so the *ratio* collapses even where `w[1]` stays
positive. A separate `λ` has nothing competing with it.

(A driver-header prediction that review would gain and `w[1]` would settle positive below 1
was **falsified on both halves** and is on the record in that run's log.)

### 4.4 The normalised form would leak in the logit

Candidate-normalised `ρ, π` in the logit make a candidate's score depend on which other
candidates share its query — odd-one-out detection on `{positive} ∪ {100 negatives}`. That is
the same class as the `standardise` defect below. **In the regulariser it is harmless,
because the term does not exist at eval time and no eval score ever reads the candidate
list.** This asymmetry is why §5.3 fixes the normalisation to be global anyway: the leak is
not the only reason, but it would be fatal on the logit side.

### 4.5 Reference branch — the causal path, and its second bug

`feature/scorer-cand-popularity` (`2ed7db8`, one commit, `model.py` + `trainer.py` +
`train_link_property_prediction.py`, +21/−9) is the **correct reference for the causal
popularity call**:

```python
# trainer.py on that branch — candidate history at each candidate's OWN query cutoff, EXCLUSIVE
cand_popularity = torch.from_numpy(
    self.walk_gen.get_node_popularity(cand_seeds_np, cand_cutoffs_np)).to(device, torch.float32)
```

**But that branch has a second, independent causality bug, and its numbers are contaminated.**
Its scorer reads

```python
popularity_v = standardise(torch.log1p(cand_popularity).unsqueeze(-1)).view(b, c)
# standardise(): dims = tuple(range(feat.dim() - 1))   -> reduces over the WHOLE [b*c] batch
```

`cand_popularity` is `[b*c]`, so mean and variance reduce over all candidates **across all
queries in the batch**. A candidate's standardised popularity depends on which other queries
share its batch. This is exactly the `f941793` cross-query leak, removed at `87b4d27`
("Pooler: fixed-scale age and hop features instead of per-batch standardisation") and
measured at **+0.046 on YouTube on its own** (clean one-variable pair: `f941793` val 0.6829 /
test 0.6046 against `87b4d27` val 0.6522 / test 0.5583, same 3 features, same 1,221 params,
same cell and seed, both `rc=0`).

So that branch is **causal in time, leaky across queries**. Two separate axes; it got one
right. Consequently the `[-geo, pop]` figure of **0.2867 on review** recorded in project
memory is **not a clean baseline**. The leak-free anchor for review is test 2's post-hoc
**0.2999**.

**`standardise` must not appear anywhere in this arm.**

---

## 5. The proposed loss term

### 5.1 The term

```
L = L_link  +  λ · mean_v [ ( ρ(v) + π(v,t) − 1 )² ]
```

with, per training step, over the batch's candidate nodes `v` at their own query times `t`:

```
pop(v,t) = indeg_t(v)                                  # edges INTO v strictly before t
π(v,t)   = clamp( log1p(pop) / C , 0, 1 )              # detached, C fixed
ρ(v)     = ( dist0(E[v]) − r_lo ) / ( r_hi − r_lo )    # r_lo, r_hi detached
```

Equivalently, and this is how to read it: `ρ + π − 1 = 0` ⟺ `r_v = r̂(v)` where
`r̂ = r_lo + (r_hi − r_lo)·(1 − π)`. So the term is a **regression of each node's radius onto
a popularity-determined target shell**, written in normalised coordinates;
`(ρ+π−1)² = (r_v − r̂)²/(r_hi−r_lo)²` and the constant folds into `λ`.

Properties:

- **`λ = 0` is master bitwise.** The arm is a strict superset and the sweep carries its own
  control.
- **The scorer is untouched**: still `w · (−d_uv)`, `model.py` ideally unmodified (§5.5),
  parameter count unchanged.
- **Gradient flows only through `dist0(E[v])`.** `π`, `C`, `r_lo`, `r_hi`, `r̂` are all
  detached — targets, not parameters.
- **Two-sided and graded**: `∂L/∂r_v = 2(ρ + π − 1)/(r_hi − r_lo)`. Popular-but-peripheral
  nodes are pulled in, obscure-but-central nodes pushed out, with force proportional to the
  disagreement, and a genuine stationary point at `r_v = r̂`.
- **Inert at initialisation.** All radii start at ≈`1e-3` (`INIT_RADIUS`), so `r_lo ≈ r_hi`
  and the band is degenerate; the term engages only once the link loss has produced radial
  spread. This is a feature: the arm starts at master.

### 5.2 Why NOT `r_v · log1p(pop)`

A plausible-looking alternative is the bare product. Rejected: `∂/∂r_v = log1p(pop) ≥ 0` is
a **one-sided** inward pressure with no equilibrium, so on its own it drives every popular
node to `r = 0`, where hyperbolic volume vanishes and popular nodes become mutually
indistinguishable — the §4.2 failure mode exactly. It is also non-monotone as a *descriptive*
statistic (on review's deciles it peaks at decile 3 at 0.339 and is low at both ends, 0.157
inner and 0.042 at the rim), because it is a product of two anti-correlated factors, so it
does not separate popular from obscure and is blind to obscure-but-central nodes.

The squared form has the interior optimum the product lacks. (The product does have one nice
reading — in hyperbolic space volume grows like `e^{(d−1)r}`, so radius is a *code length*
and `Σ pop·r` is expected code length, i.e. Huffman coding on the manifold whose optimum is
the Krioukov arrangement. But the optimum is only reached subject to a spreading constraint;
on its own the term collapses.)

### 5.3 Why the normalisation must be GLOBAL, not per candidate list

Normalising `ρ` within the candidate set is wrong for three reasons, in order of severity:

1. **It is scale-free, and the scale is what worked.** Any monotone radial ordering satisfies
   a candidate-normalised objective, including one where every radius sits within `1e-6` of
   the others. But `d_uv` reads *absolute* radii, and test 7's gain came from an arrangement
   spanning `[0.5426, 1.0079]` with `r_mean` rising to 0.9433. Per-candidate normalisation
   discards exactly the magnitude information that produced +0.0320.
2. **Degenerate gradients.** Rank-normalised `ρ` is piecewise constant → zero gradient.
   Min–max normalised `ρ` routes gradient through whichever two *other* nodes happen to be
   the extremes of that candidate list.
3. **The target becomes co-candidate-dependent** — the same node gets a different target
   depending on its random negatives. Harmless at eval (§4.4) but it injects noise into
   learning for no benefit.

`π` is likewise normalised by the **fixed global** `C`, never by batch statistics.

### 5.4 The one open choice: how `r_lo`, `r_hi` are obtained

Test 7 took them from a **frozen converged** table. A training run cannot: at init every
radius is ≈`1e-3`, so percentiles are degenerate. Three options, in preference order:

**(A) Quantile matching — RECOMMENDED.** Drop `r_lo`/`r_hi` entirely and set

```
r̂(v) = Quantile_{r(E) over train-active}( 1 − π(v) )        # detached lookup
L_rad = mean[ ( dist0(E[v]) − r̂(v) )² ]
```

This permutes which node sits at which *already existing* radius rather than inventing new
ones, so it is **mean-preserving by construction**: no expansion feedback loop, no new
constants, cannot diverge. In normalised language it is the cleanest reading of the same
formula — `ρ` becomes the node's radial *quantile* and `ρ + π − 1 = 0` says "radial rank =
inverse popularity rank". Keep the loss as `(r_v − r̂)²` so the gradient flows through `r_v`
only and rank's non-differentiability never bites. **Cost: forgoes the +0.0122 expansion
component**, isolating the +0.0320 ordering — which is the cleaner claim anyway, since radial
scale is a separate, known-potent, dataset-splitting knob (INIT_STD 0.5: +0.098 review,
−0.53 GoogleLocal) and is better not entangled with this term.

**(B) Live percentiles**, `r_lo/r_hi = p1/p99` of the current table, recomputed per step and
detached. Closest to test 7's linear band. **Risk: positive feedback** — the target pushes
mass outward (most nodes have small `π`), which raises p99, which raises the target. Requires
an `r_mean` guard (§5.7) and a hard abort if `r_mean` grows super-linearly.

**(C) Hardcode YouTube's 0.5426 / 1.0079.** Closest to what was measured, but it bakes a
previous run's outcome into the arm and does not transfer. Only for a one-off replication of
test 7 under training.

**Implement (A) and (B) behind one flag** so both can be run; report (A) as the headline.

### 5.5 Where the code goes

Keep `model.py` **unmodified** if at all possible — then "the scorer is untouched" is provable
by `git diff` rather than by argument. Everything can live in the trainer, since `dist0` and
`E` are both reachable from it.

**`link_property_prediction/trainer.py`** — the hook is `_train_batch`, immediately after the
link loss (currently `trainer.py:166-173`):

```python
logits = self._score(src_t, cand_t, t_query_t)            # [B, 1+K]
target = torch.zeros(B, dtype=torch.long, device=device)
link_loss = F.cross_entropy(logits, target)

loss = link_loss
rad_loss = None
if self.config.radial_pop_lambda > 0.0:
    rad_loss = self._radial_pop_loss(cand_t, t_query_t)   # scalar
    loss = loss + self.config.radial_pop_lambda * rad_loss

self.opt.zero_grad(set_to_none=True)
loss.backward()
self.opt.step()
```

`cand_t` is `[B, 1+K]` (positive in column 0, negatives after) and `t_query_t` is `[B]`, so
each candidate's cutoff is its own query's `t` — broadcast `t_query_t.unsqueeze(1).expand(B, 1+K)`,
exactly as `_score` already does at `trainer.py:141`.

**`scripts/train_link_property_prediction.py`** — new flags, all defaulting to the master
behaviour:

| flag | default | values |
|---|---|---|
| `--radial-pop-lambda` | `0.0` | float; `0.0` must be master bitwise |
| `--radial-pop-degree` | `in` | `in` \| `out` \| `total` |
| `--radial-pop-target` | `quantile` | `quantile` (§5.4A) \| `band` (§5.4B) |
| `--radial-pop-c` | `max` | `max` \| `p99` (the `C` choice; `max` is the validated one) |

Pass them through `TrainerConfig` and into the config dump, and record them in the log header.

### 5.6 The causal directional degree index

**`get_node_popularity` cannot supply in-degree** (§3.5, Tempest is `is_directed=False`), and
in-degree is worth +0.0320 against total's +0.0145 on YouTube. So build a directional causal
index, exactly as `scripts/test5_geom_popularity.py` does — two vectorised `searchsorted`
calls over a rank-compressed `(node, ts)` key array, built **once** before training:

```python
# full stream = train ∪ val ∪ test, the same edges the trainer ingests
tvals = np.unique(np.asarray(fg.timestamps, np.int64))     # distinct timestamps, sorted
SPAN  = np.int64(len(tvals) + 1)

# IN-degree: index the DESTINATION endpoint only.  (out-degree: sources; total: both.)
nd   = np.asarray(fg.destinations, np.int64)
ts   = np.asarray(fg.timestamps, np.int64)
keys = np.sort(nd * SPAN + np.searchsorted(tvals, ts))

def indeg_before(nodes, t):
    """#{edges into `nodes` with timestamp STRICTLY < t}."""
    base = nodes.astype(np.int64) * SPAN
    tr_  = np.searchsorted(tvals, t)        # 'left' => count of distinct times < t
    return (np.searchsorted(keys, base + tr_, side="left")
            - np.searchsorted(keys, base,      side="left")).astype(np.float32)
```

**Why this is causal and not a leak.** The index spans the full stream, but it is only ever
queried with the per-query **exclusive** cutoff, which is precisely the information set the
model's own walker gets — the trainer ingests the full graph once and relies on the per-query
cutoff for causality (`trainer.py:111-118`, and the `[tgb]`/`[tgb-seq]` ingest line in every
log). This is the established pattern in this repo. `searchsorted(tvals, t)` with the default
`side='left'` returns the number of distinct timestamps **strictly less than** `t`, which is
what makes the count strictly-before; this was validated in tests 2, 5 and 6.

Memory: two int64 arrays of length `|E|` (review 4.87M edges → ~78 MB). Build on CPU, keep the
lookup on CPU, transfer the `[B, 1+K]` result to GPU as float32. For YouTube's `B×(1+K)` per
step this is a trivial cost; if it ever shows up in the profile, cache per-node counts at the
distinct timestamps instead.

**Do not use `get_node_degrees`** — no cutoff, leaks future edges.

### 5.7 Logging and guards

Add to the epoch line:

- `lam=` the λ in force, `Lrad=` the raw (unweighted) `L_rad`, so the two objectives can be
  compared in magnitude;
- `agree=` the soft-XOR agreement `mean(ρ + π − 2ρπ)` over the batch — **logged, never
  optimised**. This exists so a null can be diagnosed: see §6.3. Both EMA nulls in this
  project were uninterpretable precisely because the mechanism's own state was not logged
  (CLAUDE.md: *"`alpha` is NOT logged per epoch, so both EMA nulls are uninterpretable at the
  mechanism level"*).
- `r_mean=`, `r_max=` already exist — keep them and watch them.

Pre-launch probe, in the driver, with a hard abort on failure:

1. **`λ = 0` is master bitwise**: build both models at the same seed and assert
   `max|logits_master − logits_arm| == 0.000e+00` on a fixed batch, and that
   `state_dict` keys and parameter count are **unchanged** (no new parameters — this arm adds
   none).
2. **The term is live**: at `λ = 1`, assert `L_rad > 0` and that `E.grad` is non-zero on rows
   that appear in the batch and zero on rows that do not.
3. **Causality**: assert `indeg_before(v, t)` equals a direct count over the full stream for
   a handful of random `(v, t)` pairs, and assert it is **non-decreasing in `t`** and
   **zero at `t = min(all timestamps)`**.
4. **No `standardise`, no batch statistics**: grep the diff and abort if `standardise` appears.
5. **Direction**: assert the index counts destinations for `--radial-pop-degree in` by
   checking a bipartite-style node that only ever receives (its out-degree must be 0).

Runtime guards:

- **NaN watch.** A NaN run reports MRR **1.0000**. That is a failure, not a result. Abort on
  non-finite loss.
- **`r_mean` runaway** (option B only): abort if `r_mean` exceeds, say, 3× its value at the
  epoch where the term first engaged, or if `r_max` crosses 8 (float32 `expmap` accuracy
  degrades past ~8; the chart stores an ambient point losing the manifold constraint near
  r = 9).
- **Train loss turning up** while `r_mean` grows linearly is this suite's known
  radius-inflation signature (seen on Yelp/WikiLink). Log it; it is a reason to stop.

---

## 6. What to run, and how to read it

### 6.1 Cells

**YouTube only in the first pass.** `nl 2`, seed 5, `d64 K5 wpn5 mwl5 lr1e-3 pat10 h32`,
`--num-epochs 100`, `--save-checkpoint`, `--export-best-embedding-table`.

| arm | λ | degree | target | note |
|---|---|---|---|---|
| control | 0.0 | — | — | must reproduce **0.5512** exactly; free bar |
| A1 | 0.3 | in | quantile | |
| A2 | 1.0 | in | quantile | expected best first guess |
| A3 | 3.0 | in | quantile | |
| B2 | 1.0 | in | band | the test-7-faithful variant |

λ scale reasoning: `L_rad` is in units of radius², and a median YouTube node sits ~0.25 from
its target, so `L_rad ≈ 0.06` once the term engages; master's link loss falls 1.6697 → 0.0322
over its 73 epochs. So `λ ≈ 1` makes the two terms comparable in the mid-training window
where the ordering has to form. Sweep around it rather than trusting it.

**Do not run review or GoogleLocal in the first pass.** Review's ordering is worth −0.0265
and GoogleLocal's entire popularity signal is +0.0007; both would only dilute the result.
Add them afterwards **as regression checks** if YouTube wins.

### 6.2 What counts as a win

- Master YouTube test = **0.5512**. Test 7's ceiling for this arrangement = **0.5950**
  (+0.0442), of which +0.0320 is ordering and the rest expansion that option (A) forgoes.
- A trained arm will not reach the frozen-table ceiling, since the link loss resists the
  pull. **Anything ≥ +0.015 test is a real effect**; +0.032 would mean the arm realised the
  full measured ordering gain.
- The external target is **SGNN-HN 0.5964**, i.e. +0.0452 from master.
- Report **both** val-selected test and max test. Drift is 0 on master's YouTube run but
  this suite has produced 0.0054 drift on other arms, and ranking on the printed
  `best_test_mrr` alone is not like-for-like.

### 6.3 How to read a null — this is why `Lrad` and `agree` are logged

| `L_rad` | `agree` | MRR | reading |
|---|---|---|---|
| stays high | stays low | unchanged | **the term did not bind** — λ too small, or the gradient is being swamped. Raise λ. |
| falls to ~0 | rises to ~1 | unchanged | **the term bound and the geometry matched the target, and it did not help** — test 7 did not transfer from a frozen table to training. That is a real, reportable negative and closes the route. |
| falls | rises | **worse** | the arrangement conflicts with what the link loss needs; look at whether `r_max` or train loss turned up. |

Without these two fields a null is uninterpretable, which is the exact mistake that made both
EMA arms unreadable.

### 6.4 Timing and scheduling

Master YouTube: **45.8 s train + 15.5 s eval per epoch**, 73 epochs ≈ 75 min on a quiet RTX.
The new term adds one `searchsorted` pair per step on `B×(1+K)` nodes and one sort/quantile of
`|train-active|` radii per step — expect **< 5% overhead**; measure it and put it in the header.

Contention caveat: an unrelated RTX job sharing the node took one of these runs from 45.8 to
58.7 s/epoch (1.28×). Budget ~25% margin.

Slurm notes for this cluster: **each QOS has its own 2-GPU budget** (a third concurrent
`long` job hits `QOSMaxGRESPerUser`; move to `verylong`/`enginf`; real ceiling ≈ 6 RTX). The
`general` partition caps at `--time=8:00:00`. The login shell has an **8 GB cgroup cap**, so
anything that loads a dataset needs `sbatch --mem`.

**Do not call YouTube before ep50.** Five arms in this project inverted between ep14 and ep40,
and master's own late escape runs ep25–50 (val 0.583 → 0.685, +0.068 in three epochs). The
ep10–40 window carries no information about the ep50+ result on this suite.

---

## 7. Risks, and what is NOT established

### 7.1 Known risks

1. **Test 7 is a frozen-table measurement.** It shows the *arrangement* ranks better; it does
   not show that training can reach that arrangement while the link loss pulls elsewhere. The
   arm is precisely the test of that, and §6.3 says how to tell which way it failed.
2. **The target is weak early in the stream.** Causal in-degree is ~0 for most nodes at early
   query times, so `π ≈ 0` and nearly everything targets `r_hi` — uniform expansion with no
   ordering. The term only becomes informative as history accumulates, so its effect is
   back-loaded within each epoch. Do not read a null at low epoch counts.
3. **Static vs per-step target.** Test 7 validated a *static* train-degree arrangement; the
   arm trains against a *per-step causal* target whose fixed point is each node's mean causal
   in-degree. On YouTube these orderings are near-identical (`R²(deg_train, deg_t) = 0.972`),
   which is why transfer is expected — but it **is** an assumption, and review is the standing
   reminder that such things do not transfer by default.
4. **Option (B) can run away** (§5.4). Guard `r_mean`.
5. **One seed.** Everything here is seed 5. The YouTube effect is large relative to this
   suite's seed noise but has not been replicated.
6. **`π` is extremely compressed** under `C = log1p(max)` (median 0.13, only 0.16% above 0.5).
   This is the validated setting, but it means the term is doing most of its work by pushing
   the unpopular mass outward, not by finely ordering the popular head.

### 7.2 Not established — the obvious follow-ups

- **The pooled-radius variant.** `r(p_v)` beats `r(E[v])` on every diagnostic (R² 0.272 vs
  0.227 on YouTube, 0.122 vs 0.063 on review) **and is dynamic**, so supervising it escapes
  the static ceiling S4 (which caps the `E` route at 78% / 64% of the popularity gain). Its
  score coupling on review is **+0.685 against −0.079** for the table radius — so it is the
  one route with a live transmission path on the dataset where the `E` route fails. The
  matching upper-bound test (displace `p_v` radially toward `r̂` on the frozen checkpoint and
  re-score) **was designed but not run**. It should be run before anyone concludes review is
  closed.
- **Why review's pooling discards E's radius** (`corr = −0.0789`). Unexplained. It is the
  real question on that dataset, not a larger λ.
- **`C = log1p(p99)` with clipping** — would use the radial band far more evenly; untested.
- **Out-degree on review.** A static out-degree reranks +0.1570 there (better than dynamic
  total degree), yet radial repositioning by out-degree is −0.0265. That gap is not understood.
- **GoogleLocal** has essentially no popularity signal at all (+0.0007), and is the one
  dataset this project currently wins. It is a regression guard, not a target.

### 7.3 Do not re-run

- Any **scorer** radius/popularity channel: four arms, net −0.041 at best, mechanism known
  (§4.3). `feature/score-d0v`, `feature/score-d0v-norm`, `feature/score-rv-perquery`,
  `tmp/duv-over-rmean`.
- Anything using **`standardise`** on a per-batch reduction: measured leak worth +0.046 on
  YouTube (§4.5).
- **`get_node_degrees`** for any causal quantity: no cutoff, leaks future edges.
- **`T_train` as a timestamp cutoff**: it is a span; returns zero edges (§3.5).
- The **optimistic-rank MRR**: inflates by 29% on tie-heavy data (§3.1).

---

## 8. Provenance

### Scripts (all on NFS in `scripts/`, uncommitted on `master`)

| script | what it does | outputs |
|---|---|---|
| `run_mckpt.sh` | master + `--save-checkpoint` + `--export-best-embedding-table`, with tree-identity and no-radius-term guards | `logs/mckpt/` |
| `test1_radius_vs_degree.py` | Spearman + deciles of radius vs degree | `logs/analysis/test1-11600705.out` |
| `test23_rerank.py` | tests 2 & 3, post-hoc rerank | `logs/analysis/test23-116012{14,15,16}.out` |
| `test5_geom_popularity.py` | tests D1–D3, S1–S4, diagnostics, score caching | `logs/analysis/test5-116012{39,40}.out` |
| `test6_popularity_source.py` | Tempest cutoff check + degree-definition comparison | `logs/analysis/test6-116012{67,68}.out` |
| `test7_radial_reposition.py` | **the decisive test**, with scale-matched controls | `logs/analysis/test7-116013{28,29}.out` |
| `radius_target_constants.py` | `r_lo`, `r_hi`, `C`, `π` distribution | `logs/analysis/const-11601365.out` |

Cached per-query score matrices (regenerated automatically if deleted):
`logs/analysis/cache/t5_{YouTube,tgbl-review}_{val,test}.npz`.

Superseded outputs are kept and renamed rather than deleted:
`logs/analysis/test23-11600728.SUPERSEDED-slow-mrr.out`,
`logs/analysis/test23-11600721.SUPERSEDED-optimistic-rule.out`,
`logs/analysis/test23-116012{14,15}.ABORTED-tie-rule.out` (if present),
`logs/analysis/superseded/test7-116013{07,08}.no-scale-control.out`.

### Commits referenced

| commit | relevance |
|---|---|
| `02f4ce4` | master, this branch's base; `LinkPredHead: compute the table's mean radius once per forward` |
| `815a938` | produced the original bars; verified equivalent to `02f4ce4` |
| `2ed7db8` | `feature/scorer-cand-popularity` — the causal `get_node_popularity` reference, and the `standardise` leak |
| `f941793` / `87b4d27` | the per-batch `standardise` leak and its removal; +0.046 on YouTube |
| `c2ba9d6` | `feature/score-rv-perquery`, the most careful rejected scorer arm |
| `78de031` | the negatives rework — changed TGB-Seq **val** only, not test |

### Environment

Artemis, `artemis-rtx-0{1,2}`, NVIDIA RTX PRO 6000 Blackwell Server Edition. Python venv at
`venv/`. Logs live in `logs/` (git-untracked); `experiment_logs/` is the curated archive and
nothing here has been copied there.

---

## 9. Summary of the decision

| question | answer | evidence |
|---|---|---|
| Is popularity worth chasing? | Yes — the largest untapped signal | +0.1529 review / +0.0617 YouTube post-hoc (§3.3) |
| Can it reach the score through geometry? | **Yes on YouTube, no on review** | +0.0320 / −0.0265 ordering, scale-controlled (§3.6) |
| Through the scorer? | No — four arms, net −0.041, mechanism known | §4.3 |
| Which geometric coordinate? | Table radius `r(E[v])` now; pooled radius `r(p_v)` is the better untested follow-up | §3.4, §7.2 |
| Which popularity definition? | **In-degree** (+0.0320 vs total +0.0145, out-degree −0.0312) | §3.6 |
| Where does popularity come from? | A **directional causal index** at per-query exclusive cutoffs — Tempest's undirected `get_node_popularity` gives total only | §3.5, §5.6 |
| What is the term? | `λ·mean[(ρ + π − 1)²]`, globally normalised, scorer untouched, master bitwise at λ=0 | §5.1 |
| Where to run it? | **YouTube only**; review and GoogleLocal as regression checks afterwards | §6.1 |
