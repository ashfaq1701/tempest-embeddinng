# Early val peak, dip, slow recovery, then patience: symptoms on the TGB suite and what the literature says

**Date:** 2026-10-26. **Code:** master `633a5a0` (`score = w * -dist(p_u, p_v)`), plus the
`feature/degree-negatives` arm `3f14de4`. **Config:** seed 5, d128, k_train 5, k_eval 100, wpn 10,
mwl 5, lr 1e-3, hidden 32, per-dataset flags from `default_conf_variations.txt`, 100-epoch cap,
patience 10. **Logs:** `logs/dneg_m633a5a0` (control), `logs/dneg_dn3f14de4` (degree).

Companion literature review with full citations and links:
`reports/early_peak_dip_rise_literature_2026-10-26.md`.

---

## 1. The symptoms

### 1.1 Three of five control cells select in the first three epochs

| dataset | test@val-ckpt | max test | best val | sel / ran |
|---|---|---|---|---|
| tgbl-wiki | 0.7834 | 0.7848 | 0.8157 | 23 / 33 |
| tgbl-review | 0.1463 | 0.1464 | 0.1340 | **2** / 12 |
| tgbl-coin | 0.7031 | 0.7036 | 0.6871 | **3** / 13 |
| tgbl-flight | 0.7777 | 0.7777 | 0.7936 | **1** / 11 |
| tgbl-comment | 0.6733 | 0.6733 | 0.6546 | **2** / 12 |

Only tgbl-wiki selects late. On the other four the run's reported number is set within three
epochs and the remaining 9-11 epochs change nothing.

### 1.2 tgbl-coin: peak at ep3, trough at ep6, six epochs of monotone recovery, patience fires

| ep | 1 | 2 | **3** | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val | 0.6792 | 0.6856 | **0.6871** | 0.6863 | 0.6710 | 0.6183 | 0.6324 | 0.6556 | 0.6607 | 0.6638 | 0.6674 | 0.6690 | 0.6675 |
| test | 0.7036 | 0.7012 | **0.7031** | — | — | — | — | — | — | — | — | — | — |
| link | 0.4279 | 0.2211 | 0.1495 | 0.1155 | 0.0958 | 0.0836 | 0.0744 | 0.0672 | 0.0619 | 0.0579 | 0.0548 | 0.0521 | 0.0499 |

Recovery from the ep6 trough: **+0.0507 over six epochs**, increments decaying
(+0.0141, +0.0232, +0.0051, +0.0031, +0.0036, +0.0016), then -0.0015 at ep13 as the counter hit
10/10. At the stop it was **0.0181 short** of ep3. Train loss fell every epoch, 0.4279 -> 0.0499.

### 1.3 tgbl-flight: peak at ep1, whole run inside a 0.0057 val band, rising when patience fired

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | **0.7936** | 0.7929 | 0.7915 | 0.7919 | 0.7892 | 0.7886 | 0.7879 | 0.7879 | 0.7880 | 0.7884 | 0.7904 |

Val bottomed at ep7-8 and rose for the last three epochs without reaching ep1. Train loss
0.0804 -> 0.0518, monotone. Test evaluated once, so 0.7777 has zero drift.

### 1.4 The degree-negatives arm is the same shape, exaggerated

tgbl-coin under `--train-negatives degree`: train loss falls 0.6620 -> 0.3824 in five epochs then
moves only 0.3824 -> 0.3578 over the next 35, while `r_mean` grows ~+0.68/epoch to 29.5 and
`r_max` to 141.6. Val sits in a 0.0092 band from ep6 to ep40 while +0.0003-scale flickers at
ep14/15/18/22/24/25/30 keep resetting patience: **34 of 40 epochs bought +0.0092 val**.
tgbl-flight and tgbl-comment in that arm both select at **ep1 of 11**.

### 1.5 Flickers resetting patience is a recurring cost, not a one-off

Already recorded in CLAUDE.md on other suites: a `rad` arm lost 0.0054 to drift when nine epochs
of +0.0006 val flickers walked the checkpointed number down; the Lorentz/ball YouTube runs had
best-val values 0.0114 apart selecting test numbers 0.031 apart. The same mechanism, on a metric
whose epoch-to-epoch noise has never been measured.

---

## 2. One analytic fact that reframes all of the above

`self.w = nn.Parameter(torch.tensor([1.0]))`, unconstrained, and the scorer is
`self.w * -dist(pooled_u, pooled_v)`. MRR ranks candidates **within a query**, and rank order is
invariant under multiplication by a positive constant.

> **While `w > 0`, changing `w` alone cannot change MRR at all. Not by a little — exactly zero.**

`w` is unconstrained in code but positive in every log we have (always printed `+`), so the
invariance holds in practice but is not enforced.

The training loss is a softmax/ranking loss over candidates and **is** scale-sensitive, so `w`
has a gradient and moves. Therefore:

- **A monotone train loss is not evidence of anything here.** Part of the loss reduction happens
  along a direction the metric is mathematically blind to. "Train loss fell every epoch" does not
  mean "not overfitting"; it means the loss is partly measuring something MRR ignores.
- Any gradient budget spent on `w` is **wasted with respect to MRR**.
- `w` is not inert: it rescales the loss and hence the gradients reaching the embeddings and
  pooler, so it modulates *learning* while being invisible to *evaluation*.
- The invariance **breaks the moment the score gains an additive term** (e.g. a popularity bias).
  That is itself a diagnostic: if a drifting scale starts moving MRR, the score is no longer a
  pure positive rescaling.

The same argument applies to the `feature/pooled-distance-normalised` arm (`3df624f`), which
divides the score by a global `r_mean`: at fixed weights its MRR is identical to master's.

---

## 3. What the scale actually does, per cell (measured)

`r_mean` is the table's mean radius (detached); `dist(p_u, p_v)` is of order `r_mean`, so the
score's magnitude is of order `w * r_mean`.

| cell | epochs | link | r_mean | r_max | w | **w · r_mean** | val |
|---|---|---|---|---|---|---|---|
| ctl tgbl-flight | 11 | 0.080→0.052 | 6.08→38.9 (6.4x) | 20.6→73.9 | 4.79→0.67 (0.14x) | **29.10→25.90 (0.89x)** | peak ep1 |
| deg tgbl-flight | 11 | 0.539→0.546 | 3.89→48.4 (12.5x) | 21.6→255.6 | 1.91→0.14 (0.07x) | **7.43→6.77 (0.91x)** | peak ep1 |
| deg tgbl-coin | 40 | 0.662→0.358 | 1.39→29.5 (21.2x) | 10.3→141.6 | 4.44→0.30 (0.07x) | 6.18→8.79, **8.69→8.79 from ep5** | peak ep30 |
| ctl tgbl-comment | 12 | 0.293→0.126 | 2.76→13.9 (5.0x) | 13.7→39.4 | 2.85→0.72 (0.25x) | 7.86→9.99 (1.27x) | peak ep2 |
| ctl tgbl-coin | 13 | 0.428→0.050 | 2.01→10.2 (5.1x) | 19.2→60.6 | 3.64→1.74 (0.48x) | 7.31→17.78 (2.43x) | peak ep3 |
| **ctl tgbl-wiki** | 33 | 1.428→0.038 | 0.16→1.67 (10.6x) | 1.2→6.2 | **1.15→2.51 (2.18x, grows)** | 0.18→4.19 (**23.1x**) | **0.7352 ep1 → 0.8157 ep23** |

**The correlation worth noticing.** In the cells that peak at ep1-2 and then flatten, the product
is constant to within 11% while `r_mean` grows 6-12x — the radius growth is being divided straight
back out, and the score's scale never changes. In **tgbl-wiki**, the one control cell that keeps
improving (val +0.0805 from ep1 to ep23), `w` **grows** alongside `r_mean` and the product rises
23x. tgbl-coin is intermediate: the product grows 2.4x and it selects at ep3 but genuinely
recovers afterwards.

So "the model inflates the radius and cancels it with `w`" is **not** a universal description —
it characterises the stuck cells specifically. n = 6 cells, one seed, so this is a **suggestive
correlation, not an established mechanism**; it is cheap to test on more cells.

---

## 4. Mechanisms from the literature

The second rise is the most diagnostic feature: it rules out the mechanisms that predict
peak-then-monotone-decline.

| mechanism | dip | **second rise** | monotone train loss | fit to our evidence |
|---|---|---|---|---|
| Metric-invariant loss progress (loss–metric mismatch) | neutral | permits any shape | yes, trivially | **strong** — §2 is a proof for our scorer |
| Effective-LR drift from norm growth | yes | **yes** (implicit annealing) | yes | **strong** — §3's pinned product is its signature |
| Epoch-wise double descent | yes | **yes, eventually past the first peak** | yes | strong, with caveats (§6) |
| Simplicity bias → memorisation | yes | **no, on its own** | yes | partial; needs pairing |
| Gradient starvation / shortcut dominance | yes | yes, if starved features later get gradient | yes | plausible — see below |
| Lazy → rich transition, catapult, edge of stability | yes | yes | short-term non-monotone | plausible, unmeasured at step granularity |
| Grokking | — | yes (sharp) | yes | **weak**: needs small data + weight decay, starts from chance |
| Adam adaptive-variance warmup | yes | yes | yes | weak: window is a fraction of ep1 at our data size |
| BN / dropout train-eval mismatch | yes | yes | yes | **N/A**: neither is in this model |
| Loss of plasticity / dormant units | no | **no — explains failure to re-reach the peak** | yes | possible secondary only |
| Winner's curse on a noisy metric | yes (regression to mean) | yes | yes | **strong for flight**, where the whole band is 0.0057 |

**Effective-LR drift** is the leading candidate. For scale-invariant parameters the gradient scales
as `1/||w||`, so growing norms *are* an unintended LR decay schedule (van Laarhoven 2017; Li &
Arora ICLR 2020 prove schedule + weight decay folds into a pure exponential LR; Zhang et al. ICLR
2019 show effective LR is weight decay's dominant mechanism for Adam too; Kosson et al. ICML 2024
extend it to Adam/AdamW/Lion and show rotation control removes the chaotic early transient).
Kodryan et al. NeurIPS 2022 identify three regimes set by effective LR — convergence, chaotic
equilibrium, divergence. That predicts our two phases exactly: a high-effective-LR noisy phase
(the ep4-6 dip) then a low-effective-LR crawl (the six-epoch recovery), with patience firing on a
run whose step size has become too small to finish.

**Epoch-wise double descent** (Nakkiran et al. ICLR 2020) is the named phenomenon for the shape,
amplified by label noise — our analogue being false negatives in sampled negatives. Heckel &
Yilmaz (ICLR 2021) argue the hump is the superposition of bias–variance tradeoffs of parts of the
network learning at different speeds, and **eliminate it by tuning per-layer step sizes**.
Stephenson & Lee (arXiv:2108.12006) find it appears only above a critical noise level.

**Gradient starvation** (Pezeshki et al. NeurIPS 2021) deserves a look here specifically: our own
record shows a popularity-only predictor **beating** the model on tgbl-review (0.2660 vs 0.1380).
A fast popularity fit peaking val in 1-2 epochs, suppressing gradient for relational structure, is
exactly that configuration, and it predicts a recovery once the easy term saturates.

---

## 5. What to do

### (a) Methodology — the cheapest and most defensible fix

**"Stop training" and "select the checkpoint" are two different decisions and need two different
rules.** Patience-on-best is a stopping rule being misused as a selection rule, and that is how
tgbl-coin and tgbl-flight were ended while val was rising.

1. **Measure the metric's noise floor first.** Evaluate one checkpoint ~5x with re-drawn eval
   negatives; bootstrap MRR over queries. This sets `min_delta` from data instead of guesswork. If
   sigma >~ 0.005, tgbl-flight's entire 0.0057 band is noise and **there is no phenomenon there to
   explain** — the ep1 peak would be a winner's-curse artefact. Do this before anything else.
2. **Require improvement over a window mean with `min_delta` >> sigma**, not over the running max.
   This kills the +0.0003-flicker-resets-patience failure directly.
3. **Adopt a progress-aware stopping criterion.** Prechelt's PQ (in *Neural Networks: Tricks of the
   Trade*, 1998) divides generalisation loss by a training-progress measure over a strip of epochs,
   on the explicit rationale that while training error is still falling fast a generalisation loss
   has a good chance of being repaired. Our coin train loss fell 0.4279 -> 0.0499 monotonically:
   **PQ would not have stopped it at ep13.**
4. **Minimum-epoch floor**, set from the observed trough location, not a round number.
5. **Never select on validation loss** — only on the metric. Guo et al. (ICML 2017) show best-val-loss
   and best-val-metric epochs differ, because NLL keeps falling via confidence inflation.
6. **Keep recording both test@val-checkpoint and max test** (already project policy) and note that
   "best of 13 epochs" and "best of 100 epochs" are different estimators, so arms with very
   different run lengths are not comparable on a selected max (Dodge et al. EMNLP 2019). Our tables
   span stop epochs 1 to 118.
7. **Checkpoint averaging** (SWA, EMA, latest-k) converts "which noisy epoch won" into a
   lower-variance estimate. Not retrospective for us: we save only the best checkpoint, so this
   needs a logging change first.

### (b) Dynamics — interventions that could make the late phase beat the early peak

Only two have evidence tied to *this curve* rather than to deep learning in general:

- **Per-part learning rates** (embedding table vs pooler). Heckel & Yilmaz demonstrably eliminate
  epoch-wise double descent this way and improve the early-stopped result. We already have a
  `--lr-pooler` knob, and an earlier attempt at it gained nothing — but that was tested as a
  regulariser, not as a double-descent fix, and not with this diagnosis in hand.
- **Longer training with an explicit decay** (cosine to zero, 5x budget, early stopping off). The
  double-descent literature predicts the second descent eventually passes the first minimum.

Supporting, weaker: **remove or freeze the degenerate scale** (strongest *local* evidence — §2 and
§3 — but no published study of an embedding-table-plus-one-scalar model); **weight decay / AdamW**
to stop the radius running away, which this project has repeatedly observed; **lower lr** (noting
Lewkowycz et al.: the best final result is often in the large-LR phase, so this may remove the dip
*and* the ceiling); **warmup / RAdam** purely to exclude the Adam-variance mechanism.

### (c) The single decisive experiment

One run on tgbl-coin: **early stopping off, 5x epoch budget, cosine decay to zero, validation 4x
per epoch**, logging `w`, `r_mean`, `r_max`, per-group `||dtheta||/||theta||`, and MRR recomputed
after rescaling the score by a fixed constant. It separates the two readings our data cannot:

- tail **passes** ep3's 0.6871 ⇒ we were **truncating**; the fix is schedule + budget + stop rule.
- tail **saturates below** ⇒ **irreversible early damage**; the fix is in the objective or the
  optimiser, and more epochs will never help.

Sub-epoch eval also answers whether ep1-3 is a peak at all, or a coarsely-sampled rising curve.

---

## 6. Honest limits

- **The pinned-product correlation is n = 6 cells, one seed.** It separates our stuck cells from
  tgbl-wiki cleanly, but six points and no replicate is a hypothesis, not a result.
- **The effective-LR theory assumes per-layer scale invariance** (BN-style). We have *one*
  degenerate direction, not an invariance group. Whether the regime picture transfers is unverified,
  and it is the biggest leap in the diagnosis.
- **No published study covers epoch-wise double descent in a ranking metric**, nor with
  sampled-negative false positives as the noise source, nor with a first descent completing in 1-3
  epochs — published curves take tens to hundreds of epochs. The transfer is a conjecture.
- **Nothing in the literature explains a second rise that plateaus *below* the early peak.** Double
  descent predicts it passes; memorisation predicts no recovery. Ours sits between, and whether that
  is truncation or damage is untested.
- **"Metric-invariant loss progress" is not a named failure mode** in the literature as far as the
  review found. Guo et al. (confidence inflation) and Huang et al. (loss–metric mismatch) are close
  but neither frames it as gradient budget spent on a direction with exactly zero metric gradient.
  Measuring the fraction of each update lying in that subspace would be a small clean contribution.
- **Checkpoint selection as a winner's-curse instance** is not treated in the literature the review
  found; epochs are highly correlated candidates, so the standard bias formulas do not apply. Treat
  any correction as an estimate.
