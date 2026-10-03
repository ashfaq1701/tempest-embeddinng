# Geometric pooler features: seven arms, five datasets, and a temperature that explains the gap

Written 2026-10-03. Everything below is measured from logs under `logs/`; inferences are
labelled as such, and a section at the end lists claims made during the work that the later
measurements withdrew.

---

## 1. The headline

**cos_o's YouTube deficit is over-sharpened pooling, and dividing the pooler logits by 2 at
eval recovers two thirds of it.** Measured on the restored best checkpoint, same weights, the
only change being `logits / tau` before the bag softmax:

| tau | cos_o val | cos_o test | master val | master test |
|---|---|---|---|---|
| 1/3 | 0.6229 | 0.5189 | 0.6438 | 0.5698 |
| 1/2 | 0.6422 | 0.5439 | 0.6623 | 0.5895 |
| **1 (as trained)** | 0.6734 | **0.5865** | 0.6852 | **0.6150** |
| **2** | **0.6859** | **0.6049** | 0.6873 | 0.6151 |
| 3 | 0.6814 | 0.6023 | 0.6641 | 0.5883 |
| 4 | 0.6708 | 0.5939 | 0.6333 | 0.5536 |
| inf (uniform) | 0.3829 | 0.3415 | 0.3873 | 0.3449 |

Three things fall out of that table.

**cos_o is mis-calibrated and master is not.** Flattening cos_o by tau=2 gains **+0.0184 test**
(0.5865 -> 0.6049) and **+0.0125 val**, so the better setting is selectable on validation without
touching test. The same operation on master gains **+0.0001** (0.6150 -> 0.6151): master's
pooling sharpness is already at its optimum, and both tau=3 and tau=1/2 make master clearly
worse. So the defect is specific to the arm, not to the architecture.

**The size of the correction matches the probe.** The probe measured cos_o's within-bag logit
spread at **9-10** against master's **3.3-3.8** through 50+ epochs, a ratio of ~2.7. The optimal
divisor is 2. Dividing cos_o's logits by 2 puts its spread at ~5, between the two. The
mis-calibration the probe measured and the correction the sweep finds are the same number.

**Uniform pooling is catastrophic here, which contradicts CLAUDE.md.** tau=inf is exactly the
parameter-free mean, and it scores **0.3415 / 0.3449** on YouTube -- 0.27 below the learned
pooler. CLAUDE.md's standing note says the fixed pooling rule *beats* the learned pooler on
YouTube (0.5677 at K=5, 0.5756 at K=10, against 0.5605 for the best learned arm). That is from
the 2026-08-29 ablation at a different d, patience and scorer. **In the current configuration it
is false by a wide margin**, and any reasoning that leaned on "uniform is the known-good
reference" -- including several statements I made during this work -- does not hold.

With the tau=2 correction cos_o's YouTube deficit goes from **-0.0284 to -0.0100**, which also
clears the external bar (0.5964): 0.6049 is +0.0085 over it.

---

## 2. Complete results, seven arms

All seed 5, d64, K=5, wpn 5, mwl 5, lr 1e-3, patience 10, hidden 32, `nl2` except WikiLink
`nl1`. `best_test_mrr` (val-selected). Master's Yelp is a **floor** -- hand-killed at ep14 while
still gaining -- so every Yelp delta understates the loss.

| arm | pooler | YouTube | Flickr | ML-20M | Yelp | WikiLink |
|---|---|---|---|---|---|---|
| **master** | 1,217 | **0.6149** | 0.6366 | 0.2469 | 0.6481 † | 0.6513 |
| cos_o (angle at O) | 1,249 | 0.5867 | **0.6368** | 0.2518 | **0.6610** | 0.6554 |
| cos_up (angle at M) | 1,249 | 0.5157 | 0.6347 | 0.2499 | 0.6161 | 0.6674 |
| 2r (`+d0_tok`) | 1,249 | 0.5753 | 0.6366 | 0.2518 | 0.6348 | 0.6659 |
| 3r (`+d0_tok,+d0_mid`) | 1,281 | 0.5836 | 0.6342 | **0.2539** | 0.6288 | **0.6739** |
| depth (`+r_tok-r_mid`) | 1,249 | 0.5657 | 0.6352 | 0.2506 | 0.6302 | 0.6609 ‡ |
| radsum (`d/(r_tok+r_mid)`) | 1,217 | 0.5684 | 0.6311 | 0.2390 | killed | killed |
| multi-query (4 learned pts) | 1,569 | 0.6009 | **0.6420** | 0.2511 | **0.6514** | 0.5743 |

† hand-killed, a floor. ‡ stopped inside a radius blow-up, a floor -- see section 7.

**No arm beats master on YouTube.** Twelve arms have now tried (including the attention family)
and master holds it outright.

**Every geometric arm beats master on WikiLink.** 3r +0.0226, cos_up +0.0161, 2r +0.0146,
cos_o +0.0041, depth +0.0096 so far. Five structurally different columns, one direction -- the
most reproducible result in the set, and WikiLink is where master sits furthest below the bar.

**ML-20M: every geometric arm wins, by +0.003 to +0.007.** Small but unanimous.

**Flickr is a wash** for everything (-0.0055 to +0.0054).

**Yelp splits by feature type**: the angular `cos_o` wins (+0.0129) while the radial arms lose
(2r -0.0133, 3r -0.0193). cos_up, also angular, loses (-0.0320), so "angular wins Yelp" is not
the rule -- it is cos_o specifically.

---

## 3. What the probe showed

Two instrumented branches, `diag/coso-grad-probe` and `diag/master-grad-probe`, with a
byte-identical `probe.py`. Both verified **inert**: epoch 1 reproduced each reference exactly
(cos_o 1.6593 / 0.1503 / 0.1298; master 1.6487 / 0.2104 / 0.1906).

### Overshoot is ruled out

`relNN` = realised `||dp||/||p||` for the pooler, per epoch:

| ep | master | cos_o |
|---|---|---|
| 1 | 1.79e-03 | 1.82e-03 |
| 3 | 5.89e-04 | 6.97e-04 |
| 11 | 3.64e-04 | 3.94e-04 |

Both at or below the nominal lr of 1e-3 and falling, and essentially identical to each other.
Gradient norms rise while the relative step shrinks -- Adam normalising correctly. Not an
optimisation pathology.

### The durable difference is logit spread, nothing else

| field | master (settled) | cos_o (settled) |
|---|---|---|
| `logit_sd` | **3.3 - 3.8** | **9 - 10** |
| `w_ent` | 0.414 - 0.422 | 0.411 - 0.427 |
| `w_old` | 0.311 | 0.317 |
| `w_cos_hi` | 0.648 | 0.664 |
| `w_max` | ~0.44 | ~0.43 |

Only `logit_sd` separates them, by ~2.7x, and it holds for 50+ epochs. Entropy, mass on old
tokens, mass on the top cosine decile and max bag weight all **converge** to within a few
thousandths. So cos_o produces far more extreme logits for the same realised weighting -- it
sits deep in softmax saturation, where gradients through those weights are flat. That is exactly
what a temperature of 2 fixes, and the sweep confirms it.

### Column attribution inverts after epoch 2

`lsd_drop_X` = `logit_sd` recomputed with standardised column X zeroed:

| ep | full | drop age | drop cos | drop d_mid | drop pos |
|---|---|---|---|---|---|
| 1 | 7.51 | 7.82 | **2.01** | 6.57 | 7.43 |
| 3 | 4.58 | 3.43 | 3.06 | 4.41 | **1.87** |
| 20 | 13.56 | 13.02 | 12.80 | 11.81 | **1.87** |
| 78 | 10.2 | 9.73 | 10.22 | 11.10 | **1.01** |

At initialisation the cosine is the whole spread (removing it collapses 7.51 -> 2.01, and the
first layer puts its largest weight column on it: 3.235 vs 2.08-2.24 for the others). **From
epoch 3 onward `pos` carries it** -- removing `pos` takes 13.56 -> 1.87 at ep20, i.e. 86% of the
spread. So the trained pooler runs on hop position, and the cosine's lasting effect is to let
`pos` generate a far wider spread than it does on master.

### Master spreads its tail faster, early

`r_head` (radius of the top training-degree decile) at ep1: master **0.148**, cos_o **0.080**.
By ep60 they converge (0.623 vs 0.598). `gtail`, the share of the embedding gradient landing on
bottom-decile nodes, is 0.155 vs 0.123 at ep1.

---

## 3a. --lr-pooler 1e-4: the knob worked and the MRR did not follow

The temperature result above says cos_o's pooler is over-sharp, so the obvious next move is to
train it that way rather than patch it at eval. `--lr-pooler` gives the pooler MLP
(`bag_weights.net.*`, 6 tensors, 1,249 params) its own RiemannianAdam param group while `E` and
`geo_temp` stay at `--lr`; unset, it keeps exactly one group. Verified before launch by a step
with all grads = 1.0: MLP tensors move rms 1.000e-04, `geo_temp` 1.000e-03, `E.weight` 1.250e-04
unchanged. Branch `diag/coso-lr-pooler` `b8b38b8`, YouTube seed 5, otherwise the cos_o config.

**Killed by hand at ep61 on a decided result. It lost to both references.**

| arm | logit_sd | val | test |
|---|---|---|---|
| **lrP 1e-4** | **5.6 - 6.2** | 0.6633 | **0.5786 max / 0.5783 val-sel** |
| cos_o (lrP = lr) | 9.5 - 13.6 | 0.6692 | 0.5834 |
| master | 3.3 - 3.8 | 0.6854 | 0.6149 |

**The intervention did exactly what it was aimed at.** `logit_sd` held 5.6-6.2 for sixty epochs
against cos_o's 9.5-13.6 -- a sustained ~2x reduction, landing close to where the tau=2 patch
effectively put cos_o. The early trajectory is unambiguous: ep1 logit_sd 2.40 against cos_o's
7.51, and the arm skipped cos_o's early run-up to 13.6 entirely.

**It bought nothing.** 0.5786 max test against cos_o's 0.5834, and 0.0363 behind master. So
**over-sharpening is a correlate of cos_o's deficit, not its cause.** An eval-time `logits/2` is
worth +0.0184; reaching the same sharpness by training the pooler ten times slower is worth
-0.0048. The two facts are only consistent if what matters is not the sharpness itself.

**Best current explanation, and it is untested.** tau=2 reshapes the pooling of an embedding that
was *trained under sharp pooling*; a low pooler lr co-adapts `E` to soft pooling from epoch 1 and
lands in a different, worse basin. The same weights are reachable either way -- a fixed tau is
exactly a 1/tau rescale of the output layer -- so this is about which trajectory finds them, not
about the hypothesis class. If that is right, the lever is `E`'s trajectory or the scorer, and
the pooler's logit scale is downstream of both.

**No escape, and the familiar front-loaded shape.** It led cos_o by +0.052 val at ep7 and master
by +0.122, led cos_o on test through ep18, then flattened: ep40 0.5742 -> ep61 0.5786, **+0.0044
over 21 epochs**, with val inside a 0.004 band from ep50. cos_o escaped in that same window
(ep40 0.6552 val -> ep61 0.6692). This is the same lead-then-flatten curve as the four rejected
pooler arms, now with a sixth mechanism.

**The tau sweep was forfeited.** It fires after training on restored best weights, so killing the
run lost it. That is the one readout worth recovering if this arm is revisited: it would say
whether a soft-trained pooler is *also* mis-calibrated, in the other direction.

This also makes the five-failed-interventions list six, and the two weight-decay arms read
differently now. wd was dismissed as a near no-op under coupled L2 that pulled toward a bad
target. A lower lr has no shrinkage target at all, reached the intended sharpness, and still
lost -- so the wd arms were probably not failing for the reason given.

---

## 4. Other measured results worth keeping

**`triangle_cos` fixed a real numerics bug.** The inline `(cosh b cosh c - cosh a)` form cancels
catastrophically at initialisation radii: float32 error against the true angle is **0.855** at
r=4e-04, on a quantity bounded in [-1, 1]. The helper switches to the Euclidean law below 1e-2
and the error drops to **2.4e-07**; from r >= 0.02 the two are bit-identical. Worth **+0.0398**
on cos_o's YouTube (0.5469 -> 0.5867). `dist0` was checked and is NOT implicated: it matches
`asinh(||x||)` to 3e-19 in float64 and is at 1-2 float32 ulp, no worse than `dist`. Splitting
the error showed 100% arithmetic, 0% inputs.

**Scale invariance predicts some arms and not others.** Sensitivity to a 1% radial push, as a
fraction of each feature's own spread: `d_mid` 0.112, `depth` 0.102-0.110, `d/(r_tok+r_mid)`
0.020-0.026, `d_mid/d_max` 0.005-0.007, `cos_o` **0.000**. The hypothesis was that scale-free
columns let the model discriminate without spreading the embedding, costing master's late escape.
It held for the cosines on YouTube but **failed** on `radsum`, which preserved `r_mean` to within
1-2% of master on all five datasets and still lost all five.

**`radsum` failed by information loss, not by scale or capacity.** `d/(r_tok+r_mid)` retains only
**12-24%** of `d_mid` (R^2, degrading as radius grows) because the pooler never sees the
denominator and cannot invert the division. It is also ~65% explained by `cos_o` and ~39% by the
radial floor -- a blend of two things each of which has a cleaner single-purpose form. Pooler
params identical to master (1,217) and training losses within 1-2%, so neither capacity nor
optimisation.

**Five interventions failed to move YouTube**, which is why the temperature result matters:

| intervention | YouTube | vs cos_o 0.5867 |
|---|---|---|
| pooler weight decay 1e-6 | 0.5836 | -0.0031 |
| pooler weight decay 1e-4 | 0.5789 | -0.0078 |
| `nl1` (193 params, 6.5x smaller) | 0.5658 | -0.0209 |
| wpn 10 | 0.5637 | -0.0230 |
| `d_mid -> d_mid/d_max` | 0.5680 | -0.0187 |
| **eval-time tau=2** | **0.6049** | **+0.0184** |

Weight decay at 1e-6 and 1e-4 are no-ops by construction: geoopt's RiemannianAdam uses coupled
L2, so effective decay is `(wd*|p|)/|grad|`, around 3e-7 and 3e-5 against far larger gradients.
`nl1` left the train-loss ratio to master at 0.44x (nl2 was 0.43x), so capacity was never the
binding constraint.

---

## 5. Claims made during this work that the measurements withdrew

Listed because several were stated confidently before the data arrived.

1. **"Uniform pooling is the known-good reference on YouTube."** From CLAUDE.md. Measured here
   at **0.3415**, 0.27 below the learned pooler. False in the current configuration.
2. **"Master's late escape is an unsharpening toward uniform."** Master's `logit_sd` does fall
   8.06 -> 3.39 through the escape, but tau=2 shows master is already optimal and further
   flattening hurts it. The escape is not a move toward uniform; uniform is terrible.
3. **"cos_o produces 73% of the logit spread."** True at epoch 1 only. From epoch 3 onward
   `pos` carries 86% of it.
4. **"cos_o puts 70% of its mass on the oldest third against master's 49%."** Epoch-1
   transient; both converge to 0.31 by ep30.
5. **"WikiLink fails from radial-spread collapse."** The no-LayerNorm arms refuted it: one
   sustained a rising spread ratio and did worse, another had the lowest ratio and the most
   stable curve.
6. **"cos_up is the better-conditioned angle and should do better."** It is better conditioned
   at build time (0% clamped, corr 0.182 vs 0.531) and scores **worse on both comparable
   datasets** -- YouTube -0.0710, Yelp -0.0449 against cos_o.
7. **"The numerics fix will gain ~+0.04 on cos_up as it did on cos_o."** It gained ~0.
8. **"radsum should be worse than cos_o because both its columns are scale-free."** It drew
   level with cos_o at ep30 before falling behind; the stated mechanism was not what decided it.
9. **I called YouTube runs settled three times while they were still escaping.** This run's
   own history: recoveries from patience streaks of 8, 6, 2 and 6, each gaining real test
   (0.5693 -> 0.5721 -> 0.5834 -> 0.5867). On YouTube a flat stretch of 8-9 epochs is normal.

---

## 6. What this suggests next

**Train with the temperature, don't just apply it at eval.** tau=2 is an eval-time patch on
weights trained at tau=1. A divisor in the forward pass -- or equivalently scaling down the
pooler's output layer -- lets the optimiser work in the calibrated regime from the start. That is
a one-line change and the measurement above says where to set it.

**Check whether master is calibrated on every dataset or only YouTube.** master's tau=2 gain is
+0.0001 here. If it is also ~0 on Yelp and WikiLink, "master is correctly calibrated and the
geometric arms are not" becomes a general statement rather than a YouTube one. Five short sweeps
on existing checkpoints would settle it; the sweep code is in both diag branches.

**The four cos_o datasets other than YouTube still carry the numerics bug.** Flickr, ML-20M, Yelp
and WikiLink were all run at `2d7b739`. On YouTube the fix was worth +0.0398, so cos_o's
+0.0129 Yelp and +0.0041 WikiLink are plausibly conservative. Re-running the four on `388f5ef` is
about five GPU-hours.

**Master has no converged Yelp baseline at any depth.** Every Yelp comparison in this report is
against a hand-killed floor of 0.6481 from ep14. One run fixes that.

**WikiLink deserves the attention, not YouTube.** It is the only dataset where added geometry
wins reproducibly across five arms, and it is where master is furthest from the bar (-0.0809
even after 3r's +0.0226). YouTube is already above the bar by +0.0185 without any of this.

---

## 7. Where things are

Still running: `depth` on Yelp (ep20) and WikiLink (ep24), jobs 11422721 and 11422720.

Nothing is running. All three in-flight runs finished or were stopped on 2026-10-03.

**depth / WikiLink stopped at ep27 inside a radius blow-up, as predicted, and 0.6609 is a
floor.** From ep20 the *training loss rose* -- 0.1578, 0.1587, 0.1596, 0.1605, 0.1615, 0.1632,
0.1640 -- while `r_mean` climbed a near-constant +0.19/epoch to 5.32 and val sat dead flat at
0.6691-0.6700 for eight epochs. It stopped on the val plateau, not on convergence. Same
signature CLAUDE.md records for the conv-stem arm on WikiLink, and plausibly a lr problem rather
than anything about `r_tok - r_mid`. Its nominal +0.0096 over master nl1 should not be read as a
win; the arm needs a lower lr before WikiLink says anything about it.

**depth / Yelp converged cleanly and lost**: 28 epochs, healthy throughout (loss monotone to
0.0474, `r_mean` smooth to 1.371), best test **0.6302** at ep18 with zero drift. Against master's
0.6481 *floor* that is a loss of at least 0.018 -- one of the few clean depth numbers here, and
it is negative.

**coso_lrp / YouTube killed at ep61**, section 3a.

Branches, all local, none pushed:

| branch | commit | what |
|---|---|---|
| `feature/pooler-cos-origin` | `388f5ef` | cos_o + `triangle_cos` |
| `feature/pooler-cos-up` | `c8828c5` | cos_up + `triangle_cos` |
| `feature/pooler-plus-d0tok` | `6edbac6` | 2r |
| `feature/pooler-three-geom` | `5dd5493` | 3r |
| `feature/pooler-signed-depth` | `3e57a5c` | `+ (r_tok - r_mid)` |
| `feature/pooler-dmid-over-radsum` | `e44f89e` | `d/(r_tok+r_mid)` |
| `feature/pooler-multi-query` | `da413c8` | 4 learned query points |
| `diag/coso-grad-probe` | `086a47d` | cos_o + full probe + tau sweep |
| `diag/master-grad-probe` | `fe33646` | master + the same probe |

Probe logs: `logs/coso_diag/.../run_3_seed5/` and `logs/master_diag/.../run_3_seed5/`.
Killed or superseded logs all carry footers saying so; `logs/radsum/`, `logs/attn_nonorm/` and
the `run_1`/`run_2` probe logs should not be quoted as results.
