# Tempest walk-supervised temporal link prediction

## TGB-Seq datasets

The suite we target: link prediction, MRR on TGB-Seq's shipped TEST negatives.
Listed ascending by edge count.

| # | dataset | edges | bipartite |
|---|---|---|---|
| 1 | GoogleLocal | 1.91M | yes |
| 2 | YouTube | 3.29M | no |
| 3 | Flickr | 7.22M | no |
| 4 | Patent | 10.8M | no |
| 5 | ML-20M | 14.5M | yes |
| 6 | Taobao | 18.85M | yes |
| 7 | Yelp | 19.8M | yes |
| 8 | WikiLink | 34.2M | no |

**Bipartite flags are authoritative from `tgb_seq/datasets/preprocess.py::bipartite_dict`,
not from notes.** The four bipartite datasets — GoogleLocal, ML-20M, Yelp, Taobao — need
`--is-bipartite`; passing it wrongly changes the negative-candidate pool.

A bare run:

```
scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset <name> \
  --use-gpu --use-gpu-tempest [--is-bipartite]
```

**Download.** TGBSeqLoader auto-downloads into a fresh dir, Taobao included. A
half-downloaded dataset (CSV present, `test_ns` missing, from an interrupted fetch) is
self-healed by `load_tgb_seq`'s preflight: it checks both files and refetches the missing
`test_ns` from `TGB-Seq/<name>` on HF.

## Where run logs live

**Every run writes to `logs/`, never to `experiment_logs/`.** `logs/` is git-untracked
(`.gitignore`), lives on this disk, and is the working tree Claude Code writes into and
reads back — it is the only place a run in flight or a run just finished can be found.

Organize under `logs/<experiment>/<cell>/<tag>/<Dataset>.log`, e.g.
`logs/simple_head_2/d64_k5/run_1/YouTube.log`. `<experiment>` names the code change under
test, `<cell>` the config (`d64_k5`), `<tag>` the replicate (`run_1`). A sweep driver's own
`DRIVER.log` lives in that same tag directory; the driver script itself belongs in `scripts/`,
parameterized so its output root points into `logs/`.

Traceability is on the log itself. Every log opens with a header naming dataset, cell, tag,
d_emb, k_train, lr, branch, **commit**, start time, and the full command line. If the working
tree carries uncommitted changes, say so in the header — a bare commit SHA that does not
describe the code that ran is worse than no SHA. Commit the code change before launching
where you can; that makes the SHA sufficient on its own.

`experiment_logs/` is the curated, git-tracked archive: **selected result logs only**, copied
in once a run is finished and judged worth keeping, and permanent thereafter. No driver
scripts, no shell scripts, no scratch or superseded logs, no in-flight runs. A log is copied
from `logs/` to `experiment_logs/` — never moved, and never written there directly by a run.

## The NN pooler's feature set: `rec, pos, rad` (measured, 2026-08-29)

Pooling weights are `softmax(MLP(features))` over the walk-token bag. **`HIDDEN` is a fixed
32, not `8 * N_FEAT`.** Under the old rule the hidden width moved with the feature count, so
every feature ablation silently changed pooler capacity too and the two effects could not be
separated. Pin it before comparing feature sets.

Four-way ablation, YouTube d=64 K=5 lr=1e-3 seed 42, no pop bias, hidden 32 in every arm,
commit `e6b6079c`. Logs: `logs/pooler_feature_ablation/d64_k5/`.

| features | params | ran | stop | test@val-ckpt | max test | max@ | escape |
|---|---|---|---|---|---|---|---|
| rec, pos | 129 | 20 | 17 | 0.5286 | 0.5293 | ep16 | ep13 |
| **rec, pos, rad** | 161 | 30 | 27 | **0.5551** | **0.5605** | ep18 | ep13 |
| rec, pos, dev | 161 | 30 | 27 | 0.5511 | 0.5526 | ep23 | ep18 |
| rec, pos, rad, dev | 193 | 28 | 25 | 0.5547 | 0.5547 | ep25 | ep17 |

`dev` = each token's geodesic distance to the bag's unweighted centroid, a per-token spread
signal. It was **removed**; recover it from `e6b6079c` if you want to re-run these arms.

**Both geometric features are individually real, and they do not compose.** Over the
no-geometry baseline `rad` is worth +0.031 max test and `dev` +0.023. But `rad+dev` lands
*below* `rad` alone: `dev` is largely redundant with `rad` and pays for the overlap in delay.

**The mechanism is escape timing, not height.** Every arm shows the same explore-then-escape
shape: a slow grind to ~0.36, then a three-epoch jump of ~+0.15, then a plateau. `rad` escapes
at ep13, `dev` at ep18 after a five-epoch stall, `rad+dev` at ep17 as a flattened ramp
(+0.044, +0.016, +0.006) rather than a jump. **`dev` delays the escape in every arm it appears
in.** Read a pooler A/B by when the escape fires and from what plateau, not by warmup height.

**Warmup ordering inverts — do not call these runs early.** `dev` led at ep1-3 (+0.014 at ep1,
the best ep1 of any arm) and finished third; `rad` trailed at ep1 and finished first. The
`1553a103` commit message justified `dev` with "+0.01-0.016 on YouTube through the warmup",
which is exactly the window that inverts. Pre-escape super-additivity misleads the same way:
`rad+dev` beat the sum of the solo gains by +0.021 at ep12 and still lost.

**Two contrasts, only one of them controlled.** `seed_all` runs before model construction, so
the two 3-feature arms (`+rad` vs `+dev`) are `Linear(3,32)` either way and draw bit-identical
initial weights — that comparison isolates the feature exactly. Arms with different input
widths draw different RNG (the 2-feature arm's first-layer weights do not match the 4-feature
arm's first two columns, and even the output layers differ), so those comparisons mix the
feature with an initialisation change. `rad`'s +0.0058 over `rad+dev` is one of them: the choice
rests on parsimony plus the clean `rad`-beats-`dev` contrast, not on that number.

**Nothing here beats the parameter-free pooler on YouTube.** Best arm 0.5605 vs the fixed
pooling rule's 0.5677 (K=5) and 0.5756 (K=10); LB #1 GraphMixer is 0.5887. The learned pooler
is still behind the rule it replaces on this dataset.

**Watch the val/test drift when reporting.** `rad` peaked 0.5605 at ep18, then nine epochs of
+0.0006 val flickers kept resetting patience and walked the checkpointed number down to 0.5551
— a 0.0054 loss to drift. `rad+dev` drifted 0.0000. Always record both test@val-checkpoint and
max test; ranking on the printed `best_test_mrr` alone is not like-for-like across arms.

## Effective walk length per dataset (measured, 2026-09-03)

How deep a backward walk actually gets before it runs out of causal history, as opposed
to the cap it was allowed. Train split ingested into Tempest, 20k seed nodes drawn
uniformly from the nodes appearing in train, K=5 backward walks each = 100k walks per
dataset, `max_walk_len=80`, ExponentialWeight start and walk bias, no cutoff time, seed
42. Numbers are `WalkData.lens`, cross-checked against the padding mask on all 800k
walks (exact agreement). Reproduce with `scripts/walk_length_stats.py`.

| dataset | train edges | mean | std | max | p50 | p90 | p99 | % at cap 80 | % >= 5 |
|---|---|---|---|---|---|---|---|---|---|
| GoogleLocal | 1,802,833 | 8.22 | 5.51 | 65 | 7 | 15 | 27 | 0.00 | 72.4 |
| YouTube | 2,730,407 | 8.15 | 5.53 | 61 | 7 | 16 | 26 | 0.00 | 69.8 |
| Flickr | 5,738,138 | 9.81 | 9.87 | 71 | 6 | 24 | 45 | 0.00 | 57.2 |
| Patent | 9,096,058 | 2.20 | 0.47 | 9 | 2 | 3 | 4 | 0.00 | 0.3 |
| ML-20M | 14,000,091 | 47.40 | 29.33 | 80 | 45 | 80 | 80 | 36.43 | 98.6 |
| Taobao | 13,981,096 | 17.49 | 11.01 | 80 | 15 | 33 | 51 | 0.01 | 93.0 |
| Yelp | 17,025,551 | 16.03 | 12.51 | 80 | 13 | 33 | 58 | 0.11 | 87.4 |
| WikiLink | 27,635,253 | 8.27 | 5.94 | 53 | 7 | 16 | 28 | 0.00 | 67.7 |

Minimum length is 2 in every dataset; no walk returns the seed alone.

**Patent walks are structurally dead at length 2.** Mean 2.20, max 9 over 100k walks, and
only 0.3% reach length 5. It is a citation graph: an edge points at prior art, so one
backward step lands on a node whose own citations are almost all *later* than the cutoff
and the walk has nowhere causal left to go. This is a better explanation for Patent's
outlier MRR (0.2140 against 0.63-0.69 elsewhere) than anything in the model — the bag is
the seed plus one neighbour. **Raising `--max-walk-len` on Patent cannot help**; the
history is not there to walk. Depth-oriented changes should exclude Patent, and a Patent
regression in a depth experiment is not evidence about depth.

**ML-20M is right-censored at the cap**: 36% of walks hit 80, so the true mean exceeds
47.4 and this row is a lower bound. 110k active nodes carry 14M edges (~127 per node), so
a walk essentially never exhausts its history.

**Training at the default `--max-walk-len 5` truncates most reachable history.** The
`% >= 5` column is the fraction of walks the default cap cuts: 98.6% ML-20M, 93% Taobao,
87% Yelp, ~70% GoogleLocal/YouTube/WikiLink, 0.3% Patent. Whether depth *helps* is
untested — this only establishes the headroom exists everywhere except Patent.

**Bipartite structure raises reachable depth**, but sparsity beats it: ML-20M 47.4,
Taobao 17.5, Yelp 16.0 against 8.2-9.8 for the non-bipartite sets, with bipartite
GoogleLocal the exception at 8.22 — it is also the sparsest at 1.8M edges over 474k nodes.

Active-node counts (nodes appearing in train): GoogleLocal 473,580; YouTube 402,422;
Flickr 233,836; Patent 1,840,152; ML-20M 110,431; Taobao 1,623,633; Yelp 1,743,769;
WikiLink 1,361,972.

## The gap to CRAFT (measured, 2026-09-13)

CRAFT, "Future Link Prediction Without Memory or Aggregation" (arXiv 2505.19408), is the
current state of the art on TGB-Seq and the bar this work has to clear. Its numbers below are
Table 2 of the paper, mean of 3 runs, MRR in **percent**. SGNN-HN is included because it
**beats CRAFT on two datasets** — GoogleLocal and YouTube — so "the bar" is the better of the
two, not CRAFT alone.

**CRAFT does not report Patent.** It evaluates seven TGB-Seq datasets; Patent is ours only.
No claim of the form "we beat CRAFT on Patent" can be made.

### Current master, seed 5 — the honest column

Ours is `best_test_mrr` (val-selected) from `experiment_logs/geometries/lorentz/*/3.log`,
commit `07dcc1e6`, d=64 K=5 lr=1e-3 patience 5, no popularity channel, single seed.

| dataset | ours (seed 5) | stop ep | SGNN-HN | CRAFT | bar | **Δ vs bar** | % walks ≥5 |
|---|---|---|---|---|---|---|---|
| GoogleLocal | **65.35** | 39 | 62.88 | 62.35 | 62.88 | **+2.47** ✅ | 72.4 |
| Flickr | **62.51** | 13 | 60.15 | 62.34 | 62.34 | **+0.17** ✅ | 57.2 |
| YouTube | 56.26 | 19 | 59.64 | 58.92 | 59.64 | −3.38 ❌ | 69.8 |
| WikiLink | 65.92 | 15 | 69.37 | 75.48 | 75.48 | −9.56 ❌ | 67.7 |
| Yelp | 62.04 | 14 | 69.34 | 72.69 | 72.69 | −10.65 ❌ | 87.4 |
| ML-20M | 24.29 | 7 | 33.12 | 35.91 | 35.91 | −11.62 ❌ | 98.6 |
| Taobao | 53.13 | 4 | 68.58 | 70.68 | 70.68 | −17.55 ❌ | 93.0 |
| Patent | 22.63 | 13 | — | — | — | (not reported) | 0.3 |

**Standing: 2 of 7.** Total deficit across the five losses is 52.8 MRR points.

### The old per-dataset records are withdrawn

Earlier notes carried a much better 4-of-7 against this bar. Those numbers are **not this
architecture and are not reproducible**: they mix seeds, carry a per-node popularity bias table
(`head params` ~= one scalar per node), and come from a `[-geo, cos, rad]`-family scorer that is
now measured to be unstable rather than merely suspected of it.

**The instability, measured 2026-09-14** on `w.[-d_H, cos]`, ML-20M seed 5, one variable against
the baseline. It peaked at **epoch 2** — val 0.2955, test 0.2539, the best of any head tried —
and never recovered:

| ep | 1 | **2** | 3 | 4 | 5 |
|---|---|---|---|---|---|
| val | 0.2854 | **0.2955** | 0.2908 | 0.2909 | 0.2936 |
| link | 0.5936 | 0.4549 | 0.4204 | 0.4018 | 0.3904 |

Training loss fell every epoch while validation turned over after one. The mechanism is visible
in the radius: `cos` is exactly scale-invariant (measured `d(cos)/d(radius) = 3.7e-09`), so it
gives a cheap **angular** way to separate candidates and the embedding never needs to spread —
r_mean 0.049 at ep1 against the baseline's 0.225, r_max collapsing to 1.038 while the baseline
climbed to 3.116. The objective gets solved without geometry, and then there is nothing left to
generalise with.

**Do not quote the old records, and do not treat them as headroom.** A number that survives two
epochs is not a result.

### Two confounds in the seed-5 column

**Deficit and early stopping correlate across the whole table**, which is a confound: an
under-trained run and a capacity-limited one look identical here. Taobao has the largest deficit
(−17.55) and the shortest run (**stop ep4**); WikiLink loses 9.56 and stops at **ep15**, where
an earlier long run on this dataset reached **ep44**. Neither has been checked for convergence.
ML-20M is the one place it has been: its val genuinely saturates and declines after ep7, so
there the deficit is real. **Before reading Taobao or WikiLink as capacity losses, run them long.**

**Walk truncation is not a single-axis explanation.** Sorting by `% walks ≥5`, truncation still
separates the three worst deficits (Yelp/Taobao/ML-20M, all ≥87%) from everything else, and
depth is worth testing there — but WikiLink sits at 67.7% truncation and still loses by 9.56,
in between two wins. Patent must stay out of any depth conclusion; its walks are structurally
dead at length 2.

### Where to attack first

ML-20M: biggest *verified* deficit, cheapest to iterate (242 s/epoch, stops by ep12, ~55 min a
run, against Taobao's 1381 s and Yelp's 1780 s), and the extreme point on truncation. Find the
mechanism there, then **confirm on Taobao before believing it generalises** — ML-20M is an
outlier in density too (127 edges/node against Taobao ~8.6 and Yelp ~9.8), so a fix that
exploits deep dense history need not transfer.

## Four poolers, one null result (measured, 2026-09-28/29)

Four architectures were built off master, run on 3-5 datasets each and rejected in one session.
All seed 5, d=64, K=5, wpn 5, mwl 5, lr 1e-3, patience 10, hidden 32, against **arm B = current
master `9f24967`** with a depth-matched `n_layers_pooler 2` pooler (1,217 params).

| arm | commit | pooler params | YouTube | Flickr | ML-20M | Yelp | WikiLink |
|---|---|---|---|---|---|---|---|
| **master (arm B)** | `9f24967` | 1,217 | **0.6149** | **0.6366** | **0.2469** | 0.6481 † | 0.6500 † |
| hierarchical GRU | `be74a7a` | 7,234 | 0.4722 ‡ | 0.5165 ‡ | 0.2413 ‡ | — | — |
| hierarchical conv | `cb3405e` | 3,586 | 0.5842 | 0.6242 | **0.2556** | 0.6589 ‡ | 0.6632 ‡ |
| conv + pointwise stem | `6005d39` | 6,434 | 0.6129 | **0.6369** | — | **0.6592** | 0.5530 ✗ |
| tangent displacement | `46e195a` | 1,217 | 0.4968 ‡ | 0.6053 ‡ | 0.2478 | — | — |

† lower bound, run killed before convergence (ep14 / ep11). ‡ killed mid-run, not converged.
✗ diverged. Bold = best converged number per dataset.

**The three converged head-to-heads are −0.002, +0.0003 and +0.009.** Nothing in
this family beat master, and the conv arms cost **1.3-1.6x master per epoch** (YouTube 61.6s vs
43.5s train, Flickr 93.6s vs 60.0s, ML-20M 193s vs 115.8s) at 3-5x the parameters. The GRU cost
**3-4x** (YouTube 181s, ML-20M 958s) for the same null.

### The shape they all share: front-load, then flatten

Three of the four led master by **+0.05 to +0.18 val in the ep8-25 window and all lost**, because
master has a late second escape none of them reproduce — YouTube val 0.583 -> 0.685 over ep25-50,
+0.068 in three epochs. The conv arm led by +0.143 val at ep15 and crossed under at ep30; the
tangent arm led by +0.073 at ep18 and was -0.092 by ep35.

**Capacity is not the constraint, and this is measured, not inferred.** The conv arm reached
*identical training loss* to master on YouTube (ep50 link 0.0387 vs 0.0388) while scoring 0.031
worse on test. The hierarchy arms' **val->test gap was 0.008-0.015 WIDER** than master's
(YouTube 0.086 vs 0.071). They fit the objective as well and generalised less.

**Do not re-run these.** A hierarchy over `[Q, K, L]` cannot enlarge the hypothesis class: any
non-negative weight matrix `W[k,l]` factors as `w_walk[k] * w_tok[k,l]` with
`w_walk[k] = sum_l W[k,l]`, so GRU and conv were *reparameterisations* of master's flat softmax,
not extensions. Deeper convs, bigger kernels, GRU+conv hybrids and transformer set encoders
(1 win / 4 losses, -0.126 on WikiLink) are all the same move.

### What the pointwise stem fixed, and what it did not

`6005d39` put a `Conv1d(C, H, kernel_size=1)` — a per-token Linear — in front of the two `k=3`
convs. Master is two per-token Linear->GELU stages; the 2-conv arm had **zero** purely per-token
stages, its first layer projecting 4->32 *and* mixing three positions in the same weights. The
stem makes the conv arm a superset of master and recovered **+0.029 on YouTube and +0.013 on
Flickr**, taking both from a real deficit to a tie. Receptive field is unchanged at 5, which at
mwl 5 already spans the whole walk, so a third `k=3` stage buys capacity and no new context.

### Two instabilities the conv arms have and master does not

**WikiLink diverged** on the stem arm: stopped on val at ep8, then from ep9 the *training loss
rose* (0.1925 -> 0.1988) while `r_mean` climbed a constant +0.20/epoch to 3.92 and val thrashed
between 0.335 and 0.640. **Yelp** stopped at ep18 with `r_max` running 5.67 -> **8.14**. Both are
radius blow-up, and both are plausibly fixable with a lower lr or a pooler-specific lr — so they
are *not* evidence about hierarchy, and WikiLink's -0.097 should not be quoted as a capacity loss.

### Reading these numbers later

**Master has never been run to convergence on Yelp or WikiLink.** Its references above (0.6481 at
ep14, 0.6500 at ep10) come from runs killed by hand. On WikiLink the val increments had decayed to
+0.0002/epoch and test had already turned over (0.6500 at ep10 -> 0.6477 at ep11), so that one is
close to its plateau; Yelp's was still gaining +0.003/epoch. **Also: the arm-B WikiLink reference
is `n_layers_pooler 1` (162 params) while every arm above ran `nl 2`** — so every WikiLink delta
in this table mixes in a pooler-depth change. A clean WikiLink baseline does not exist.

**Drift decides ties.** The stem arm's YouTube *val* (0.6906) and *max test* (0.6157 at ep64) both
beat master's (0.6854 / 0.6149), yet its val-selected checkpoint reports 0.6129 because val
flickered up at ep68 and ep73 while test fell. Record both numbers or the ranking is not
like-for-like.

### Where this leaves the attack

Four unrelated mechanisms — recurrence, convolution, per-token depth, leaving the convex hull —
each landed within 0.009 of master and none of them ahead. **The pooler is not where the headroom is.** The open question
is what the flat softmax does between ep25 and ep50 on YouTube that none of these reproduce, and
the untested axes remain walk depth (`--max-walk-len`, 70-99% of reachable history is discarded at
5), breadth (`--num-walks-per-node`; wpn 10 measured at only **1.11x** the train cost of wpn 5,
one epoch, then killed) and the scorer.

## Walk-sequence poolers and step columns: master wins 4/5 (measured, 2026-10-05)

Two independent families, run as two tables. All seed 5, d64, K5, wpn 5, mwl 5, lr 1e-3,
patience 10, hidden 32. **The baseline is the POST-`triangle_cos` master** (`logs/cosomaster`,
commit `7e68aa7`), which is not the same as the older `logs/coso` references — see "the baseline
moved" below.

### Table A: does the pooler need walk structure?

Five arms. `ema` = single-rate normalised EMA over the older hops (one learned fade rate `alpha`);
`ph` = the one older hop through a learned linear map. `nl1`/`nl2` is pooler depth, and each arm
reduces to master **bitwise** at its degenerate setting (`alpha -> 0`, `w_prev = 0`).

| arm | params | YouTube | Flickr | ML-20M | Yelp | WikiLink |
|---|---|---|---|---|---|---|
| **cos_o (master)** | 1,249 / 193 nl1 | **0.5867** | **0.6377** | 0.2515 | **0.6523** | **0.6607** |
| ema-nl1 | 194 ᴰ | 0.5668 | 0.6312 | 0.2513 | 0.6348 | 0.6596 |
| ema-nl2 | 1,250 | 0.5708 | 0.6329 | 0.2539 | 0.6304 | 0.6652 ◊ |
| ph-nl1 | 225 ᴰ | 0.5584 † | 0.6302 | 0.2500 | 0.6307 | **0.6647** |
| ph-nl2 | 2,273 | 0.5652 | 0.6381 | **0.2542** | ✗ dropped | ✗ dropped |

† **FLOOR, not a result**: hit the `--num-epochs 100` cap with its best AT ep100, val still
rising. ᴰ **nl1 arm against an nl2 master** — on YouTube/Flickr/ML-20M/Yelp these two rows mix
the mechanism with a 5-6x capacity cut, because a cos_o **nl1** run exists only for WikiLink.
Only the nl2 rows are depth-matched there. ◊ **no depth-matched baseline exists** — 1,250 params
against master's nl1 193 — NOT a delta, do not quote it as one. ✗ both ph-nl2 heavy cells were killed
when the family was dropped -- WikiLink at ep5 (max 0.6449) and Yelp at ep13 (max 0.6392, still
gaining, against a cos_o reference that ran 34 epochs). Both are floors; neither is a result.
WikiLink was uninterpretable anyway (2,273 params against master's nl1 193). A clean cos_o
**nl2** WikiLink run has never been made.

**So the Yelp column has no ph-nl2 entry**, and ema-nl2's 0.6304 (-0.0219) is the only
depth-matched nl2 Yelp number.

**VERDICT, called 2026-10-05: the recurrence family is a no-go and was dropped.** Master is
complete at 5/5 and wins every dataset except WikiLink. Across four arms and two mechanisms the
only win is ph-nl1's +0.0040 on WikiLink. The last two runs were cancelled mid-flight rather than
finished, so **18 of 25 cells are converged results and 2 are floors that must not be quoted**;
nothing from this family is queued and no GPU work remains.

**Master wins 4 of 4 completed datasets.** Deltas vs master: YouTube −0.016 to −0.028 for all
four arms, Yelp −0.018 to −0.022, Flickr −0.005 to −0.008 (ph-nl2 +0.0004, a tie), ML-20M −0.002
to **+0.0027**. ML-20M is the only dataset where arms lead, and by ~0.0025.

**The one legitimate win is ph-nl1 on WikiLink, +0.0040**, depth-matched (225 vs 193). It is also
the only dataset where any arm beats master, now across fifteen attempts.

**nl2 > nl1 within each mechanism** (ema +0.0040, ph +0.0068 on YouTube) but **"nl2 beats master"
does not generalise** — it holds only on ML-20M. An earlier read of partial columns suggested
depth mattered more than mechanism; the final numbers do not support it.

**alpha is NOT logged per epoch, so both EMA nulls are uninterpretable at the mechanism level.**
A null cannot distinguish "memory unused, `alpha -> 0`, i.e. master" from "memory used and
unhelpful". Evidence points at collapse: on WikiLink ema-nl1 tracked cos_o nl1 to within 0.001
train loss, 0.002 `r_mean` and 0.0003 val at matched epochs, which is what `alpha -> 0` looks
like. **Add an `alpha=` field to the epoch line before running any further EMA arm.**

### The one door left open, and it was not tested

The family was dropped on the numbers above, which is defensible, but **the nl2 losses are
confounded with INITIALISATION and that confound was never controlled.** Table A's arms do not
start at master: `ema` initialises `alpha_init = 0.5`, i.e. the memory fully ON before any
gradient step, and `ph` leaves `w_prev` at the default `nn.Linear` draw, i.e. the prior hop
contributes RANDOMLY at init. Table B's step arms zero-initialise and so start BITWISE as master.
The arms that start at master cost ~0.000-0.003; the arms that start far from it cost 0.016-0.022.

The tell is ema-nl2: it is a strict superset of master with **one** extra parameter, and `alpha ->
0` recovers master bitwise (verified) -- yet it lost 0.016 on YouTube and 0.022 on Yelp. A single
scalar that can switch the mechanism off should not cost that much if the optimiser could get
back. So these runs measure "can the optimiser switch recurrence off from a bad start", not "does
recurrence help". Same shape as the `--lr-pooler` result, where an eval-time tau=2 gained +0.018
but training at a low pooler lr gained nothing.

**If this is ever reopened, the test is two runs:** ema-nl2 with `alpha_init ~ 1e-3` and ph-nl2
with `w_prev` zero-initialised, making both exact supersets of master at init. Deltas collapsing
toward 0 would mean the 0.02 was initialisation and recurrence is neutral; deltas staying at 0.02
would confirm the verdict on its own terms. Neither run was made.

### Table B: feed the previous hop as feature columns instead

Three columns added to master's four, all **zero-initialised with master's exact RNG
consumption**, so each arm starts BITWISE identical to master (verified: forward equal, max|diff|
0.000e+00) and the columns enter only by gradient.

`d_step = d(x_l, x_{l-1})`, `cos_step = triangle_cos(r_tok, r_prev, d_step)`,
`age_step = log1p(age_{l-1} - age_l)`.

| arm | params | YouTube | Flickr | ML-20M | min train loss (YT / FL / ML) |
|---|---|---|---|---|---|
| **cos_o (master)** | 1,249 | **0.5867** | **0.6377** | 0.2515 | 0.0290 / 0.0602 / 0.3967 |
| step-geo (`d`,`cos`) | 1,313 | 0.5865 | 0.6348 | 0.2539 | 0.0282 / 0.0562 / 0.3874 |
| step-geo-age (+`age`) | 1,345 | 0.5837 | 0.6349 | **0.2546** | **0.0255 / 0.0561 / 0.3792** |

**Both arms undercut master's training loss on all three datasets and generalised no better.**
That is the project's standing law holding again: driving train link loss below master's does not
transfer. step-geo-age has the lowest loss everywhere and the worse MRR of the two on 2 of 3.

**These columns are nearly free, unlike Table A.** Deltas are −0.0002/−0.003 on YouTube and
−0.003 on Flickr against Table A's −0.016 to −0.028, and +0.002/+0.003 on ML-20M. So **the large
YouTube damage in Table A tracks the ARCHITECTURAL change, not the sequence information** —
feeding the previous hop as standardised columns costs almost nothing; restructuring the pooler
to consume it costs 0.02+.

`logit_sd` is **unavailable** for Table B: these drivers derive from `run_cosomaster_arm.sh`,
which carries no probe. That diagnostic is missing, not null.

### Two things that will mislead the next reader

**The baseline moved, and in both directions.** Four of five old `logs/coso` references predate
`triangle_cos` (commit `2d7b739`). Re-running master post-fix gave Flickr 0.6368 -> **0.6377**,
ML-20M 0.2518 -> **0.2515** (down, via drift: max was 0.2521), Yelp 0.6610 -> **0.6523** (down
0.0087, the largest revision), WikiLink 0.6554 -> **0.6607** (up 0.0053). That WikiLink revision
alone flipped ema-nl1 from a +0.0042 win to a −0.0011 loss. **Always state which baseline a delta
uses.** YouTube reproduced 0.5867 exactly, which also confirms the `[Q,K,L]` walk_tokens refactor
(`4b1a919`) is inert end-to-end through a full 78-epoch run, not just in its unit test.

**Four Table-A cells ran on A40, not RTX** — cos_o ML-20M, ema-nl2 Flickr, ph-nl2 YouTube and
Flickr (headers show `gpu=NVIDIA A40`). Per-epoch timings are not comparable and bitwise
reproduction is not guaranteed. Table B's ML-20M column therefore compares RTX arms against an
A40 baseline.

### The direction trap, which cost five runs

Walks are stored in **time order**: index 0 is the OLDEST hop, `lens-1` is the seed. Measured on
a real chain walk (`0->1->...->5` at t=10..50, backward from node 5 at cutoff 60) the ages come
out 50, 40, 30, 20, 10, 0 along increasing index. So **increasing index is forward in time** and
the previous hop is `l-1`.

The first versions of all three sequence arms read `l+1` and so conditioned every token on its
own FUTURE. The cause was a docstring: `l+1` is "the hop it was reached from" in walk-GENERATION
order, because the sampler starts at the seed and steps backward — and generation order is the
reverse of time. Five runs were launched and killed on it. The same trap bit `age_step`, where
the natural-looking `age_l - age_{l-1}` is NEGATIVE here and `log1p` of it is NaN (verified).
**Every driver for these arms now pins the direction with an abort guard.** Do not describe `l+1`
as "the hop it was reached from".
