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
`--is-bipartite`. The flag sets what a negative can be: destinations only when bipartite,
sources ∪ destinations otherwise. Training negatives come from the train split; our eval
negatives (TGB-Seq val, DyGLib val/test) from the full dataset (train ∪ val ∪ test), the splits
CRAFT draws both val and test from. Runs before this change drew eval negatives from the train
split only, so their val MRRs (and DyGLib test MRRs) are not directly comparable with runs after.

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

### OBSOLETE (2026-10-10): the `ours` column below is withdrawn

**Only `ours` and `Δ vs bar` are obsolete. The SGNN-HN / CRAFT bar is still the bar** — those
come from the papers, not from a run of ours.

Those `ours` numbers were `best_test_mrr` (val-selected) from
`experiment_logs/geometries/lorentz/*/3.log` at commit `07dcc1e6`, d=64 K=5 lr=1e-3
**patience 5**, no popularity channel, single seed. **That path no longer exists**: the
`experiment_logs/` archive was emptied on 2026-10-10 (`79339b1`); working copies survive in
`logs/lorentz`, and the deleted files are in git history at `79339b1^`.

Four reasons not to quote the column:

- the **linear-skip pooler** is now master (`a035e0e`): `softmax(skip(f) + MLP(f))` with
  `skip = Linear(3,1,bias=False)` init `[-1,-1,0]`, which `07dcc1e6` does not have;
- the **`triangle_cos` numerics fix** (`7e68aa7`) landed after it and is worth **+0.0398 on
  YouTube** on its own;
- the **negatives rework** (`78de031`, 2026-10-10) draws eval negatives from train ∪ val ∪ test
  instead of the train split, so val MRRs before and after are not comparable;
- **patience 5** truncates the late escape this suite shows between ep25 and ep50 — every stop
  epoch in the table is 4-39, and YouTube's ep19 stop is exactly the window where a +0.068
  three-epoch jump has since been measured.

**Current skip-master TGB-Seq numbers live in**
`logs/seq_skip/d64_k5_wpn5_mwl5_lr1e-3_pat10_h32_nl2_skip/run_1_seed5/` (WikiLink in the `nl1`
sibling), seed 5, patience 10: GoogleLocal 0.6701, YouTube 0.6087, Flickr 0.6313, ML-20M 0.2468,
Yelp 0.6537, WikiLink 0.6533. Against the bar above that is **3 of 6 clear** (GoogleLocal +4.13,
YouTube +1.23, Flickr +0.79 in MRR %), not 2 of 7 — but those six were run before `78de031`, so
they will need re-running on current master before they go in a paper.

| dataset | ~~ours (seed 5)~~ | stop ep | SGNN-HN | CRAFT | bar | ~~**Δ vs bar**~~ | % walks ≥5 |
|---|---|---|---|---|---|---|---|
| GoogleLocal | **65.35** | 39 | 62.88 | 62.35 | 62.88 | **+2.47** ✅ | 72.4 |
| Flickr | **62.51** | 13 | 60.15 | 62.34 | 62.34 | **+0.17** ✅ | 57.2 |
| YouTube | 56.26 | 19 | 59.64 | 58.92 | 59.64 | −3.38 ❌ | 69.8 |
| WikiLink | 65.92 | 15 | 69.37 | 75.48 | 75.48 | −9.56 ❌ | 67.7 |
| Yelp | 62.04 | 14 | 69.34 | 72.69 | 72.69 | −10.65 ❌ | 87.4 |
| ML-20M | 24.29 | 7 | 33.12 | 35.91 | 35.91 | −11.62 ❌ | 98.6 |
| Taobao | 53.13 | 4 | 68.58 | 70.68 | 70.68 | −17.55 ❌ | 93.0 |
| Patent | 22.63 | 13 | — | — | — | (not reported) | 0.3 |

~~**Standing: 2 of 7.** Total deficit across the five losses is 52.8 MRR points.~~
**Stale** — see the obsolescence note above; on the skip-master runs the standing is 3 of 6.

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

## Poincare ball vs Lorentz hyperboloid: equivalent on 5 of 6 (measured, 2026-10-06)

Six datasets, one seed (5), d64 K5 wpn5 mwl5 lr1e-3 h32, nl2 except WikiLink nl1. Lorentz is
`4649142` (`logs/ballinit`), the ball is `6bb43ea` (`logs/poincare1`). **YouTube ran 200 epochs
/ patience 20 on both sides; the other five ran 100 / 10 on both sides.** val is best-val, test
is the val-selected `best_test_mrr`.

| dataset | Lorentz val/test | stop | ball val/test | stop | d (L-P) | L r_max | P r_max |
|---|---|---|---|---|---|---|---|
| GoogleLocal | 0.6849 / 0.6574 | 44 | 0.6830 / **0.6577** | 44 | -0.0003 | 1.05 | 1.15 |
| YouTube | 0.6823 / 0.5933 | 118 | 0.6855 / **0.6075** | 118 | **-0.0142** | 3.12 | 3.00 |
| Flickr | 0.6706 / **0.6417** | 28 | 0.6680 / 0.6378 | 28 | **+0.0039** | 2.73 | 3.24 |
| ML-20M | 0.2927 / **0.2510** | 9 | 0.2916 / 0.2500 | 9 | +0.0010 | 4.85 | 4.56 |
| Yelp | 0.6701 / 0.6528 | 25 | 0.6691 / **0.6534** | 24 | -0.0006 | **7.75** | **6.21** |
| WikiLink | 0.6812 / 0.6723 | 16 | 0.6815 / **0.6728** | 17 | -0.0005 | **7.28** | **6.21** |

**Five of six land within 0.004 and four pairs stop at the IDENTICAL epoch** (44, 118, 28, 9).
Summed across six the ball is ahead by 0.0107, all of it YouTube. Lorentz wins Flickr +0.0039
and ML-20M +0.0010; the ball wins YouTube -0.0142 and three cells by <=0.0006.

### This comparison is only meaningful because the init was matched first

The original sweep was NOT a chart-only A/B, and the difference it showed was mostly an init
artefact. Two things had to be established:

**The pooler is the same operation in both charts.** geoopt's `PoincareBall.weighted_midpoint`
is the Einstein/gyromidpoint; `lorentz.py::weighted_midpoint` is the Lorentzian centroid (Law et
al. 2019). They are the same point in two charts -- both are the Klein-model affine average, with
Klein coords `k_i = x'_i/x0_i` and `gamma_i = x0_i` making the Einstein form equal `s'/s0`.
Measured agreement across the chart map: **1e-12 to 1e-10** on bags of 2 to 125 with uniform,
one-hot and skewed weights. So the pooler is NOT a difference between the arms.

**The inits differed by 2.3x on three stacked conventions**, fixed in `4649142`: `irange` was a
per-coordinate scale while geoopt's `std` is a tangent-norm scale (factor sqrt(n)); the box is
anisotropic; and PoincareBall's `dist0` is `2 artanh|x|` against this chart's `asinh|x'|`
(factor 2). At the same nominal 1e-3 and n=64 the old master started at mean radius 4.6125e-03
and the ball at 1.9929e-03. On YouTube at 200/20 that was worth **+0.0168** to Lorentz
(0.5765 -> 0.5933) and it changed the stopping epoch from 104 to 118 -- the SAME epoch the ball
stops at. The remaining 0.0142 is the chart.

### The ball clamps at r_max 6.2126 and training goes past it

geoopt's `project` caps the coordinate norm at 0.99599993, i.e. **r = 2 artanh(0.996) = 6.2126
in float32** (12.2061 in float64). Both large datasets pinned there exactly: Yelp and WikiLink
read `r_max=6.213` while Lorentz, whose `projx` is the identity, reached 7.75 and 7.28 on the
same data. Earlier master runs reached r_max 12.09, and the worst across all logs is 15.89.

**`r_max = 6.213` in a ball log means CLAMPED, not converged.** Nothing in the log says so. On
WikiLink the ball spent its last six epochs pinned, every outward gradient discarded, still
printing plausible val increments. Converting a Lorentz model at r=9 into the ball loses 2.79 of
radius to the clamp; at r=12.09 it loses 5.88.

So "just bound r below 5 and use the ball" is not free: **training does not stay under 5 on its
own** (7.3-7.8 on two of six datasets), and capping it is an untested modelling change.

### geoopt's ball `dist` is far less accurate than ours, including below r=5

Against a 60-digit mpmath reference. `to_poincare` is exact (1e-16..1e-13), so every discrepancy
is geoopt's `dist`, which uses `2 artanh(sqrt(c)|(-x)+y|)` -- Mobius addition plus `artanh` near
1 -- where this file uses the `2 asinh(sqrt(w/2))` rearrangement.

| r | ours f64 | geoopt f64 | ours f32 | geoopt f32 |
|---|---|---|---|---|
| 0.001 | 2.3e-16 | 8.7e-14 | 1.1e-07 | 5.6e-05 |
| 0.5 | 2.2e-16 | 2.0e-09 | 1.4e-07 | 1.9e-07 |
| 2.0 | 1.4e-16 | 2.7e-08 | 7.6e-08 | 4.7e-07 |
| 5.0 | 1.2e-16 | 3.1e-07 | 6.4e-08 | **1.3e-04** |
| 8.0 | 6.1e-17 | 3.9e-06 | 4.3e-08 | **2.3e-02** |
| 10.0 | 8.8e-17 | **1.5e-01** | - | - |

Nearby pairs at separation 1e-3 -- what the ranking loss compares -- f32: ours 1.2e-05 / 8.4e-05
/ 2.1e-03 at r = 0.5 / 2 / 5 against the ball's 7.2e-05 / 2.5e-04 / **4.1e-01**. `transp`
isometry error f32: ours 3.0e-05 / 3.9e-05 / 9.8e-05 against 7.8e-04 / 2.3e-03 / 1.1e-02 --
**26x to 110x better inside r<5**. `expmap` is comparable below r=5. The f32 coordinate LATTICE
is equivalent in the two charts (`sinh(r) eps` vs `2 cosh^2(r/2) eps`, ratio 0.46 at r=1 rising
to 1.000 by r=5), so the precision gap is the algorithm, not the chart.

### Standing read

**The chart is worth nothing on five of six datasets and ~0.013 to the ball on YouTube** (0.0129
on max test; patience 8-14 would have given Lorentz its own 0.5946, see below). Against that,
Lorentz represents what training actually does rather than clamping it, and its `dist` -- the
scorer's own function -- is 6x to 2000x more accurate. One seed, so the YouTube number is not
replicated.

**YouTube's reported number is set by where patience cuts a flat val surface.** The three 200/20
runs have best-val 0.6741 / 0.6823 / 0.6855 -- 0.0114 apart -- selecting test numbers 0.031
apart. On the Lorentz run, **patience 15 is the minimum** that reaches the ep118 checkpoint (the
binding gap is a 14-epoch val drought, ep82 -> ep97), and patience 8-14 would have reported
**0.5946**, better than patience 20's 0.5933: val gained +0.0062 after ep82 while test lost
0.0013. Record both numbers.

**Reproduction is exact at fixed seed.** The pat20 runs reproduced their pat10 twins bit-for-bit
to four decimals on val and test -- 72/72 epochs on Lorentz, 63/63 on the ball -- so a differing
number means a differing input, not variance.

## `cos_o` vs no `cos_o`: the ablation the lineage never got (measured, 2026-10-06/07)

One variable. `cos_o` is the angle at the origin between a walk token and the bag centre,
`F.cosine_similarity(xt, mid)`; the no-`cos_o` arm drops that column and nothing else.
Features `[log1p(age), pos, d_mid, cos_o]` (n_feat 4, pooler 1,249 at nl2 / 193 at nl1)
against `[log1p(age), pos, d_mid]` (n_feat 3, pooler 1,217 / 161).

Seed 5, d64 K5 wpn5 mwl5 lr1e-3 h32, nl2 except WikiLink nl1, **100 epochs / patience 10
on both arms including YouTube**, Lorentz chart, matched init, same GPU model per cell.

| dataset | `cos_o` | no `cos_o` | d | stop ep (`cos_o` / no) |
|---|---|---|---|---|
| YouTube | 0.5946 | **0.6105** | **+0.0159** | 82 / 50 |
| GoogleLocal | 0.6574 | **0.6695** | **+0.0121** | 44 / 46 |
| Flickr | **0.6417** | 0.6347 | -0.0070 | 28 / 28 |
| ML-20M | **0.2510** | 0.2477 | -0.0033 | 9 / 9 |
| Yelp | **0.6528** | 0.6513 | -0.0015 | 25 / 18 |
| WikiLink | **0.6723** | 0.6608 | -0.0115 | 16 / 17 |

**All six cells are now final** (2026-10-07, both closed `rc=0`): Yelp stopped at ep18 of 28
with val 0.6711 -> test 0.6513, WikiLink at ep17 of 27 with val 0.6711 -> test 0.6608. Every
number in the table is the val-selected `best_test_mrr`. Max test differs on one cell only:
WikiLink peaked at **0.6611 at ep12**, so its val-selected 0.6608 carries 0.0003 of drift;
everywhere else max test equals the reported number.

The two cells that were provisional both landed where they had been sitting, and neither
sign changed. Yelp's earlier swing (+0.0173 at ep9 -> -0.0041 at ep16) settled at -0.0015,
so the "genuinely open" reading of its sign was resolved by running it out, not by variance.
WikiLink closed at **-0.0115** and remains the one dataset that has never favoured removal,
having held -0.009 to -0.012 from ep5 onward.

**Standing: removing `cos_o` wins 2 of 6.** But read the shape, not the count: the two wins
are +0.0159 and +0.0121 while three of the four losses are <=0.0070, so the summed effect
favours removal. WikiLink at -0.0115 is the only loss comparable in size to a win, and it is
the same cell that prefers `cos_o` in every other experiment on this suite.

**Both closing runs ended in radius inflation, which is why the last ten epochs bought
nothing.** WikiLink's train loss ROSE for its final twelve epochs (0.1464 at ep15 ->
0.1524 at ep27) while `r_mean` climbed a near-constant +0.18/epoch to 5.26 and `r_max`
reached **7.62**; Yelp's `r_max` ran to **8.18** by ep27. Both are past the float32 band
where `expmap` loses accuracy, and in both cases val sat inside a 0.0008 band throughout.
+0.0003 val flickers kept resetting patience with no gain, so on this suite a run whose
train loss has turned up and whose `r_mean` is growing linearly can be stopped early.

**YouTube's `cos_o` number is DERIVED, not run at 100/10.** The run was launched at 200/20;
patience only decides when to stop, never the trajectory (verified: the 200/20 and 100/10
twins were bit-identical for all 72 shared epochs). Replaying the stop rule on its log,
patience 10 stops at ep92 and restores ep82 -> val 0.6761, test **0.5946**. The 0.5933
quoted elsewhere is the 200/20 number, 0.0013 lower because running to ep118 let val tick
up while test fell.

### The mechanism is the val->test gap, not the fit

`cos_o` LOWERS training loss and WIDENS the gap. On YouTube the loss ratio (with/without)
climbs 1.00 at ep6 -> **2.15 at ep18**, the two arms reach the SAME best val (0.6823 vs
0.6819, 0.0004 apart), and the gap to test is **0.0890 with against 0.0714 without** --
which accounts for essentially all of the +0.0159. Not a checkpoint artefact: at matched
epochs the gap is 0.083-0.089 with against 0.065-0.072 without. Same ordering on Yelp
(gap 0.029-0.032 with, 0.019-0.021 without, `cos_o` loss lower at every epoch).

**WikiLink is the one cell where the ordering FLIPS** -- there the no-`cos_o` arm has the
*wider* gap (0.0089 vs 0.0079 at ep6) -- and it is also the one cell `cos_o` wins. That is
consistent, not coincidental: where the angle is genuinely predictive it helps, where it is
not it substitutes a cheap signal for radial structure and the model reaches the same
validation with less usable geometry.

### Spreading: the tail, not the bulk

Removing `cos_o` leaves `r_mean` unchanged and lengthens the radial tail, on BOTH a winning
and a losing dataset, so spread is NOT what separates them:

| dataset | ep | r_mean (with / without) | r_max (with / without) | ratio |
|---|---|---|---|---|
| WikiLink | 9 | 1.936 / 1.980 | 3.302 / **4.857** | 1.47x |
| YouTube | 30 | 0.553 / 0.574 | 1.656 / **2.272** | 1.37x |
| YouTube | 60 | 0.678 / 0.699 | 2.112 / **2.750** | 1.30x |

`geo_temp` moves inversely, the scorer compensating for the wider spread. Practical note:
on Yelp and WikiLink the no-`cos_o` arm crosses `r_max` 5 by ep17/ep13 and kept going to
**8.18 and 7.62** by the time both runs closed, well into the band where float32 `expmap`
steps lose accuracy -- an argument for bounding radius, not against the ablation.

### Why no clean ablation existed before

`cos_o` entered at `2d7b739`, whose parent `9f24967` differs by one CLAUDE.md commit -- so
that WAS a one-variable test, but `2d7b739` carried the `triangle_cos` numerics bug, fixed
later at `7e68aa7` and worth **+0.0398 on YouTube**. No bug-free ablation had ever been run.
The confounded lineage read -0.0216 on YouTube, +0.0051 Flickr, +0.0041 ML-20M, +0.0210
WikiLink, 0.0000 Yelp. The clean ablation reproduces the YouTube sign and size (-0.0159 for
keeping it) but **not** the claimed +0.0210 on WikiLink, which comes out at -0.0112 the
other way. Treat every lineage-based `cos_o` delta as contaminated.

### k_train interacts with it

`cos_o` also makes the model intolerant of more training negatives. YouTube, same config,
k_train 5 vs 10:

| | K=5 | K=10 | d from doubling K |
|---|---|---|---|
| with `cos_o` | 0.5946 | 0.5448 (stop ep24) | **-0.0498** |
| without `cos_o` | **0.6105** | 0.5981 (stop ep56) | **-0.0124** |

Removing `cos_o` makes the K=10 penalty **4x smaller** and the run survives to ep56 instead
of dying at ep24 (best val 0.6837 vs 0.6637). The K=10 gap with `cos_o` blew out to 0.1189
and never recovered; without it the gap peaked at 0.1223 around ep31 and came back to 0.0885.
More negatives still cost something; the catastrophic version was `cos_o`-mediated.

### Do not call these runs before ep50

Five arms today inverted between ep14 and ep40 -- both LayerNorm arms, `nl3`, the no-`cos_o`
K=10 arm, and the no-`cos_o` YouTube arm itself, which sat at **-0.1457 val at ep16** and
finished **+0.0159 up**. The ep10-40 window carries no information about the ep50+ result on
this suite.

## TGB suite: `cos_o` vs no `cos_o` -- PLACEHOLDER, nothing closed yet

The same ablation on the TGB suite (`--data-suite tgb`, restored 2026-10-06). Paired runs,
one variable: seed 5, d64 K5 wpn5 mwl5 lr1e-3 h32 nl2, 100 epochs / patience 10,
`bipartite=no`, RTX. `no cos_o` -> `logs/tgb_anchor`, `cos_o` -> `logs/tgb_coso`.

| arm | tgbl-wiki | tgbl-review | tgbl-coin | tgbl-comment | tgbl-flight |
|---|---|---|---|---|---|
| `cos_o` | pending | pending | pending | pending | running |
| no `cos_o` | 0.7481 ◊ | 0.1288 | 0.4045 ◊ | pending | pending |

◊ max-test-so-far on a live run, not a val-selected final. tgbl-review is the only closed
cell: **0.1288**, stopped at ep6 of 100 after val fell for ten straight epochs
(0.1345 -> 0.1086, back to its ep1 level) -- it overfits within six epochs.

**These five numbers are not comparable with each other, nor with any TGB-Seq number.** TGB
ships a different per-positive negative count per dataset (tgbl-review serves K=100), and
serves pre-generated negatives for BOTH val and test where TGB-Seq uses our sampler for val.
The only valid comparison for a TGB cell is the same cell in the other arm, or the TGB
leaderboard.

Note the TGB arms show almost NO val->test gap -- tgbl-review ran val/test within 0.004 at
every epoch, tgbl-coin's test (0.4045) is above its val (0.3971) -- because both splits use
the same negative protocol. The val->test gap that drives every TGB-Seq conclusion is a
property of TGB-Seq's eval design, not of the model.

## Dynamic node classification: frozen encoder beats every published number on Wikipedia AND Reddit (measured, 2026-10-08)

DyGLib protocol (Wikipedia/Reddit ban prediction, ROC-AUC over the whole split, backbone
frozen, classifier on the source node only, 5 runs, std with ddof=1). Run 1 trains link
prediction on `--data-suite dyglib --is-bipartite --k-train 5 --k-eval 5` (linear-skip
pooler) and saves the best-val checkpoint; Run 2 freezes it and trains a ~2.4k-param MLP.
Branch `feature/node-classification-geo-walk-hist`, recipe commit `85662e1`; drivers
`scripts/lp_seeds_wiki.sh`, `scripts/nc_final.sh`; logs `logs/node_cls_tempest_hist/k5/seed*/`
(recipe), `logs/node_cls_final*/k5/seed*/` (ablations, earlier numpy-history version).
Full write-up: `reports/node_classification_2026-10-08.md`.

### 5 checkpoints per dataset (seeds 42, 0, 1, 2, 3), test AUC

| classifier input | Wikipedia | Reddit |
|---|---|---|
| **12 geometric + 5 walk-time + 5 Tempest history** (the recipe) | **90.28 ± 0.76** | **73.63 ± 0.77** |
| same, history from a numpy scan of the stream (first version) | 89.91 ± 0.87 | 73.87 ± 0.95 |
| 12 geometric + 5 walk-time (`--no-history`) | 89.91 ± 0.67 | 68.63 ± 0.87 |
| 12 geometric only (`--geometry-only`) | 89.68 ± 0.97 | 64.84 ± 2.26 |
| best published | JODIE 88.99 ± 1.05 | TAWRMAC 71.45 ± 0.92 |
| DyG-Mamba / TAWRMAC / TIDFormer / DyGFormer | 88.58 / 87.69 / 87.53 / 87.44 | 70.79 / 71.45 / 69.59 / 68.00 |

Baselines: DyGLib Table 15 (arXiv 2303.13047), DyG-Mamba Table 11 (2408.06966), TIDFormer
Table 2 (2506.00431), TAWRMAC Table 7 (2510.09884) -- all read from the PDFs, not memory.

**The recipe was fixed on Wikipedia and applied to Reddit unchanged** -- no Reddit-specific
selection of any kind. Reddit LP checkpoints (seeds 42/0/1/2/3) restore epochs 40/38/42/33/41, val MRR 0.966 each.

**The two datasets disagree on what carries the signal.** On Wikipedia geometry alone is
enough (89.68) and history adds nothing; on Reddit geometry alone is 64.84 (still above
JODIE/DyRep/TGN/GraphMixer at 60-64) and the walk-time (+3.8) and exact history (+5.2)
features are what lift it past TAWRMAC. Report both; do not generalise the Wikipedia ablation.

**History comes from Tempest, not a side scan.** `get_node_popularity` (count),
`get_node_recency` / `get_latest_events` (last edge time AND partner), a second latest-event
lookup with cutoff = t_last (last gap), and the last partner's popularity and recency. Never
use `get_node_degrees` here: it takes no cutoff, so with the full graph ingested it leaks
future edges.

**The geometry carries the ban signal on its own.** Twelve numbers from the frozen hyperbolic
encoder -- radius of `E[u]` and of the pooled point, bag spread (pooler-weighted, unweighted,
and under softmax(-d) attention keyed on `p_u`), nearest/farthest token, pooling entropy --
beat every published method. The single strongest feature is `d0(E[u])` (univariate AUC
0.88 train / 0.84 val, inverted): banned users sit near the origin, i.e. radius encodes
activity. The exact history lookups (count, recency, gaps) add nothing on test.

### How the recipe was chosen (test sealed)

Sweeps ran on a cached feature table (scratch, seed-42 checkpoint) and printed train/val
only; test went to a sealed log, opened once at a milestone and then only for the final
recipe. Validation has 17 positives (AUC s.e. ~0.03-0.04), so the deciding criterion became
chronological CV over train ∪ val (4 folds, ~120 positives), fixed BEFORE the final test.

What moved the number, in order: adding `d0(E[u])` and the other radii (val 0.69 -> 0.85);
the full geometric set + walk-time + history (0.89); 20 walks per node at classification time
(+0.007 val over 5; 50 walks lost); shuffled classifier batches (+0.010 CV -- in time order
the rare positives arrive in clumps; features are already causal so shuffling leaks nothing).

### Negative results -- do not re-run

- **Edge features, every form**: mean / pooler-weighted / last edge / last-3, as a separate
  MLP branch (val 0.884 vs 0.890, more variance), as a 2-branch net (test 0.739 vs 0.779),
  as 4 PCA components (CV 0.878 vs 0.892). Standalone L2 logistic regression on the LIWC
  vectors reaches only val 0.65 / CV 0.71. There is little ban signal in them here.
- **Fine-tuning E during classification**: lr 1e-5 / 1e-4 tie frozen (val 0.890); lr 1e-3
  overfits hard (train 0.99, val 0.75). 156 positives cannot steer a 590k-param table.
- **Learned token attention** over per-token geometric descriptors ties the hand summaries
  (val 0.896, CV 0.887 vs 0.897 / 0.889).
- Wider/deeper heads, dropout 0.3, pos-weight, weight decay, batch 1000, lr 1e-4 (DyGLib's):
  all tie or lose. Burst counts (edits in last 1h/1d/1w, distinct pages): CV 0.899 vs 0.902.
- More features is worse: adding the 6 item/user-split radii drops val 0.890 -> 0.869.

### Two bugs found on the way

- `BagWeights` holds `E` as a submodule, so `bag_weights.requires_grad_(False)` silently
  re-freezes `E`. Set `E`'s flag AFTER the pooler's when fine-tuning.
- Tempest has no per-call seed and its RNG advances per call: a repeated walk call draws
  different walks; a fresh instance with the same seed replays exactly. The encoder therefore
  builds a fresh walker per pass (re-ingest ~0.2 s on Wikipedia).
