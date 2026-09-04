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

## Reproduction is stable to ~0.0002 MRR

Same seed, same config, independent launches land on the same number to the third
decimal. Walk sampling is the only nondeterminism, and it moves results far less than
the ±0.01 band the older notes assumed. **Single-seed deltas are readable at the third
decimal here** — a +0.005 difference between two configs is signal, not noise.

Measured, all seed 42:

| replicate pair | run A | run B | Δ |
|---|---|---|---|
| GoogleLocal d=32 | 0.6303 | 0.6301 | 0.0002 |
| ML-20M d=32 | 0.2222 | 0.2224 | 0.0002 |

Patent d=64 with `--use-cand-pop` was launched twice and tracked epoch for epoch:

| epoch | run A test | run B test |
|---|---|---|
| 1 | 0.0990 | 0.0990 |
| 2 | 0.0836 | 0.0837 |
| 3 | 0.1107 | 0.1106 |
| 4 | 0.1531 | 0.1531 |
| 5 | 0.1939 | 0.1940 |
| 6 | 0.2241 | 0.2242 |

Two consequences. An A/B does not need multiple seeds to separate configs that differ by
more than ~0.001 — the remaining reason to run several seeds is to report mean±std against
a leaderboard, not to establish an ordering. And a run that fails to reproduce is a real
signal: suspect a code or config change rather than variance.

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
no-geometry baseline `rad` is worth +0.031 max test and `dev` +0.023 — both well clear of the
~0.01 band where initialisation luck is a live explanation. But `rad+dev` lands *below* `rad`
alone: `dev` is largely redundant with `rad` and pays for the overlap in delay.

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
arm's first two columns, and even the output layers differ), so a gap under ~0.01 there is not
separable from init luck. `rad`'s +0.0058 over `rad+dev` is inside that band: the choice rests
on parsimony plus the clean `rad`-beats-`dev` contrast, not on that number.

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
