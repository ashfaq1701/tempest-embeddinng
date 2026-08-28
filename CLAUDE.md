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
