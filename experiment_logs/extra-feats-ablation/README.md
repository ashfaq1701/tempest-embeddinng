# Extra candidate-feature ablation

One question: does adding a cheap, non-geometric scalar about the **candidate** to the
scorer beat the plain distance baseline, and does it beat it everywhere?

The head is the master baseline with the distance temperature folded into a bias-free
`Linear(n, 1)` and the extra channels appended. Init is `[1, 0, ...]`, so at step zero the
score is **exactly** `geo_temp * (-d_H)` with `geo_temp` pinned at 1 and every extra channel
contributing nothing. A channel can only ever *earn* weight; it is never granted any.

| arm | branch | features | init |
|---|---|---|---|
| `pop/` | `feature/duv-with-cand-pop` | `[-d_H, log1p(cand_popularity)]` | `[1, 0]` |
| `rec/` | `feature/duv-with-cand-rec` | `[-d_H, log1p(cand_recency)]` | `[1, 0]` |
| `rec-and-pop/` | `feature/duv-with-cand-pop-and-rec` | `[-d_H, log1p(pop), log1p(rec)]` | `[1, 0, 0]` |

All arms: d=64, K=5, lr=1e-3, seed 5, patience 5, cap 100, softmax CE, no popularity bias
table. Bipartite flags per `bipartite_dict`.

## The two features

`cand_popularity` — edges the candidate participated in strictly before the query cutoff,
via `get_node_participation_counts` Backward_In_Time. **0 for a node with no prior edge.**

`cand_recency` — an **age**, `cutoff - t_last(v)`, via `get_latest_events_for_nodes`.
Tempest returns `t_last = -1` for a node with no prior edge, so a cold candidate gets
`cutoff + 1`: maximally stale. **Larger means staler**, so a useful recency weight is
expected to be *negative*, opposite in sign to a useful popularity weight.

**`log1p`, not `log`, on both.** Popularity is 0 for a cold node and `log(0) = -inf` would
NaN the first batch. The age is always >= 0 so `log` is not unsafe there, but it spans raw
timestamps (`log1p(1e9) = 20.7`) and `log1p` keeps the channels in a comparable range.

## Read the weight trajectory, not just the final number

`w[1]` has behaved completely differently on the two datasets finished so far, and the
trajectory diagnoses which case you are in without waiting for the final number:

- **YouTube** — jumped to 0.527 in epoch 1, then sat in a 0.47-0.50 band for fifteen epochs
  while `w[0]` climbed linearly to 35. Popularity acted as a fixed additive offset and the
  geometry trained normally underneath it. Real, persistent signal.
- **GoogleLocal** — rose to 0.653, then **decayed monotonically through zero at ep26** and
  kept going to **-0.25**. On this dataset a *popular* candidate is *less* likely to be the
  true next destination, and encoding that penalty is worth a small gain.

Neither was the capture failure mode (`w[1]` running away while `w[0]` collapses), which
every badly-finishing head on this suite has shown.

## Prior worth holding when reading a gain

TGB-Seq test negatives are drawn **uniformly** over the destination pool and are
popularity-blind: measured `corr(log10 train-popularity, log10 test-negative-frequency)`
= +0.0137 on ML-20M, with rare pool nodes averaging 1340.2 appearances as negatives against
1340.6 for popular ones. So a popularity channel cannot help by gaming the sampler. It can
only help because true positives follow preferential attachment while negatives do not,
which makes popularity genuinely discriminative *on the test set itself*.

Separately, a per-node popularity **bias table** was part of the withdrawn historical-best
configs that `CLAUDE.md` records as unstable — they climbed then fell sharply. This is a
different mechanism (one global scalar on a per-candidate count, not a learned per-node
bias), but the family has misled this project before.

## Results

Baseline is commit `07dcc1e6`, seed 5, from `experiment_logs/geometries/lorentz/*/3.log`.

| dataset | baseline test | pop test | delta | stop (base -> pop) | final `w[1]` |
|---|---|---|---|---|---|
| YouTube | 0.5626 | **0.5965** | **+0.0339** | 19 -> 11 | +0.486 |
| GoogleLocal | 0.6535 | **0.6559** | **+0.0024** | 44 -> 33 | -0.246 |

Both arms also reached their peak in **fewer epochs** than the baseline needed.

Peak test vs test@val-checkpoint, which are not the same number: YouTube peaked 0.5984 at
ep10 and the val-selected checkpoint was ep11 at 0.5965, a 0.0019 drift. Record both.
