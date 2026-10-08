# Dynamic node classification with a frozen Tempest encoder — results, 2026-10-07/08

**Headline.** A small classifier on 22 numbers read from the *frozen* link-prediction encoder
beats every published dynamic-node-classification result on both DyGLib datasets.

| test AUC-ROC, 5 runs (mean ± std, ddof=1) | Wikipedia | Reddit |
|---|---|---|
| **ours** | **89.91 ± 0.87** | **73.87 ± 0.95** |
| best published | JODIE 88.99 ± 1.05 | TAWRMAC 71.45 ± 0.92 |
| margin | **+0.92** | **+2.42** |

Code: branch `feature/node-classification-geo-walk-hist` (recipe commit `83c6911`).
Logs: `logs/node_cls_final*/k5/seed*/`, link-prediction checkpoints in `logs/node_cls_linskip/k5/`.

---

## 1. Protocol

DyGLib's dynamic node classification (Yu et al., NeurIPS 2023), verified against its source
(`master` @ `3aacc36`):

- **Data.** Wikipedia (157,474 interactions, 217 bans) and Reddit (672,447 interactions, 366
  bans), the processed files of Zenodo record 7213796. The task: predict, per interaction,
  whether the source user is banned after it.
- **Split.** Chronological by timestamp quantile: train `≤ q0.70`, val `(q0.70, q0.85]`, test
  `> q0.85`. Positives train/val/test: Wikipedia 156 / 17 / 44, Reddit 190 / 82 / 94.
- **Frozen backbone.** DyGLib freezes the backbone ("follow previous work") and trains only the
  classifier. We do the same.
- **Metric.** ROC-AUC over the whole split (not per batch), test read at the best-validation
  classifier state, 5 runs. Each run's classifier sits on that run's own link-prediction
  checkpoint (seeds 42, 0, 1, 2, 3).
- **Causality.** Every feature for interaction `(u, t)` uses only edges with `t_edge < t`
  (exclusive cutoff). That excludes the interaction itself and every edge sharing its
  timestamp. DyGLib's `find_neighbors_before` uses the same strict inequality.

**Run 1, link prediction.** The existing model (Lorentz embedding + walk-bag pooler with the
linear skip from `feature/pooler-linear-skip`) is trained on the train split via
`--data-suite dyglib --is-bipartite --k-train 5 --k-eval 5`. The best-validation checkpoint
is saved with `--save-checkpoint`.

**Run 2, classification.** The checkpoint is loaded and frozen (`eval()`,
`requires_grad_(False)`, `no_grad`; a weight hash is asserted unchanged). Each split is
encoded once, then a ~2.4k-parameter MLP is trained on the features.

## 2. The recipe

Per interaction, 22 scalars:

| group | features |
|---|---|
| **geometry (12)**, from the frozen walk bag | `d0(p_u)`; spread `Σ w_i d(x_i, p_u)`; unweighted spread; nearest / farthest token; mean token radius; `d0(E[u])`; `d(E[u], p_u)`; pooling entropy and peak weight; spread and entropy under `softmax(−d(x_i, p_u))` (attention keyed on `p_u`) |
| **walk time (5)** | log #real edges; min / mean / pooler-weighted log-age; #distinct nodes in the bag |
| **exact history (5)** | log #prior edges of `u`; log time since last / first; log mean / last gap |

Here `p_u` is the frozen pooler's point, `w` its weights, `x_i` the bag's token points and `d0`
the hyperbolic radius.

**Classifier:** `BatchNorm(22) → 22 → GELU → 32 → GELU → 32 → GELU → Dropout(0.1) → 1`.
Adam at lr 1e-3, BCE, batch 200, **shuffled** batches, 100 epochs / patience 20 on validation
AUC. **20 walks per node** at classification time; the checkpoints trained with 5.

## 3. Results

### Per run, test AUC

| seed | Wikipedia | Reddit |
|---|---|---|
| 42 | 0.8923 | 0.7458 |
| 0 | 0.9050 | 0.7237 |
| 1 | 0.8923 | 0.7348 |
| 2 | 0.8943 | 0.7438 |
| 3 | 0.9114 | 0.7453 |
| **mean ± std** | **89.91 ± 0.87** | **73.87 ± 0.95** |

The link-prediction checkpoints are consistent across seeds. Wikipedia best epochs are
42/43/54/70/36 (val MRR 0.964–0.966, test 0.956–0.960). Reddit best epochs are 40/38/42/33/41
(val 0.966, test 0.960).

### Against all published results (from the papers' PDFs, not memory)

| method | Wikipedia | Reddit | source |
|---|---|---|---|
| JODIE | 88.99 ± 1.05 | 60.37 ± 2.58 | DyGLib Table 15 |
| DyRep | 86.39 ± 0.98 | 63.72 ± 1.32 | DyGLib Table 15 |
| TGAT | 84.09 ± 1.27 | 70.04 ± 1.09 | DyGLib Table 15 |
| TGN | 86.38 ± 2.34 | 63.27 ± 0.90 | DyGLib Table 15 |
| CAWN | 84.88 ± 1.33 | 66.34 ± 1.78 | DyGLib Table 15 |
| TCL | 77.83 ± 2.13 | 68.87 ± 2.15 | DyGLib Table 15 |
| GraphMixer | 86.80 ± 0.79 | 64.22 ± 3.32 | DyGLib Table 15 |
| DyGFormer | 87.44 ± 1.08 | 68.00 ± 1.74 | DyGLib Table 15 |
| DyG-Mamba | 88.58 ± 0.92 | 70.79 ± 1.97 | arXiv 2408.06966, Table 11 |
| TIDFormer | 87.53 ± 1.12 | 69.59 ± 1.70 | arXiv 2506.00431, Table 2 |
| TAWRMAC | 87.69 ± 0.78 | 71.45 ± 0.92 | arXiv 2510.09884, Table 7 |
| **ours** | **89.91 ± 0.87** | **73.87 ± 0.95** | this report |

### Ablations (5 runs each, same checkpoints)

| classifier input | Wikipedia | Reddit |
|---|---|---|
| geometry only (12), `--geometry-only` | 89.68 ± 0.97 | 64.84 ± 2.26 |
| geometry + walk time (17), `--no-history` | 89.91 ± 0.67 | 68.63 ± 0.87 |
| full recipe (22) | 89.91 ± 0.87 | 73.87 ± 0.95 |

**On Wikipedia the geometry is enough on its own.** Twelve numbers read off the frozen
hyperbolic encoder beat every published method. The strongest single feature is `d0(E[u])`, the
radius of the user's own point (univariate AUC 0.88 train / 0.84 val, inverted). Banned users
sit near the origin, which in this geometry means low activity. The exact history lookups add
nothing on test.

**On Reddit it is not.** Geometry alone (64.84) still beats JODIE, DyRep, TGN and GraphMixer,
but the walk-time features (+3.8) and the exact history (+5.2) are what carry the margin over
TAWRMAC. The Wikipedia ablation does not generalise; report both.

## 4. How the recipe was chosen

All exploration ran on the **Wikipedia seed-42 checkpoint**, on a cached feature table in
scratch. Sweeps printed train and validation AUC only; test AUC went to a sealed log.

- Validation has 17 positives (AUC standard error about 0.03–0.04), too few to rank close
  candidates. The deciding criterion became **chronological CV over train ∪ val**: 4 folds,
  about 120 positives, each fold trained only on earlier rows. It was fixed *before* the final
  test numbers were opened.
- The sealed test log was opened once mid-way, at a milestone for an earlier configuration
  (val-selected, test 0.898). It did not drive any later choice.
- **The recipe was frozen on Wikipedia and applied to Reddit unchanged.** There was no
  Reddit-specific selection of any kind.

What moved the number:

| step | effect |
|---|---|
| starting classifier, `[d0(p_u), Σ w d(x, p_u)]` | val 0.69, test 0.779 |
| + `d0(E[u])` (radius of the user's own point) | val 0.69 → 0.85 |
| full geometric set + walk time + exact history | val 0.890 ± 0.003 (5 seeds) |
| 20 walks per node instead of 5 (50 lost) | +0.007 val, on both criteria |
| shuffled classifier batches | +0.010 CV; in time order the rare positives arrive in clumps |

## 5. What did not work (do not re-run)

- **Edge features, in every form.** Tried: the mean, pooler-weighted, last edge and last-3
  edge features; as a separate MLP branch (val 0.884 vs 0.890, with more variance); as a
  two-branch net (test 0.739 vs 0.779); and as 4 PCA components (CV 0.878 vs 0.892). A
  standalone L2 logistic regression on the 172-d LIWC vectors reaches only val 0.65 /
  CV 0.71. There is little ban signal in them here.
- **Fine-tuning `E` during classification.** lr 1e-5 and 1e-4 tie frozen (val 0.890); lr 1e-3
  overfits hard (train 0.99, val 0.75). 156 positives cannot steer a 590k-parameter table.
- **Learned token attention** over per-token geometric descriptors ties the hand summaries
  (val 0.896 / CV 0.887 vs 0.897 / 0.889).
- **More features is worse.** Item/user-split radii lowered val from 0.890 to 0.869; burst
  counts (edits in the last hour/day/week) gave CV 0.899 vs 0.902.
- **Head and optimiser variants** all tie or lose: wider, deeper, dropout 0.3, positive
  weighting, weight decay, batch 1000, lr 1e-4 (DyGLib's own).

## 6. Bugs found on the way

1. **`BagWeights` holds `E` as a submodule.** `bag_weights.requires_grad_(False)` therefore
   silently re-froze `E`, so the first "fine-tuning" runs were frozen. When fine-tuning, set
   `E`'s flag *after* the pooler's.
2. **Tempest has no per-call seed, and its RNG advances with every call.** A repeated walk call
   draws different walks; a fresh instance with the same seed replays exactly. The encoder
   builds a fresh walker per pass (re-ingest about 0.2 s on Wikipedia), so every pass is
   deterministic.
3. **Reddit's zip is Deflate64.** Python's `zipfile` cannot read it; the loader falls back to
   Info-ZIP `unzip`.
4. **Reddit's timestamps are in milliseconds.** The loader scales to integers by the smallest
   exact power of ten (×1000) instead of truncating. The split still cuts on DyGLib's float
   quantiles.

## 7. Caveats

- **Selection and reporting share a checkpoint.** The seed-42 Wikipedia checkpoint was used for
  selection (train and validation only) and is also one of the 5 reported runs. Its test number
  (0.8923) is the lowest of the five, so this did not inflate the mean.
- **Our link-prediction backbone differs from DyGLib's.** It is trained on the full train split,
  where DyGLib's LP split also holds out 10% of nodes for inductive testing. It uses our
  negatives (uniform, `k=5`, MRR), not DyGLib's 1:1 AP/AUC. The node-classification protocol
  itself (frozen backbone, splits, metric, source-only classifier) matches.
- **Our classifier settings differ from DyGLib's defaults:** lr 1e-3 vs 1e-4, shuffled vs
  ordered batches, 20 walks vs the checkpoint's 5. These are choices of our model, selected on
  validation and CV. The baselines likewise use their own tuned configurations.
- **Seeds are 42, 0, 1, 2, 3; DyGLib uses 0–4.** The standard deviation is computed the same way
  (ddof = 1).
- **One hardware setup** (RTX 2000 Ada laptop GPU). Reddit runs shared the GPU three ways.
  Results are deterministic per seed, but per-epoch timings are not comparable across runs.

## 8. Earlier in the session

- **tgbl-wiki link prediction peaks at epoch 1 and then declines** on master, at both lr 1e-3
  and lr 1e-4. Embedding weight decay (`--wd-e 1e-4`) changed nothing (val 0.7605, test
  0.7250 at epoch 1). A fixed-prior pooler diagnostic (MLP replaced by
  `softmax(−log1p(age) − pos)`) dropped to val 0.47 / test 0.42, so the learned pooler is worth
  about 0.3 MRR there.
- **The linear-skip pooler fixes that pattern on DyGLib Wikipedia.** Master peaks at epoch 1
  (val 0.9390, near-untrained `E`, `r_mean` 0.017). The linear skip climbs to epoch 42
  (val 0.9639 / test 0.9569, `r_mean` 0.457): +0.025 / +0.029, and a trained encoder for
  classification. One seed for the master comparison.
- **TGB downloads work again** (`py-tgb ≥ 2.3.0`, new host). A bad download now raises a clear
  error instead of leaving a stub zip. This is on `master`.
- **New `dyglib` data suite** (auto-download by name from Zenodo, MD5-checked), plus
  `--save-checkpoint` for link prediction, the `node_classification/` package, and
  `scripts/train_node_classification.py`.
