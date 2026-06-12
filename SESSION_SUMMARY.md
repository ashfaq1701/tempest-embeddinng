# Session summary — pair-feature integration (attempt 2), 2026-06-11/12

Branch: `feature/pair-feature-integration-attempt-2` (base `74a6bae` = master cross-GRU,
GRU depth 2, reported 0.7345 val / 0.6926 test). All work committed and pushed.

**Goal:** beat TPNet's tgbl-wiki MRR (reported ~0.84) by adding pair features to the
cross-GRU link-prediction model.

**One-line outcome:** found a clean, robust pair-feature win (exact recurrence + history,
**+0.015–0.020 test, 3-seed confirmed**), falsified everything else (co-reachability,
interaction MLP), resolved the baseline-vs-doc mystery, and pinned the honest gap to
TPNet at its **exact** batch sizes: **e2_hist 0.8025 val / 0.7744 test vs TPNet ~0.84** —
a real ~0.04–0.056 **core-model** gap, not a pair-feature gap.

---

## 1. Phase A — analysis & planning (docs)

Produced, in `pair-feature-integration.md` and `tpnet-knowledge-base.md`:
- **TPNet pair-feature inventory** — the temporal walk matrix `A^(k)(t)`, time-decay
  score, JL random-feature reps, and the Gram pair feature with its semantic blocks
  (direct l-hop connectivity = recurrence, co-reachability, self-structure), scaling, MLP.
- **What Tempest returns** — walk nodes / timestamps / lens / (optional) edge features.
- **Candidate pair-feature list** — 9 features scored on (covered-by-embedding?, why,
  fast/vectorizable?, source, expected gain).
- **A 10-hour campaign decision tree** with guardrails (additive-before-MLP,
  smooth-curve, noise ≥0.015 or 3-seed, one GPU job at a time).

Key a-priori fact that drove ordering: tgbl-wiki is ~66–71% repeat pairs, so recurrence
is the dominant lever; co-reachability was reserved for the ~30% new-edge slice.

---

## 2. Phase B — infrastructure built (all flag-gated; baseline byte-identical when off)

| component | file | what |
|---|---|---|
| Reusable sparse store | `tempest_walks/sparse_store.py` | `SparseStreamStore`: pandas `Index` hash table, int64 key → named int64 cols with per-col batch reducers (max/min/add/last). O(#keys), vectorized, scales to tgbl-comment's 995k nodes. (Replaced a hand-written 120-line open-addressing table after evaluating cykhash/pandas.) |
| Pair recency store | `tempest_walks/pair_store.py` | `PairRecencyStore`: ~30-line view (key = canonical undirected pair, cols `last_ts`=max / `count`=add). Strict-causal: query at scoring, update after, reset per epoch. 0 mismatches vs brute force. |
| Co-reachability | `tempest_walks/coreach.py` | `--use-coreach`: exact walk-derived shared-neighbour count (scipy.sparse membership, dense-source + per-source matvec, ~16 s/epoch). 0 mismatches vs brute force. |
| Head features | `tempest_walks/link_pred_head.py` | flags: `--use-pair-recency` (#1), `--use-pair-history` (#2 ever-bit+count), `--use-ctx-term` (#5 learned h·h), `--use-coreach` (#3), `--use-pair-mlp` (joint interaction decoder). |
| Campaign driver | `scripts/pair_campaign.sh` | sequential (1 GPU), unbuffered, `EVAL_BS`/`TRAIN_BS`/`SEED` env, logs to `logs/pair_features/RESULTS.tsv`. |

New deps: `pandas` (already transitive), `scipy` (added to requirements). No others —
`cykhash` was evaluated and rejected (no vectorized batch ops).

---

## 3. Phase C — experiments (full ledger: `logs/pair_features/RESULTS.tsv`)

### Wave 1 — streaming features (eval-bs 50, seed 42; base = 0.7715 / 0.7362)

| run | flags | val | test | Δtest |
|---|---|---|---|---|
| base | — | 0.7715 | 0.7362 | — |
| e1_rec | #1 | 0.7857 | 0.7555 | +0.0193 |
| **e2_hist** | **#1+#2** | 0.7851 | **0.7581** | **+0.0219** |
| e1b_ctx | #5 | 0.7680 | 0.7322 | −0.0040 |
| e1_rec_ctx | #1+#5 | 0.7845 | 0.7553 | +0.0191 |
| e2_hist_ctx | #1+#2+#5 | 0.7842 | 0.7578 | +0.0216 |

→ recurrence works; history (#2) adds a bit more; **learned co-reach (#5) hurts.**

### Wave 2 — exact co-reachability (#3)

| run | val | test | Δtest |
|---|---|---|---|
| w2_hist_coreach (#1+#2+#3) | 0.7853 | 0.7585 | +0.0223 (= e2_hist, noise) |
| w2_coreach (#3 alone) | 0.7670 | 0.7292 | −0.0070 |
| w2_rec_coreach (#1+#3) | 0.7839 | 0.7525 | +0.0163 (< #1 alone) |

→ **exact co-reach falsified** — alone it hurts, with #1 it drags, on #1+#2 it's noise.

### Wave 3 — joint pair-MLP interaction decoder

| run | val | test | shape |
|---|---|---|---|
| w3_pairmlp (#1+#2+#3 via MLP) | 0.7717 | 0.7484 | overfits, peak ep4 |

→ **pair-MLP falsified** — worse than additive, overfits. No conditional co-reach signal.

### Multi-seed confirmation of the winner (e2_hist, eval-bs 50)

| seed | base test | e2_hist test | Δtest |
|---|---|---|---|
| 42 | 0.7362 | 0.7581 | +0.0219 |
| 1 | 0.7337 | 0.7541 | +0.0204 |
| 7 | 0.7384 | 0.7567 | +0.0183 |

→ **Δtest = +0.0202 ± 0.0015, Δval = +0.0121 ± 0.0011** — robust, clears the noise band.

### Eval-granularity sweep (smaller eval-bs = fresher strict-causal state = higher MRR)

| eval-bs | 200 | 100 | 50 | 25 |
|---|---|---|---|---|
| base — test | 0.6932 | 0.7111 | 0.7362 | 0.7513 |
| e2_hist — test | 0.7217 | 0.7419 | 0.7581 | 0.7710 |
| e2_hist — val | 0.7524 | 0.7697 | 0.7851 | 0.7972 |

→ **`base @ bs=200` (val 0.7339) ≈ the doc's 0.7345** — the doc used the bs=200 default;
this resolved the whole "base looks higher than the doc" mystery (this campaign ran bs=50).

### TPNet's EXACT batch sizes (train 200 / eval 20 — verified in their code)

| config | val | test |
|---|---|---|
| base (master) | 0.7928 | 0.7597 |
| **e2_hist (#1+#2)** | **0.8025** | **0.7744** |
| **Δ pair features** | **+0.0097** | **+0.0147** |
| gap to TPNet (reported ~0.84 / ~0.83) | ~−0.038 | ~−0.056 |

TPNet evals wiki at `batch_size=20` (`TGB_TPNet/train_link_prediction.py:99–102`,
`evaluate_link_prediction.py:96`). Their reported MRR (~0.84) is cited pending
confirmation against the official TGB leaderboard — not asserted from memory.

---

## 4. Findings

1. **Ship `--use-pair-features`** — exact pairwise `(u,v)` recurrence
   (time-since-last-interaction) + ever-bit + decayed count, from the streaming store,
   added additively to the chord logit. **+0.015–0.020 test, 3-seed confirmed, smooth.**
   This is exactly TPNet's `A^(1)_{u,v}` recurrence block, computed exactly (no JL sketch).
   (During the campaign these were two flags `--use-pair-recency` + `--use-pair-history`;
   post-campaign they were collapsed into the single `--use-pair-features` flag and all
   falsified code — co-reach, ctx-term, pair-MLP — was removed from the branch.)
2. **Do NOT ship** co-reach (#3 exact, #5 learned) or the pair-MLP — all
   neutral-to-harmful. The GRU walk-encoder already captures shared-neighbour structure,
   so explicit co-reach only adds variance; the MLP overfits.
3. **The pair-feature win shrinks as eval granularity rises** (+0.0147 test @ bs20 vs
   +0.0202 @ bs50): at fine granularity the base already sees fresh causal state and
   captures recency implicitly.
4. **The remaining ~0.04–0.056 gap to TPNet is a CORE-MODEL gap, not a pair-feature gap.**
   Root cause is the inverse of TPNet's regime: our GRU base is *strong* (≈0.79 at fine
   eval), so structural pair features are largely redundant; TPNet's base is *weak* (≈0.34),
   so its pair features are load-bearing. Exact recurrence captures essentially all the
   available pair-feature headroom on a strong walk-encoder base.

## 5. What we built that is reusable

- `SparseStreamStore` — a clean, sparse, vectorized substrate for any future per-pair or
  per-node streaming feature (degree/popularity stores subclass it directly).
- `coreach.py` — exact walk-derived co-reachability (kept, off by default; useful on
  weak-base datasets where it may matter).
- `pair_campaign.sh` — env-driven sweep driver (EVAL_BS / TRAIN_BS / SEED).

## 6. Suggested next steps (in memory `tempest-pair-feature-campaign-2026-06-12`)

- **Test recurrence+history on the WEAK-base / cold-start datasets (tgbl-review).** Pair
  features should matter *more* where the base is weaker and recurrence isn't already
  implicit — the natural place this win pays off.
- **Closing the wiki gap to TPNet is core-model work, not pair features**: a joint
  decoder `MLP([h_u, h_v, f_uv])` over a *link-trained* representation, or an MLP-Mixer /
  attention backbone over the walk neighbourhood (TPNet's architecture), is where the
  remaining ~0.04–0.056 lives.
- Always evaluate at **bs≈20** on wiki to match TPNet (bs=50 under-reports).

## 7. Commit trail (this session, on the branch)

```
0f6ee45 docs: apples-to-apples at TPNet's exact batch sizes (train200/eval20)
efbdc72 chore: driver TRAIN_BS env override
e1922fa docs: multi-seed confirmation + eval-granularity curve + fair TPNet comparison
ce7fb30 docs: campaign results — recurrence+history wins, co-reach falsified
896a546 feat: --use-pair-mlp joint interaction decoder
78a4089 chore: pair_campaign driver — EVAL_BS/SEED env
848014d feat: #3 exact walk-derived time-decayed co-reachability
c1d8059 chore: pair_campaign driver — unbuffered + CUDA frag guard
d23caf8 refactor: pandas-backed reusable SparseStreamStore; drop hand-written hashmap
d01ca34 feat: pair features #1/#2/#5 + scalable pair store
82797eb / 22f6874 / 6d14626 / e4afc1c  docs: decision tree, feature list, Tempest
                                       returns, TPNet pair-feature inventory
67af560 / 94a9ecb / 2e271f7 / 96f6f95  TPNet knowledge base + motivation/plan
```

---

## 8. Follow-up — source-side-only link head (a clear win)

Branch `feature/source-side-walks-only` (off master). Changed the link head from the
**symmetric cross** encoder (sample walks for every unique node = sources + candidates;
score `-scale·(‖E[u]-ĥ[v]‖ + ‖E[v]-ĥ[u]‖)`) to a **source-side-only** head: sample walks
for the **sources alone**, GRU-encode to `h[u]`, and score the single chord
`-scale·‖ĥ[u] - E[v]‖` — "does the candidate's raw embedding match the source's recent
walk context". Candidate recency (`t_query - t_last[v]`) now comes from a per-node
last-seen store (`NodeLastSeenStore`, reuses `SparseStreamStore`) since candidate walks
are gone; pair features unchanged (kept as the `--use-pair-features` flag).

Runs at TPNet's bs (train 200 / eval 20, seed 42):

| head | pair feats | peak val / test | peak ep | per-epoch | GPU |
|---|---|---|---|---|---|
| **source-side** | **on** | **0.8064 / 0.7791** | ep3 | ~100 s | 0.7 GB |
| cross (master) | on | 0.8025 / 0.7744 | ep3 | ~210 s | 3.5 GB |
| **source-side** | off | 0.7948 / 0.7659 | ep7 | ~100 s | 0.7 GB |
| cross (master) | off | 0.7928 / 0.7597 | ep7 | — | — |

### Findings

1. **The source-side head beats the symmetric cross head in BOTH regimes** — with pair
   features (+0.0039 val / +0.0047 test) and without (+0.0020 val / **+0.0062 test**).
   It's the head design itself, not an interaction with pair features. Single-seed;
   multi-seed confirmation is the natural follow-up.
2. **~2× faster and ~5× less GPU memory** (~100 s/epoch vs ~210 s; 0.7 GB vs 3.5 GB):
   candidate-walk sampling + encoding — the dominant cost — is removed. Walks are now
   sampled for ≤ B unique sources instead of B + B·C unique nodes.
3. **Stable curves** — gentle post-peak tails like master's (source+pf:
   0.8064→0.8059→0.8045→0.8018, −0.0046/3 ep).
4. **source-side + pair features (0.8064 / 0.7791) is the new best on wiki**, narrowing
   the gap to TPNet (~0.84) to ~0.034 val / ~0.05 test — while being far cheaper to run.

Net: a strictly better, simpler, cheaper head. Merged to master (single-seed; the
+0.004–0.006 MRR gain plus the large speed/memory win justified shipping; multi-seed
confirmation tracked as follow-up).

---

## 9. Follow-up — re-adding the source-seeded alignment loss

> Documentation only. The alignment-loss CODE lives on branch
> `feature/readd-alignment-loss` and was **not merged** (the experiment falsified it);
> only this writeup is on master, for the record.

Branch `feature/readd-alignment-loss` (off the merged master). Ported the InfoNCE
**alignment loss** verbatim from the alignment+uniformity backup
(`tempest-embeddinng-bak/` @ detached `9208aff`, "source-seeded alignment"):
`L_total = L_link + L_align`, single RiemannianAdam (E is the sphere ManifoldParameter,
so alignment's gradient on E goes through the manifold-aware optimizer). L_align pulls
`E[seed]` toward its backward-walk context nodes — word2vec NEG partition over the batch
pool (`c(v)^0.75`), per-position weight `(1-γ)·hop + γ·recency`, chunked +
gradient-checkpointed partition. `--use-alignment`; `--detach-link-e` toggles regime.
**Link-pred head unchanged.** Two runs at TPNet's bs (train 200 / eval 20), seed 42,
**with `--use-pair-features`**. γ=0.4, τ=0.5, recency_scale = train mean inter-arrival.
(These numbers predate the source-side head — the "baseline" here is the cross head.)

| variant | flags | peak (ep) | val | test | post-peak val decline (4 ep) |
|---|---|---|---|---|---|
| **baseline** (pair-feats only) | — | ep3 | **0.8025** | **0.7744** | −0.008 (gentle) |
| no-detach (E ← link + align) | `--use-alignment` | ep2 | 0.8005 | 0.7743 | −0.021 |
| detach (E ← align only) | `--use-alignment --detach-link-e` | ep2 | 0.8002 | 0.7724 | −0.024 |

Full val curves: detach 0.7944→**0.8002**→0.7990→0.7933→0.7872→0.7767→0.7644;
no-detach 0.7943→**0.8005**→0.8004→0.7974→0.7935→0.7798. Both early-stopped ~ep7.

### Findings

1. **Alignment does not help on top of the link + pair-features stack** (this protocol,
   single seed). Both variants peak at-or-below the pair-features-only baseline (val
   ~−0.002; no-detach ties on test at 0.7743) **and are less stable** — they peak early
   (ep2) and decline faster than the baseline's gentle tail.
2. **no-detach > detach** on both axes: higher peak test (0.7743 vs 0.7724) and a
   gentler post-peak decline. Keeping the link gradient on E (alignment as a
   regulariser) beats letting alignment shape E alone.
3. **Mechanism (visible in the logs):** every epoch the align loss falls and val drops
   in lockstep — alignment pulls E toward walk-context-optimal geometry, which is
   slightly *off* the link-optimal geometry the strong cross-GRU + pair-features stack
   wants. detach (E ← align only) drifts fastest; no-detach's link gradient slows but
   doesn't prevent it. This is the inverse of the regime where alignment originally
   helped (a weak link head needing E pre-shaped).
4. **Overhead negligible:** ~+2 s/epoch (train 63.8 → 65.7 s); eval unchanged
   (alignment is train-only). bs=20 eval (~146 s) dominates per-epoch time.

### Control: alignment WITHOUT pair features (isolating the alignment effect)

To test whether alignment was redundant only *because* pair features already shaped E,
re-ran all three variants with **no pair features** (same protocol, seed 42). If
alignment helps a weaker base, the no-pair alignment runs should beat the no-pair base.

| (no pair features) | peak val / test | peak ep | post-peak val tail (stability) |
|---|---|---|---|
| **base** | **0.7921 / 0.7597** | ep7 | −0.010 (stable) |
| no-detach align | 0.7878 / 0.7582 | ep3 | −0.043 (steep) |
| detach align | 0.7842 / 0.7557 | ep3 | −0.045 (steepest) |

Full 2×3 landscape (peak val/test; tail = val decline over the 4 epochs after the peak):

| regime | base | + no-detach align | + detach align |
|---|---|---|---|
| **no pair feats** | 0.7921 / 0.7597 (stable, −0.010) | 0.7878 / 0.7582 (−0.043) | 0.7842 / 0.7557 (−0.045) |
| **+ pair feats** | 0.8025 / 0.7744 (stable, −0.008) | 0.8005 / 0.7743 (−0.021) | 0.8002 / 0.7724 (−0.024) |

### Verdict — alignment does NOT help, on both axes, in both regimes

- **Peak:** in *both* columns `base > no-detach > detach`. Alignment **lowers** the peak
  with OR without pair features (−0.004 val no-pair, −0.002 val with-pair). The
  "alignment helps the weaker base" hypothesis is **refuted** — the no-pair base is
  weaker (0.792 vs 0.802) yet alignment hurts it *more* (−0.004 vs −0.002), not less.
- **Stability:** the plain base is the most stable in both regimes (gentle ~−0.01 tail).
  Every alignment run peaks **earlier** and falls into a **much steeper** decline
  (−0.02 to −0.045) — val drops in lockstep as `L_align` minimises, i.e. alignment pulls
  E off the link-optimal geometry from the start. detach (E ← align only) drifts fastest.
- **no-detach > detach** everywhere (higher peak, gentler tail) — so it's the variant to
  keep IF pursued, but as configured (γ=0.4, τ=0.5, full-weight `L_align`) it is a net
  cost in every cell.

**Recommendation: ship pair-features alone; the source-seeded alignment loss is not
additive and not a standalone win on wiki — it lowers the peak and destabilises training
regardless of pair features.** Untried levers if revisited: a small `λ_align` weight
(light regulariser), warmup-then-decay, γ/τ sweep, or — the one regime not yet tested —
a genuinely cold-start dataset (tgbl-review) where the link head isn't already capturing
the geometry.

---

## 10. Follow-up — multi-bias walks (falsified on wiki)

> Documentation only. The CODE lives on branch `feature/multi-bias-walks-v2`
> (off the source-side master) and was **not merged** — the experiment falsified it.

Idea: instead of sampling all walks under one bias, sample some walks under EACH of
several `(start_bias, walk_bias)` configs and merge them, giving the GRU a mix of walk
"shapes". Built `walks.merge_walks(*walks)` (vararg, merges multiple Tempest results
into one, preserving per-seed row grouping, pads to common L) and
`walks.multi_bias_walks(trw, seeds, bias_configs, K, L)` (one Tempest call per config +
merge). `--num-walks-per-node` → `--num-walks-per-node-per-bias` (default 3, so each
seed gets `3 × len(configs)` mixed-bias walks); `--walk-bias`/`--start-bias` removed
(configs in `walks.MULTI_BIAS_CONFIGS`, env-overridable via `MULTI_BIAS_CONFIGS`).

Runs on the source-side head + pair features, TPNet bs (train 200 / eval 20, seed 42):

| walk config | peak val / test | peak ep | vs single-bias |
|---|---|---|---|
| **single bias (ExpW)** | **0.8064 / 0.7791** | ep3 | — |
| mixed-walk: start=ExpW; walk ∈ {ExpW, Uniform, Linear} | 0.8016 / 0.7772 | ep2 | −0.0048 / −0.0019 |
| homogeneous: {ExpW/ExpW, Linear/Linear, Uniform/Uniform} | 0.7945 / 0.7722 | ep3 | −0.0119 / −0.0069 |

### Finding — multi-bias walks hurt, monotonically with ExpW dilution

The more Uniform/Linear walks in the mix, the lower the peak: single-bias ExpW (0/3
diluted) > mixed-walk (2/3 walk-bias diluted) > homogeneous (2/3 fully non-ExpW). On
this recurrence-heavy dataset **ExponentialWeight (recency-biased sampling) is the
load-bearing sampler**; Uniform/Linear walks are noise the GRU has to average over, so
adding them dilutes rather than diversifies. Both multi-bias variants also peak ~1
epoch earlier and aren't more stable. Cost: ~15% slower per epoch (3 Tempest calls +
9 vs 5 walks/seed; source-side keeps it cheap). **Single-bias ExpW stays the winner;
the branch is a documented negative result.** The `merge_walks` / `multi_bias_walks` /
env-selectable-config infrastructure is reusable if multi-bias is ever wanted on a
dataset where a single bias isn't dominant.

---

## 11. Tempest sliding-window sweep (`--tempest-batch-window-multiplier`)

On master (source-side head + pair features), TPNet bs (train 200 / eval 20, seed 42).
The multiplier caps Tempest's retained history: `max_time_capacity =
round(mult · batch_size · mean_inter_arrival)`; `-1` = unbounded.

| window | max_time_capacity | peak val / test | peak ep | vs unbounded |
|---|---|---|---|---|
| mult = −1 (unbounded) | ∞ | 0.8064 / 0.7791 | ep3 | — |
| mult = 3 | 10,452 | 0.8073 / 0.7803 | ep3 | +0.0009 / +0.0012 |
| **mult = 5** | 17,420 | **0.8074 / 0.7811** | ep3 | +0.0010 / +0.0020 |

### Finding — a bounded window marginally beats unbounded (within noise)

Both bounded windows edge out unbounded by ~+0.001 val / +0.001–0.002 test, with the
larger window (mult=5) slightly best. The bounded runs start *lower* at ep1 (~0.789 vs
0.798 — sparser walks early) but catch up and crest marginally higher at the same epoch,
with the same stable gentle tail. Two take-aways: (1) a sliding window does **not**
starve the source-side walks — the peak is unaffected-to-slightly-better; (2) the old
model's "unbounded is best" max-time-capacity finding does **not** carry over to the
source-side + pf stack. The gains are **single-seed and inside the noise band**, so not
a confirmed win — multi-seed on mult=5 (or a slightly larger window, mult=8/10) would
be needed to tell signal from noise. No code change (the flag already exists);
documented for the record.
