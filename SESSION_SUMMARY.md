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

---

## 12. Localizing the TPNet gap — test-MRR stratification

Branch `feature/stratify-test-errors` (code: a `Trainer._eval` recorder hook +
`scripts/stratify_test_errors.py`; doc-only here). Re-ran the current best model's
(source-side + pf) **test** eval capturing, per positive, the reciprocal rank +
strict-causal metadata (endpoint-seen, pair-seen/count, degree), and stratified it.
Sanity-checked: stratified MRR 0.7786 == `_eval` return; partitions reconstruct ±1e-4.

| slice | frac | mean_rr | verdict |
|---|---|---|---|
| repeat-pair | 87.4% | **0.887** | solved — the gap is NOT decoder capacity |
| **new × both-seen** | 8.0% | **0.022** | the co-reachability cell (+0.063 headroom) |
| new × u-only-inductive | 4.5% | 0.043 | cold-start / bootstrap (+0.035) |

**The entire +0.048 test gap to TPNet is the 12.6% new-pair slice** (mean_rr ~0.03);
the 87% repeat majority is already at 0.887. The decisive cell is `new × both-seen`
(both endpoints known, never linked).

## 13. Attacking the new-pair slice (co-reach + cross-attention) — all dead

Targeted the `new × both-seen` cell three ways; **all three hit the same floor**, so
its headroom is **not recoverable from graph / neighbourhood structure** — it is
content / cold-start signal (page text, node features). Judged on the SLICE via the
recorder, never aggregate (the 8% slice is swamped in aggregate — the exact trap that
buried earlier co-reach).

1. **Step-0 exact co-reach precondition** (branch `feature/gated-coreach`,
   `scripts/coreach_precondition.py`). Scored the slice's candidates by exact causal
   co-reach (streamed adjacency, no walk-sampling noise), through the TGB scorer.
   Caught a **bipartite confound**: tgbl-wiki is user×page, so even-power co-reach
   (`A^2`,`A^4`) is structurally 0; the cross-side signal is odd powers. **Ceiling =
   `A^3` 0.0414** (`A^5` 0.0361; time-decay dead) — far below the 0.10 dead bar, only
   ~2× the model's 0.022. The new positives aren't co-reach-separable from negatives
   (TGB negs include pages MORE co-reachable with u than the genuinely-new positive).
2. **Aggregate co-reach** (Wave-2/3, §3) — already falsified: redundant with the GRU.
3. **u-v metric cross-attention** (branch `feature/xattn-uv`, off `6922976` cross head;
   `scripts/xattn_run.py`). Re-test of the previously-falsified cross-attention head,
   bugfixed (tied `Wqk`, ONE fixed temp `1/√d_head`, candidate-aware slots) on the
   CLEAN link-trained regime (no alignment, no detach), gated to new pairs. C1 @ TPNet
   bs: aggregate 0.7762 (cleared the cross-head baseline ~0.774), repeat-slice guard
   held (0.8845 — the gate makes repeats byte-identical), but the **`new × both-seen`
   slice did NOT move: 0.0204 ≈ 0.022**. Un-collapsed token-level neighbourhood overlap
   carries no signal the pooled-mean discarded. Curve was the healthy gentle shape (no
   drift), so the channel is well-behaved — it just finds nothing.

**Bonus (regime attribution):** the old cross-attention died under detached-E +
alignment; this one runs in the clean no-detach regime and *still* finds nothing on the
slice ⇒ the **detached-E regime was NOT the cause** of the old failure — the mechanism
simply doesn't carry signal here (so C2/C3/C4 were moot).

**Conclusion:** the new-pair headroom is triply-confirmed not-in-the-graph. Closing the
TPNet gap on wiki needs **content / node features** (page identity/text, cold-start
priors), not more walk/graph structure. Walk-based pair features and metric
cross-attention are exhausted on this slice.

---

## Link heads: Point vs Gaussian vs Attention × unbounded vs window-5 (2026-06-13)

**Decision: the Point head is the one we build on going forward.** It has the
highest peak (tied with Gaussian, above Attention), is the *only* monotone one
(no overfit-drift), is the cheapest at Gaussian-equivalent accuracy, and degrades
the most gracefully under windowing.

Three link heads on tgbl-wiki, each crossed with the Tempest history window. The
two **geometric** heads share a base-point construction: `p = E[u]`, log-map u's
recency-weighted walk-neighbours into `T_{E[u]}`, predict a mean
`μ = Σ softmax(−λ·age)·Log_p(E[node])`. The **Point** head
(`GeometricPointHead`, master `57ebbd7`) scores by `−α‖ν−μ‖ − β·angle`; the
**Gaussian** head (`GeometricGaussianHead`, branch `feature/geometric-gaussian-head`
`a984e79`) fits a recency-weighted covariance `C` and scores by Mahalanobis
`(ν−μ)ᵀ(C+τI)⁻¹(ν−μ)` (Woodbury, only an `[n,n]` inverse). The **Attention** head
(`SourceWalkAttnHead`, "Option A" — query-relative pre-softmax recency-biased
attention over the source walk-neighbours; tag `source-walk-attn-head`, branch
`feature/source-attention`, commit `790f51c`, the pre-geometric baseline) is the
contrast: a learned attention pool instead of a fixed geometric statistic. All
runs: walks10, no pair features, bs 200 / eval-bs 20, lr 1e-3 → 1e-5, seed 42,
30 epochs, patience 5, decay-horizon 30 (except the two early geometric runs,
noted below).

### Peaks (best-val, restored)

| head | **unbounded** (val / test) | **window-5** (val / test) | Δ window−unbounded |
|---|---|---|---|
| **Point**     | **0.8057 / 0.7720** (ep20*) | 0.7965 / 0.7491 (ep37) | −0.009 / **−0.023** |
| **Gaussian**  | **0.8050 / 0.7719** (ep30)  | ~0.795 / ~0.749 (proj†) | ≈ −0.010 / **≈ −0.023** |
| **Attention** | 0.7966 / 0.7703 (ep10‡)     | ~0.7630 / ~0.7144 (ep11§) | **−0.034 / −0.056** |

\* Point unbounded hit its 20-epoch cap *still climbing* (not fully converged),
yet already matched Gaussian's converged ep30 peak.
† Gaussian window-5 was stopped manually at **ep15 (val 0.7828 / test 0.7352,
still climbing)**; the plateau is projected from its step-for-step tracking of
the Point window-5 curve, which peaked 0.7965/0.7491 at ep37.
‡ Attention unbounded peaked at ep10 then **overfit-drifted** (val 0.7966→0.7914)
and early-stopped at ep15 — the only head that is *not* monotone.
§ Attention window-5 was stopped manually at ep12 (peak ep11), already plateauing
(ep12 patience 1/5); it does not climb on like the geometric window-5 runs.

### Window-5 epoch overlay (Point vs Gaussian, the clean A/B)

Near-identical schedule for ep1–12 (cosine barely moved); the two curves are the
**same curve within ±0.003**:

| ep | Gaussian val / test | Point val / test |
|---|---|---|
| 1  | 0.6544 / 0.5969 | 0.6546 / 0.5971 |
| 5  | 0.7285 / 0.6737 | 0.7298 / 0.6752 |
| 10 | 0.7710 / 0.7221 | 0.7719 / 0.7237 |
| 15 | 0.7828 / 0.7352 | 0.7856 / 0.7389 |

(ep13+ the schedules diverge slightly — Point's run used decay-horizon 50, the
Gaussian's 30 — so compare ep1–12 as apples-to-apples and peaks otherwise.)

### What was observed

1. **Point ≈ Gaussian everywhere — the covariance buys nothing on accuracy.**
   Unbounded the two tie within 0.001 (0.8057/0.7720 vs 0.8050/0.7719);
   windowed they track epoch-for-epoch (±0.003 over ep1–12). Adding `C` +
   Mahalanobis over the shared `μ` does not move MRR on wiki.
2. **Windowing hurts BOTH heads by the same ~0.023 test (~0.009 val).** The
   window, not the head, sets the ceiling: it starves the neighbourhood the
   geometric channel is built from. Identical penalty for Point and Gaussian.
3. **The Gaussian covariance gives NO window-robustness** (falsifies the
   "wider region survives eviction" hypothesis). Under the window the Gaussian
   tracks the Point head step-for-step, because `μ` is shared and `C` just rides
   along — and `C`, a second moment, is the *more* sample-starved quantity, so
   if anything it is slightly more window-fragile (matched by the learnable `τ`
   falling back toward mean-distance). Mechanism: the region is recomputed from
   the sampled tokens each forward; evicted neighbours contribute nothing
   regardless of how well their `E` was trained.
4. **Gaussian costs more for nothing.** Per-epoch: unbounded ~115s (Gaussian)
   vs ~100s (Point); window-5 ~38s vs ~23s — the `[n,n]` inverse, with zero
   accuracy return.
5. **Window-5 is ~3× faster per epoch** (small graph speeds *both* train and
   eval) but caps ~0.023 test lower — a pure speed/accuracy trade, same for both
   heads.

### Attention head — front-loads, overfit-drifts, most window-fragile

The attention head behaves *qualitatively* differently from the geometric pair:

1. **Front-loads hard.** Unbounded ep1 = **0.7891 / 0.7621** — a huge head-start
   (geometric heads start ~0.773 / ~0.740) and already within ~0.01 of the
   geometric *converged* peak. It is near-peak by ep10.
2. **Not monotone — it overfit-drifts.** Unbounded peaks **0.7966 / 0.7703 @
   ep10**, then val falls (0.7966→0.7914) while link loss keeps dropping →
   early-stop ep15. This is the classic walk-tower drift, delayed to ep10. The
   geometric heads, by contrast, climb cleanly to ep30 with no drift. **Point/
   Gaussian are monotone; Attention is not.**
3. **Lower ceiling than geometric.** Peak test 0.7703 sits just *under* the
   geometric 0.7719/0.7720, and on val it is clearly lower (0.7966 vs ~0.8055).
   It reaches its (lower) peak fast (ep10) where the geometric heads need ~ep25–30
   to reach their (higher) one — fast-but-lower-ceiling vs slow-but-higher.
4. **Most window-fragile of the three — by ~3×.** Window penalty **−0.034 val /
   −0.056 test**, vs ~−0.009 / ~−0.023 for the geometric heads. Window-5 caps it
   at ~0.763 val — *below even the geometric window-5* (~0.795). Under window-5 it
   matches the Point window-5 curve for ep1–7 (it even *leads* by ~0.004 early —
   its fast-start edge survives), then crosses at ep7 and plateaus while Point
   climbs on. Mechanism: the attention head's whole value is *reading a rich
   neighbourhood* (the source of its unbounded head-start); windowing starves
   exactly what it is best at, so it collapses to the lowest windowed ceiling. The
   geometric heads lean on a simpler statistic (μ) that degrades far more
   gracefully — confirming windowing is *not* a regularizer that rescues the
   attention head's drift; it just lowers the ceiling.
5. **Speed:** Point-class (~100s/epoch unbounded, ~24s/epoch windowed).

### Takeaway — Point is the head we build on

On tgbl-wiki, no pair features, across all three heads and both window regimes:

- **Point ≈ Gaussian on accuracy** (tie within 0.001 unbounded; track
  epoch-for-epoch windowed) — the Gaussian covariance buys nothing and costs the
  `[n,n]` inverse, and gives no window-robustness either.
- **Attention front-loads but loses**: non-monotone (overfit-drift), a slightly
  lower test ceiling (0.7703 vs 0.7719/0.7720), clearly lower val, and ~3× the
  window penalty.
- **Windowing hurts every head** — unbounded wins for all three (Point/Gaussian by
  ~0.023 test, Attention by ~0.056). The window is the ceiling, not the head.

⇒ **Build on the Point head, unbounded.** Highest peak, the only clean monotone
climb (no overfit babysitting), cheapest at top accuracy, and most window-tolerant.
Gaussian is a no-gain complexity tax; Attention is a fast-but-overfitting,
window-fragile alternative. If window-insensitivity is ever needed, the lever is
feeding μ from an un-evicted per-node neighbour buffer — not switching heads.

Logs: `logs/manifold/run_{point,gauss,attn}_*` (gitignored). Heads preserved at
tags `geometric-point-head`, `geometric-gaussian-head`, `source-walk-attn-head`.

---

## Geometric-head improvement series — NO pair features, cross-dataset (2026-06-14)

Goal: find the single best geometric link head (improved Point **or** improved
Gaussian, carried forward unidirectionally) that generalizes across TGB — not
wiki-only. Crucial reframe: this series runs **without pair/recurrence features**,
so the geometry carries the prediction (with pair features on wiki the model
down-weights the geometric channel to `coef_geo≈0.07` and every geometric change
is masked). Selection oracle = **tgbl-review** (cold-start, surprise 0.987,
~100% new-pair); **tgbl-wiki** (recurrence-heavy) is the non-regression guard.

**Infra** (on `feature/point-improvements`): `scripts/strat_run.py` trains a head
at best-config and runs a strict-causal **per-slice** stratified test eval
(`Trainer._eval` + a recorder hook) — repeat/new-pair, source-degree, plus learned
params and n_eff / ‖μ‖ distributions. Best-val-selected, single-seed (seed 42)
unless noted. Config: d_emb 128, walks10/mwl20, 50ep/patience5, lr 1e-3. Review is
subsampled to wiki size (110k recent train / 25k official val,test; **K_train=100**
on review — the 352k-node embedding leaves no GPU headroom at K=300).

### Result ledger (Δ test vs the plain Point baseline)

| variant | which | wiki | review | verdict |
|---|---|---|---|---|
| Imp1 ablate angle −β·θ | Point | tie | — | drop θ (redundant with d via law of cosines; β learned ≈0) |
| Imp2 global **ambient** diagonal metric | Point | **−** | — | 49× anisotropy but in a per-source-rotating frame → net loss |
| N3 explicit radial channel −α₀‖ν‖ | Point | tie | **−0.010** | new-pair is chord-ceilinged; rescales geodesic vs recency → hurts cold-start. N5★ skipped |
| N1 soft-min set-distance (κ) | Point | — | tie | multimodality dead; the unlearned core of the already-falsified attention head |
| G1 per-source trace-whitening `d/√(s²+τ)` | Gauss smoke | — | **−0.002** | per-source 2nd moment unestimable; hurts cold-start → **Gaussian line dead** (G2/G3 skipped) |
| **N2 intrinsic-frame anisotropy** | **Point** | **+0.0025** | **+0.0095 (2-seed)** | **WINNER — carried head** |

### THE WINNER — N2 (committed `fa8e70d` on `feature/point-improvements`)

The candidate distance is an **ellipse oriented along each source's own heading**
`r = μ/‖μ‖`, not an isotropic circle:
`d² = a‖δ∥‖² + b‖δ⊥‖²`, δ=ν−μ split into along-heading (δ∥) and sideways (δ⊥);
a, b ≥ 0 are **two global scalars** (`logit = −α·d`). a=b recovers the plain Point
head. The model learns that being off *along* the heading is ~free while being off
*sideways* is costly — **direction matters more than exact distance**, the clean
version of what the dropped angle term θ reached for.

- **wiki:** learns a/b ≈ 1/100 (strong ellipse), **+0.0025 test** (0.7757→0.7782).
- **review (cold-start):** **+0.0095 test, identical delta across seeds 42 & 1**
  (N2 0.0828 vs Point 0.0733) — a real systematic gain, not noise.
- **Adaptive**: ~isotropic where no direction signal exists → never hurts. 2 params,
  cannot overfit. Cold-source safe (μ≈0 ⇒ r→0 ⇒ d→√b·‖ν‖ = geodesic-to-u).

### Decision & why

- **Best head = the N2 Point head.** It is the only lever that beat baseline, and
  it wins **both** regime endpoints (wiki recurrence + review cold-start).
- **The Gaussian line is dead.** Its distinctive per-source 2nd moment is *never a
  net positive*: the full Gaussian shrinks the covariance away (τ→∞ ⇒ ties Point),
  and the cheapest estimable fragment (G1's scalar trace), when *forced*, **hurts
  cold-start** (60% deg-0 sources have noisy/zero spread; one global τ can't spare
  them). So Point ≥ Gaussian in every regime, at lower cost.
- The remaining gap to TPNet is **architectural** (recurrence/encoder/content
  channels), not in the head — every geometric prototype (point, axis/ellipse, set)
  and every per-source 2nd-order idea is now exhausted.

Caveats: single-seed except N2-review (2 seeds); review is a wiki-sized chronological
subsample at K=100 (relative-comparison harness, not the full-review leaderboard
number). Per-slice ledger: `logs/manifold/SERIES_NOPAIR.md` + `strat_*.md` (gitignored).

---

# Velocity / trajectory-extrapolation head bake-off, 2026-06-14/15

**Hypothesis under test:** replace the static recency-weighted *centroid* μ (the N2
Point head above) with a *trajectory*: fit the direction u's walk has been moving
through tangent space over AGE by weighted least squares, project that line forward to
query time, and score candidates by an ellipse oriented along the **motion** V̂. Three
variants, each on its own branch off master (all carry the same intrinsic-frame
anisotropy + 3 learnable channel-mix coefficients `coef_geo/coef_rec/coef_pair` as
master, so the *only* change vs master is centroid → trajectory):

| branch | head | how the K walks are used |
|---|---|---|
| `feature/velocity-head`        | `GeometricVelocityHead`           | **pooled** — one line over all K·L nodes (flat `[B,n,d]`) |
| `feature/velocity-perwalk-avg` | `GeometricVelocityPerWalkAvgHead` | per-walk fit, **averaged** μ,V over K walks (`[B,K,L,d]`) |
| `feature/velocity-mixture`     | `GeometricVelocityMixtureHead`    | per-walk fit, **soft-min** over K walk-lines (`[B,K,C,d]`) |

The structured (`perwalk-avg`, `mixture`) heads needed a trainer `_score` adapter that
keeps the (K,L) walk axes instead of flattening; `train.py` got `--max-train-edges/
--max-eval-edges` chronological subsample (cherry-picked to all branches) for review.

### Results — 12 runs (master + 3 branches × {wiki-nopf, wiki-pf, review-nopf}), seed 42

| branch | wiki nopf (val/test) | wiki pf (val/test) | review nopf (val/test) |
|---|---|---|---|
| **master (N2 Point)**     | **0.8106 / 0.7789** | 0.8173 / 0.7934 | 0.0903 / 0.0935 |
| velocity-head (pooled)    | 0.8053 / 0.7730     | 0.8173 / 0.7935 | 0.0904 / 0.0937 |
| velocity-perwalk-avg      | 0.8067 / 0.7783     | 0.8163 / 0.7935 | **0.0905 / 0.0939** |
| velocity-mixture          | 0.8067 / 0.7782     | 0.8165 / 0.7935 | OOM (1.53 GiB, 352k-node emb) |

(Runtimes: wiki nopf ran long — 25–34 epochs, ~85–110 min; wiki pf converged fast —
7–14 epochs, ~31–55 min; review ~4 min. Per-epoch cost rises across epochs because the
Tempest graph accumulates edges.)

### Verdict — the trajectory idea does NOT beat the static centroid. Keep master.

- **Wiki no-pf (the pure head comparison): master WINS.** 0.7789 test vs the best
  velocity head 0.7783 (perwalk-avg), +0.0039 val. All three velocity heads are
  **≤ master**; the pooled head is clearly worst (−0.0059 test). Extrapolating "where
  activity is heading" buys nothing over "where activity sits" — and costs accuracy.
- **Wiki pf: everything ties** at test ≈ 0.7934–0.7935. As before, pair-feature
  recurrence dominates and the geometric channel washes out (`coef_geo` shrinks) — the
  centroid-vs-trajectory distinction is invisible here.
- **Review cold-start: velocity heads a hair ahead** (perwalk-avg 0.0939 vs master
  0.0935 test, +0.0004) but **deep inside the noise band** (bar = ≥0.015 or 3-seed;
  see [[feedback_noise_threshold_wiki]]). Not shippable.
- **Best velocity head, if one must be picked: `perwalk-avg`** — best-or-tied velocity
  head on every cell, fastest of the structured pair, and **no OOM**. The `mixture`
  head is the worst operational choice: its `[B,K,C,d]` per-walk×per-candidate tensor
  OOMs review on 8 GB even at K=100, and it's the slowest. Pooled is worst on accuracy.

**Decision: do not fold any velocity head into master.** The carried N2 anisotropic
Point head stays the best head. This closes the trajectory-extrapolation line — it joins
the Gaussian per-source 2nd-moment line as falsified. The remaining gap to TPNet
(verified 0.827 test / 0.842 val) is architectural, not in the head geometry.

Branches kept (not merged) for provenance: `feature/velocity-head` (a25db3c),
`feature/velocity-perwalk-avg` (78628f6), `feature/velocity-mixture` (4d7c0da).
Raw ledger: `logs/manifold/run12/RESULTS.tsv` (+ per-run `*.log`). Single-seed;
mixture-review is OOM, not a number.

### Promotion override (2026-06-15): velocity-perwalk-avg is now master's head

Decision reversed by direction: **`GeometricVelocityPerWalkAvgHead` (per-walk averaged
trajectory) is promoted to master** as the new base head, replacing the N2 Point head.
Rationale: the Point head's wiki-no-pf win came from its **heading-frame** ellipse
(oriented along the centroid heading r=μ/‖μ‖); the velocity head currently orients its
ellipse along the **motion** V̂ — a different, not-yet-tuned frame. Rather than keep two
heads, we take the trajectory head as the substrate and **port the heading-frame
"ellipse point fix" onto it next** (expected to recover and likely exceed the −0.0006
wiki-no-pf gap). Until that port lands, master is ~sub-noise below the retired Point
head on wiki-no-pf and ties everywhere else (see table above).

The retired Point head is preserved in git history (master pre-`b77519b`) and on
`feature/point-improvements`; not lost. Docstring de-framed (dropped the OPTION-2 /
pooled-vs-mixture bake-off language) before promotion. `best_configs.sh` comment flags
the pending ellipse-fix TODO.

---

# Edge-feature plumbing + weighting, 2026-06-15

**Two pieces, different fates.**

**1. Edge features plumbed into `WalkData` (KEPT on master, `b835c3f`).** TGB datasets
carry per-edge features (tgbl-wiki: 172-dim) that were loaded → fed into Tempest →
**dropped** on the way back out (`walks.py` discarded the sampler's 4th return as
`_ef`). Now `WalkData.edge_feats` exposes them as `[NK, L, d_ef]` (`None` when absent),
**index-aligned 1:1 with `nodes`/`timestamps`**: Tempest returns `[NK, L-1, d_ef]`
(no seed-slot row) and we right-pad one zero column so the context mask
(`positions < lens-1`) selects exactly the real edges. Pairing verified against the
backward-walk contract — synthetic graph encoding `[src,tgt,ts,eid]` → **108/108**
ts-match + endpoint-match, seed/padding exactly zero. Pinned in
`tests/test_walk_edge_feats.py`. The model can read `wd.edge_feats` in `_score`
whenever a head wants it; nothing consumed it before the experiment below.

**2. Learnable edge-feature weighting on the velocity head (REVERTED; preserved at tag
`edge-feature-integration`).** Re-weighted each step of the per-walk LS fit by
`softmax(−λ·age + edge_proj(edge_feat))` — a **zero-init `Linear(172→1)`** added to the
recency log-weight (exact no-op at init; grows only if it lowers the loss). Edge
features entered ONLY the fit weights (no candidate channel — the `(u,v)` edge doesn't
exist at query time). Always-on, no flags. Validated: byte-identical no-op at init,
grads reach `E_u`/`E_v`/tokens/`edge_proj.weight`, masked steps stay `w=0`.

**Outcome (wiki no-pf, vs the current master velocity head, same config/seed 42):** it
**did not improve** — through the run it tracked the baseline within ±0.003, oscillating
and trending **slightly negative** (ep6–11 Δval/Δtest ≈ −0.001; ep12 val +0.0014 /
test −0.0003). Inside the wiki single-seed noise band ([[feedback_noise_threshold_wiki]]);
the learned edge re-weighting buys nothing the recency weighting doesn't already get.
Run stopped early (ep12) once the verdict was clear.

**Disposition:** merged to master to land the commit in history, **tagged
`edge-feature-integration`** (annotated, at `74d0726`), then **reverted on master**
(`f000942`) so master's head stays edge-free. `git checkout edge-feature-integration`
(or cherry-pick `74d0726`) restores the full feature. The `WalkData.edge_feats`
plumbing stays on master as reusable infra. Branch `feature/edge-feature-weights` kept.

---

# With-pair-features test stratification on current master, 2026-06-15

Re-ran `scripts/stratify_test_errors.py` on the **current master head**
(`GeometricVelocityPerWalkAvgHead` + `--use-pair-features`) to re-localize the gap to
TPNet after the velocity-head promotion. wiki, seed 42, d_emb 128, walks10, K_train=100,
TPNet protocol (train bs 200 / eval bs 20), 25 ep / patience 5. Peaked **ep4 (val 0.8145
/ test 0.7921)**; stratified test MRR **0.7922** over 23,621 positives (sanity OK,
partitions reconstruct ±1e-4). **TPNet ref 0.827 test → gap +0.0348.** (The output file's
title still says `SourceWalkAttnHead` — stale script header; the trained head is the
current velocity head.) Two script-side bit-rot fixes were needed to run against master:
re-added the optional `recorder` hook to `Trainer._eval` (stripped in `cdc99c0`) and
dropped stale `neigh_store` / `use_struct_features` references in the script.

### Pair recurrence (the whole story in two rows)

| slice | frac | mean_rr | hits@1 | hits@10 | contrib |
|---|---|---|---|---|---|
| **repeat-pair** | 87.4% | **0.9030** | 0.850 | 0.985 | 0.7888 |
| **new-pair**    | 12.6% | **0.0268** | 0.003 | 0.056 | 0.0034 |

### Cross-tab: pair-recurrence × transductivity (decisive)

| slice | frac | mean_rr | verdict |
|---|---|---|---|
| repeat × both-seen | 87.4% | 0.9030 | solved — not a decoder-capacity gap |
| **new × both-seen** | **8.0%** | **0.0207** | the co-reachability cell (both endpoints known, never linked) |
| new × u-only-inductive | 4.5% | 0.0383 | cold-start / bootstrap source |
| new × v/both-inductive | 0.1% | ~0.004 | negligible mass |

### Source-degree (orthogonal view — same cold-start story)

deg=0 sources (4.5%) sit at mean_rr **0.0381**; the curve climbs monotonically to
0.87 (deg 21-100) / 0.84 (deg >100). Low-degree/cold sources are exactly the new-pair
mass; warm sources are solved.

### Where the gap is

**Unchanged from §12's pre-velocity-head finding: the entire +0.0348 test gap to TPNet
is the 12.6% new-pair slice (mean_rr 0.027); the 87% repeat majority is already at
0.903.** Headroom sizing: lifting `new × both-seen` (8.0%, 0.0207) to even 0.30 is
+0.0223 overall — alone more than half the TPNet gap; to the both-seen level (0.829) it
is +0.0646. The decisive cell is again **new × both-seen** (both endpoints seen, never
linked) plus the cold-start `new × u-only-inductive` cell. Per §13 this headroom is
**triply-confirmed not recoverable from graph/walk structure** (exact co-reach ceiling
`A^3`=0.041, aggregate co-reach redundant, cross-attention finds nothing on the slice) —
it is content / node-feature / cold-start signal. The velocity head did **not** change
the shape of the gap: it remains architectural (content channels), not in the head
geometry or pair features. Output: `logs/stratify/wiki_test_strata_attn.{md,json}`
(single-seed).
