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

1. **Ship `--use-pair-recency --use-pair-history`** — exact pairwise `(u,v)` recurrence
   (time-since-last-interaction) + ever-bit + decayed count, from the streaming store,
   added additively to the chord logit. **+0.015–0.020 test, 3-seed confirmed, smooth.**
   This is exactly TPNet's `A^(1)_{u,v}` recurrence block, computed exactly (no JL sketch).
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
