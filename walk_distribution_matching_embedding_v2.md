# Walk-Distribution-Matched Temporal Embeddings

**Design & Execution Plan (v2.3 — Phase S architecture locked, loss-family search integrated)**

---

## 0. What changed from v2.0 → v2.1 → v2.2 → v2.3

**v2.1** fixed five issues with v2.0's Phase S frame: Group E added (head structure searchable); anchor validation §3 (3 seeds, gates Phase S); Group A split A1/A2/A3 (weighting / on-off / target); §4.4 floor = anchor-validated mean; P4 broken out as ~1-week phase.

**v2.2** added §4.6's deduplicate-by-effective-compute-graph clause so redundant Phase S cells are skipped when search-space dimensions collapse (e.g., A2 × `λ_link` under E.2).

**v2.3 (current)** records Phase S progress so far and integrates the loss-family amendment as a revised Group A3:

- **Wiki anchor (§3) CONFIRMED** at test 0.7070 ± 0.0016 across seeds {42, 7, 13}.
- **Group A2 (alignment on/off) SEARCHED.** A2-off wins by +0.0019 test MRR (0.7089 vs 0.7070). Per v2.2 §4.4 ties-go-to-simpler, A2-off is the wiki Phase S A2 winner.
- **Group E (link MLP head) SEARCHED under A2-off.** E.2 (Component-0-only head, 99-dim input) ties E.1 (cross-table, 1123-dim) at 0.7079 ± 0.0005 vs 0.7089 ± 0.0012 — Δ −0.0010, inside both stds. Simpler wins: E.2 is the wiki Phase S E winner under A2-off.
- **Phase 0.5 over-training cliff confirmed.** 2-ep test 0.7070 vs 50-ep test 0.4269 under alignment+uniformity. Cross-table column norms grow 5× over 50 epochs. The cliff is the load-bearing diagnostic for v2.3's loss-search direction.
- **Group A3 (supervision target) REVISED** — original spec (per-position / endpoint / multi-endpoint, all within the same inner-product-similarity family) is replaced by a loss-family search per §4.7 below. Triplet, InfoNCE, SGNS, plus a diagnostic-derived norm-brake regularizer. Decision: pick a primary that eliminates the cliff and transfers cross-dataset; wiki peak is expected to *tie* the anchor regardless of loss change.
- **Loss-family amendment integrated.** The standalone `loss_function_search_ammendum.md` is archival as of v2.3; its load-bearing content lives in §4.7.

**Execution structure (v2.3, post wiki A2+E lock):**

**Loss-family search §4.7 (6 cells on wiki, ~3 hours)** → **Apply winning loss to tgbl-review-v2** → **P1 / P2 / P3 conditional refinement** → **Cross-dataset deployment (server / A40 for tgbl-coin / -flight / -comment)** → **P4 deferred until production architecture is locked**.

The thesis ("walk sampler's distribution is the supervision signal") was directly tested by Group A2 and *did not survive on wiki* — alignment loss is at best neutral on a recurrence-saturated dataset. v2.3 keeps the thesis as a cross-dataset claim, to be re-tested on review and the server datasets.

---

## 1. Thesis

> The walk sampler's distribution is the supervision signal. The alignment loss's job is to make the embedding inner product approximate `log P(walk ends at v | seed = u)` under that distribution. Time decay, distance decay, and recurrence emerge from this matching rather than being hand-coded — the sampler already encodes them.

A separate, **now confirmed** claim: the link prediction head needs explicit time-since-last-event signals at scoring time. Phase 0.5 diagnostic confirmed Component 0 (time encoding + cold-start bits) drives test MRR from ~random to provisionally 0.71 at 2 epochs with no walks supervision contributing usefully.

The open question, surfaced by the diagnostic: **does walks-supervision help, hurt, or have a sweet spot on this dataset?** Phase S finds out.

---

## 2. What we know going in (locked findings — Phase 0.5 + wiki Phase S A2/E)

These are non-negotiable for v2.3 and downstream phases:

1. **Time encoding works substantially on wiki.** Component 0 + random-init cross-table → Test MRR **0.7070 ± 0.0016 at 2 epochs** across 3 seeds (anchor confirmed §3). The `is_cold_start_uv` bit alone is doing most of the work (99.1% of test pairs are uv-cold-start; the bit carries the EdgeBank-recurrence signal natively).

2. **Cold-start bits do not get LayerNorm-washed.** Column-norm analysis showed the 3 bits get amplified to 1.78× the cross-table mean (50-epoch model). The §12.1 mitigation is unnecessary.

3. **Walks-as-currently-supervised hurt at 50 epochs (the over-training cliff).** Cross-table column norms grow 5× from 2-ep (~0.36) to 50-ep (~1.99); test MRR drops 0.28 from 0.7070 → 0.4269 under alignment+uniformity. The embeddings are learning *something*, and that something correlates negatively with eval. **Eliminating this cliff is the load-bearing win condition for §4.7's loss-family search**, not lifting wiki peak.

4. **Walks-supervision is at best neutral on wiki (Group A2 result).** A2-off (`λ_align = 0`) tests at 0.7089 ± 0.0012 vs anchor 0.7070 ± 0.0016 — Δ +0.0019, just above anchor std. Per v2.2 §4.4 ties-to-simpler, **A2-off locks on wiki**. The thesis "walks distribution IS the supervision signal" was directly tested and did not produce a measurable lift; remains an open question for non-recurrence-saturated datasets.

5. **Cross-table reads of uniformity-only embeddings are noise (Group E result under A2-off).** E.2 (Component-0-only head, 99-dim) at 0.7079 ± 0.0005 ties E.1 (8-block cross-table, 1123-dim) at 0.7089 ± 0.0012 — Δ −0.0010 within both stds. **E.2 locks** under A2-off. Cross-seed std drops to 0.0005 (no embedding-table init variance contributes when embeddings are unread at scoring), confirming the dedup-via-compute-graph reasoning empirically.

6. **The strict-causal protocol works.** No leak shape surfaced in the diagnostic; the protocol's order (walks pre-ingest, ingest post-scoring last) was audited 3× in code.

7. **TGB serves ~999 random negatives per positive at eval.** For tgbl-wiki, ~99% of negatives are cold-start at the (u,v) level. The eval task is dominantly "is this pair recurring?" — which is why wiki is recurrence-saturated and why §4.7 expects wiki peak to tie the anchor regardless of loss choice.

---

## 3. Anchor validation (30 min, BEFORE Phase S)

The "Component 0 only, 2-epoch, Test 0.71" finding from the diagnostic was a single seed. The entire Phase S frame anchors on this number. Validate it before committing.

### 3.1 Anchor validation protocol

Run the Phase 0.5 architecture (Component 0 + dual identity tables + 8-block cross-table link MLP + current alignment+uniformity loss) with three seeds: {42, 7, 13}. Each run: 2 epochs only, otherwise default config.

Report: mean ± std of val MRR and test MRR across the three seeds.

### 3.2 Anchor validation decision gate

- **If mean test MRR ≥ 0.70 with std ≤ 0.02:** anchor confirmed. Phase S anchors at the mean. Proceed.
- **If 0.65 ≤ mean < 0.70:** anchor partially validated. Phase S anchors at the verified mean (whatever it is), not the 0.71 from the smoke. §4.4 success criterion adjusts accordingly.
- **If mean < 0.65 or std > 0.04:** the 0.71 smoke was lucky. Stop. Investigate before Phase S. Likely causes to check: did the diagnostic's training loop use a different config than v2.1 expects? Was the 2-epoch run somehow different (different batch ordering, different walk-gen state)?

### 3.3 What we lock in after anchor validation

The verified mean test MRR becomes the **Phase 0.5 baseline** for all downstream comparisons. Every Phase S configuration is judged against this number, not against 0.71.

---

## 4. Phase S: search for the right architecture (12 hours)

This is not a "run a sweep" phase. It's a "let Claude Code explore and report findings" phase, with bounded scope and clear decision criteria.

### 4.1 What Phase S is allowed to vary (status per group as of v2.3)

**Group A1 — Within-family alignment weighting:** PENDING — only relevant if a primary loss in §4.7 stays within the alignment family.
- Current: `1/K · (1 + Δt/τ)^(-β)`
- `1/K` only
- Uniform `α = 1` over walk positions

**Group A2 — Alignment on/off (`λ_align`):** **WIKI COMPLETE.** A2-off (`λ_align = 0`) wins by +0.0019 (0.7089 vs 0.7070). Locked on wiki; revisited per-dataset.

**Group A3 — Supervision objective:** **REVISED into a loss-family search** — see §4.7. The original spec (per-position / endpoint / multi-endpoint, all within inner-product-similarity) is replaced by three primaries (InfoNCE / Triplet / SGNS) plus a diagnostic-derived norm-brake regularizer.

**Group C — Joint training:** PENDING — depends on §4.7 outcome (under E.2, `λ_link` is dedup-moot; under E.1 with a new primary loss it becomes relevant again).
- `λ_link ∈ {0, 0.1, 0.3, 1.0}`

**Group D — Embedding regularization:** PARTIALLY SUBSUMED by §4.7's normbrake auxiliary. Weight decay variants stay relevant as the §4.7 4-way ablation control to distinguish normbrake's novelty from reparameterised WD.
- Embedding-table dropout (0, 0.1, 0.3)
- Weight decay on `E_target, E_context` (0, 1e-5, 1e-4)
- Stop-gradient on alignment loss past epoch N

**Group E — Link MLP head structure:** **WIKI COMPLETE under A2-off.** E.2 (Component-0-only, 99-dim) ties E.1 (8-block cross-table, 1123-dim) at 0.7079 ± 0.0005. Locked on wiki under A2-off. **§4.7 loss search rolls back to E.1** because under E.2 the embeddings are unread at scoring, which dedup-collapses any loss-family search.

### 4.2 What Phase S is NOT allowed to vary (locked)

- Component 0 stays (time encoding + cold-start bits at the link MLP input)
- Strict-causal protocol stays
- TGB Evaluator for all reported numbers
- No edge features at the scoring head
- No walk encoder (deferred)
- No multi-view (deferred to P2)
- Dual identity tables `E_target`, `E_context` (cross-table use is in Group E)
- Xavier-uniform init, no feature-based init

### 4.3 Search budget

12 hours total wall-clock, ~40 runs at ~15 min each (Tempest CPU + RTX 2000 Ada). Suggested allocation:

- **3 runs:** A2 first (alignment on/off comparison, single seed each). Cheapest, most informative.
- **6 runs:** A1 within-family weighting on the A2 winner side
- **6 runs:** A3 supervision objective (only if A2 says "alignment on")
- **3 runs:** Group E (head structure) on top of A1 winner
- **4 runs:** Group C (joint training λ_link) on top of A1 + E winner
- **6 runs:** Group D (regularization) on top of A1 + E + C winner
- **4 runs:** best-of-each combinations across groups
- **8 runs:** multi-seed validation of the top 3 configurations (3 seeds × 3 configs minus 1 already-run baseline)

If something surprising happens that wasn't on the search grid, follow the surprise (see §4.6 guidance).

### 4.4 Phase S success criterion

The search succeeds if it produces a single configuration with:

**Test MRR ≥ Phase 0.5 baseline mean** (from anchor validation §3.3) **reproducibly across 3 seeds within ±0.02**, AND a clear val-MRR peak with early-stopping patience=5 (the protocol terminates naturally, not at max epochs).

Note the floor is the *verified* Phase 0.5 number from anchor validation, not 0.71 from the smoke and not 0.65. If anchor validation pins the floor at 0.69, that's the floor.

If no Phase S configuration meets this bar:

- **If no-alignment (A2-off) is the highest-scoring**: lock in "Component 0 + no alignment + Group E head choice." That's the honest outcome — walks-supervision is harmful on wiki.
- **If alignment configs tie with no-alignment**: lock in the no-alignment config (simpler model wins ties). Mark walks-supervision as "neutral on wiki, retest on other datasets" for the paper.
- **If everything regresses below Phase 0.5**: stop and investigate. Likely a bug in the new infrastructure, not a thesis failure.

### 4.5 Phase S deliverables

At the end of Phase S, the agent reports:

1. **Anchor validation results** (3 seeds, mean ± std)
2. **Best configuration**: which choice from each of A1, A2, A3, C, D, E
3. **Comparison matrix**: all explored points sorted by test MRR. Annotated with which Group(s) each varies. Include the A2 off configuration explicitly.
4. **Interpretive summary**: does walks-supervision help, hurt, or break even on wiki? Under what conditions? What does Group E's outcome say about whether the cross-table embeddings carry signal at all?
5. **Recommended base for P1–P3**: the locked configuration. P1+ phases run on top of this.
6. **Per-epoch divergence shape** for top 3 configurations (val MRR + test MRR per epoch). We want to see whether the alignment-on configs diverge after some epoch or stabilize.

### 4.6 Phase S guidance for Claude Code

The agent has latitude in execution. Guidance for that latitude:

- **Start with A2 (alignment on/off).** This is the cheapest, most informative experiment. If A2-off wins, the rest of the search is a tuning exercise on regularization and head structure, not on loss form. Single seed for the initial A2 comparison is fine; multi-seed validation comes later for the top-N winners.
- **Use early stopping everywhere.** Every run reports the best-val-MRR checkpoint, not the final-epoch checkpoint.
- **Log per-epoch test MRR even though you select on val.** We want to see the divergence shape — does the alignment loss pull test MRR down monotonically, or does it overfit at some point and then stabilize?
- **Report negative results.** A run that does poorly is informative; don't filter the comparison matrix.
- **Don't try to ship a 0.82+ number from Phase S.** The goal is a stable base, not a leaderboard chase. Save tuning for P1–P3.
- **Two-seed minimum for any decision.** Wiki has ±0.02 single-seed noise.
- **If something surprising happens, follow the surprise.** Examples: one loss variant produces stable training while others diverge — investigate why; joint training helps massively when alignment is absent but not when alignment is present — that's a finding about coupling; Group E "no cross-table" wins — that's a paper-defining finding worth multi-seed validation immediately.
- **Deduplicate by effective compute graph, not nominal hyperparameters.** When Phase S configurations would be mathematically equivalent (e.g., Option E.2 means embeddings are never read at scoring, so A2-on/off and `λ_link` collapse to the same gradient flow), the agent should NOT run duplicate experiments. Specifically: under E.2 the embeddings receive no link-BCE gradient regardless of `λ_link`, and A2-on differs from A2-off only in whether the (unread) embeddings are also trained by the alignment loss. Recognise these collapses and skip the redundant cells in the comparison matrix.
- **Phase S budget is 12 hours.** Don't expand into a multi-day project. If you run out of budget, report what you have.

---

## 4.7 Group A3 (REVISED): loss-family search

The original Group A3 (per-position / endpoint / multi-endpoint contrastive) was three variants within the *same* inner-product-similarity family. The Phase 0.5 diagnostic (5× column-norm growth correlating with −0.28 test MRR decline) and the A2 result (alignment+uniformity is neutral on wiki) jointly suggest the *loss family itself* is the candidate to revise, not just its weighting or target position. §4.7 replaces the original A3 spec.

### 4.7.1 Candidates

Three primary loss families, plus one diagnostic-derived auxiliary regularizer. All four are specified in detail in the archived `loss_function_search_ammendum.md`; §4.7 keeps the load-bearing summary here.

**A3.1 — Multi-positive InfoNCE with positional weighting.**
For each walk, anchor = `target(seed)`, positives = `context(walk_position_i)` weighted by `w(i) = 1/K · (1+Δt/τ_pos)^(-β)`, negatives = in-batch other anchors' contexts + uniform random destinations. Temperature `τ_contrastive = 0.1`. **`η_uniform = 0` (mandatory).** Sample negatives from unigram^0.75 over training destinations to mitigate heavy-tail false-positive collisions on wiki. Direct empirical precedent: NeurTWs (Jin et al., NeurIPS 2022) Table 4 Ablation 5 shows BCE deteriorates vs multi-negative contrastive on every continuous-time-graph dataset they tested.

**A3.2 — Triplet/margin loss with semi-hard mining.**
For each walk, sample one positive `p ~ Cat({n_0..n_{L-2}}, weights={w(i)})`, one uniform-random negative `q`. **Cosine similarity** (not raw dot product), margin `m = 0.5`. Semi-hard mining mask: only triplets with `pos_sim − neg_sim < m AND neg_sim < pos_sim` contribute gradient. **`η_uniform = 0`; add `weight_decay_emb = 1e-4`** to supply the norm control uniformity used to provide. One triplet per walk (over-sampling causes spurious gradient correlation). NEVER enable hard mining — bipartite graphs collapse under it.

**A3.3 — SGNS (Skip-gram with negative sampling).**
For each (anchor `u`, walk position `v` with weight `w(i)`): `pos = σ(target(u)·context(v))`, sample `k=5` negatives `v_- ~ unigram^0.75` over training destinations, `neg = σ(−target(u)·context(v_-))`. Loss = BCE-with-logits. Mikolov-style subsampling `t=1e-5` discards frequent positives. Linear lr decay `0.025 → 1e-3` over 5 epochs (Mikolov schedule); without it expect early-epoch divergence on negative gradients. **`η_uniform = 0`.** The only candidate with a principled stopping criterion: Levy & Goldberg (NIPS 2014) factorization → stop when Frobenius distance between `target·contextᵀ` and shifted PMI plateaus.

**A3.x_normbrake — Custom diagnostic-derived norm-brake regularizer (auxiliary; composes with any primary).**
Per-column L2 hinge: `excess = max(0, ||E[:,j]|| − threshold)`, loss = mean(excess²) over `E_target` and `E_context`. Threshold = `1.5 × anchor_col_norm` (calibrated once from the anchor's best-val checkpoint; from the diagnostic's 2-ep mean ≈ 0.36, threshold ≈ 0.54). `λ_normbrake = 0.1`. Derived directly from the diagnostic finding that the cliff is caused by unbounded column-norm growth — penalises growth past the empirically observed "safe" regime. **No published precedent in this exact form** (matrix-factorisation column-norm regularisation is well-known, but the empirical calibration to the cliff threshold is custom).

### 4.7.2 Why each candidate, ranked by cliff-elimination strength

| Criterion | InfoNCE (A3.1) | **Triplet (A3.2)** | SGNS (A3.3) |
|---|---|---|---|
| Cliff elimination mechanism | none (NPC coupling keeps pulling) | **literal ∇=0 once margin clears** | sigmoid saturation (soft) |
| Theoretical guarantee on stopping | none | **strongest (mathematical)** | shifted-PMI plateau (soft) |
| Cosine vs raw-dot scale stability | raw-dot scale issue | **cosine bounded [-1,1]** | sigmoid bounded |
| Predicted wiki peak (amendment §4) | 0.700 ± 0.020 | 0.690 ± 0.020 | 0.695 ± 0.020 |
| Predicted wiki cliff (50-ep − peak) | −0.05 to −0.10 | **±0.01 (largely eliminated)** | −0.05 |
| Predicted cross-dataset uplift | +0.03–0.08 | +0.02–0.05 | **+0.04–0.07 (best transfer rep)** |
| Implementation cost | ~30 LOC | **~20 LOC (lowest)** | ~50 LOC (unigram cache + lr sched) |
| Failure mode | false-neg in-batch on heavy-tail | margin mis-tune | wrong neg dist, anisotropy |

**The cliff is the load-bearing diagnostic.** Triplet (A3.2) has the strongest theoretical fit — its gradient structurally bounds at convergence, which is exactly what the diagnostic's 5×-column-norm-growth failure mode demands.

### 4.7.3 Pre-registered predictions (commit before running)

| Cell | Config | Predicted test MRR | Predicted cliff (50 ep − peak) | Confidence |
|---|---|---|---|---|
| 1 | InfoNCE alone | 0.70 ± 0.02 | −0.05 | medium |
| 2 | **Triplet alone** | **0.69 ± 0.02** | **±0.01** | **high (theory)** |
| 3 | SGNS alone | 0.695 ± 0.02 | −0.05 | medium |
| 4 | InfoNCE + normbrake | 0.70 ± 0.02 | ±0.02 | medium |
| 5 | Triplet + normbrake | 0.69 ± 0.02 | ±0.01 (normbrake dormant) | high |
| 6 | SGNS + normbrake | 0.695 ± 0.02 | ±0.02 | medium |

**Prior winner (pending data): Cell 5 (Triplet + normbrake), or Cell 2 (Triplet alone) if normbrake is dormant.** Updated to Cell 1/3/4/6 only if data show > 0.005 advantage over Cell 2 on multi-seed reproduction.

### 4.7.4 Execution plan (six single-seed cells, then multi-seed top-2)

Six cells per amendment §5.1, executed under E.1 head (NOT E.2 — under E.2 the dedup clause collapses all six to one):

```
Cell 1: --primary-loss infonce  --lambda-normbrake 0
Cell 2: --primary-loss triplet  --lambda-normbrake 0
Cell 3: --primary-loss sgns     --lambda-normbrake 0
Cell 4: --primary-loss infonce  --lambda-normbrake 0.1
Cell 5: --primary-loss triplet  --lambda-normbrake 0.1
Cell 6: --primary-loss sgns     --lambda-normbrake 0.1
```

All cells: seed 42, `--num-epochs 50 --early-stop-patience 5`, `--head-mode cross_table` (E.1).

**Normbrake threshold calibration (BEFORE Cells 4–6):** load the seed-42 anchor's best-val checkpoint, compute joint mean per-column L2 norm of `E_target` ∪ `E_context`, lock `normbrake_threshold = 1.5 × that_mean`. ~30 seconds, not a training cell.

**Wall budget:** 6 × ~9 min single-seed = ~1 hour. Plus ~30–40 min for multi-seed top-2 on seeds {7, 13}. Within the v2.2 §4.3 12-hour Phase S budget by a wide margin.

### 4.7.5 Per-cell logging

For each cell:
1. Best epoch (per patience=5 early stopping)
2. Best val MRR + best test MRR (pinned to the same epoch)
3. **Per-epoch test MRR trajectory** (the cliff diagnostic)
4. **Per-epoch column-norm trajectory** of `E_target ∪ E_context` (cliff cause signal)
5. **Per-epoch `L_normbrake`** (for Cells 4–6 only — should be near zero for Cell 5 if triplet self-limits as designed; non-zero for Cells 4/6 if cliff occurred)
6. Wall time / epoch
7. Hyperparameter values used

### 4.7.6 Decision rules

After Cells 1–3 (primary alone):
- **A — clear winner:** any of A3.1/A3.2/A3.3 beats A2-off (0.7092) by > 0.0016 reproducibly across seeds → that's the primary. Multi-seed validate, lock.
- **B — all tie within noise (EXPECTED on wiki):** loss form isn't the wiki peak bottleneck. Pick the cell with cleanest cliff shape (smallest decline from peak over 4 epochs post-stop). Per §2 goal-shaping, this is the *expected* outcome and is fine.
- **C — a cell shows no peak by epoch 10:** that loss is genuinely fixing the cliff. Run it longer. Multi-seed validate immediately — most important deliverable.
- **D — all three regress below A2-off (0.7092):** loss-family change is harmful on wiki under E.1. Two possibilities: implementation bug (verify) OR alignment+uniformity was actually right for this task (a paper finding in its own right). Proceed to Cells 4–6 to test whether normbrake rescues anything.

After Cells 4–6 (primary + normbrake):
- **E — normbrake fixes cliff (column norms stay < `1.5 × threshold` across 50 ep AND test MRR holds within 0.02 of peak):** paper-novel finding. Multi-seed validate.
- **F — normbrake dormant (`L_normbrake ≈ 0` throughout):** primary already self-limits (likely Cell 5 with triplet). Drop normbrake from production. No paper finding, no harm done.
- **G — normbrake always-active (`L_normbrake > 0.5` throughout):** threshold too tight. Re-calibrate from a post-Cell-1 checkpoint instead of the anchor, OR raise `1.5×` multiplier to `2.5×`.
- **H — normbrake hurts (test MRR regress > 0.005):** drop. Report as informative.
- **I — normbrake duplicates weight decay:** run the 4-way ablation (no-aux / WD-only / normbrake-only / both) on the winning primary BEFORE any v2.3+ lock. **This is paper-integrity-critical** — without it, the normbrake contribution claim is unsupported.

### 4.7.7 Stop conditions (don't burn budget)

- Stop if any cell gives test MRR ≥ 0.7090 with clean cliff (≤ 0.02 decline from peak over 50 ep). Strong evidence; multi-seed validate immediately.
- Stop if all 6 cells regress below A2-off (0.7092) by > 0.005. Lock A2-off + E.2 (current Phase S winner) and move to cross-dataset validation.
- DO NOT stop just because peaks tie within noise. The cliff shape is the more important signal per §2.

### 4.7.8 What's NOT in §4.7 (explicitly removed)

- **EdgeBank-as-teacher distillation.** Considered in the amendment v1.1, removed in v1.2: distilling a known heuristic into the embeddings dilutes the contribution claim from "we found a useful walks-supervised loss family" to "we trained embeddings to encode EdgeBank." Out of scope for this paper.
- **Decoupled Contrastive Loss (DCL):** marginal expected gain over A3.1; possible cliff regression.
- **Barlow-Twins / VICReg:** inductive-bias mismatch (augmented-views vs role asymmetry of target/context).
- **Hawkes-process log-likelihood:** implementation cost too high for Phase S; pilot in a separate workstream.
- **Supervised contrastive, BYOL/SimSiam, mutual-information bounds, masked walk modeling, BPR:** ruled out (no class labels / inductive mismatch / dominated by NeurTWs / misaligned with task).

The archival `loss_function_search_ammendum.md` (in repo root) preserves the full reasoning behind each inclusion and exclusion.

---

## 5. Refinement phases (after Phase S)

These run on top of the Phase S winner ("the base"). Each is one overnight session.

### P1: `max_time_capacity` window sweep (1 hr)

Configure Tempest's recency-view sampling window.

`max_time_capacity ∈ {6h, 1d, 3d, 7d, ∞}`. Three seeds for the chosen window.

Decision criterion: if any window beats `∞` by ≥0.02 test MRR (mean across seeds), lock in.

**Skip P1 if Phase S locked in A2-off (no alignment).** Without walks supervision, `max_time_capacity` is irrelevant — walks are never used. Go straight to P2 pre-flight or skip to P3.

### P2: Two-view multi-bias (3 hrs)

**Skip P2 if Phase S locked in A2-off.**

**Pre-flight (mandatory, 10 min): walk-distribution divergence test, stratified by node degree.**

If aggregate JS divergence is < 0.05 across all degree buckets → skip P2 entirely. The structural view collapses to the recency view on wiki.

If divergence is meaningful in mid_80pct or low_decile bucket → add structural view with `TemporalNode2Vec(p=1.0, q=0.25)`, `E_context_S` table, view-specific projections per Component 1.5.

Link MLP gets 2 additional structural-view interaction blocks (Hadamard interactions only, per v1.5 §4 Component 3). If Phase S Group E selected "no cross-table," P2 reverts to "no cross-table + Φ(Δt) + bits" and the multi-view comparison is moot — skip P2.

Decision criterion: +0.02 over P1 base = proceed.

### P3: Ablation matrix + error analysis (2 hrs)

**Within-method ablations (4–6 training runs depending on which earlier phases applied):**

1. Phase 0 reference (the original 0.33 number — current baseline)
2. Phase 0.5 (Component 0, current alignment, 50 ep — the over-trained version, 0.39)
3. Anchor-validated Phase 0.5 (Component 0, early-stopped, ~0.71)
4. Phase S winner (locked base)
5. P1 winner (if applicable; otherwise skip this row)
6. P2 winner (if applicable; otherwise skip this row)
7. **Capacity-matched control** (if P2 applies): single-view at `d_emb = 192` vs P2's two-view at `d_emb = 128`. Per v1.5 §6 row 7b: `3·128 = 2·192 = 384·N` table parameters strictly matched.

**Error analysis (post-hoc, no extra training):** split the best model's eval predictions by:
- (u, v) recency bucket: never seen, < 1h, < 1d, < 1w, older
- Source node degree: low / mid / high terciles
- Whether v is a hub (top-K most popular destinations)

Identifies where the multi-view (or windowed-walk) signal pays off, if at all.

### P4: Honest-protocol re-baseline (its own phase, ~1 week) — **DEFERRED**

> **Status (2026-05-19): DEFERRED.** Right now the optimisation track
> (Phase S → P1 → P2 → P3) is what pushes OUR number up. P4 is a
> paper-narrative move; it does not improve our test MRR, only the
> framing in which we publish it. Resume after P3 lands or whenever
> the user explicitly un-defers. The rest of this section is preserved
> as the scope-of-work for that future resumption.

This is the paper-defining experiment from §8, broken out per v2.1 pushback 5.

**Why this needs its own phase, not a P3 row:** realistically, re-running TPNet under strict-causal requires:
- Reading their code (lxd99/TGB_TPNet)
- Identifying state-update points (their leak shape candidates differ from ours)
- Modifying the training loop without breaking numerics
- Validating "as-published" reproduction matches the leaderboard within ±0.01 BEFORE applying the protocol fix
- Then applying the protocol fix and measuring
- Repeating for DyGFormer (yule-BUAA/DyGLib_TGB) and TGN (same repo)

That's 1–2 days per method minimum. Budget ~1 week total for the three.

**P4 deliverable:** a 3-method table showing as-published vs strict-causal test MRR for TPNet, DyGFormer, TGN. Plus a same-protocol comparison row for our method.

**P4 is not gated on Phase S, P1, P2, or P3.** Could run in parallel with refinement once the implementer is set up. But it has its own resource cost and shouldn't be assumed to fit in a Phase S window.

If P4 numbers don't drop meaningfully (i.e., the published methods are already strict-causal): the paper's narrative shifts. The honest-protocol re-baseline is no longer a contribution; the paper becomes about walks-supervision-is-dataset-dependent (the Phase S finding) and dataset-specific tuning. Still a paper, but a different one. P4 outcomes affect §8 narrative.

---

## 6. Architecture (the parts that don't move regardless of Phase S outcome)

### 6.1 Component 0: Time encoding at the link MLP (LOCKED)

For each scored pair `(u, v, t)`:

```
Δt_u  = t - last_event_time[u]
Δt_v  = t - last_event_time[v]
Δt_uv = t - last_edge_time[u, v]

Φ(Δt) = [cos(ω_1·Δt), sin(ω_1·Δt), ..., cos(ω_k·Δt), sin(ω_k·Δt)] ∈ ℝ^{2k}
```

`ω_i` learnable, `k = 16` (so `d_time = 32`).

Cold-start: `last_event_time[*]` init to 0; `last_edge_time[u,v]` returns 0 for unseen pairs. Three binary flags `is_cold_start_u, is_cold_start_v, is_cold_start_uv`. Δt clamped to `time_scale × 100` before passing through Φ.

State maintenance in post-scoring block, strict-causal. `last_edge_time` in sparse dict. Symmetry across positives and negatives confirmed.

### 6.2 Embedding store: dual tables (LOCKED at table level; usage in link MLP is Group E)

- `E_target[u] ∈ ℝ^{N × d}` (seed-side of alignment)
- `E_context[u] ∈ ℝ^{N × d}` (walk-internal positions or endpoint depending on Phase S A3 choice)

Xavier-uniform init. No feature-based init.

How these tables are *used* in the link MLP (Group E) is part of Phase S. The tables themselves stay.

### 6.3 Node feature integration (preserved from v1.5)

When node features present, three projections (proj_t, proj_c) + three fusion layers (target_final, context_final). No-op when `d_n = 0`. tgbl-wiki sits in bottom-left of regime matrix (ef present, nf absent → plain identity lookups).

### 6.4 Link MLP head (Phase S Group E result + §4.7 roll-back to E.1)

**Wiki Phase S E result (post v2.2):** E.2 (Component-0-only, 99-dim) ties E.1 (cross-table, 1123-dim) at 0.7079 ± 0.0005 vs 0.7089 ± 0.0012 under A2-off. Simpler-wins-ties → **E.2 locks on wiki under A2-off**.

**§4.7 roll-back:** the loss-family search uses **E.1** as the default head, because under E.2 the embeddings are unread at scoring and any loss-family change becomes dedup-moot (§4.6). If §4.7 lifts the production number under E.1, the final architecture is E.1 + the §4.7 winner. If §4.7 doesn't lift, fall back to the simpler E.2 + A2-off.

Possible heads:

**Option E.1 — 8-block cross-table (current):**
```
phi(u, v, t) = [
  target(u), context(v), target(u) ⊙ context(v), |target(u) - context(v)|,
  target(v), context(u), target(v) ⊙ context(u), |target(v) - context(u)|,
  Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv),
  is_cold_start_u, is_cold_start_v, is_cold_start_uv,
]
```

**Option E.2 — Component-0-only:**
```
phi(u, v, t) = [
  Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv),
  is_cold_start_u, is_cold_start_v, is_cold_start_uv,
]
```

**Option E.3 — 8-block cross-table with cross-table-output dropout (0.1, 0.3):** same as E.1 with dropout applied to the 8d cross-table portion of the input before LayerNorm.

In all cases: LayerNorm → 2-layer GELU MLP → 1 logit. BCE-with-logits.

If E.2 wins: the design's narrative simplifies dramatically. The embeddings become inert to the scoring path (still trained by alignment loss for representation quality, but the link MLP doesn't read them). That's a major paper finding.

### 6.5 What Phase S can change

- Alignment loss form (Group A1) — including dropping it entirely (Group A2)
- Supervision target (Group A3) — endpoint vs walk-position
- Optimizer joint vs separate (Group C)
- Embedding regularization (Group D)
- Link MLP head structure (Group E)

Everything else in §6.1–6.3 is locked.

---

## 7. Strict-causal protocol (LOCKED)

```
1. seeds ← unique(batch.src ∪ batch.tgt)
2. walks ← walk_gen.walks_for_nodes(seeds)             # PRE-ingest
3. L_emb ← (alignment + uniformity, whatever Phase S settles on)
   embedding_optimizer.step()
4. negs ← neg_sampler.sample(batch)                     # PRE-batch reservoir
5. Compute Δt features + cold-start flags (state ≤ B-1)
   L_link.backward(); link_optimizer.step()
6. POST-SCORING:
   neg_sampler.observe(batch.src, batch.tgt)
   walk_gen.add_edges(...)
   update last_event_time, last_edge_time
```

Per-epoch: `walk_gen.reset()` once at start of each training epoch. At eval, Tempest state carries through.

---

## 8. Paper narrative (what we're going for)

**The strongest version of the contribution:**

> "We identify two non-obvious properties of temporal link prediction on TGB benchmarks: (1) within-batch state-update leak shapes inflate leaderboard MRRs in TGN-family methods (validated by P4 honest-protocol re-baselines); (2) on memorization-saturated datasets like tgbl-wiki, walks-supervised embeddings can hurt eval performance by pulling node representations toward walk-co-occurrence geometry, which is anti-correlated with the eval task's reward function (validated by Phase S A2). We propose a walk-distribution-matching framework that (a) uses time-encoded recurrence signals at the scoring head as the primary predictor, (b) tunes walks-supervision empirically per dataset rather than committing to fixed alignment objectives, and (c) provides honest-protocol re-baselines for TPNet, DyGFormer, TGN."

This survives review even if the absolute MRR doesn't beat 0.82.

**Specific claims for the ablation matrix:**
1. Time encoding alone closes most of the gap to leaderboard methods — Phase 0 (0.33) vs anchor-validated Phase 0.5 (~0.71)
2. Walks-supervision contribution is dataset-dependent — Phase S A2 result (alignment on vs off)
3. Cross-table blocks are useful / inert / harmful depending on supervision regime — Phase S Group E result
4. (If P2 applies) Time-windowed walk sampling improves recency-view supervision — P1 vs Phase S base
5. (If P2 applies) Multi-view supervision helps in the long-tail bucket — error analysis
6. Honest-protocol re-baselines: TPNet/DyGFormer/TGN under strict-causal vs as-published — P4

**The paper-defining experiment is P4.** If those methods drop substantially under honest protocol and our method holds, the contribution is sharp regardless of absolute MRR.

---

## 9. Expected outcomes (revised, anchored)

| Stage | Expected Test MRR | What it means |
|---|---|---|
| Phase 0 (the 0.33 baseline) | 0.33 | The 50-ep walks-supervised reference |
| Anchor validation (Component 0 + alignment, 2 ep) | TBD via §3 | Single-seed smoke was 0.71; pending 3-seed verification |
| Phase S winner | ≥ anchor-validated number | Stable across 3 seeds |
| P1 (+ windowed walks, if applicable) | +0.02–0.05 over Phase S base | If recency tuning helps |
| P2 (+ multi-view, if applicable) | +0.03–0.07 over P1 | If divergence pre-flight passes |
| P3 final | Same as P2 (no new training) | Just the ablation matrix |
| P4 (honest-protocol re-baseline) | Other methods drop 0.10–0.30 | If the leak shape is real and our method holds |

The leaderboard target is "competitive under honest protocol." Absolute MRR ranges depend on what anchor validation pins down. The paper's contribution does not depend on the leaderboard ranking.

---

## 10. Implementation watch-list

- **Early stopping protocol.** Patience=5 on val MRR. Report best-val-MRR checkpoint, not final-epoch. Implement once, use everywhere.
- **Cold-start bits.** Resolved per Phase 0.5 diagnostic — no LayerNorm-wash mitigation needed.
- **`last_edge_time` storage.** Sparse dict on wiki; hash table for tgbl-coin / tgbl-flight.
- **Per-epoch test MRR logging.** Always on. We want divergence shape.
- **Three-seed minimum for any locked decision.** Two-seed for intermediate filtering within Phase S, three for Phase S → P1+ handoffs.
- **Phase S budget enforcement.** 12-hour cap, ~40 runs. Don't let the search expand into a multi-day project.
- **P4 budget.** ~1 week, separate from Phase S/P1/P2/P3. Treat as parallel-track work.
- **Anchor validation is gating.** Phase S does not start until §3 produces a verified number.

---

## 11. What's deliberately NOT in v2.3 (preserved across v2.1 → v2.3)

- Walk encoder (deferred; the diagnostic suggests adding sequence models on top of an unstable base is the wrong direction)
- TGN-style memory (roadmap; honest raw-message-store version only)
- Edge features at the scoring head (literature audit confirmed leak shape)
- Hand-rolled MRR (TGB Evaluator only)
- Three-view or higher
- Feature-based init for E_target

---

## 12. Hyperparameters (defaults; Phase S can override)

| Parameter | Default | Notes |
|---|---|---|
| `d_emb` | 128 | overnight confirmed 192 overfits under prior architecture |
| `d_n` | dataset-specific | 0 disables node-feature path |
| `d_hidden_link` | 128 | link MLP hidden dim |
| `d_time` | 32 | k=16 frequencies |
| `max_walk_len` | 20 | revisit in P1 |
| `num_walks_per_node` | 5 | per view |
| `target_batch_size` | 200 | B=1000 regresses |
| `num_epochs` | up to 50, early-stopped | patience=5 on val MRR |
| `early_stop_patience` | 5 | epochs with no val improvement |
| `K_neg_walk` | 5 | negatives per walk endpoint (if endpoint contrastive) |
| `max_time_capacity` | TBD P1 | ∞ default |
| `λ_link` | TBD Phase S Group C | joint training weight |
| `λ_align` | TBD Phase S Group A2 | overall alignment weight (could be 0) |
| `dropout_emb` | TBD Phase S Group D | embedding-table dropout |
| `weight_decay_emb` | TBD Phase S Group D | weight decay on E_* |
| `link_head` | TBD Phase S Group E | E.1 / E.2 / E.3 |
| `η_uniform` | 1.0 | unchanged |
| `γ_uniform` | 2.0 | unchanged |
| `num_neg_per_pos` | 10 | training K for link BCE |
| `hist_neg_ratio` | 0.5 | matches TGB eval mix |
| `reservoir_size` | 32 | per-source Vitter R reservoir |
| `cold_start_dt_clamp` | `time_scale × 100` | for Φ input |
| `cold_start_sentinel_init` | 0 | for last_*_time |
| `emb_lr` | 1e-3 | Adam |
| `link_lr` | 1e-3 | Adam |
| `seeds` | {42, 7, 13} | three seeds for Phase S handoff decisions |

---

## 13. Leaderboard reference

| Method | tgbl-wiki-v2 Test MRR |
|---|---|
| Random | 0.0075 |
| EdgeBank-inf | 0.495 |
| EdgeBank-tw | 0.571 |
| GraphMixer | 0.594 |
| DyRep | 0.665 |
| TGN | 0.690 |
| CAWN | 0.711 |
| TNCN | 0.718 |
| DyGMamba | 0.739 |
| DyGFormer | 0.798 |
| HyperEvent | 0.810 |
| Heuristic(LocalGlobal) | 0.821 |
| **TPNet (#1)** | **0.827** |
| **This design (target)** | competitive under honest protocol |

Per May 2026 audit, top performers appear leak-free under within-batch state-update analysis. P4 will verify by re-running TPNet/DyGFormer/TGN under strict-causal protocol.

---

## 14. Process for v2.3 → v2.4+

v2.3 records the wiki Phase S A2 + E results and integrates the loss-family amendment as §4.7. The standalone `loss_function_search_ammendum.md` is archival as of v2.3 — its load-bearing content lives in §4.7; the full reasoning trail stays on disk for future reference.

**Next steps:**
- **v2.4 (post-§4.7):** lock the §4.7 winner (primary + optional normbrake). If §4.7 lifts wiki past 0.7089 reproducibly across seeds, production architecture is E.1 + §4.7 winner; otherwise fall back to E.2 + A2-off (the current Phase S wiki winner at 0.7079 ± 0.0005).
- **Apply v2.4-locked architecture to tgbl-review-v2** on the laptop. Verifies cross-dataset transfer before committing.
- **Ship to A40 server for tgbl-coin / -flight / -comment / -review-full** (these need >8 GB VRAM, not laptop-feasible).
- **v2.5 (post-P1/P2/P3):** within-method ablation matrix with the locked architecture across all five datasets.
- **v2.6 (post-P4):** honest-protocol re-baseline. Currently deferred (paper-narrative work, not number-improvement work).

---

*Document version: v2.3, May 2026. Changes from v2.2: (1) §0 integrates the wiki Phase S A2 and E results (A2-off wins by +0.0019; E.2 ties E.1 under A2-off and locks via simpler-wins); (2) §2 promotes the over-training cliff (5× column-norm growth → 0.28 test MRR drop over 50 epochs) to a load-bearing locked finding; (3) §4.1 adds status flags per Phase S group; (4) §4.7 REVISES Group A3 as a loss-family search — three primaries (InfoNCE / Triplet / SGNS) plus a diagnostic-derived norm-brake regularizer, integrating the previously-standalone `loss_function_search_ammendum.md`; (5) §4.7 includes pre-registered predictions (Cell 5 Triplet+normbrake = prior winner) and a comparative ranking; (6) §6.4 records the E.2 lock on wiki + the §4.7 roll-back to E.1 for the loss search; (7) §14 sketches the v2.3 → v2.4 → v2.5 → v2.6 evolution. Preserved from v2.2: §4.6 deduplicate-by-effective-compute-graph clause (applies to §4.7 as well — under E.2 all primaries dedup-collapse, which is why §4.7 uses E.1). Preserved from v2.1: Group E, anchor validation §3, Group A split (now A1/A2 only — A3 is revised), §4.4 anchor-validated floor, P4 broken out (currently deferred). Companion to Tempest paper (Salehin et al., arXiv:2605.16182).*
