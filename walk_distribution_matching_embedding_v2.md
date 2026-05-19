# Walk-Distribution-Matched Temporal Embeddings

**Design & Execution Plan (v2.2 — anchored Phase S frame)**

---

## 0. What changed from v2.0

v2.0 introduced the Phase S search frame as a replacement for v1.5's fixed eight-phase progression. The agent review of v2.0 surfaced five issues, all valid:

1. The link MLP head structure was over-locked. The diagnostic showed cross-table column norms growing 5× while test MRR dropped 0.28 — the head structure should be part of the search, not assumed.
2. The 0.71 anchor was a single 2-epoch smoke, not a multi-seed result. The whole Phase S frame was anchored on an unvalidated number.
3. Group A conflated three different bets: within-family weighting (A1), alignment on/off (A2), and supervision objective (A3). Mixed comparisons produce uninterpretable matrices.
4. Success criterion of "≥ 0.65" was soft given the 0.71 anchor — the actual floor should be early-stopped Phase 0.5 baseline.
5. The honest-protocol re-baseline of TPNet/DyGFormer/TGN was buried as one row of P3, when realistically it's its own multi-day effort.

**v2.1 fixes all five.** Structure becomes:

**Anchor validation (30 min)** → **Phase S (12 hours)** → **P1 (window sweep)** → **P2 (multi-view, conditional)** → **P3 (within-method ablation matrix + error analysis)** → **P4 (honest-protocol re-baseline, ~1 week)**.

The thesis ("walk sampler's distribution is the supervision signal") survives. The paper-defining experiment (P4) is now correctly budgeted.

---

## 1. Thesis

> The walk sampler's distribution is the supervision signal. The alignment loss's job is to make the embedding inner product approximate `log P(walk ends at v | seed = u)` under that distribution. Time decay, distance decay, and recurrence emerge from this matching rather than being hand-coded — the sampler already encodes them.

A separate, **now confirmed** claim: the link prediction head needs explicit time-since-last-event signals at scoring time. Phase 0.5 diagnostic confirmed Component 0 (time encoding + cold-start bits) drives test MRR from ~random to provisionally 0.71 at 2 epochs with no walks supervision contributing usefully.

The open question, surfaced by the diagnostic: **does walks-supervision help, hurt, or have a sweet spot on this dataset?** Phase S finds out.

---

## 2. What we know going in (locked findings from Phase 0.5)

These are non-negotiable for v2.1 and downstream phases:

1. **Time encoding works substantially on wiki.** Component 0 + random-init cross-table → Test MRR provisionally 0.71 at 2 epochs (subject to anchor validation in §3). The `is_cold_start_uv` bit alone is doing most of the work (99.1% of test pairs are uv-cold-start; the bit carries the EdgeBank-recurrence signal natively).

2. **Cold-start bits do not get LayerNorm-washed.** Column-norm analysis showed the 3 bits get amplified to 1.78× the cross-table mean (50-epoch model). The §12.1 mitigation is unnecessary.

3. **Walks-as-currently-supervised hurt at 50 epochs.** Cross-table column norms grow 5× from 2-ep to 50-ep; test MRR drops 0.28. The embeddings are learning *something*, and that something correlates negatively with eval.

4. **The strict-causal protocol works.** No leak shape surfaced in the diagnostic.

5. **TGB serves ~999 random negatives per positive at eval.** For tgbl-wiki, ~99% of negatives are cold-start at the (u,v) level. The eval task is dominantly "is this pair recurring?"

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

### 4.1 What Phase S is allowed to vary

**Group A1 — Within-family alignment weighting:** when alignment loss is active, what weighting?
- Current: `1/K · (1 + Δt/τ)^(-β)`
- `1/K` only
- Uniform `α = 1` over walk positions

**Group A2 — Alignment on/off:** is walks-supervision useful at all on this dataset?
- Alignment on (the A1 winner)
- Alignment off (`λ_align = 0`; link MLP only, on random-init embeddings)

**Group A3 — Supervision objective (if A2 says "on"):** if alignment is useful, what target?
- Current per-position alignment (the A1 winner from above)
- Endpoint contrastive (SGNS on walk endpoints, the v1.5 Phase 2 idea)
- Multi-endpoint contrastive (sample K endpoint-like positions per walk)

**Group C — Joint training:** does link BCE backprop into embeddings help?
- `λ_link ∈ {0, 0.1, 0.3, 1.0}`

**Group D — Embedding regularization (NEW):**
- Embedding-table dropout (0, 0.1, 0.3)
- Weight decay on `E_target, E_context` (0, 1e-5, 1e-4)
- Stop-gradient on alignment loss past epoch N (early-stop just the embedding-side)

**Group E — Link MLP head structure (NEW per v2.0 → v2.1):**
- 8-block cross-table (current)
- **No cross-table blocks** (only Φ(Δt) + cold-start bits — the minimal Component-0-only head)
- 8-block cross-table with cross-table-output dropout

Group E exists because the diagnostic suggests the cross-table blocks may be net-negative when the embeddings have over-trained into the wrong geometry. The "no cross-table" variant is the cheapest representation of "trust Component 0 only." If it ties or beats the 8-block version, that's a major finding.

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

### P4: Honest-protocol re-baseline (its own phase, ~1 week)

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

### 6.4 Link MLP head (FLOATING — Phase S Group E decides)

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

## 11. What's deliberately NOT in v2.1

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

## 14. Process for v2.1 → v2.2+

After anchor validation completes: confirm or adjust the Phase 0.5 baseline number in v2.2. After Phase S completes: write v2.2 with the locked base configuration as a real committed plan (no more search). After P1/P2/P3 land: v2.3 has the ablation matrix and error analysis. After P4 lands: v2.4 has the honest-protocol comparison and final paper narrative.

v2.1 is a search frame with bounded latitude and a clearly-budgeted multi-phase plan. v2.2+ are committed plans built on Phase S's locked outputs.

---

*Document version: v2.2, May 2026. Fix from v2.1: §4.6 adds a "deduplicate by effective compute graph, not nominal hyperparameters" guideline so the agent skips redundant cells in the comparison matrix when search-space dimensions collapse (e.g., under E.2 head, embeddings are never read at scoring, so A2-on/off and `λ_link` differ only in whether the unread embeddings are alignment-trained — most combinations there are mathematically equivalent at the link MLP). Preserved from v2.1: (1) Group E added to Phase S — link MLP head structure searchable (E.1 cross-table / E.2 Component-0-only / E.3 cross-table+dropout); (2) anchor validation §3, 30-min pre-Phase-S step, 3 seeds, gates Phase S launch; (3) Group A split into A1 (within-family weighting), A2 (alignment on/off), A3 (supervision target); (4) §4.4 success criterion floor explicit at anchor-validated baseline; (5) honest-protocol re-baseline extracted from P3 into its own ~1-week P4. Companion to Tempest paper (Salehin et al., arXiv:2605.16182).*
