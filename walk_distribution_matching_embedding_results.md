# Walk-Distribution-Matched Temporal Embeddings — Running Results & Session Transcript

Companion to `walk_distribution_matching_embedding_v2.md` (v2.2, anchored
Phase S frame; superseded the v1.5 fixed 8-phase progression after the
Phase 0.5 diagnostic). Records what was implemented, what was run, what
the numbers were, and what decision was made at each phase.

**Branch:** `feature/walk-distribution-embedding` (off `master` @ `b246b87`).
Plan v2.2 committed at `7ceeebe`.

**Plan-document trail:**
- v1.5 (`b246b87`): fixed 8-phase plan, master.
- v2.0: introduced Phase S search frame as replacement for v1.5's fixed
  progression. Five issues flagged on review.
- v2.1: addressed all five (Group E added; anchor validation §3; A split
  into A1/A2/A3; floor = anchor-validated mean; P4 broken out).
- **v2.2 (`7ceeebe`, current):** adds §4.6 "deduplicate by effective
  compute graph" clause so redundant Phase S cells are skipped when
  search-space dimensions collapse (e.g., under Option E.2 head, the
  embeddings are not read at scoring, so most A2/`λ_link` combinations
  are mathematically equivalent at the link MLP).

**Starting state confirmed:**
- `master` is at the v3 baseline (no walk encoder, no DyG, no memory, no
  EdgeBank feature). Walk-encoder family lives on feature branches and is
  NOT pulled into this one.
- Phase 0 reference number (overnight measurement): test MRR 0.331,
  val 0.4015 (Phase 0 = cross-table 8-block link MLP + alignment +
  uniformity, B=200, K=5, L=20, d=128, undirected).

---

## Phase 0.5 — Time encoding ablation

**Status:** COMPLETE. Decision: "composes additively" — keep Component 0
(LOCKED in v2.2 §6.1). Next gate is anchor validation per v2.2 §3.

> Note: the original v1.5 successor was "Phase 1" (alignment-weighting
> ablation). v2.2 supersedes that lattice — see §0 of v2.md and the
> anchor validation / Phase S sections below.

**Implementation commit:** `199de30` on `feature/walk-distribution-embedding`.

**Implementation summary:**
- New module `tempest_walks/timestate.py` — NodeTimeState class with
  per-node `last_event_time` (np.ndarray) + per-pair `last_edge_time`
  (sparse dict, symmetric key). Unit-tested with 7 cases (empty init,
  single event, max-reduce per node, pair-symmetric alias, duplicate-pair
  max-reduce, reset, empty update).
- `TimeEncoder` (model.py): k=16 learnable ω_i with dataset-aware
  geometric init from `time_scale`. Verified Φ(0) = [1,0,1,0,...].
- `LinkPredictor` extended: when `use_time_encoding=True`, input grows
  from `8·d` to `8·d + 3·d_time + 3` (3 cold-start bits as scalars).
- Trainer + Evaluator wiring: read state pre-scoring, write state post-
  scoring (LAST line of post-scoring block, after walk_gen.add_edges).
- Strict-causal audit (3×): writes in post-scoring only, reads pre-
  scoring only, reset per training epoch only. Documented in commit msg.
- Smoke tests: 20-batch end-to-end training + 1 val eval batch, no NaN,
  link BCE drops 0.70 → 0.47, time_state accumulates correctly.

**Configuration for the training run:**
- Phase 0 defaults: B=200, K=5, L=20, d=128, hist_neg_ratio=0.5
- Component 0 ENABLED: time_enc_k=16, cold_start_dt_clamp_factor=100.0
- Component 1.5 inactive on wiki (no node features)
- time_scale derived: 93132.6 (span/20 formula)
- 50 epochs

**Training-loss trajectory (key checkpoints):**

| epoch | align | uniform | link | epoch_time |
|---|---|---|---|---|
| 1   | 0.8905 | -3.8902 | 0.1916 | 19.8 s |
| 5   | 0.875… | -3.92  | ~0.14  | ~17 s |
| 10  | 0.875  | -3.92  | ~0.12  | ~17 s |
| 25  | 0.874  | -3.93  | ~0.11  | ~17 s |
| 50  | 0.8741 | -3.9341 | 0.0962 | 16.9 s |

Total wall clock: ~14 min training + ~3 min eval = ~17 min.

**Results:**
- **Val MRR: 0.4377** (vs Phase 0 baseline 0.4015 → **Δ = +0.0362**)
- **Test MRR: 0.3940** (vs Phase 0 baseline 0.3313 → **Δ = +0.0627**)
- Decision criterion bucket: **"+0.03 to +0.10" → composes additively**
- **Decision: KEEP Component 0; lock in for all downstream phases.**

**Notes for paper:**
- The test gain is roughly comparable to the +0.051 we saw overnight from
  the discrete `is_v_in_u_history` EdgeBank-style feature. Component 0's
  `Φ(Δt_uv)` plus `is_cold_start_uv` is the CONTINUOUS version of the
  same signal — at much lower implementation cost (no K_history buffer,
  no per-batch history reads of variable-sized neighbor sets).
- Link BCE at epoch 50: 0.096 (lower than Phase 0's 0.11) — the time
  features are giving the link MLP measurable signal it didn't have.
- Alignment loss is essentially unchanged from Phase 0 (~0.874) because
  Component 0 doesn't touch the alignment-side at all. This confirms
  Component 0 is correctly orthogonal to the alignment supervision.

---

### Phase 0.5 — Diagnostics (Diagnostic 1, 2, 3 from user's plan)

Diagnostic script: `scripts/phase0_5_diag.py`. Trains Phase 0.5 from scratch
then runs all three probes in-process. Two runs captured (2-epoch toy smoke
and full 50-epoch).

#### Diagnostic 1: cold-start prevalence

| split | is_cold_start_uv | is_cold_start_u | is_cold_start_v |
|---|---|---|---|
| val  | 99.2% | 5.8% | 3.0% |
| test | 99.1% | 5.9% | 0.5% |

Δt_uv channel is virtually always "cold-start" at val/test — because TGB
serves ~999 random destinations per positive, the (u, v) pair has never
been seen for ~99.9% of scored rows. **The `is_cold_start_uv` bit, not the
continuous `Φ(Δt_uv)`, is the dominant Component 0 signal.** This is
essentially the EdgeBank "is this pair recurring?" lookup made
differentiable as a binary feature.

#### Diagnostic 3: column-norm analysis (50-epoch model)

| position group | mean L2 column norm | ratio vs cross-table |
|---|---|---|
| cross-table (slots 0:8·d)            | 1.985 | 1.0× |
| time encoding (slots 8·d:8·d+3·d_t)  | 1.218 | 0.61× |
| cold-start bits (last 3 slots)       | 3.526 | **1.78×** |

The model AMPLIFIED the cold-start bit columns to 1.78× the cross-table
mean (and 2.89× the time-encoding mean). §12.1's LayerNorm-wash concern
is therefore RESOLVED for the 50-epoch model — proceed without the
parallel-MLP mitigation.

#### Diagnostic 2: zero-out ablation (50-epoch)

| split | full Phase 0.5 | Component 0 zeroed | drop |
|---|---|---|---|
| val  | 0.4654 | 0.0531 | **−0.412** |
| test | 0.4269 | 0.0280 | **−0.399** |

User's decision threshold: `>0.04 drop ⇒ proceed`. Actual drop is 10×
that — Component 0 is doing essentially all of the model's predictive
work. The walks-supervised cross-table embeddings, when isolated, score
roughly at random.

#### Surprise finding — 2-epoch beats 50-epoch by 0.28 test MRR

The diagnostic smoke (2-epoch training, otherwise identical) recorded:

| training length | Val MRR (full) | Test MRR (full) | Test (zeroed) |
|---|---|---|---|
| **2 epochs**  | **0.7451** | **0.7070** | 0.0170 |
| 50 epochs | 0.4654 | 0.4269 | 0.0280 |

A 2-epoch Phase 0.5 model reaches Test 0.71 on wiki — within range of
TGN (0.690) and CAWN (0.711). The SAME architecture trained for 50
epochs drops to 0.43.

Why: at init, the link MLP can't use the random cross-table embeddings,
so it learns the simple "trust `is_cold_start_uv`" rule → near-EdgeBank
accuracy after 2 epochs. Over training, alignment+uniformity pull the
cross-table embeddings toward walk-co-occurrence geometry; the link MLP
tries to USE that signal alongside cold-start; the result is a model
that's WORSE at TGB-style link prediction than the 2-epoch checkpoint.

Cross-table column norms grow from 0.36 (2-ep) to 1.99 (50-ep) — the
embeddings ARE learning structure. But that structure hurts prediction.

#### Decision following diagnostics

Per the user's stated decision rule (zero-out drop > 0.04 ⇒ proceed),
the gate is satisfied. But the 2-epoch finding is structurally
important: the v1.5 plan locks in Phase 0.5 at 0.39 when the
architecture's TRUE early-stopping ceiling is ~0.71. Downstream phases
should be measured against the early-stopping number, not the
over-trained one.

**This finding triggered the v1.5 → v2.0 → v2.1 → v2.2 plan rewrite.**
The original linear "Phase 1 / 1.5 / 2 / 3 / 4 / 5 / 6" lattice is
replaced by an anchored bounded-search frame (see plan v2.2). The
sections below follow that new structure; the v1.5 phase placeholders
are retired.

---

## Anchor validation (v2.2 §3)

**Status:** PENDING — gates Phase S launch.

**Configuration (verbatim from v2.2 §3):**
- Re-run Phase 0.5 config with early stopping (patience=5 on val MRR).
- 3 seeds: {42, 7, 13}.
- 2 epochs each — matches the single-seed checkpoint that hit Test 0.7070.
- Wall-clock budget: ~30 min total.

**Decision gate (verbatim from v2.2 §3.2):**
- **mean test MRR ≥ 0.70 with std ≤ 0.02** ⇒ anchor confirmed; Phase S
  anchors at the mean. Proceed.
- **0.65 ≤ mean < 0.70** ⇒ anchor partially validated. Phase S anchors
  at the verified mean (whatever it is), not the 0.71 from the smoke.
  v2.2 §4.4 success criterion adjusts accordingly.
- **mean < 0.65 or std > 0.04** ⇒ the 0.71 smoke was lucky. Stop.
  Investigate before Phase S. Likely causes to check: did the diagnostic's
  training loop use a different config than v2.2 expects? Was the 2-epoch
  run somehow different (different batch ordering, different walk-gen
  state)?

**Lock-in after anchor validation (v2.2 §3.3):** the verified mean test
MRR becomes the **Phase 0.5 baseline** for all downstream comparisons.
Every Phase S configuration is judged against this number, not against
0.71.

**Sanity checks before launch (all passed; see commit `3727f63`):**
- [x] Trainer respects `--num-epochs 2` and stops cleanly (`trainer.py:268`).
- [x] NodeTimeState resets between seeds (fresh Trainer per seed → fresh
  `NodeTimeState`; `.reset()` at epoch start).
- [x] TGB Evaluator state is independent per seed (load_val_ns/test_ns
  once before loop, idempotent; fresh `TGBNegativeSampler` per seed).
- [x] Logged outputs distinguish seeds (per-seed stdout tag +
  `runs/anchor_validation_<ts>.json`).
- [x] WalkGenerator + Tempest fresh per seed (`walks.py:48` constructs
  new `TemporalRandomWalk` per `Trainer.__init__`).
- [x] Per-seed RNG: `np.random.seed`, `torch.manual_seed`,
  `cuda.manual_seed_all` before `Trainer.__init__`;
  `HistoricalNegativeSampler` takes `seed=config.seed`.

**Results (run `20260519_173221`, ~4.2 min wall, RTX 2000 Ada + Tempest CPU):**

| Seed | Val MRR | Test MRR | Train s | Eval s |
|---|---|---|---|---|
| 42         | 0.7447  | 0.7088  | 37.2 | 50.0 |
| 7          | 0.7427  | 0.7060  | 33.6 | 49.4 |
| 13         | 0.7420  | 0.7062  | 33.4 | 49.8 |
| **mean ± std** | **0.7431 ± 0.0014** | **0.7070 ± 0.0016** | — | — |

**Decision: CONFIRMED** (v2.2 §3.2 gate). Test mean 0.7070 ≥ 0.70 ✓,
std 0.0016 ≤ 0.02 (massively under) ✓. The 0.71 smoke reproduces
tightly across {42, 7, 13}. Per-epoch align/uniform/link numbers are
identical within 0.001 across seeds, meaning Tempest's unsealed walk
RNG is producing effectively deterministic walks for this protocol
(input timestamps and seed sets fully determine the trajectory). The
~0.01–0.02 noise concern from the pre-launch audit was overcautious.

**Phase 0.5 baseline locked at test MRR = 0.7070** (mean across the
three anchor seeds). Phase S configurations are judged against this
floor; gains must be > 0.0016 (anchor std) to count as a real "win"
per v2.2 §4.4.

#### Init-divergence sanity check (post-anchor, before Phase S)

The bit-tight 0.001 cross-seed agreement on per-epoch loss values is
unusual enough to verify seed plumbing isn't broken before committing
12 hours to Phase S's multi-seed validation in §4.3 (which becomes
meaningless if seeds aren't actually independent).

Script: `scripts/init_divergence_check.py`. Dumps `E_target[0:3]`,
`E_context[0:3]`, `link_mlp.net[0].weight[0, 0:3]`, and the
negative-sampler's first 5 RNG draws right after `Trainer.__init__`,
before any forward pass, for each anchor seed.

| Channel | Identical across {42, 7, 13}? |
|---|---|
| E_target init                       | False ✓ |
| E_context init                      | False ✓ |
| link_mlp first-Linear weight        | False ✓ |
| neg_sampler.rng first-5 draws       | False ✓ |
| time_encoder.omegas                 | True (deterministic geometric schedule from `k=16`, no randomness — expected) |

**Verdict: seed plumbing is healthy.** Init genuinely varies; the
bit-tight trajectory reproduction is a real property of the loss
surface, not a plumbing bug.

**Paper finding:** at 2 epochs on tgbl-wiki, Component 0's recurrence
signal is so dominant that Xavier-uniform inits differing by ~0.02 in
absolute value still collapse to the same loss within 0.001 across
independent seeds. Worth a methodology-section sentence regardless of
Phase S outcomes.

---

## Phase S — Bounded search frame (v2.2 §4)

**Status:** blocked on anchor validation.

**Budget:** 12 hours wall clock. Hard stop. Each run uses early stopping
(patience=5 on val MRR) so under-trained variants don't burn budget on
the over-training cliff.

**Groups (v2.2 §4.1):**

| Group | Knob | Cells | Notes |
|---|---|---|---|
| A1 | within-family weighting | A / B / C from v1.5 | only when alignment is on |
| A2 | alignment loss | on / off | gates whether A1 has any effect |
| A3 | supervision target | walk-position / walk-endpoint (Phase 2 of v1.5) | conditional on A2=on |
| C  | joint training `λ_link` | {0, 0.1, 0.3, 1.0} | gradient flow from BCE into embeddings |
| D  | embedding regularisation | none / weight-decay / freeze-after-epoch-2 | counters over-training drag |
| E  | link MLP head | E.1 cross-table / E.2 Component-0-only / E.3 cross-table+dropout | the v2.0 → v2.1 fix |

**Deduplication clause (v2.2 §4.6):** When configurations collapse to
the same effective compute graph, do NOT run duplicates. Specifically
under **E.2** (Component-0-only head), embeddings are not read at
scoring, so the link BCE provides no gradient to them regardless of
`λ_link`; A2-on vs A2-off then differ only in whether the unread
embeddings are alignment-trained. Skip those redundant cells.

**Success floor (v2.2 §4.4):** anchor-validated baseline mean (from §3
above), NOT 0.65. A Phase S cell is a "win" only if it beats this floor
by > anchor std.

**Sanity checks before launch:**
- [ ] Each group's CLI flags wired into `scripts/train.py`.
- [ ] Early-stopping criterion logged (val_mrr, patience, best epoch).
- [ ] Run-log captures: group, cell ID, walltime, val/test MRR,
  early-stop epoch, BCE@best, align@best.
- [ ] Compute-graph collapse detector before launching each cell (avoid
  E.2 × A2 × `λ_link` duplicates).

**Results matrix:** (filled in run-by-run)

| Cell ID | Group | Config | Seed | Val MRR | Test MRR | Best ep | Walltime |
|---|---|---|---|---|---|---|---|
| anchor   | A2 (on)  | λ_align=1.0, 2 ep flat                       | 42 | 0.7447 | 0.7088 | — | 87 s |
| anchor   | A2 (on)  | λ_align=1.0, 2 ep flat                       |  7 | 0.7427 | 0.7060 | — | 83 s |
| anchor   | A2 (on)  | λ_align=1.0, 2 ep flat                       | 13 | 0.7420 | 0.7062 | — | 83 s |
| A2-on    | A2 (on)  | λ_align=1.0, early-stop patience=4, ≤12 ep   | 42 | 0.7444 | 0.7083 | 2 | ~7 min |
| A2-off   | A2 (off) | λ_align=0.0, early-stop patience=4, ≤12 ep   | 42 | 0.7448 | 0.7092 | 4 | ~9 min |
| A2-off   | A2 (off) | λ_align=0.0, early-stop patience=4, ≤12 ep   |  7 | 0.7458 | 0.7099 | 3 | ~8 min |
| A2-off   | A2 (off) | λ_align=0.0, early-stop patience=4, ≤12 ep   | 13 | 0.7438 | 0.7075 | 3 | ~8 min |

**Group-level summary so far:**

| Group | Config | Test mean ± std (3 seeds) | Δ vs anchor | Verdict |
|---|---|---|---|---|
| A2 (on)  | λ_align=1.0 (anchor) | 0.7070 ± 0.0016 | — (floor)  | locked floor |
| **A2 (off)** | **λ_align=0.0** | **0.7089 ± 0.0012** | **+0.0019** | **A2-off wins by > anchor std; v2.2 §4.4 ties-go-to-simpler ⇒ A2-off locked** |

**Implications:**
- Group A1 (within-family weighting), A3 (supervision target) are now moot — no alignment supervision exists to weight or re-target.
- Group C (joint training `λ_link`): becomes the relevant remaining question. Under A2-off, embeddings are uniformity-only. Does enabling BCE backprop into embeddings (`λ_link > 0`) recover anything?
- **Group E (head structure)** is the natural next: under A2-off + Group E.2 (Component-0-only head, drops cross-table reads), the head no longer reads the uniformity-only embeddings. If E.2 ≈ E.1 cross-table, that says cross-table reads of those embeddings are noise; if E.2 wins, dropping them is a simplification.

**Phase S deliverables (v2.2 §4.5):**

1. **Anchor validation results** (3 seeds, mean ± std) — recorded under §3 above.
2. **Best configuration:** which choice from each of A1, A2, A3, C, D, E.
3. **Comparison matrix:** all explored points sorted by test MRR.
   Annotated with which Group(s) each varies. Include the A2-off
   configuration explicitly.
4. **Interpretive summary:** does walks-supervision help, hurt, or break
   even on wiki? Under what conditions? What does Group E's outcome say
   about whether the cross-table embeddings carry signal at all?
5. **Recommended base for P1–P3:** the locked configuration. P1+ phases
   run on top of this.
6. **Per-epoch divergence shape** for top 3 configurations (val MRR +
   test MRR per epoch). We want to see whether the alignment-on configs
   diverge after some epoch or stabilize.

**Phase S exit summary:**
- Cells run / total budget used:
- Winning config:
- Δ over anchor floor:

---

## P1 — Window sweep on Phase S winner

**Status:** blocked on Phase S.

**Skip condition (v2.2 §5 P1):** Skip P1 if Phase S locked in A2-off
(no alignment) — without walks supervision, `max_time_capacity` is
irrelevant. Go straight to P2 pre-flight or skip to P3.

**Sweep:** `max_time_capacity ∈ {6h, 1d, 3d, 7d, ∞}` (seconds:
`{21600, 86400, 259200, 604800, -1}`) on the Phase S winner. Three
seeds for the chosen window.

**Decision criterion (v2.2 §5 P1):** if any window beats `∞` by ≥0.02
test MRR (mean across seeds), lock in.

**Results:**
| window | Val MRR | Test MRR | Notes |
|---|---|---|---|

**Decision:** chosen `max_time_capacity_R` =

---

## P2 — Multi-view (conditional)

**Status:** blocked on P1.

**Skip conditions (v2.2 §5 P2):**
- Skip if Phase S locked in A2-off (no alignment).
- Skip if Phase S Group E selected "no cross-table" (Option E.2) — the
  multi-view comparison is moot under a Component-0-only head.
- Skip if aggregate walk-distribution JS divergence is < 0.05 across
  all degree buckets — the structural view collapses to the recency
  view on this dataset.

**Pre-flight (mandatory if not skipped, ~10 min):** walk-distribution
JS by degree bucket on `mid_80pct` / `low_decile` / `high_decile`
seeds.

**Pre-flight results:**
| bucket | mean JS | p50 JS |
|---|---|---|

**Training (if proceeding):** add `walk_bias="TemporalNode2Vec", p=1.0,
q=0.25` view; add `E_context_S` + `proj_c_S` + `context_S_final`; dual
contrastive loss `L_R + λ_S · L_S` with `λ_S = 1.0`. Link MLP gets 2
additional structural-view interaction blocks (Hadamard interactions
only, per v1.5 §4 Component 3).

**Decision criterion (v2.2 §5 P2):** +0.02 over P1 base = proceed.

**Results:**
- Val MRR:
- Test MRR:
- Δ vs P1:

**Decision:**

---

## P3 — Within-method ablation matrix + error analysis

**Status:** blocked on P2. Budget ~2 hrs per v2.2 §5 P3.

**Ablation matrix (v2.2 §5 P3 — 4–6 rows depending on which earlier
phases applied):**

| Row | Config | Val MRR | Test MRR | Δ vs row above |
|---|---|---|---|---|
| 1 | Phase 0 reference (the 0.33 baseline)                          |   |   | — |
| 2 | Phase 0.5 (Component 0, current alignment, 50 ep, over-trained) | 0.4377 | 0.3940 |   |
| 3 | Anchor-validated Phase 0.5 (Component 0, early-stopped)         |   |   |   |
| 4 | Phase S winner (locked base)                                    |   |   |   |
| 5 | P1 winner (if P1 applied; otherwise skip)                       |   |   |   |
| 6 | P2 winner (if P2 applied; otherwise skip)                       |   |   |   |
| 7 | Capacity-matched control (if P2 applied): single-view d_emb=192 vs P2's two-view d_emb=128 — `3·128 = 2·192 = 384·N` table parameters strictly matched (v1.5 §6 row 7b) |   |   | (vs row 6) |

**Error analysis** (post-hoc, no extra training; v2.2 §5 P3):
- By positive `(u, v)` recency bucket (never seen / <1h / <1d / <1w / older):
- By source degree (low/mid/high tercile):
- Whether v is a hub (top-K most popular destinations):

Identifies where the multi-view (or windowed-walk) signal pays off, if
at all.

---

## Cross-dataset extension (optional, post-P3)

Repeat P3 row 8 (final config) on:
- tgbl-coin
- tgbl-flight
- tgbl-review
- tgbl-comment

**Cross-dataset results:**
| dataset | Val MRR | Test MRR | Leaderboard #1 reference |
|---|---|---|---|

---

## P4 — Honest-protocol re-baseline (paper-defining) — **DEFERRED**

**Status (2026-05-19): DEFERRED** per user direction. The optimisation
track (Phase S → P1 → P2 → P3) is what improves OUR test MRR; P4 is a
paper-narrative move that re-baselines competitor methods under
strict-causal protocol but does not raise our number. Resume after P3
lands or whenever explicitly un-deferred. Scope preserved below for
future resumption.

**Original scope:** scheduled separately, ~1 week effort (v2.2 §5). Not in the
Phase S 12-hour budget.

**Scope:** TPNet, DyGFormer, TGN re-run under strict-causal regime
(pre-ingest walk supervision, post-scoring state update). If those
numbers drop AND ours holds, the paper defends as "leaderboard
inflation + first leak-free competitive method."

**Re-baseline results:**
| Method | Original test MRR | Honest-protocol test MRR | Δ |
|---|---|---|---|

---

## Running session log (chronological)

Use this section as a flat timeline of decisions, bugs, surprises, and
fixes — anything that wouldn't fit in the per-phase tables but matters
for paper write-up.

**Session start:** branched `feature/walk-distribution-embedding` off
`master @ b246b87` (the v3 baseline). Verified master cleanly contains
only Phase 0 architecture (EmbeddingStore + LinkPredictor, no walk
encoder / DyG / memory / EdgeBank-feature from the overnight session).

**Phase 0.5 — implementation:**
- Created `tempest_walks/timestate.py` from scratch. Symmetric pair-key
  design (`(min, max)`) avoids tracking the same (u, v) ⇄ (v, u) twice.
- Used `np.maximum.at` for the per-node max-reduce — handles duplicate
  indices correctly (a normal `[idx] = max(...)` would silently drop
  duplicates).
- TimeEncoder ω_i geometric init: 1 / (time_scale × 1000^(i/(k-1))).
  Verified Φ(0) = [1,0,1,0,...] (cos(0)=1, sin(0)=0).
- Cold-start handling: Δt clamped to time_scale × 100 BEFORE Φ; the
  binary `is_cold_start` bit is the actual signal. Per v1.5 §12.1, the
  bit might get LayerNorm-attenuated; flagged as watch-list, not
  pre-emptively complicated.
- 3× strict-causal audit clean (writes only in post-scoring after
  walk_gen.add_edges; reads only pre-scoring; resets per training epoch
  only). All call sites grep'd and verified.

**Phase 0.5 — result (50 epochs, the v1.5 plan default):**
- Val 0.4377 / Test 0.3940 vs Phase 0 (0.4015 / 0.3313). Δ test = +0.063.
- Falls in the "composes additively" decision band. Keep, proceed.
- Link BCE settles ~0.10 (vs Phase 0's ~0.11) → time features give the
  head measurable signal; alignment loss unchanged at ~0.874 → Component
  0 is correctly orthogonal to the alignment supervision (it never
  touches that loss).

**Phase 0.5 diagnostics (post hoc):** the 50-epoch number is artificially
low. Component 0 + an untrained cross-table reaches Test 0.71 at 2 epochs;
50 epochs degrades it to 0.43 because alignment+uniformity supervision
pulls the cross-table embeddings into a geometry that's WORSE for TGB-
style link prediction than the random init. See Diagnostics section
below.

**Plan rewrite v1.5 → v2.0:** the 2-epoch finding broke v1.5's fixed
linear progression. Phase 1 (alignment-weighting ablation) was already
implemented (commit `68d32f1`, `align_weighting` A/B/C variants) and
its first variant launched; we stopped it before the v2.0 plan was
finalised. v2.0 introduced the Phase S bounded-search frame.

**v2.0 review pushback (five issues):**
1. Head structure was over-locked — cross-table column norms grew 5×
   while test MRR dropped 0.28; the head should be searchable.
2. The 0.71 was a single seed, not a multi-seed result; anchor unvalidated.
3. Group A conflated within-family weighting (A1), alignment on/off
   (A2), and supervision target (A3).
4. Success floor of "≥ 0.65" was soft given the 0.71 anchor.
5. Honest-protocol re-baseline of TPNet/DyGFormer/TGN was buried as one
   row of P3; realistically a multi-day effort of its own.

**v2.1:** addressed all five (Group E added to Phase S; anchor
validation §3 as 30-min pre-Phase-S step; A split into A1/A2/A3;
floor = anchor-validated baseline mean; P4 broken out as ~1-week
phase). Re-reviewed.

**v2.2 (`7ceeebe`, current):** added the §4.6 "deduplicate by effective
compute graph, not nominal hyperparameters" clause. Catches the
collapse under Option E.2 (Component-0-only head): embeddings are not
read at scoring, so most A2/`λ_link` combinations are mathematically
equivalent at the link MLP and should not be run twice.

**Pending implementation:** anchor validation (v2.2 §3) is the next
gate — 3 seeds {42, 7, 13} × 2 epochs, ~30 min total. No experiment
launched yet under the v2.x plan.

---
