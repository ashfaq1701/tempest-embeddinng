# Walk-Distribution-Matched Temporal Embeddings — Running Results & Session Transcript

Companion to `walk_distribution_matching_embedding.md` (v1.5). This document
records what was implemented, what was run, what the numbers were, and what
decision was made at each phase. Maintained as the implementation proceeds.

**Branch:** `feature/walk-distribution-embedding` (off `master` @ `b246b87`).

**Starting state confirmed:**
- `master` is at the v3 baseline (no walk encoder, no DyG, no memory, no
  EdgeBank feature). Walk-encoder family lives on feature branches and is
  NOT pulled into this one.
- Phase 0 reference number (overnight measurement): test MRR 0.331,
  val 0.4015 (Phase 0 = cross-table 8-block link MLP + alignment +
  uniformity, B=200, K=5, L=20, d=128, undirected).

---

## Phase 0.5 — Time encoding ablation

**Status:** COMPLETE. Decision: "composes additively" — keep Component 0,
proceed to Phase 1.

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
**we satisfy the gate to advance to Phase 1**. But the 2-epoch finding
is structurally important: the v1.5 plan locks in Phase 0.5 at 0.39
when the architecture's TRUE early-stopping ceiling is ~0.71. Downstream
phases should be measured against the early-stopping number, not the
over-trained one.

Three paths forward, in priority order (waiting on user confirmation):

1. **Early-stop Phase 0.5 lock-in.** Re-train with `num_epochs ∈ {2, 3,
   5, 10}` and pick the val-MRR peak. Use that as the Phase 0.5 base for
   Phase 1+.
2. **Phase 1 (loss-weighting A/B/C) at both 2-epoch and 50-epoch
   budgets.** Tells us whether variants B (1/K only) or C (uniform) dodge
   the over-training drag.
3. **Bigger rethink.** The 2-epoch result suggests that
   alignment+uniformity is actively *fighting* the EdgeBank-recurrence
   signal on this dataset. The walks would still be useful as supervision
   for OTHER targets (e.g., Phase 2's endpoint contrastive on walk
   endpoints rather than positions), but not for the current cross-table
   pull.

---

## Phase 1 — Loss-weighting ablation

**Status:** blocked on Phase 0.5

(template below; fill in after Phase 0.5 result)

**Variants:**
- A: `1/K · (1 + Δt/τ)^(-β)` (control, current)
- B: `1/K` only
- C: uniform `α = 1`

**Results:**
| Variant | Val MRR | Test MRR | Notes |
|---|---|---|---|

**Decision:**

---

## Phase 1.5 — Joint training ablation

**Status:** blocked on Phase 1

**Sweep:** `λ_link ∈ {0, 0.1, 0.3, 1.0}` on Phase 1 winner.

**Results:**
| λ_link | Val MRR | Test MRR | Notes |
|---|---|---|---|

**Decision:**

---

## Phase 2 — Walk-endpoint contrastive (single view)

**Status:** blocked on Phase 1.5

**Implementation note:** replace alignment loss with SGNS-style contrastive
matching on walk endpoints. Uses Tempest's `nodes[w, 0]` as the endpoint
(deepest valid past node in chronological-return convention). K_neg=5 per
walk; negatives uniform over `train_destinations`.

**Results:**
- Val MRR:
- Test MRR:
- Fallback applied (if any):

**Decision:**

---

## Phase 3 — `max_time_capacity` sweep

**Status:** blocked on Phase 2

**Sweep:** `max_time_capacity ∈ {6h, 1d, 3d, 7d, ∞}` (in seconds:
`{21600, 86400, 259200, 604800, -1}`).

**Results:**
| window | Val MRR | Test MRR | Notes |
|---|---|---|---|

**Decision:** chosen `max_time_capacity_R` =

---

## Phase 4 — Two-view multi-bias

**Status:** blocked on Phase 3

**Pre-flight (mandatory):** walk-distribution divergence by degree bucket.
Run `compute_per_seed_js_divergence` from v1.5 §9 on `mid_80pct` /
`low_decile` / `high_decile` seeds.

**Pre-flight results:**
| bucket | mean JS | p50 JS |
|---|---|---|

**Decision rule check:**
- `mid_80pct` mean JS > 0.1 → proceed; <0.05 → re-evaluate; in between → proceed with caution.

**Training:** add `walk_bias="TemporalNode2Vec", p=1.0, q=0.25` view; add
`E_context_S` + `proj_c_S` + `context_S_final`; dual contrastive loss
`L_R + λ_S · L_S` with `λ_S = 1.0`.

**Results:**
- Val MRR:
- Test MRR:
- Δ vs Phase 3:

**Decision:**

---

## Phase 5 — Per-view tuning

**Status:** blocked on Phase 4

(Up to ~11 runs across structural-view `max_time_capacity` × `q` × `λ_S`.)

**Results:** (best per-knob)
| knob | best value | Val MRR | Test MRR |
|---|---|---|---|

**Final hyperparameters:**

---

## Phase 6 — Ablation matrix + error analysis

**Status:** blocked on Phase 5

**Ablation matrix (9 training runs):**
| Row | Config | Val MRR | Test MRR | Δ vs row above |
|---|---|---|---|---|
| 1   | Phase 0 baseline                         |   |   | — |
| 2   | + Component 0 (time encoding)            |   |   |   |
| 3   | + Phase 1 weighting winner               |   |   |   |
| 4   | + Phase 1.5 joint-training winner        |   |   |   |
| 5   | + Phase 2 endpoint contrastive           |   |   |   |
| 6   | + Phase 3 max_time_capacity_R            |   |   |   |
| 7a  | + Phase 4 multi-view (d=128)             |   |   |   |
| 7b  | Single-view d=192 (parameter-matched)    |   |   | (vs 7a) |
| 8   | Phase 5 final (tuned, multi-view)        |   |   |   |

**Central ablation claim (row 7a vs 7b):**

**Error analysis** (post-hoc on row 8 / Phase 5 final):
- By positive `(u, v)` recency bucket (never seen / <1h / <1d / <1w / older):
- By source degree (low/mid/high tercile):
- Hub-positive vs non-hub-positive split:

---

## Cross-dataset extension (optional, post-Phase-6)

Repeat Phase 6 row 8 (Phase 5 final config) on:
- tgbl-coin
- tgbl-flight
- tgbl-review
- tgbl-comment

**Cross-dataset results:**
| dataset | Val MRR | Test MRR | Leaderboard #1 reference |
|---|---|---|---|

---

## Paper-defining experiment (optional, biggest upside)

Honest-protocol re-baselines for TPNet, DyGFormer, TGN under strict-causal
regime. The contribution: if those numbers drop AND ours holds, the paper
defends as "leaderboard inflation + first leak-free competitive method."

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

---
