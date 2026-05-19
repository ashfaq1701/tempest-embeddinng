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

**Status:** pending implementation

**Goal:** measure how much Component 0 (time encoding at the link MLP)
alone moves the needle on the master baseline (Test MRR ~0.33).

**Implementation checklist:**
- [ ] Add `last_event_time` array (n_nodes, int64) and `last_edge_time`
      sparse dict to `Trainer`
- [ ] Maintain both in the post-scoring block (training AND eval), AFTER
      `walk_gen.add_edges` and `reservoir.observe`
- [ ] Reset `last_event_time` (zeros) at the start of each training epoch;
      `last_edge_time` is a fresh dict per epoch
- [ ] Add `time_encoder` module: learned ω_i, k=16, d_time=32
- [ ] At link-MLP forward: compute Δt_u, Δt_v, Δt_uv; clamp to
      `time_scale × 100` before Φ; compute 3 binary cold-start bits
- [ ] Extend `LinkPredictor` input from `8·d` to `8·d + 3·d_time + 3`
- [ ] Smoke-test: confirm Δt features differ between (u, v_pos, t) and
      (u, v_neg, t), and that `last_*_time` updates only in post-scoring
- [ ] Quick sanity run (5 batches, watch link BCE) before launching 50-epoch

**Configuration for the training run:**
- All Phase 0 defaults (B=200, K=5, L=20, d=128, hist_neg_ratio=0.5)
- Component 0 enabled
- Component 1.5 (node-feature fusion) preserved as-is — no-op on wiki

**Decision criterion (from v1.5 §6 Phase 0.5):**
- ≥ +0.10 over Phase 0 (0.33): time encoding doing major work; lock in.
- +0.03 to +0.10: composes additively; keep.
- < +0.03: surprising. Debug `last_edge_time` population first.

**Results:** (to be filled in)
- Val MRR:
- Test MRR:
- Δ vs Phase 0 baseline (0.331):
- Decision:

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

---
