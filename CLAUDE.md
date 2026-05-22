# Tempest walk-first temporal embeddings — minimal production

This codebase is the embedding-side companion to the Tempest paper
(2026 ACM submission). Its conclusion: *"Tempest opens a path to
walk-native temporal embedding methods that consume causal walks
directly."* This repo implements that.

**Honest test MRR on TGB v2 (strict-causal protocol, no within-batch leak):**

| Dataset | Test MRR |
|---|---|
| tgbl-wiki-v2 | **0.7100** |
| tgbl-review-v2 | **~0.31** (preliminary; 6-ep sampled-eval) |

Leaderboard reference (most carry the TGN within-batch memory leak;
see Lesson 1):

| Method | wiki | review |
|---|---|---|
| Random | 0.0075 | — |
| EdgeBank-tw | 0.571 | — |
| GraphMixer | 0.594 | 0.521 |
| TGN | 0.690 | — |
| DyGFormer | 0.798 | — |
| TPNet | 0.827 | — |
| **Ours (honest)** | **0.71** | **0.31** |

---

## Architecture (fixed — no variant knobs)

```
                  ┌─────────────────┐
                  │ TGB / Tempest   │ ← ingest AFTER scoring (strict-causal)
                  │  edge stream    │
                  └────────┬────────┘
                           │ walks_for_nodes(union(src, tgt))
                           ▼
              ┌─────────────────────────┐
              │ per-walk WalkData        │  seed at nodes[lens-1]
              │  (nodes, ts, lens, ef)   │  chronological order
              └──────────┬──────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
  alignment_loss                   uniformity_loss
  pull target(seed) toward         spread target(u) on
  context(walk-position)           the unit hypersphere
        │                                 │
        └────────────────┬────────────────┘
                         ▼
                   EmbeddingStore
              (E_target, E_context  ∈  R^[N, d=128])
              + optional node-feat residual (proj_t, proj_c)
              + edge_feat_proj (consumed by alignment loss only)
                         │
                         ▼  + normbrake regularizer (column-norm clamp)
                         │
              ──────────────────────────────────────────
              At each scoring row (u, v, t):
              ──────────────────────────────────────────
                         │
        ┌────────────────┼────────────────────┐
        ▼                ▼                    ▼
  source side       destination side   Component 0
  WalkEncoder       E_target[v]        Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv)
  (1-layer GRU      E_context[v]       + 3 cold-start bits
   over walks                          (Xu et al. 2020 +
   for u)                              Phase 0.5)
        │                │                    │
        └────────────────┴────────────────────┘
                         ▼
              LinkPredictor (8-block cross-table + Component 0)
              input dim = 8·d_emb + 3·d_time + 3 = 1123
              3-layer GELU MLP → 1 logit
                         ▼
              BCE-with-logits (train)
              TGB Evaluator.eval (val/test)
```

**Two optimizers, decoupled supervision:**

- `emb_optimizer`  ← `alignment + η·uniformity + λ·normbrake`
- `link_optimizer` ← BCE only, `weight_decay=1e-4` (cliff fix)
  - Includes `walk_encoder.parameters()` AND `time_encoder.parameters()`
  - BCE backprops through the encoder INTO E_target/E_context via
    per-step lookups — gradient flow that absorbs link-MLP runaway.

## Strict-causal protocol (NON-NEGOTIABLE)

Every batch — training AND evaluation — runs in this exact order:

```
1. walks   = walk_gen.walks_for_nodes(seeds = unique(batch.src ∪ batch.tgt))
             ← Tempest state contains events strictly ≤ batch B-1.
2. (train) l_emb = alignment(walks) + η·uniformity + λ·normbrake
           emb_optimizer.step()
3. negs    = neg_sampler.sample(batch)
           ← Training: 50/50 historical reservoir + uniform random.
             Reservoir contains observations through batch B-1.
             Eval: TGB pre-generated negatives (50/50 mix).
4. walk_repr_u = walk_encoder(walks_for_nodes(unique(batch.src)))
   score    = link_predictor(walk_repr_u, target(v),
                             context(u), context(v),
                             Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv),
                             is_cold_*)
5. (train) l_link = BCE(score, labels); link_optimizer.step()
   (eval)  evaluator.eval(...)  ← TGB Evaluator, leaderboard-identical.
6. POST-SCORING BLOCK (all for batch B+1):
        if HistoricalNegativeSampler: neg_sampler.observe(batch.src, batch.tgt)
        time_state.update(batch.src, batch.tgt, batch.ts)
        walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)   ← LAST
```

If step 6 happens before step 5, walks for source `u` include the edge
`(u, v_pos, t)` we're about to score, alignment pulls `E_target[u]`
and `E_context[v_pos]` together, and the link MLP trivially scores the
just-strengthened pair high. v1/v2 reported inflated 0.508 / 0.7630
numbers from exactly this leak.

## Walk-structure rules (Tempest)

`temporal_random_walk.TemporalRandomWalk.get_random_walks_and_times_for_nodes`
returns walks in chronological order:

```
nodes:      [n_0, n_1, ..., n_{lens-1}, -1, -1, ...]   # n_{lens-1} = seed
timestamps: [t_0, t_1, ..., t_{lens-1}, -1, -1, ...]
edge_feats: [ef_0, ef_1, ..., ef_{lens-2}, ...]        # ef[p] = edge (n_p, n_{p+1})
lens:       int — number of valid node positions
```

**Seed is at `nodes[lens-1]`** (Lesson 7). NEVER assume `nodes[0]`.

## WalkEncoder (source-side, mandatory)

1-layer unidirectional GRU. Per-step input:

```
step_input_i = concat([
  E_target[node_i] if i == lens-1 else E_context[node_i],    # role-aware lookup
  Φ(t_seed - t_i),                                            # time delta
  edge_features[i] (zero-padded at i=0)                       # if dataset has ef
])
```

GRU input dim = `d_emb + d_time + (d_emb if has_edge_feat else 0)`.

Output: `walk_repr[u] = mean_k(GRU(step_inputs_k).hidden_at_seed_position)`
over K=5 walks per seed.

At the link MLP, `walk_repr_u` REPLACES the previously-static
`E_target[u]` slot in the cross-table 8-block. `E_target[v]`,
`E_context[u]`, `E_context[v]` are still static-table lookups.

## Locked hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| `d_emb`, `d_hidden_link` | 128 | Lesson 15 (capacity hurts wiki) |
| `max_walk_len` | 20 | Tempest default |
| `num_walks_per_node` (K) | 5 | Tempest default; K=1 within noise |
| `walk_bias` | ExponentialWeight | Tempest default |
| `temporal_decay_exp` (β) | 0.5 | alignment-loss recency power |
| `alignment_time_scale` | train_span / L_REF=20 | Lesson 11 (decoupled from max_walk_len) |
| `eta_uniform` | 1.0 | Stage 5 Scenario A (defaults win) |
| `uniformity_temperature` | 2.0 | Wang & Isola default |
| `uniformity_cap` | 20000 | Stage 5 (cap never fires at B=200) |
| `time_enc_k` | 16 | → d_time = 32 |
| `cold_start_dt_clamp_factor` | 100.0 | Δt clamp to 100×time_scale |
| `lambda_normbrake` | 0.1 | Stage 2 (only architectural fix that helps) |
| `normbrake_threshold` | 3.87 (wiki) / 31.32 (review) | 1.5 × col_norm at ep 1-2 |
| `weight_decay_link` | 1e-4 | Stage 3 BREAKTHROUGH (cliff drop -0.014) |
| `num_neg_per_pos` (K_train) | 10 | TGB default |
| `hist_neg_ratio` | 0.5 | Stage 4 (matches TGB eval distribution) |
| `reservoir_size` | 32 | Vitter-R per-source reservoir |
| `emb_lr`, `link_lr` | 1e-3 | Adam |
| `target_batch_size` | 200 | Lesson 15 (small batches better) |

All other knobs from the experimental phase were demonstrated to
either hurt or be non-load-bearing and were stripped (see SKIP list
below).

## Files

| Path | Role |
|---|---|
| `tempest_walks/config.py` | Locked production hyperparameters (23 fields) |
| `tempest_walks/model.py` | EmbeddingStore + TimeEncoder + LinkPredictor |
| `tempest_walks/walk_encoder.py` | 1-layer GRU walk encoder (source-side) |
| `tempest_walks/losses.py` | alignment + uniformity + normbrake + link_bce |
| `tempest_walks/trainer.py` | Strict-causal step + early-stop + snapshot/restore |
| `tempest_walks/evaluator.py` | TGB-Evaluator-backed scorer with chunked link forward |
| `tempest_walks/timestate.py` | NodeTimeState for Component 0 |
| `tempest_walks/walks.py` | Tempest CPU walk-generator wrapper |
| `tempest_walks/negatives.py` | TGB negatives + historical reservoir |
| `tempest_walks/data.py` | TGB dataset loader |
| `scripts/train.py` | Entry point (~10 CLI args) |
| `scripts/anchor_validate.py` | 3-seed × 2-ep gate (Gate A) |

## Running

```bash
# Wiki (default thresholds calibrated for wiki):
.venv/bin/python -m scripts.train --tgb-name tgbl-wiki --use-gpu \
  --num-epochs 50 --early-stop-patience 5

# Review (override normbrake threshold + use sampled eval):
.venv/bin/python -m scripts.train --tgb-name tgbl-review --use-gpu \
  --num-epochs 6 --early-stop-patience 2 \
  --normbrake-threshold 31.32 \
  --monitor-sample-pct 0.05 --skip-final-full-eval

# Anchor validation (Gate A):
.venv/bin/python -m scripts.anchor_validate --tgb-name tgbl-wiki --use-gpu
```

---

## Lessons learned (consolidated)

### Lesson 1 — TGN memory has a within-batch leak

Standard `memory_update_at_start=False` mode (Rossi et al. 2020) applies
the batch's positive edges to the GRU memory state BEFORE scoring.
Both endpoints of the positive have been freshly updated; negatives'
v-side has not. Model learns "freshness" rather than the link rule.
Most TGN/DyRep/DyGFormer/TPNet leaderboard implementations carry this
leak, so leaderboard numbers are not strict-causally comparable to ours.

### Lesson 2 — Walks supervise embeddings, NOT transformer inputs

Feeding walks as per-batch input through a transformer and training
end-to-end on link BCE alone plateaued at ~0.011 test MRR on wiki.
Walks must supervise the embedding tables via alignment+uniformity;
the link MLP then becomes a thin decoder over the supervised geometry.

### Lesson 3 — Ingest order at TRAINING matters

If ingest happens BEFORE walks at training, walks for source `u`
include the current `(u, v_pos)` edge → alignment pulls `E_target[u]`
toward `E_context[v_pos]` → link MLP trivially scores the pair high
→ training MRR inflates, eval doesn't. v1/v2's 0.508/0.7630 were
exactly this leak. This codebase ingests AFTER scoring at training AND
at eval, always.

### Lesson 4 — Historical negatives are RIGHT under walks-supervise-embeddings

`hist_neg_ratio=0.5` matches TGB's eval-time 50/50 historical/random
mix. Without historical negatives at train, the train/eval
distributions diverge and val MRR is harder to optimize. The reservoir
is observed POST-batch (strict-causal — batch B's positives are NOT
in the reservoir when batch B is scored).

### Lesson 5 — Memorization is THE signal on tgbl-wiki

EdgeBank-tw at 0.571 reveals that any architecture not designed to
exploit historical recurrence will underperform. The alignment-loss
geometry encodes recurrence by pulling `E_target[u]` and
`E_context[v]` together whenever `v` appears in u's past walks.
Repeated edges → repeated alignment pull → tight embeddings → high
link score.

### Lesson 6 — TGB Evaluator, not hand-rolled MRR

Pessimistic-only MRR (`(neg ≥ pos).sum() + 1`) differs from TGB's
official Evaluator at score-tie boundaries. Reported numbers go
through `tgb.linkproppred.evaluate.Evaluator.eval(...)`.

### Lesson 7 — Walks at seed position lens-1, NEVER 0

Tempest's backward walks are reversed in place so callers see
chronological order. Seed at `nodes[lens-1]`. Treating `nodes[0]` as
seed (an old bug) trains alignment against the deepest-past node and
cost 0.020 test MRR.

### Lesson 8 — Exploit Tempest's per-hop timestamps + edge features

Alignment loss uses per-hop timestamps via `(1 + Δt/time_scale)^(−β)`
and per-hop edge features through `proj_e` inside `context_walk`.
Both are no-ops when dataset doesn't provide them.

### Lesson 9 — Cross-table link MLP > within-table

Alignment trains the `target ↔ context` cosine. The link MLP must
consume exactly that interaction. Within-table products
(target⊙target, context⊙context) have no direct supervision.
Cross-table 8-block (`target(u)⊙context(v)` for u→v and
`target(v)⊙context(u)` for v→u, plus L1 diffs) gave +0.04 test MRR
on wiki by itself.

### Lesson 10 — Seed walks on union(src, tgt), not just src

On bipartite-flavored datasets (users→pages on wiki, users→reviews on
review), seeding only on src starves `target(v)` and `context(u)` of
alignment supervision. Union seeding doubles seed diversity per step;
cost is negligible.

### Lesson 11 — `time_scale` must be decoupled from `max_walk_len`

The alignment loss's recency term `(1 + Δt/time_scale)^(−β)` is
extremely sensitive to `time_scale`. Coupling it to `max_walk_len`
(old derivation `span / max_walk_len`) means bumping L=20→L=50
collapses `time_scale` from 93k→37k and crushes recency weight.

Right derivation: `time_scale = (t_max − t_min) / L_REF` with
`L_REF=20` FIXED. On wiki this is 93k seconds ≈ 1.08 days regardless
of `max_walk_len`. Empirically ~0.02 test MRR over the per-node mean
inter-event time.

### Lesson 12 — Edge-feat index at walk position p is `edge_feats[p]`

`timestamps[p]` and `edge_feats[p]` describe the SAME hop (between
`nodes[p]` and `nodes[p+1]`). Right-pad edge_feats so position p in
the [W, L] grid carries `edge_feats[p]`. Left-padding off-by-one was
a bug.

### Lesson 13 — Cross-pair attention with 4-channel link MLP regresses

Replacing the 12-block link MLP with a 4-channel head fed by
cross-pair attended W cost ~0.05 test MRR on wiki. The hand-crafted
Hadamard/L1 cross-table interactions do more work than cross-pair
attention adds on a 110K-edge dataset. If cross-pair is added back
later, layer it ON TOP of the cross-table 8-block, not as a
replacement.

### Lesson 14 — Direct (u, v) recurrence is the biggest unpulled lever

EdgeBank-tw's 0.571 on wiki shows that direct (u, v) pair-recurrence
is the dominant signal. Phase 6 historical experiments showed +0.051
test MRR from adding an explicit "is v in u's recent K-history?"
recency-aware feature. **Not yet ported to this production codebase**
— biggest single likely gain remaining.

### Lesson 15 — Capacity scaling HURTS on wiki

d_emb 128 → 192 regressed test MRR by 0.03. Deeper link MLP n=5
regressed -0.04 over 50 ep. The architecture is at the right capacity
for 110K-edge / 9.2k-node wiki. Stripped: `link_mlp_n_layers` knob,
`d_emb` is fixed at 128.

### Lesson 16 — Memory module adds marginal value once recurrence is captured

Raw-message-store memory added only +0.003 test MRR on top of walk
encoder + co-occurrence. On wiki, K_history=32 already covers the
relevant time window. Memory is more useful on datasets where the
recurrence horizon exceeds K_history.

### Lesson 17 — Alignment+uniformity has a 50-epoch over-training cliff

50-ep training on wiki: val MRR collapses 0.7448 → 0.4625 (drop -0.28).
Diagnosed mechanism:

1. **Primary driver — embedding magnitude runaway.** col_norm of E
   grows 2.08 → 10.76 (5.2×). Adam's accumulated momentum keeps moving
   E_context even after its per-batch grad collapses (0.005 → 0.0001
   by ep 7).
2. **Secondary driver — link MLP weight runaway.** link_w_norm grows
   0.28 → 2.02 (7×) to track the growing embedding magnitudes.
3. **Universal context-side grad collapse.** E_context grad drops to
   ~0.0001 by ep 7 across EVERY fix tested — alignment loss saturates
   on the context side almost immediately.

### Lesson 18 — Normbrake (per-column L2 hinge) halves the cliff

`L_normbrake = λ · Σ_c relu(||col_c||₂ − threshold)²` on both tables.
With `λ=0.1` and `threshold = 1.5 × col_norm at ep 1-2`:
- Wiki threshold = 3.87
- Review threshold = 31.32

Result: val drop reduces from -0.28 → -0.11. Clean clamp behavior
(col_norm freezes at threshold). E_target grad stays HEALTHY (0.092
vs baseline 0.012 — 8× better). Peak val MRR unchanged.

Does NOT close the cliff alone — link_w_norm still runs away.

### Lesson 19 — Joint training (λ_link > 0) collapses contrastive walk-supervision

Letting link BCE backprop into the embedding tables with `λ_link > 0`
IMMEDIATELY collapses val MRR across ALL contrastive loss families
and ALL `hist_neg_ratio` values. At `λ_link=0.1` on wiki, val drops
0.7432 → 0.5109 by ep 2.

Universal: confirmed for InfoNCE, alignment+uniformity, AND across
hist_neg_ratio ∈ {0, 0.25, 0.5, 0.75}. Mechanism is BCE-into-embeddings
fighting alignment regardless of negative type.

**Locked: `λ_link = 0` (no CLI knob).**

### Lesson 20 — Triplet wins wiki, loses review (cross-dataset)

Wiki §4.7 loss-family search:
- Triplet: 0.7105 ± 0.0014 multi-seed (best peak wiki)
- SGNS+normbrake: 0.7113 (tied)
- alignment+uniformity: 0.7079 anchor

Review sweep (6 ep sampled-eval):
- alignment+nb: **0.3135**
- Triplet: ~0.16 (decisive loss)

**Locked: alignment + uniformity** as cross-dataset-robust choice
(user-imposed within-±0.01 rule). Triplet / InfoNCE / SGNS removed
entirely from this production codebase.

### Lesson 21 — Architectural fixes (dropout, depth) don't fix the cliff

Stage 2 ran 6 cells × 50 ep on wiki testing dropout/depth:

| Cell | Val drop |
|---|---|
| normbrake λ=0.1 | **-0.11** |
| kitchen sink (normbrake + dropout + deeper) | -0.12 |
| link MLP dropout 0.3 | -0.19 |
| deeper MLP n=5 | -0.23 |
| embedding dropout 0.3 | -0.27 |
| baseline | -0.28 |

Normbrake is the only meaningful fix. Dropout/depth knobs stripped.

### Lesson 22 — Adam constructor changes induce CUDA non-determinism drift

Same seed, same hyperparams, but adding `weight_decay=0.0` to the
`Adam(...)` constructor explicitly (vs default-omitted) caused
mechanism-preserving bit-tight col_norm/grad trajectories BUT val MRR
drift of -0.030 over 50 epochs. Anchor-validation tolerance is ±0.005
— well below this drift level. Be suspicious of drift > 0.005.

### Lesson 23 — weight_decay_link closes the residual cliff (Stage 3 BREAKTHROUGH)

After normbrake halves the cliff (-0.28 → -0.11), the residual is
link_w_norm runaway. Adding `weight_decay=1e-4` to the link_optimizer
holds link_w_norm flat (0.18 → 0.17 vs unfixed 0.28 → 2.02). Result:

| Fix | Val drop peak→50ep |
|---|---|
| baseline | -0.28 |
| normbrake only | -0.11 |
| **normbrake + WD_link=1e-4** | **-0.014** |
| normbrake + WD_link=1e-3 | -0.001 (over-suppresses link MLP; less robust cross-dataset) |

WD_link=1e-4 is the locked production value.

### Lesson 24 — Single-table architecture fails on cliff shape (Step 6)

A single `E[N, d]` table + `P_src` + `P_tgt` projections (instead of
two separate tables) was tested on wiki:

| Metric | dual-table | 1T_asym |
|---|---|---|
| Best test | 0.7096 | 0.7090 |
| ep 50 val | 0.7335 | 0.7105 |
| Drop peak→50 | -0.011 | **-0.035** |

Peak ties within noise (Δ -0.0006) but cliff shape regresses 3.2×.
Per smooth-curve rule, dual-table locked.

### Lesson 25 — Walk encoder ties locked-v2 on peak, smooths the cliff

Step 7 result:

| Cell | Best test | ep 50 val | Drop peak→50 |
|---|---|---|---|
| W_off (encoder OFF) | 0.7096 | 0.7261 | -0.019 |
| W_gru (K=5) | 0.7080 | 0.7404 | -0.005 |
| W_gru_k1 (K=1) | 0.7089 | 0.7431 | -0.002 |

Encoder ties locked-v2 on peak (within anchor std) AND produces
4-10× smoother long-training trajectory. **Locked ON in this
production codebase.** K=5 retained as Tempest convention default;
K=1 verified equivalent within noise and 5× cheaper at scoring time.

### Lesson 26 — Freeze-tables sanity confirms wiki tables are dead weight

Phase 1 sanity check (freeze E_target / E_context, retrain encoder +
link MLP only on wiki):

| Cell | Best test |
|---|---|
| W_gru_k1 (encoder, tables train) | 0.7089 |
| **Sanity (encoder, tables FROZEN)** | **0.7094** |

Δ +0.0005 — essentially identical. The link MLP's cross-table weights
collapsed to ~0.01 (down from 0.17 with tables training), confirming
the link MLP voluntarily ignores the static-table contribution.
Wiki performance comes from walk encoder + Component 0 +
recurrence-via-time-features, NOT learned per-node identity.

This is the canonical signature predicted by cross-domain literature
(CAWN ICLR 2021; GraphMixer review #1; TGB v2 leaderboard pattern).
Strong indicator that walks-only architectures (drop E_target
entirely; CAWN-style anonymous identity) would help on review
(surprise index 0.987) without hurting wiki (surprise index 0.108).
Walks-only is documented as future work below.

### Lesson 29 — Normbrake IS load-bearing under coherent walks (2026-05-22)

**Pre-Step-3 hypothesis (FALSIFIED).** Across all three Step-3 cells
(W_off, W_gru_k1, Sanity) the normbrake loss `nb` saturated at
0.0017 or lower — well below noise. The natural read was "the brake
is dormant, contributes nothing to the loss, can be stripped."

**W_no_nb ablation (2026-05-22, 50 ep, seed 42, dual-table fixed,
encoder ON, K=5, --lambda-normbrake 0).** Result:
  - Best at ep 8: val 0.7445 / test **0.7086** (unchanged vs brake-on).
  - Ep 50: val 0.5519 → cliff drop **-0.193**.

Comparison to brake-ON Step-3 cells:
  | Cell | Best test | Cliff |
  |---|---|---|
  | W_off (brake ON, K=5)   | 0.7081 | -0.057 |
  | W_gru_k1 (brake ON, K=1) | 0.7075 | -0.079 |
  | **W_no_nb (brake OFF, K=5)** | 0.7086 | **-0.193** |

The cliff is **2.5–3.5× worse without the brake**.

**Why the dormancy reading was wrong.** The brake's loss
`L = mean_j relu(||E[:, j]||₂ − threshold)²` is a HINGE. When E's
column norms are at-or-below threshold, the loss is exactly 0 and
gradient is zero. When norms cross threshold, the loss jumps to a
small positive value AND a clamping gradient kicks in. In Step 3 we
observed E_col_norm pinned ≈ 4.17 (just above threshold 3.87) and
nb stuck at 0.0017. That looked dormant. But the brake's job isn't
to ADD a meaningful loss — it's to provide the clamping gradient that
HOLDS E at the threshold. Holding takes zero loss budget once
saturated, but is what prevents runaway. The dormancy IS the brake
working.

Removing the brake → no clamping → E.weight grows unbounded → link
MLP overfits the growing geometry → severe cliff.

**Decision (executed): KEEP normbrake.** Master stays at
`locked-v2-fixed`. The `experiment/normbrake-ablation` branch is
preserved as a paper artifact showing the brake's load-bearing role.

**Lesson 18 (normbrake halves the cliff, pre-fix) status:**
**CONFIRMED + STRENGTHENED.** Pre-fix the brake reduced cliff from
-0.28 → -0.11; post-fix it reduces cliff from -0.193 → -0.057
(W_off) or -0.079 (W_gru_k1). Different absolute magnitudes (because
the cliff itself is different under coherent walks), but the brake's
relative effect — preventing E magnitude runaway — survives the
bug-fix.

---

### Lesson 28 — Two coupled strict-causal bugs: `shuffle_walk_order=True` (walks) and reservoir leak (negatives) (2026-05-21)

**This is the most consequential finding in the project's history.**
Every walks-supervision diagnostic from Phase 2 onward (Lessons 17-26,
the Stage 2-5 cliff analyses, the Step 6 single-table ablation, the
Step 7 walk-encoder verification, and the Lesson 27 migration cell)
was measured under randomized seed-to-row mapping. The historical
empirical claims are provisional until re-collected.

A second strict-causal bug — the historical-negative reservoir
leaking across epochs — surfaced during the post-shuffle-fix code
audit (Bug 2 below). Both are coupled in that the reservoir leak's
effect was masked by the shuffle bug (alignment trained random
signal regardless of negative choice). After both fixes the
pipeline finally trains on coherent supervision.

## Bug 1 — `shuffle_walk_order` defaults to True

`temporal_random_walk.TemporalRandomWalk.__init__` accepts a
`shuffle_walk_order` kwarg. Per the Tempest source
(`temporal_random_walk/src/common/const.cuh`):

```cpp
constexpr bool DEFAULT_SHUFFLE_WALK_ORDER = true;
```

When True, Tempest randomly interleaves the [N·K, L] walk-output
array across all seeds. The Python binding's docstring states this
explicitly. `tempest_walks/walks.py:48-53` constructed
`TemporalRandomWalk(...)` without passing `shuffle_walk_order`, so the
True default silently activated. `grep -rn shuffle_walk *.py` returned
zero hits across the codebase before the fix.

**Where it broke the pipeline.** Three downstream paths assume the
output is grouped as `[seed_0×K, seed_1×K, ..., seed_{N-1}×K]`:

1. **`losses.py` alignment_loss**:
   `e_target_seed.repeat_interleave(K, dim=0)` interleaves
   `[E_target[s_0]×K, E_target[s_1]×K, ...]`. The cosine sim at
   `losses.py:56` then pulls `E_target[s_i]` toward
   `E_context[walk_row_i]` — but `walk_row_i` was sampled from a
   randomly-different seed. Alignment is structurally randomized at
   the (seed, context-walk) pairing.

2. **`walk_encoder.py` reshape**:
   `walk_repr_per_walk.view(N, K, d).mean(dim=1)` averages K rows
   assuming they came from the same seed. With shuffled output,
   `walk_repr[i]` is the mean of K rows from K randomly-mixed seeds.

3. **`trainer.py::_compute_walk_repr_for`**: builds `idx_map` from
   `walks.seeds`, the INPUT seed array (unshuffled), and looks up
   rows for each batch.src entry — returning the wrong rows because
   Tempest shuffled them but the input array didn't reflect that.

`t_query` is `batch.t_max` broadcast across all seeds at the trainer
level, so the `repeat_interleave(t_query)` at `losses.py:60` is
shuffle-immune. Component 0 (Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv) + 3 cold-start
bits) goes through `NodeTimeState`, which never touches the walks
pipeline; it is also shuffle-immune.

**Predictive coherence with the historical record.** The bug's
mechanism predicts the exact pattern we've been measuring:

| Historical observation | Predicted under the bug |
|---|---|
| Sanity (Lesson 26): freeze E_target / E_context → test MRR Δ = +0.0005 ("tables are dead weight") | Tables never received coherent supervision; their training was effectively pulling toward random walks → behaved like random parameters → "frozen random = trained" is exactly what randomized supervision produces |
| W_off (encoder OFF) = 0.7096, W_gru_k1 (encoder ON, K=1) = 0.7089, Δ ≈ 0 (Lesson 25) | Encoder's per-seed walk_repr was the mean of K randomly-mixed seeds — structurally noise — so the link MLP learned to ignore it |
| Link-MLP cross-table weights collapsed 0.17 → 0.01 in sanity (Lesson 26) | Cross-table block reading meaningless inputs → WD_link drives weights to 0 |
| 0.71 wiki ceiling across 25 architectural variants (Phases 2-7) | 0.71 IS Component 0's ceiling. Phase 0.5's zero-out diagnostic (Component 0 → 0 drops test MRR by -0.40) puts Component 0's contribution at ~0.40. Subtracting from 0.71 leaves ~0.31 — which is roughly the EdgeBank-tw / GraphMixer-style memorization floor reachable via NodeTimeState's Δt features. Walks-supervision was contributing ~0 |
| Cliff dynamics (Lesson 17): E_context grad collapses to ~1e-4 by ep 7 | Alignment loss saturated against random context — no stable gradient signal |
| Triplet wins wiki / loses review (Lesson 20) | Both losses operate on randomized walks; the cross-dataset difference reflects which loss-shape produces less bias under noise, NOT which captures walks structure better |

The Sanity check is the cleanest evidence — *every* prediction the
bug's mechanism makes lines up with what was observed.

**Fix.** One kwarg in `tempest_walks/walks.py`:

```python
self.trw = TemporalRandomWalk(
    is_directed=is_directed,
    use_gpu=use_gpu,
    enable_weight_computation=True,
    timescale_bound=timescale_bound,
    shuffle_walk_order=False,   # ← THE FIX
)
```

**Cross-check verification.** `scripts/_shuffle_diagnostic.py` (5-node
graph, K=3 walks per seed, deterministic timestamps):
  - Pre-fix: **13 of 15** output rows had `nodes[lens-1] != seeds[row // K]`.
  - Post-fix: **0 of 15** rows misaligned. Layout matches the
    `[seed_0×K, seed_1×K, ...]` convention the codebase assumes.

## Bug 2 — HistoricalNegativeSampler reservoir leaks across epochs

`Trainer.train()`'s docstring at trainer.py:333 says "Per epoch: reset
Tempest + time_state + reservoir". The implementation reset only the
first two — `walk_gen.reset()` and `time_state.reset()`. The
`HistoricalNegativeSampler.reset()` method (negatives.py:102-104,
zeros the reservoir + count) was never called outside `__init__`.

**Failure mode.** Each training epoch iterates the chronological train
set. By the END of epoch 1, every source's reservoir reflects the
ENTIRE training timeline (reservoir_size=32 vs ~20-30 avg unique
destinations per source on wiki — most sources saturate). At the
START of epoch 2:

  - walk_gen and time_state are empty (fresh chronological pass).
  - But the reservoir still holds every destination each source ever
    interacted with.
  - Historical negatives for batch B in epoch 2 sample uniformly from
    that full pool.
  - Many of those destinations are the source's FUTURE positives
    within epoch 2 (B' > B in the same chronological pass).
  - The false-negative guard at negatives.py:135-137 only invalidates
    `hist_neg == batch.tgt`; it does NOT know which (u, v) pairs are
    upcoming positives.

So the model is trained to score (u, v_future_positive) LOW for batch
B, then trained to score the same pair HIGH at batch B' > B. The
"historical" channel of the negatives is a structurally
self-contradictory training signal from epoch 2 onward, on a real
fraction of every batch.

**Why this masked under Bug 1.** With shuffled walks, the embedding
tables and walk encoder were never receiving coherent supervision —
contradictions in negative sampling couldn't damage signal that
didn't exist. The HistoricalNegativeSampler docstring already warned
against using historical negatives on wiki for an *adjacent* reason
(scoring eval-time historical positives LOW collapses MRR). The
reservoir leak amplifies that risk by guaranteeing future-positive
contamination.

**Magnitude.** wiki has 9227 nodes / 110K train edges / reservoir_size
32. Average source has ~20-30 unique destinations; most fit in the
reservoir by end of epoch 1. By epoch 2 each source's reservoir
approximates its full destination-history distribution. EdgeBank-tw's
0.571 wiki score quantifies the underlying recurrence — a substantial
fraction of historical negatives served at training time are pairs
that genuinely should be classified positive at later batches in the
same epoch.

**Fix.** Two changes:

1. `NegativeSampler` ABC gains a no-op default `.reset()`. Subclasses
   that own state (`HistoricalNegativeSampler`) override; stateless
   subclasses (`UniformNegativeSampler`, `TGBNegativeSampler`) inherit
   the no-op.

2. `Trainer.train()` calls `self.neg_sampler_train.reset()` at the top
   of the per-epoch loop, immediately after `walk_gen.reset()` and
   `time_state.reset()`. Unconditional — the ABC default makes it
   safe across sampler choices.

**Status of historical lessons.** Provisional pending re-anchor:

  - Lesson 17 (cliff mechanism) — col_norm runaway is real but the
    *cause* (alignment loss saturating early) may be artifact of
    randomized supervision rather than a fundamental training-dynamics
    issue.
  - Lesson 18 (normbrake halves cliff) — may be regularizing a
    parameter that already had no useful signal; effect under
    coherent supervision unknown.
  - Lesson 19 (λ_link > 0 collapses) — collapse mechanism may differ
    under coherent supervision; possibly the BCE-into-embeddings
    pressure was fighting *random alignment* rather than a real loss
    surface.
  - Lesson 22 (Adam-constructor drift) — likely still valid (CUDA
    non-determinism is independent of supervision quality), but
    sensitivity scale may change.
  - Lesson 23 (WD_link breakthrough) — held link_w_norm flat while
    the link MLP read meaningful inputs from walks? Or while it read
    noise? Unknown until re-tested.
  - Lesson 24 (single-table cliff-shape regression) — measured under
    the bug; both architectures shuffled.
  - Lesson 25 (walk encoder ties peak) — almost certainly artifact;
    the encoder's `walk_repr[u]` was structurally noise.
  - Lesson 26 (frozen tables sanity) — the cleanest case of the
    bug's signature; explained entirely by the mechanism above.
  - Lesson 27 (single-table migration) — pre-registered before the
    bug was identified; deferred until anchor revalidates.

Lesson 4 ("historical negatives are RIGHT under walks-supervise-
embeddings") is provisional too: under the reservoir leak, training
historical negatives were sampling from "destinations the source
ever interacted with across all of train" rather than "destinations
the source interacted with up to batch B-1". The lesson's strict-
causal premise wasn't fully met. The mix of historical vs random
training negatives still matters; only the *historicity* of the
historical channel was broken.

Lessons 1-3, 5-16 and the strict-causal protocol are independent of
the walks pipeline (memory leak analysis, ingest order, evaluator,
TimeEncoder, etc.) and stand as-is.

**Re-verification protocol.**
  - Step 1a ✓ commit walks.py shuffle fix (`905bfa4`).
  - Step 1b ✓ commit reservoir-reset fix (`50c7d32`).
  - Step 2 ✓ anchor validation under both fixes on wiki (3 seeds × 2 ep):
    val mean **0.7435 ± 0.0010** / test mean **0.7086 ± 0.0007**.
    Verdict: CONFIRMED vs 0.7070 ± 0.0016 target. Delta vs pre-fix
    anchor (test mean 0.7087): -0.0001 — **identical within noise**.
    Outcome B: at 2-epoch scale on wiki, neither bug was load-bearing.
    Component 0 dominates the 2-ep signal; walks-supervision adds
    nothing measurable at this depth. The discriminators are deeper
    training (Step 3) and review (Step 4).
  - Step 3 — 50-ep wiki cells under both fixes (seed 42, --log-debug,
    --early-stop-patience 999, --num-walks-per-node 1 where indicated):

    | Cell | Best (ep, val/test) | Ep-50 val | Cliff drop | Pre-fix (Lesson 25/26) |
    |---|---|---|---|---|
    | W_off (no encoder, tables train, K=5) | ep 3: 0.7433 / **0.7081** | 0.6860 | **-0.057** | 0.7096 best, -0.019 cliff |
    | W_gru_k1 (encoder ON, tables train, K=1) | ep 6: 0.7435 / **0.7075** | 0.6646 | **-0.079** | 0.7089 best, -0.002 cliff |
    | Sanity (encoder ON, tables FROZEN, K=1) | ep 1: 0.7449 / **0.7105** | 0.7426 | **-0.002** | 0.7094 best, -0.002 cliff |

    **The Sanity result is the headline finding.** Sanity has:
      - The HIGHEST peak test MRR of the three (0.7105 vs 0.7081 / 0.7075).
      - The SMALLEST cliff (-0.002 vs -0.057 / -0.079, ~30× better).
      - The fastest convergence (best at ep 1, not ep 3 or ep 6).

    Translation: **under coherent walks, training the embedding tables
    via alignment+uniformity ACTIVELY HURTS on wiki**. Frozen random
    Xavier-init tables + walk encoder + Component 0 + link MLP
    collectively reach a stronger, more stable optimum than ANY
    configuration that trains the tables. The pre-fix Sanity result
    (0.7094, indistinguishable from W_gru_k1 0.7089) was hiding this —
    under the shuffle bug, trained tables were random anyway, so
    "trained" and "frozen random" were both ~equivalent random.

    Note on the W_off vs W_gru_k1 ordering: W_off (0.7081) edges out
    W_gru_k1 (0.7075) in peak. Once tables are training, the encoder
    adds nothing on wiki — it just over-fits faster (deeper cliff).

    Architectural implications:
      - On wiki, the source-side identity is irrelevant; only the
        downstream channels (encoder, Component 0, link MLP cross-table)
        matter. This is the canonical CAWN/anonymous-identity signature
        predicted by cross-domain literature.
      - The single-table+dual-projection migration is now under-
        motivated: if tables don't matter, halving them doesn't help
        either. But complexity reduction is still valuable.
      - normbrake is observably dormant (nb ≈ 0.0017 across W_off/
        W_gru_k1; literally 0.0 in Sanity since E.weight has no
        gradient at all). Strippable.
      - There's a stronger move available: **freeze the tables, or
        remove the alignment+uniformity loss entirely**. The user
        flagged complexity-reduction first; this third move is a
        candidate but not in the current execution path.

    Key observations:
      - **Peak test MRR is unchanged within noise** vs pre-fix
        (Δ ≈ -0.001 to -0.002). The walks-supervision pipeline is
        coherent now, but Component 0 + memorization-via-time-features
        still dominate the wiki ceiling at the 0.71 level.
      - **Cliff is now SIGNIFICANTLY worse** under coherent walks:
        W_off -0.057 vs -0.019 (3× deeper), W_gru_k1 -0.079 vs -0.002
        (~40× deeper). The pre-fix observation of "encoder smooths the
        cliff" (Lesson 25) **does not survive** under coherent
        supervision. With real alignment signal, the embeddings drift
        further over training even with the encoder.
      - Link loss now CONVERGES MUCH FURTHER under coherent walks
        (W_gru_k1 link loss 0.10 at ep 50 vs locked-v2's 0.15 at ep
        50). The link MLP is fitting the now-meaningful cross-table
        signal harder, which is the proximate cause of the deeper
        cliff (overfit to overly-confident embeddings).
      - Normbrake's contribution remains tiny: nb stuck around
        0.0015-0.0018 across both cells, well below noise floor.
        L_normbrake is essentially dormant under coherent walks too,
        confirming the prior on Lesson 28b/Lesson 18 redundancy.
        (Stripping pending architectural decision.)
      - The encoder's "smoothing" property (Lesson 25) was an artifact
        of the bug — under randomized walks, the encoder's noisy
        output never trained anything coherent, so no overtraining-
        induced cliff. Under coherent walks, the encoder doesn't help
        smooth the cliff at all — it actually shifts the optimum
        earlier (ep 6 vs W_off's ep 3 best, but bigger collapse after).
  - Step 4: review run under both fixes (6-ep sampled-eval). _Deferred
    (user instruction): re-evaluate after the architectural
    simplification (single-table + possibly normbrake removal) lands._
  - Step 5: synthesize, decide which historical lessons re-collect.

**Status of historical lessons (refined post-Step-3).**

  - Lesson 17 (cliff mechanism, col_norm runaway) — **CONFIRMED real**.
    The cliff is not an artifact of randomized supervision; under
    coherent walks it is in fact DEEPER. Mechanism analysis (E grad
    saturating early, Adam momentum runaway) still applies, but the
    saturation is now on real signal rather than noise.
  - Lesson 18 (normbrake halves cliff) — **provisional under coherent
    supervision**. The pre-fix improvement was on a randomized-noise
    cliff; whether it still helps the now-deeper cliff is untested.
    nb is dormant (0.0017) in Step 3, suggesting it's not engaging
    enough to matter — likely still strippable as Lesson 28-followup.
  - Lesson 23 (WD_link breakthrough) — **provisional**. WD_link held
    link_w_norm flat in Step 3 cells, but the cliff still appeared
    via deeper link-loss convergence (different mechanism). Possibly
    still load-bearing but in a different way.
  - Lesson 25 (walk encoder ties peak, smooths cliff) — **FALSIFIED
    under coherent walks for the cliff-smoothing claim**. Peak still
    ties (W_off 0.7081 vs W_gru_k1 0.7075, within noise), but
    encoder's cliff is WORSE (-0.079 vs W_off's -0.057). The
    "smoothing" interpretation was a bug artifact.
  - Lesson 26 (frozen tables sanity) — **CONFIRMED + STRENGTHENED**.
    Pre-fix Sanity tied W_gru_k1 (0.7094 vs 0.7089) and was framed as
    "tables are dead weight". Post-fix Sanity BEATS W_gru_k1 (0.7105
    vs 0.7075) AND has a 30× smaller cliff. The interpretation is no
    longer "tables are dead weight"; it's "training the tables hurts".
  - Lesson 27 (single-table migration) — re-evaluated after Step 3
    lands (see Decision Path below).

**Decision path after Step 3 (executed autonomously per user
authority).** 
  1. Merge the fix-branch to master, tag the result.
  2. Re-run the single-table + dual-projection migration on the fixed
     dual-table master. Acceptance: peak val within ~0.005 of fixed
     dual-table baseline AND cliff not noticeably worse. If pass, the
     dual-table architecture is gratuitous complexity → strip to
     single-table.
  3. With single-table locked, run a 50-ep `--lambda-normbrake 0`
     ablation. If peak + cliff unchanged within noise, strip
     normbrake entirely.

The diagnostic script `scripts/_shuffle_diagnostic.py` is retained
in-tree for reproducibility of the pre-/post-fix witness; it is not
part of the production pipeline.

---

### Lesson 27 — Single-table + dual-projection migration (PRE-REGISTERED 2026-05-21, DEFERRED 2026-05-21)

> **STATUS — DEFERRED pending Lesson 28's shuffle_walk_order bug fix.**
> This pre-registration was committed on 2026-05-21, then the
> migration's single-seed wiki run was launched and reached ep 40
> before being halted at user request. During that run a critical
> walks-supervision bug was identified (Tempest's `shuffle_walk_order`
> defaults to True; the codebase has never disabled it). Every
> diagnostic this migration was retesting — Step 6 (Lesson 24), the
> Sanity check (Lesson 26), the cliff-mechanism analyses (Lessons
> 17-23, 25) — was measured against shuffled walks. The migration is
> therefore comparing two architectures whose supervision signal was
> randomized in the same way. See Lesson 28 for the bug, the
> mechanism, the predictive coherence with the sanity-check signature,
> and the fix. The refactor code is preserved on branch
> `experiment/single-table-dual-projection` as a paper-ablation
> artifact. The migration's empirical claims will be re-evaluated
> after Lesson 28's anchor revalidation under the fix.

This section is committed BEFORE the migration runs as a pre-registered
prediction. Final empirical result is appended at the bottom of this
lesson once Cell A + multi-seed verification complete.

**Motivation.** Lesson 26 sanity-check froze E_target / E_context on wiki
and the test MRR moved by Δ=+0.0005 — the dual tables contribute
essentially nothing to wiki link prediction once the walk encoder is
on. The asymmetry that the cross-table 8-block exploits lives at the
LINK MLP HEAD (target(u) ↔ context(v) pairings), not in the embedding
table identities. A single shared table E ∈ R^[N, d] with two linear
projections P_src, P_tgt expresses exactly that asymmetry in fewer
parameters:

  - dual-table E_target + E_context on wiki: 9229 × 128 × 2 = 2.362M params.
  - single-table E + P_src + P_tgt on wiki:  9229 × 128 + 2 × (128 × 128) = 1.214M params.

The migration halves the embedding-side parameter count and makes the
"the asymmetry is at the head" finding architecturally explicit.

**Why this is NOT a re-run of Step 6 (Lesson 24).** Step 6 tested
1T_asym BEFORE Stage 3's weight_decay_link=1e-4 landed and BEFORE the
walk encoder shipped. The dual-table peak tied at the noise floor
(Δ=-0.0006) but cliff shape regressed 3.2× (-0.011 → -0.035). Two of
the three cliff-stabilizers we now ship (WD_link + walk encoder) hadn't
yet been measured at that point. This migration retests the same
architectural primitive under the present locked stack.

**Pre-registered prediction (timestamped 2026-05-21).** On wiki:
  - Peak test MRR ties locked-v2 within the anchor std (0.7070 ± 0.0016).
    Confidence: high. The walk encoder is the load-bearing source-side
    component; the static table contribution is verifiably ~0
    (Lesson 26). Moving table identity into a projection should be
    transparent to peak MRR.
  - Cliff shape (peak val → ep-50 val drop) is the open question.
    Possibility A: WD_link + walk encoder absorb the cliff-shape
    regression Step 6 saw → drop matches locked-v2 (~-0.011).
    Possibility B: single-table E magnitude dynamics differ enough that
    even normbrake doesn't fully stabilize → drop somewhere between
    -0.011 and Step 6's -0.035.
  - Threshold-recalibration is mandatory. Dual-table threshold=3.87 was
    calibrated against E_target.weight's per-column norm at ep 1-2; the
    single-table E.weight absorbs gradients from BOTH the P_src side
    (alignment + uniformity) AND the P_tgt side (alignment context-walk
    + uniformity-of-target ub). Magnitude dynamics should differ. The
    2-epoch calibration cell measures the new col_norm and sets
    threshold = 1.5 × col_norm_at_ep2.

On review (sampled 6-ep eval): peak ties or marginally improves vs
locked-v2's 0.3135 (Lesson 26 predicts walks-only-style architectures
should help on high-surprise-index datasets, and single-table is a
small step in that direction by removing one of the two static
identity tables).

**Decision rule (locked before running).**
This is a complexity-reduction migration, not an optimization. MERGE if
ALL of the following hold:
  - Wiki peak val MRR within ±0.005 of locked-v2 peak (≥ 0.7399, ≤ 0.7499
    against the 0.7449 reference).
  - Wiki cliff drop (peak → ep-50) within ±0.01 of locked-v2's -0.011
    (i.e. drop ≤ -0.021, so ep-50 val ≥ 0.7235 against the 0.7335
    reference).
  - Multi-seed (42, 7, 13) mean within ±0.005 of locked-v2 mean. Single
    seed is insufficient — Lesson 22 documents CUDA non-determinism
    drift of ~±0.030 on identical seeds.

DO NOT MERGE if any decision-rule check fails. The branch becomes a
paper-ablation artifact; master stays at locked-v2.

**Partial empirical result (HALTED 2026-05-21 before completion).**
The migration's seed-42 cell ran for ~40 of 50 epochs on the
`experiment/single-table-dual-projection` branch with calibrated
threshold 4.13, then was halted at user request because the
shuffle_walk_order bug (Lesson 28) was identified. The collected
numbers are recorded for the historical chronology but are NOT to be
interpreted as evaluating the migration on its merits; both
architectures (dual-table locked-v2 and single-table experiment) were
trained against randomized walks.
  - Calibrated single-table threshold (E.weight, ep 2, walks shuffled):
    1.5 × 2.7507 = **4.13** (delta vs dual-table 3.87 is +6.7%).
  - 2-epoch calibration with λ_normbrake=0: val 0.7443 / test 0.7111
    (within noise of locked-v2 2-ep anchor val 0.7449 / test 0.7081).
  - Wiki seed 42, 50 ep at threshold 4.13:
      * Best at ep 28: val 0.7453 / test 0.7096.
      * Ep 50: val 0.6921 (cliff -0.053, far worse than locked-v2's
        -0.011 under the same shuffle-bug condition).
      * Trajectory: nb saturated tiny (0.0017 from ep 10 onward);
        E.weight clamped near 4.17; P_src_norm RAN AWAY (13 → 38);
        link_w_norm 10 → 13.
  - Wiki seeds {7, 13}: NOT RUN (halted).
  - Decision: DEFERRED. Migration verification re-runs after the
    Lesson 28 fix lands and the post-fix anchor is established.

---

## Historical chronology (consolidated)

| Phase | Outcome | Test MRR |
|---|---|---|
| **Phase 0** — v3 baseline (alignment+uniformity, cross-table E.1) | locked anchor v0 | 0.3313 |
| **Phase 0.5** — Component 0 (TimeEncoder + cold-start bits) | anchor v2.2 §3 | **0.7070 ± 0.0016** |
| **Phase 2** — GRU walk encoder over per-position context | architectural baseline | 0.4289 |
| **Phase 4–5** — DyG + co-occurrence + memory | marginal +0.011 | 0.4396 |
| **Phase 6** — EdgeBank-style direct (u,v) recurrence channel | +0.051 (biggest single jump) | 0.4902 |
| **Phase S** — A2/E head sweeps | E.1 cross-table + A2-on locked | 0.7079 ± 0.0005 |
| **§4.7** — Loss-family search (Triplet/InfoNCE/SGNS vs alignment) | Triplet wiki marginal lead | 0.7105 ± 0.0014 |
| **Review sweep** — cross-dataset loss validation | alignment LOCKED (Triplet lost on review) | 0.3135 review |
| **Stage 2** — architectural fixes for cliff (dropout, depth, normbrake) | normbrake the only fix | drop -0.11 |
| **Stage 3** — λ_link + WD_link sweep | WD_link=1e-4 BREAKTHROUGH | drop -0.014 |
| **Stage 4** — hist_neg_ratio × λ_link (2×4 grid) | λ_link=0 confirmed; hist=0.5 default | — |
| **Stage 5** — uniformity hyperparameter sweep | defaults win (Scenario A) | — |
| **Step 6** — single-table 1T_asym ablation | dual-table wins on cliff shape (pre-WD_link, pre-encoder) | — |
| **Step 7** — source walk encoder | ties locked-v2 on peak, smoother cliff | **0.7100** |
| **(this codebase)** | minimal locked production | **0.7100 wiki** |
| **Step 8** — single-table + dual-projection retest under locked-v2 stack | pre-registered 2026-05-21 (Lesson 27); result pending | — |

Final wiki single-seed: val 0.7423 / **test 0.7100** at ep 1 with the
minimal production architecture (smoke test 2026-05-21).

---

## What was stripped (`SKIP` per the cleanup)

Removed entirely from this production codebase:

- `align_weighting` B/C variants of the alignment loss (variant A always wins)
- `cross_table_dropout`, `link_mlp_n_layers > 3`, `link_mlp_dropout`,
  `embedding_dropout` (Stage 2 demonstrably hurt)
- `lambda_link` and joint-training code path (Lesson 19 — universal collapse)
- `primary_loss` ∈ {triplet, infonce, sgns} + their hyperparameters
  (Triplet decisively lost on review; alignment locked)
- `head_mode=component_0_only` (E.2 head — Phase S settled E.1 cross-table)
- `lambda_align=0` (A2-off ablation — Phase S settled A2-on)
- `single_table` mode (Step 6 cliff-shape regression)
- `freeze_tables` flag (Phase 1 sanity check — done, no longer needed)
- All sweep wrapper scripts (one-off experiment drivers, reproducible
  from this fixed train.py)

---

## Open / future directions

Not in this codebase but documented for future work:

1. **Direct (u,v) recurrence channel** (Lesson 14 — biggest unpulled
   lever, +0.051 historical). Port honestly into the link MLP input.
   Implementation: a `K_history=32` per-source ring buffer of recent
   destinations + recency-decayed similarity to v.

2. **Walks-only source representation.** Drop `E_target` entirely from
   the source side. Encoder source-side input = CAWN hitting-count
   anonymous identity + `E_context[walk_node]` + Φ(Δt) + edge_feats.
   Predicted to help on review (high surprise index) more than wiki.
   Lesson 26 motivation.

3. **Multi-scale time encoding.** Current Φ(Δt) uses a single
   `time_scale`. Multi-scale (short ~minutes + long ~days) might
   capture both within-session and cross-session patterns.

4. **Honest-protocol leaderboard baselines.** Re-run TPNet / DyGFormer
   / TGN under raw-message-store memory to compare apples-to-apples.
   May reveal that 0.82 is leak-inflated and honest ceiling on wiki is
   ~0.75.

5. **tgbl-coin / -comment / -flight scaling.** Edge-feat + node-feat
   support is already in. Per-dataset normbrake threshold needs
   calibration (1.5 × col_norm at ep 1-2).

---

## Branches in this repo

- `master` (THIS) — minimal production at `locked-v2` tag.
- `experiment/embedding-table-variations` — single-table 1T_asym
  ablation (Lesson 24, preserved for paper reproducibility).
- `experiment/add-source-walk-embedding` — Step 7 walk encoder history
  (with experimental knobs that have been minimized away here).
- `feature/walk-distribution-embedding` — experimental playground
  (Stages 1–5 history). Archived as
  `~/CLionProjects/tempest-walk-embedding-intermediate-archive-20260521.tar.gz`.

---

## Operating notes

- **Tempest CPU mode.** Walk generator runs on CPU regardless of model
  device. The arena would collide with PyTorch's allocator on 8 GB
  VRAM. Walks are deterministic regardless of device.
- **Default training negatives.** K=10, 50% historical reservoir + 50%
  uniform random over `train.destinations` (NOT full node set).
- **Default batch size.** 200 — empirically optimal for wiki (Lesson
  15: larger batches regress).
- **Default time scale.** Derived (`train_span / L_REF=20`). On wiki
  this is 93k seconds ≈ 1.08 days. Override via
  `Config.alignment_time_scale` for ablations.
- **Default `is_directed`.** Dataset-specific (heuristic in
  `data.py::_UNDIRECTED`). Pass `--is-directed` / `--no-is-directed`
  to override. On wiki, undirected matches TGB's eval semantics.
- **Eval `row_budget`** chunks the link-MLP forward at ~100K rows per
  pass to fit 8 GB. Math identical to unchunked path.
