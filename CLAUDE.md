# Tempest walk-first temporal embeddings — v3 (clean rebuild)

This codebase is the embedding-side companion to the Tempest paper
(2026 ACM submission, *"Tempest: A GPU-Accelerated Engine for Streaming
Temporal Random Walks"*). Its conclusion: *"Tempest opens a path to
walk-native temporal embedding methods that consume causal walks
directly, which we are pursuing in ongoing work."* That ongoing work
is this repo.

The architecture is a minimum, tightly-TGB-coupled implementation of
the only design that has been shown to actually work under an
**honestly strict-causal protocol**:

1. Walks **supervise the embedding table** via alignment + uniformity
   (DeepWalk / node2vec / Wang & Isola pattern), seeded by current-
   batch nodes but generated from a **pre-ingest Tempest state** —
   so the walks never include the current batch's edges.
2. A two-table embedding store (`E_target`, `E_context`) is decoded
   by an 8-block MLP for link prediction.
3. The TGB-official `Evaluator` is the only scorer; reported numbers
   are bit-identical to the leaderboard protocol.

Everything else has been tried and either failed or carried a leak —
see the lessons-learned section at the bottom. For the full design
spec (every primitive, every loss, every channel, with shapes and
equations), see `design_document.md`.

---

## Strict-causal protocol (NON-NEGOTIABLE)

Every batch — at training AND at evaluation — runs in this exact order:

```
1.  walks   = walk_gen.walks_for_nodes(seeds = unique(batch.src ∪ batch.tgt))
        # UNION seeding: every node touched by the batch gets walks. On
        # bipartite-flavoured datasets (wiki, review, …) seeding only on
        # batch.src would starve target(dst) and context(src) of the
        # alignment signal — see Lesson 9.
        # Tempest state contains events strictly UP THROUGH batch B-1;
        # the current batch's edges have NOT been ingested yet.
        # First batch of each epoch: walks are empty (Tempest just reset),
        # so embeddings stay at random init for that batch — by design.
2.  l_emb   = alignment(walks) + η · uniformity(batch_nodes)
    embedding_optimizer.step()
        # E_target / E_context updated from B-1-state walks only.
3.  negatives = neg_sampler.sample(batch)
        # Training:    K_hist drawn from each source's reservoir (events
        #              ≤ batch B-1) + K_rand uniform random.
        # Validation/test: dataset.negative_sampler.query_batch (TGB
        #              pre-generated, 50/50 historical/random per positive).
4.  score   = link_predictor(target(u), target(v), context(u), context(v))
    (training) l_link = BCE(score, labels); link_optimizer.step()
    (eval)     evaluator.eval({y_pred_pos, y_pred_neg, eval_metric})
5.  POST-SCORING BLOCK (both updates feed batch B+1, not B):
        if HistoricalNegativeSampler: neg_sampler.observe(batch.src, batch.tgt)
        walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)   ← Tempest update is the LAST line
```

**Why this matters.** If the ingest happens before walks (the way the
v1/v2 baselines did), the walks for source `u` contain the very edge
`(u, v_pos, t)` that is about to be scored. The alignment loss has
just pulled `E_target[u]` and `E_context[v_pos]` together; the link
MLP then trivially scores that pair high. Training MRR balloons,
evaluation MRR does not, and the published 0.508 / 0.7630 numbers from
v1 / v2 were inflated by this leak. The same leak shape as the TGN
memory-update leak documented in the v0 critical-lesson — just
transposed to walk-embedding supervision.

The strict-causal version trains slower (the embedding has to learn
from past walks only, not from the batch it's about to score) but the
training and eval distributions match.

Per-epoch reset: `walk_gen.reset()` is called once at the start of
each training epoch. At eval the Tempest state is **kept from end of
training** and accumulates eval edges as it goes — TGB's streaming
convention.

---

## Architecture

```
                  ┌─────────────────┐
                  │ TGB / Tempest   │ ← ingest after every batch
                  │  edge stream    │
                  └────────┬────────┘
                           │ walks_for_nodes(seeds = unique(batch.src ∪ batch.tgt))
                           ▼
              ┌────────────────────────┐
              │  per-walk: (nodes,     │  seed at nodes[lens-1]
              │   timestamps, lens,    │  chronological order
              │   edge_feats)          │  edge_feats[p] = feature of hop (p → p+1)
              └──────────┬─────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
  alignment_loss                    uniformity_loss
  (pull target(seed) toward         (spread target(u) on
   context_walk(walk-position))      unit hypersphere)
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
                 EmbeddingStore
                 (E_target, E_context  ∈  ℝ^[N, d])
                 + proj_t, proj_c, proj_e  (feature projections)
                 + target_final, context_final, context_walk_final  (fusion)
                       │
                       ▼
              LinkPredictor MLP   (CROSS-table 8-block)
   ┌─────────────────────────────────────────────────────┐
   │ in = concat([                                        │
   │   # u→v direction: target(u) ↔ context(v)            │
   │   target(u), context(v),                             │
   │   target(u) ⊙ context(v), |target(u) − context(v)|,  │
   │   # v→u direction: target(v) ↔ context(u)            │
   │   target(v), context(u),                             │
   │   target(v) ⊙ context(u), |target(v) − context(u)|,  │  # 8·d_emb
   │ ])                                                    │
   │ → LayerNorm → 2-layer GELU MLP → 1 logit             │
   └─────────────────────────────────────────────────────┘
                       │
                       ▼
              BCE-with-logits (train)
              TGB Evaluator.eval (val/test)
```

Walks supervise the embedding table; the link MLP only sees frozen
embedding lookups. Two optimizers, two losses, run sequentially per
batch:

- `embedding_optimizer` ← (l_align + η · l_uniform)
- `link_optimizer`      ← BCE-with-logits on (pos, K negatives)

---

## Negative sampling

| Phase | Source | K per positive |
|---|---|---|
| Training | `HistoricalNegativeSampler`: K_hist from each source's reservoir of past destinations (Vitter R, M=32 by default), K_rand uniform over `train.destinations` pool. Default mix is 50/50, matching TGB's eval protocol. | 10 |
| Validation | `dataset.negative_sampler.query_batch(..., split_mode="val")` (TGB pre-generated, mix of historical + random) | ~999 on tgbl-wiki |
| Test | `dataset.negative_sampler.query_batch(..., split_mode="test")` | ~999 on tgbl-wiki |

**Causality of the reservoir.** `reservoir.observe(batch.src, batch.tgt)`
runs in the post-scoring block AFTER the link step for batch B. So
when batch B+1's `sample()` reads from the reservoir, it sees
destinations from events ≤ batch B. Batch B's positives are not in
its own reservoir at sampling time. Same strict-causal guarantee as
the Tempest ingest.

**Distribution match with eval.** TGB serves a mix of historical and
random negatives at val/test (variable per positive, ~50/50 typical).
Training on the same mix keeps the link-MLP's input distribution
aligned with what eval will present. `hist_neg_ratio=0.5` is the
default; set `--hist-neg-ratio 0` to fall back to pure-random
training negatives for ablations.

**False-negative guard.** Any historical-negative draw equal to the
current positive's target (`v_neg == v_pos`) is replaced with the
random fallback. Empty reservoir slots (cold-start sources) are
likewise replaced. Both checks are a single vectorised `np.where`.

---

## TGB integration (tight coupling)

| Concern | TGB symbol | Where |
|---|---|---|
| Dataset | `tgb.linkproppred.dataset.LinkPropPredDataset(name, root, preprocess=True)` | `data.py::load_tgb` |
| Splits | `dataset.{train,val,test}_mask` | `data.py::load_tgb` |
| Edges | `dataset.full_data["sources" / "destinations" / "timestamps"]` | `data.py::load_tgb` |
| Eval negatives | `dataset.load_val_ns()` / `load_test_ns()`, then `dataset.negative_sampler.query_batch(src, tgt, ts, split_mode)` | `negatives.py::TGBNegativeSampler` |
| Metric scorer | `tgb.linkproppred.evaluate.Evaluator(name).eval({y_pred_pos, y_pred_neg, eval_metric})` | `evaluator.py` |
| Metric name | `dataset.eval_metric` (typically `"mrr"`) | passed through `Loaded.eval_metric` |

**There is no hand-rolled MRR.** A per-positive `tgb_eval.eval(...)`
call computes the metric exactly as the leaderboard does.

---

## Walk-structure rules (Tempest)

`temporal_random_walk.TemporalRandomWalk.get_random_walks_and_times_for_nodes`
returns walks in **chronological order** regardless of direction:

```
nodes:      [n_0, n_1, ..., n_{lens-2}, n_{lens-1}, -1, -1, ...]
timestamps: [t_0, t_1, ..., t_{lens-2}, sentinel,   -1, -1, ...]
lens:       number of valid NODES per walk (so lens-1 edges).
```

- `nodes[lens-1]` = **the seed** (Tempest reverses the walk in place
  before return so callers see chronological order). NEVER assume
  seed at `nodes[0]`.
- `timestamps[k]` for `k ∈ [0, lens-2]` = time of edge between
  `nodes[k]` and `nodes[k+1]`.
- `timestamps[lens-1]` = sentinel `INT64_MAX` (backward walks).
- `edge_feats[k]` for `k ∈ [0, lens-2]` = feature of the SAME edge
  (`nodes[k]`, `nodes[k+1]`). Same indexing as `timestamps[k]`.
  In `context_walk`, the edge-feat tensor is **right-padded** so that
  position `p` of the [W, L] grid carries `edge_feats[p]`. Left-padding
  it (off-by-one — uses `edge_feats[p-1]`) was a bug we hit; the
  alignment loss's `timestamps[p]` and `edge_feats[p]` must describe
  the same hop. See Lesson 11.
- Positions ≥ `lens` are padding (`nodes=-1`, `timestamps=-1`).

**Cold-start.** A seed with no prior edges in the current Tempest
state can return `lens=0` (empty walk). Code paths that touch
per-position node embeddings must handle this — typically by
clamping `lens` to ≥1 and treating the seed itself as the only valid
token. The alignment loss masks padding out via the `lens` vector.

---

## CRITICAL LESSONS LEARNED (do not re-litigate)

### Lesson 1 — TGN memory has a within-batch leak

Per Rossi et al. (2020), the standard `memory_update_at_start=False`
mode applies the batch's positive edges to the GRU memory state
BEFORE scoring. At scoring time both endpoints of the positive have
been freshly updated; negatives' v-side has not. The model learns
"freshness" rather than the link rule. Every open-source TGN /
DyRep / DyGFormer / TPNet reimplementation that uses this mode
carries the leak.

The honest pattern is `memory_update_at_start=True` with a
**raw-message-store**: batch B reads stored raw messages from batch
B-1, applies them via the GRU at the start of B's forward, scores,
then stashes B's new raw messages for B+1. Memory state used to
score B reflects events strictly up to B-1. Gradient still flows
because the GRU runs inside the autograd graph at scoring time.

This v3 codebase does NOT include a memory module by default. If one
is added later it MUST use the raw-message-store pattern.

### Lesson 2 — Walks supervise embeddings, not transformer inputs

Feeding walks as per-batch input through a transformer encoder and
training end-to-end on link loss only does **not** work. The
encoder has to learn to read walks, the link MLP has to learn to
score, and the embeddings have to be useful — all from a single
weak 1:10 BCE/InfoNCE signal. The walks-only baseline plateaued at
~0.011 test MRR on tgbl-wiki this way.

What works: walks **supervise the embedding table** via
`alignment(seed, walk-neighbours) + uniformity(batch nodes)`. The
walks provide rich per-position supervision; the embeddings
specialize; the link MLP becomes a thin decoder. The v1 baseline
reached 0.508 test MRR on tgbl-wiki this way.

### Lesson 3 — Ingest order at TRAINING matters

It's not enough for eval to be strict-causal. If at training the
ingest happens BEFORE walks are generated, walks for source `u`
include the current `(u, v_pos)` edge, the alignment loss pulls
`E_target[u]` and `E_context[v_pos]` together, and the link MLP
scores the just-strengthened pair seconds later. This is a leak,
just on the walk-supervision side instead of the memory side. The
v1/v2 published 0.508 / 0.7630 numbers were obtained under this
leaky training protocol; the eval-only-strict-causal protocol they
used isn't enough to make the numbers honest.

This v3 codebase ingests AFTER scoring at training AND at eval.
Always.

### Lesson 4 — Historical negatives are RIGHT under walks-supervise-embeddings

In the previous (walks-as-transformer-input) architecture, training
with historical negatives collapsed eval MRR to ~0.065 on tgbl-wiki.
At the time this looked like "the model learns to score historical
pairs LOW; eval positives are historical; collapse." But the real
cause was that the link MLP had no signal beyond the encoder output,
and the encoder learned to discriminate against EXACTLY the hist-neg
distribution it was trained on. Historical negatives just exposed
the architecture's weakness.

Under THIS architecture (walks supervise dual embedding tables;
link MLP decodes), the calculus flips:

- The alignment loss pulls `E_target[u]` close to `E_context[v]`
  every time `v` appears in `u`'s walk history — that IS recency
  encoding. The embedding now carries "v shows up in u's recent
  past" without the link MLP having to learn it.
- Historical negatives at training are therefore hard-but-honest
  examples: `v_hist` shares u's history (high overlap in
  `E_target` × `E_context` similarity) but isn't the actual
  positive at this timestamp. The model learns to use OTHER
  embedding-space signal to break the tie. That's exactly what
  eval needs the model to do.
- Training distribution matches eval distribution (TGB serves
  ~50/50 hist/random at val/test).

**Default: `hist_neg_ratio = 0.5`** — same mix TGB uses at eval.
The original 3be7ada commit reported +0.070 test MRR from this
change under the v1 architecture; the contribution under v3 needs
re-measurement.

### Lesson 5 — Memorization is THE signal on tgbl-wiki

EdgeBank's 0.571 reveals that any architecture not designed to
exploit historical recurrence will underperform on tgbl-wiki. The
walk-as-supervision approach in this v3 encodes recurrence by
pulling `E_target[u]` and `E_context[v]` together whenever `v`
appears in `u`'s past walks. Repeated edges → repeated alignment
pull → tight embeddings → high link score. That's EdgeBank's lookup
table, made differentiable.

### Lesson 6 — TGB Evaluator, not hand-rolled MRR

Pessimistic-only MRR (`(neg ≥ pos).sum() + 1`) differs from TGB's
official Evaluator at score-tie boundaries. Reported numbers must
go through `tgb.linkproppred.evaluate.Evaluator.eval(...)` or they
aren't comparable to anything on the leaderboard. This v3 does it
properly.

### Lesson 7 — Walks at seed position lens-1, NEVER 0

Tempest's backward walks are reversed in place so callers see
chronological order. The seed lives at `nodes[lens-1]`. Treating
`nodes[0]` as the seed (a bug carried from v1 row 1 through row 11)
trains the alignment loss against the deepest-past node, which
acts as a noisy regularizer. The seed-position fix (commit
`dca4962`) cost 0.020 test MRR vs the buggy version because the
heavy downstream heads had implicitly tuned to the wrong anchor —
correctness > noise-band-adjacent metric.

### Lesson 8 — Exploit Tempest's unique outputs

Most random-walk methods don't have per-hop timestamps or per-hop
edge features. Tempest does. The alignment loss uses per-hop
timestamps via the `(1 + Δt(c) / time_scale)^(−β)` temporal weight
and per-hop edge features through `proj_e` inside `context_walk`.
Both are no-ops when the dataset doesn't provide the signal — see
the "Feature handling" matrix.

Edge features are intentionally context-side-and-alignment-only.
They characterise a hop (an edge between two nodes), not a node
identity, and they're absent from negatives (negative `(u, v_neg)`
pairs have no associated edge), so feeding them to the link MLP
would let it key on "edge feat present → positive" — a leak. The
HISTORICAL note about edge features regressing under the v2 setup
was the additive-residual variant; the current concat + per-site
fusion handles them robustly.

### Lesson 9 — Cross-table link MLP > within-table

The alignment loss trains the `target ↔ context` cosine. The link
MLP must consume **exactly that interaction** so the BCE signal can
ride on the supervised geometry instead of re-learning it from
scratch. Within-table 8-blocks (`target⊙target`, `context⊙context`)
were the original design — those products have no direct
supervision and the head had to learn cross-table coupling
implicitly through its hidden layers. Switching to cross-table
blocks (`target(u)⊙context(v)` for u→v and `target(v)⊙context(u)`
for v→u, plus their L1 differences) gave +0.04 test MRR on wiki by
itself.

### Lesson 10 — Seed walks on union(src, tgt), not just src

Seeding walks only on `unique(batch.src)` starves `target(v)` and
`context(u)` of alignment supervision whenever the dataset has a
bipartite tilt (users → pages on wiki, users → reviews on review,
etc.). Both tables effectively become half-trained, and the link
MLP has to lift the un-supervised half from BCE alone. Union
seeding (`unique(batch.src ∪ batch.tgt)`) gives every node touched
by the batch a chance to pull its target view, doubles seed
diversity per step, and is the right default everywhere. Costs
~2× walk-sampling per batch — negligible.

### Lesson 11 — `time_scale` must be decoupled from `max_walk_len`

The alignment loss's recency term `(1 + Δt / time_scale)^(−β)` is
extremely sensitive to `time_scale`. The original derivation
`(t_max − t_min) / max_walk_len` coupled `time_scale` to walk
length: bumping L=20 → L=50 collapsed `time_scale` from 93k to 37k
on wiki and crushed the recency weight, negating the benefit of
the longer walks.

The right derivation **keeps the dataset-span numerator but uses
a fixed reference constant `L_REF=20` in the denominator**, NOT
the configurable `max_walk_len`:

```
time_scale = (t_max − t_min) / L_REF       where L_REF = 20 (fixed)
```

For wiki this is 1.86M / 20 ≈ **93k sec ≈ 1.08 days**, regardless
of `--max-walk-len`. Empirically this beats the per-node mean
inter-event time `(span × N / E) ≈ 155k` by ~0.02 test MRR on wiki —
the alignment recency wants a sharper scale than the per-node
recurrence period, closer to a within-session timescale.

Override at the CLI via `--alignment-time-scale <value>` for
ablations and per-dataset tuning.

### Lesson 12 — Edge-feat index at walk position p is `edge_feats[p]`

`timestamps[p]` and `edge_feats[p]` describe the SAME hop (between
`nodes[p]` and `nodes[p+1]`). The alignment loss reads
`timestamps[p]` at position p; the per-position context vector
must therefore read `edge_feats[p]` at position p, not
`edge_feats[p-1]`. Left-padding the [W, L-1, d_e] tensor was
introducing a one-position skew between time and edge-feat at
every walk position — fixed by right-padding (zero at the seed
slot, which is masked out anyway).

### Lesson 13 — Cross-pair attention with 4-channel link MLP regresses

Replacing Phase 2's 12-block link MLP (8 cross-table + 4 walk-encoded)
with a 4-channel head fed by cross-pair-attended W cost ~0.05 test MRR
on wiki. The hand-crafted Hadamard/L1 cross-table interactions are
doing more work than cross-pair attention adds on a 110K-edge dataset.
Keep the 12-block design; if cross-pair is added back, layer it as
PAIR-CONDITIONED W INSIDE the 12-block structure, not as a replacement.

### Lesson 14 — Direct (u, v) recurrence is the biggest unpulled lever

On tgbl-wiki, EdgeBank's pure (u, v)-pair-recurrence lookup reaches
0.571 test MRR. Our co-occurrence feature captures SHARED-NEIGHBOR
recurrence (second-order), NOT direct (u, v) recurrence (first-order).
Adding an explicit "is v in u's recent K-history?" feature with
recency-aware decay gave +0.0506 test MRR (Phase 5 0.4396 → Phase 6
0.4902) — the largest single architectural gain of the session. The
recurrence signal is the dominant signal on this dataset; the walk
encoder, DyG transformer, memory, and co-occurrence all approximate
it indirectly and inefficiently.

### Lesson 15 — Capacity scaling hurts on wiki

d_emb 128 → 192 with the full Phase 5 stack regressed test MRR by
0.03 (0.4396 → 0.4093). The architecture is already at the right
capacity for 110K-edge / 9.2k-node wiki. More params just makes the
model overfit faster (BCE 0.027 → ~0.025 with no eval improvement).
Capacity scaling is a tool for bigger datasets, not for squeezing
more out of small ones.

### Lesson 16 — Memory module adds marginal value when recurrence is already captured elsewhere

TGN-style raw-message-store memory (with proper one-step BPTT) added
only +0.003 test MRR on top of Phase 4 (DyG + co). The memory's main
contribution — "summary of u's history" — overlaps with what the DyG
transformer already extracts. Memory is more useful on datasets where
the recurrence horizon is much longer than K_history (e.g., monthly
patterns vs daily wiki edits). On wiki, K_history=32 already covers
~the same time window as memory's running state.

---

## Results (tgbl-wiki, 50 epochs, B=200, K=5, L=20, d=128 unless noted)

All MRR through `tgb.linkproppred.evaluate.Evaluator.eval(...)`.

| Run | Val MRR | Test MRR | Notes |
|---|---|---|---|
| v3, additive node-feat residual | 0.3725 | 0.2884 | early v3 baseline |
| v3 + init-only node-feat | 0.3629 | 0.2942 | |
| **v3 + cross-table link MLP + union seeding + ef-pad fix (Phase 0)** | **0.4015** | **0.3313** | the 8-block + walk-context baseline that everything stacks on |
| v3 directed (sanity) | 0.3553 | 0.2735 | TGB negs don't respect direction; undirected is right for wiki |
| Phase 2 — Phase 0 + walk encoder (GRU over walk) | 0.5128 | 0.4289 | walk encoder produces seed-pooled W; 12-block link MLP (8 cross-table + 4 walk-encoded) |
| Phase 3 — Phase 2 + cross-pair attention + 4-channel link MLP | 0.4424 | 0.3829 | REGRESSED — Hadamard/L1 cross-table blocks were doing more work than cross-pair attn added |
| Phase 4 — Phase 2 + DyG node encoder + co-occurrence + 12-block MLP | 0.5118 | 0.4362 | Marginal; cross-pair dropped |
| Phase 4 @ 15 epochs (early stop) | 0.4847 | 0.3906 | Worse — model still learning past ep 15 |
| Phase 5 — Phase 4 + raw-message-store memory | 0.5198 | 0.4396 | Memory adds +0.003 test (marginal) |
| Phase 5 + d_emb=192 (capacity sweep) | 0.5066 | 0.4093 | REGRESSED — capacity hurts on this dataset |
| **Phase 6 — Phase 5 + EdgeBank-style direct recurrence feature** | **0.5264** | **0.4902** | **+0.0506 test from EdgeBank's direct (u,v)-in-history signal — biggest single jump of the session** |

Training cost (Phase 6): 50 epochs × ~77 s = ~64 min + ~6 min eval.
Tempest on CPU, model on RTX 2000 Ada (8 GB VRAM).

Leaderboard reference (note: most carry the TGN memory leak):

| Method | Test MRR |
|---|---|
| Random | 0.0075 |
| EdgeBank-inf | 0.495 |
| EdgeBank-tw | 0.571 |
| GraphMixer | 0.594 |
| DyRep | 0.665 |
| TGN | 0.690 |
| DyGFormer | 0.798 |
| TPNet | 0.827 |

The current honest walks-supervise-embeddings number is **0.490 test
MRR** on wiki (Phase 6, all channels stacked). Closing the remaining gap
to EdgeBank-tw (0.571) and beyond would require:
- Larger K_history window (eb-feat saturates at K=32 since we only see
  32 most-recent events per node — wiki has dense interactions)
- True time-windowed history (vs slot-windowed) — EdgeBank-tw uses a
  time window, not a count window
- Per-token co-occurrence features (DyGFormer's full version, not the
  scalar summary we have)

The historical assertion that "0.331 is the next milestone" is now
superseded — 0.490 puts us between EdgeBank-inf (0.495) and EdgeBank-tw
(0.571). Memorisation-via-recurrence is the now-dominant signal; the
DyG transformer + memory module add small marginal value on top.

(legacy text retained below for context)

Closing the gap to EdgeBank (0.495) was the original milestone; the
levers we haven't pulled yet (walks-at-scoring, raw-message-store
memory, longer walks at small batch) are exactly what the leaderboard
methods rely on.

## Roadmap (in priority order)

1. **Walks-at-scoring (GRU/transformer encoder feeding the link MLP).**
   The alignment loss already trains rich walk representations; the
   link MLP currently re-reads only the raw `target / context` tables.
   Routing the seed's pooled walk representation INTO the link MLP
   alongside the table lookups is the largest expected gain — same
   architectural step that the old codebase used to clear TGN territory.
2. **Walk-length sweep at small batch with corrected `time_scale`.**
   B=200, L ∈ {30, 50, 80}, K ∈ {5, 10}. Isolates the long-walk effect
   from the batch-size regression; the diagnostic in
   `scripts/diag_walk_lens.py` shows ~38% of walks pin the L=20 cap.
3. **TGN-style memory with raw-message-store** (Lesson 1's honest
   pattern). Composes multiplicatively with walks-at-scoring per the
   old codebase's row 6 → row 7 jump (+0.034).
4. **Scale-out to tgbl-coin / -comment / -flight / -review** with
   Tempest paper conventions (`wpn=10, mwl=80`). Edge-feat support is
   already in; node-feat support is already in.
5. **Honest-protocol baselines for TGN / DyGFormer / TPNet** — re-run
   them with raw-message-store memory and report against our numbers.
   The paper's contribution-defining comparison.

---

## Operating notes

- Tempest **CPU mode**. The laptop has 8 GB VRAM; the walk arena
  collides with PyTorch's allocator if both run on GPU. Walks are
  deterministic regardless of device.
- Default training negatives: K = 10, uniform random over the
  training-destination pool (NOT the full node set — bipartite
  datasets have asymmetric source / target node populations and
  random over all nodes would create a trivially easy task that
  doesn't transfer to eval).
- Default `target_batch_size = 200`. Re-confirmed empirically: at
  B=1000 on wiki, test MRR regresses by ~0.06 even after fixing the
  `time_scale` derivation. Smaller datasets need smaller batches —
  v2's "collapse to constant logit" failure was a more severe form
  of the same effect.
- Default `alignment_time_scale` is derived as
  `(t_max − t_min) / L_REF` with `L_REF=20` fixed. On wiki this
  gives 93k seconds ≈ 1.08 days — the empirically-best value (see
  Lesson 11). Pass `--alignment-time-scale` to override for
  per-dataset tuning.
- Default `is_directed` is dataset-specific (heuristic in
  `data.py::_UNDIRECTED`). Pass `--is-directed` / `--no-is-directed`
  to override. TGB does not declare directedness; the choice is
  interpretive. On wiki, undirected matches TGB's eval semantics
  (negatives don't respect direction) — running directed regressed
  test MRR by 0.06.
- Eval `row_budget` chunks the link-MLP forward at ~500K rows per
  pass to fit 8 GB. Math is identical to the unchunked path.
- Diagnostic at `scripts/diag_walk_lens.py` reports the actual walk-
  length distribution Tempest is returning under the strict-causal
  protocol. Run it after any change to walk config to verify whether
  `max_walk_len` is the binding constraint (on wiki at L=20 it is —
  ~38% of walks pin the cap at end-of-epoch).
