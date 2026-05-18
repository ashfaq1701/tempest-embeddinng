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
see the lessons-learned section at the bottom.

---

## Strict-causal protocol (NON-NEGOTIABLE)

Every batch — at training AND at evaluation — runs in this exact order:

```
1.  walks   = walk_gen.walks_for_nodes(seeds = unique(batch.src))
        # Tempest state contains events strictly UP THROUGH batch B-1.
        # The current batch's edges have NOT been ingested yet.
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
4.  score   = link_predictor(E_target/E_context lookups for batch.src/tgt + negatives)
    (training) l_link = BCE(score, labels); link_optimizer.step()
    (eval)     evaluator.eval({y_pred_pos, y_pred_neg, eval_metric})
5.  POST-SCORING BLOCK (both updates feed batch B+1, not B):
        if HistoricalNegativeSampler: neg_sampler.observe(batch.src, batch.tgt)
        walk_gen.add_edges(batch.src, batch.tgt, batch.ts)   ← Tempest update is the LAST line
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
                           │ walks_for_nodes(seeds=batch.src)
                           ▼
              ┌────────────────────────┐
              │  per-walk: (nodes,     │  seed at nodes[lens-1]
              │   timestamps, lens,    │  chronological order
              │   edge_feats? unused)  │
              └──────────┬─────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
  alignment_loss                    uniformity_loss
  (pull E_target[seed] toward       (spread E_target on
   E_context[walk-neighbours])       unit hypersphere)
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
                 EmbeddingStore
                 (E_target, E_context  ∈  ℝ^[N, d])
                       │
                       ▼
              LinkPredictor MLP
   ┌─────────────────────────────────────────────┐
   │ in = concat([                               │
   │   E_target[u], E_target[v],                 │
   │   E_target[u]·E_target[v], |.−.|,           │
   │   E_context[u], E_context[v],               │
   │   E_context[u]·E_context[v], |.−.|,         │  # 8·d_emb blocks
   │ ])                                           │
   │ → LayerNorm → 2-layer GELU MLP → 1 logit    │
   └─────────────────────────────────────────────┘
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
edge features. Tempest does. The alignment loss already uses
per-hop timestamps via the
`(1 + Δt(c) / time_scale)^(−β)` temporal weight. Edge features
are not yet wired in (HISTORICAL row 10x noted they regressed on
wiki under the v2 setup, but tgbl-comment / coin / flight have
real signal in edges and the integration is on the roadmap).

---

## Baseline result (v3, strict-causal, walks-supervise-embeddings only)

**tgbl-wiki, 50 epochs, B=200, hist_neg_ratio=0.5, K=5 walks × L=20, d=128**

| | Val MRR | Test MRR |
|---|---|---|
| v3, additive node-feat residual + edge-feat alignment | 0.3725 | 0.2884 |
| v3, init-only node-feat + edge-feat alignment (current)  | 0.3629 | **0.2942** |

Training: 50 epochs × ~15.3s = ~13 min total. Tempest on CPU, model on
RTX 2000 Ada. Link BCE 0.27 → 0.124 over 50 epochs; alignment plateaus
at ~0.62 by epoch 5; uniformity stable around −3.89.

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

The v3 baseline is the honest walks-only number under the strict-
causal protocol. Memory, walks-at-scoring, edge features, and longer
walks are the next levers (see roadmap below).

## Roadmap (in priority order, after the v3 baseline lands)

1. **Reproduce 0.508 baseline test MRR on tgbl-wiki** under the
   strict-causal protocol. If MRR ≪ 0.508 then the v1 number was
   leak-inflated; report whatever the honest number is.
2. **Add edge features to alignment** (per-hop `edge_feats[w, k]`
   projected and concatenated into the context vector). Bypass the
   wiki regression by adding a `--use-edge-feats` flag.
3. **Add a TGN-style memory module with raw-message-store** (the
   honest version of Lesson 1). Combine with walk-supervised
   embeddings.
4. **Scale-out to tgbl-coin / -comment / -flight / -review** with
   `wpn=10, mwl=80` (Tempest paper conventions).
5. **Honest-protocol baselines for TGN / DyGFormer / TPNet** —
   re-run them with raw-message-store memory and report against
   our numbers. This is the paper's contribution-defining
   comparison.

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
- Default `target_batch_size = 200`. Larger batches dilute the
  per-positive gradient and let the model collapse to a constant
  logit (observed empirically in v2 — fix was small batches).
- Eval `row_budget` chunks the link-MLP forward at ~500K rows per
  pass to fit 8 GB. Math is identical to the unchunked path.
