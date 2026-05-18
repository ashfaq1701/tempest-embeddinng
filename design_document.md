# Tempest Walk-Supervised Temporal Embeddings — Design Document

## 1. Goal

Build a **walk-first temporal-embedding method** for TGB link-prediction
benchmarks that uses Tempest-generated causal random walks to supervise
two embedding tables, then scores link prediction with a thin MLP head
on top of those tables. The design is the embedding-side companion to
the Tempest paper (2026 ACM submission), whose conclusion is:

> "Tempest opens a path to walk-native temporal embedding methods that
> consume causal walks directly, which we are pursuing in ongoing work."

This codebase is that ongoing work.

The whole architecture is built around one non-negotiable principle —
**strict causality**: nothing the model uses to score a positive edge
has seen that edge or any other edge in the same batch. Both the walks
and the negative samples for batch B come from a state that contains
events strictly through batch B−1.

## 2. Non-negotiable: the strict-causal protocol

Every batch — training AND evaluation — runs in this exact order:

```
1.  seeds   ← unique(batch.src ∪ batch.tgt)
2.  walks   ← walk_gen.walks_for_nodes(seeds)         (PRE-ingest Tempest)
3.  L_emb   ← L_align(walks) + η · L_uniform(unique batch nodes)
            embedding_optimizer.step()
4.  negs    ← neg_sampler.sample(batch)               (PRE-batch reservoir)
5.  L_link  ← BCE-with-logits(score(positives, negatives), labels)
            link_optimizer.step()
6.  reservoir.observe(batch.src, batch.tgt)           (for batch B+1)
7.  walk_gen.add_edges(batch.src, batch.tgt, ts, ef)  (for batch B+1)  ← LAST
```

If steps 6 and 7 happen before step 5, the walks for source `u` already
contain the edge `(u, v_pos, t)` we're about to score, the alignment
loss has pulled `target(u)` and `context(v_pos)` together, and the link
MLP trivially scores the just-strengthened pair high. Training MRR
inflates, eval MRR doesn't — the classic walk-side analog of the TGN
memory-update leak. **This codebase ingests AFTER scoring.** Every
prior baseline that claimed leak-free behaviour while ingesting first
was wrong; the strict-causal protocol is what makes our numbers
honest.

Per-epoch reset: `walk_gen.reset()` is called once at the start of
each training epoch. At eval the Tempest state carries through from
the end of training and accumulates eval edges as scoring proceeds —
this is the TGB-standard streaming convention.

## 3. Architecture overview

```
                          ┌──────────────────┐
                          │ TGB / Tempest    │ ← ingest after every batch
                          │  edge stream     │
                          └────────┬─────────┘
                                   │ walks_for_nodes(unique(src ∪ tgt))
                                   ▼
                       ┌───────────────────────┐
                       │ WalkData             │
                       │   nodes      [W, L]   │  seed at nodes[lens-1]
                       │   timestamps [W, L]   │  chronological
                       │   edge_feats [W,L-1]  │  per-hop
                       │   lens       [W]      │
                       └────────┬──────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
     alignment_loss                       uniformity_loss
     (pull target(seed)                   (spread target(u)
      ↔ context_walk(p))                   on unit hypersphere)
                │                               │
                └───────────────┬───────────────┘
                                ▼
                        EmbeddingStore
                        (E_target, E_context  ∈  ℝ^[N, d])
                        + node-feat projections (proj_t, proj_c)
                        + edge-feat projection  (proj_e)
                        + per-site fusion Linears
                                │
                                ▼
                       LinkPredictor (8-block MLP, CROSS-table)
                                │
                                ▼
                       BCE-with-logits (train)
                       TGB Evaluator.eval     (val/test)
```

Two optimizers run sequentially per batch:

- `embedding_optimizer` ← `L_align + η · L_uniform`
- `link_optimizer`      ← BCE-with-logits over (pos, K negatives)

## 4. Walk structure (Tempest convention)

`temporal_random_walk.get_random_walks_and_times_for_nodes` returns
walks in **chronological order** regardless of `walk_direction`:

```
nodes:       [n_0, n_1, ..., n_{lens-2}, n_{lens-1}, -1, -1, ...]
timestamps:  [t_0, t_1, ..., t_{lens-2},  sentinel,  -1, -1, ...]
edge_feats:  [eps_0, eps_1, ..., eps_{lens-2}]   (last entry is the hop INTO the seed)
lens:        number of valid NODES (so lens-1 hops)
```

For backward walks (the only kind we use):

- `nodes[lens-1]` = **the seed**. Tempest reverses the walk in place
  before returning. Treating `nodes[0]` as the seed is the v1 bug —
  documented in our `walks.py` header and confirmed to cost ~0.02 test
  MRR when present.
- `nodes[0]` = the oldest reachable past neighbour along the walk.
- `timestamps[p]` for p in [0, lens-2] = time of edge between
  `nodes[p]` and `nodes[p+1]`.
- `timestamps[lens-1]` = `INT64_MAX` sentinel (no out-edge from the seed
  in this direction).
- `edge_feats[p]` for p in [0, lens-2] = feature of the same edge
  (`nodes[p]`, `nodes[p+1]`). **Critical**: edge-feat at walk position
  `p` is `edge_feats[p]`, not `edge_feats[p-1]` — the indexing must
  match `timestamps[p]`. The implementation right-pads the edge-feat
  tensor at the seed slot.
- Positions ≥ lens are padding (`nodes = -1`, `timestamps = -1`).

## 5. Embedding store

Two identity tables, optional feature residuals, per-role learned fusion.

### 5.1 Tables (always)

```
E_target  ∈ ℝ^{N × d}
E_context ∈ ℝ^{N × d}
```

Both Xavier-uniform init. **No feature-based init** — that would freeze
node features at construction time and break streaming-feature datasets.
Features flow in as learned residuals at every lookup instead.

### 5.2 Role asymmetry (the design principle)

- `target(u)` is the canonical lookup for "u as the present focal point —
  the node from which the model is predicting forward".
- `context(u)` is the canonical lookup for "u as someone that has shown
  up in another node's recent past".

The two tables earn their keep only if each receives genuinely different
gradient signal. The alignment loss is the **only bridge** between them:

- `E_target` is touched ONLY as the seed end of alignment (and
  uniformity, and target-side link blocks).
- `E_context` is touched ONLY at walk-internal positions of alignment
  (and context-side link blocks).
- No walk-internal position is allowed to pull `E_target[u]` toward
  another `E_target[s]` — that would blur the role split and re-derive
  a one-table model with extra parameters.

### 5.3 Per-feature projections (only when feature is present)

```
proj_t : ℝ^{d_n} → ℝ^d        (node feat → target role)
proj_c : ℝ^{d_n} → ℝ^d        (node feat → context role)
proj_e : ℝ^{d_e} → ℝ^d        (edge feat)
```

`proj_t` and `proj_c` are distinct Linears: same input but two role-
specific projections, each supervised through its respective side of
the alignment loss.

### 5.4 Per-role fusion (only when its channel is present)

These are the learned mixers at sites where the identity table is
concatenated with a projected feature:

```
target_final       : Linear(2d, d)   — fuses E_target ‖ proj_t(nf)     (if nf)
context_final      : Linear(2d, d)   — fuses E_context ‖ proj_c(nf)    (if nf)
context_walk_final : Linear(2d, d)   — fuses context(u) ‖ proj_e(eps)  (if ef)
```

Concat-and-project (rather than addition) is used because identity
tables and projected features have different magnitudes; an additive
residual is scale-fragile. The Linear learns per-dimension mixing
weights and can zero a channel out if it's noise. Each fusion site has
its own weights — no parameter sharing across roles, since each role's
gradient signal is structurally different.

### 5.5 Lookup primitives

```
target(u):
  no nf:    E_target[u]
  with nf:  target_final([E_target[u] ‖ proj_t(nf[u])])

context(u):
  no nf:    E_context[u]
  with nf:  context_final([E_context[u] ‖ proj_c(nf[u])])

context_walk(u, eps):
  no ef:    context(u)
  with ef:  context_walk_final([context(u) ‖ proj_e(eps)])
```

All three return ℝ^d.

### 5.6 Walk-position vector

For walk `w` at position `p ∈ {0, ..., lens_w − 2}` (non-seed):

```
z_{w,p} = context_walk(walk_nodes[w, p], edge_feats[w, p])
```

The edge index is `p`, the same index `timestamps[p]` uses. Both index
the hop OUT of position `p` toward position `p+1`. The seed position
gets a zero-padded edge slot (right-padded in the implementation), which
the alignment loss masks out anyway.

## 6. Alignment loss

Pulls `target(seed_w)` toward `z_{w,p}` for every valid non-seed walk
position, weighted by hop distance and temporal recency.

```
q_w        = target(seed_w)                   # left operand
z_{w,p}    = context_walk(u_{w,p}, eps_{w,p}) # right operand

K_{w,p}    = lens_w - 1 - p                   # hop distance to seed
dt_{w,p}   = max(t_query - timestamps[w, p], 0)
alpha_{w,p} = (1 / K_{w,p}) × (1 + dt_{w,p} / time_scale)^(-beta)

cos_{w,p}  = (q_w · z_{w,p}) / (‖q_w‖ · ‖z_{w,p}‖)

L_align    = sum_{(w,p) valid, p<lens_w-1}  alpha_{w,p} · (1 - cos_{w,p})
             / sum_{(w,p) valid, p<lens_w-1}  alpha_{w,p}
```

The valid mask excludes (a) padding positions (p ≥ lens_w) and (b) the
seed position (p = lens_w − 1), where the cosine would be a self-pull.

### 6.1 Per-hop weighting

The weight is a product of two terms:

- **Distance**: `1/K` — older walk positions contribute less. The
  closest non-seed position (K=1) gets weight 1; the deepest (K=L−1)
  gets weight 1/(L−1).
- **Recency**: `(1 + Δt / time_scale)^(-β)`. Δt is measured from the
  current batch's t_max to the edge's time at position p. Recent edges
  → small Δt → weight near 1; deep past → big Δt → small weight.

The two terms compose: the closest hop in time AND space dominates;
deep-past or deep-walk positions are softly down-weighted instead of
hard-cut.

### 6.2 `time_scale` derivation (decoupled from walk length)

`time_scale` sets the natural unit of "one walk hop in time" for the
recency term. The derivation:

```
time_scale = (t_max − t_min) / L_REF      where L_REF = 20 (fixed)
```

For tgbl-wiki: 1.86M sec / 20 ≈ **93,100 sec ≈ 1.08 days**.

Properties:

- **Walk-length-independent.** `L_REF` is a fixed reference constant,
  NOT `max_walk_len`. Bumping `--max-walk-len` from 20 to 50 does NOT
  perturb the temporal decay rate. Tying it to `max_walk_len` was the
  original Lesson 11 bug — bumping L collapsed the scale from 93k → 37k
  and crushed the recency weight.
- **Dataset-adaptive through the span numerator.** A dataset with a
  longer training span gets a proportionally larger time_scale, so
  the relative decay rate across hops stays comparable across datasets.
- **Overridable** via `--alignment-time-scale` CLI flag for ablations
  and per-dataset tuning.

The alternative derivation `(span × n_nodes / n_edges)` — the mean per-
node inter-event time — was tried as a "more principled" formula but
empirically lost ~0.02 test MRR on wiki. The alignment recency wants
a sharper, session-level scale, not the average between-event gap. The
fixed-`L_REF` form is the current default.

## 7. Uniformity loss (Wang & Isola 2020)

Spreads `E_target` on the unit hypersphere, preventing the alignment
pull from collapsing everything into a single direction.

```
U          = unique(batch.src ∪ batch.tgt)
tilde_e_u  = target(u) / ‖target(u)‖
L_uniform  = log E_{u ≠ v ∈ U} [exp(-γ · ‖tilde_e_u − tilde_e_v‖²)]
```

Implemented as all-pairs cosine on normalised target embeddings with
the diagonal masked, log-mean-exp of pairwise squared distances. `U` is
subsampled to a cap (default 20K) at very large batch sizes to bound
quadratic cost.

Only `target` is regularised, not `context`. Context spread emerges
naturally because many different seeds pull on the same context node
through their walks — multiple anchors create natural spread.

## 8. Embedding-side objective

```
L_emb = L_align + eta · L_uniform
```

`L_emb.backward()` updates:

```
E_target, E_context,
proj_t, proj_c, proj_e,
target_final, context_final, context_walk_final.
```

Default `eta = 1.0`.

## 9. Link predictor (CROSS-table 8-block MLP)

The alignment loss trains the `target ↔ context` cosine geometry —
that's the supervised interaction. The link MLP must expose **exactly
that interaction** so the BCE signal leverages it directly instead of
re-learning it from scratch.

For ordered pair (u, v):

```
phi(u, v) = [
  target(u),  context(v),  target(u) ⊙ context(v),  |target(u) − context(v)|,   # u→v
  target(v),  context(u),  target(v) ⊙ context(u),  |target(v) − context(u)|,   # v→u
]  ∈ ℝ^{8d}

score = MLP(LayerNorm(phi))  ∈ ℝ      (raw logit; BCE-with-logits)
```

The 8 blocks are organised in two directions (u→v and v→u) because the
ordered (src, dst) pair has inherent asymmetry on directed graphs and
the head should let the MLP weight the directions itself instead of
committing to one at design time. Each direction contributes 4 blocks:
target-side embedding, context-side embedding, Hadamard, L1 difference
— the standard link-prediction feature set, lifted to cross-table.

The MLP is a 2-layer GELU stack with LayerNorm at the input, dropout
0 by default, hidden dim 128. Returns raw logits paired with BCE-with-
logits — no separate sigmoid layer to keep numerics stable.

Edge features are intentionally **not** fed to the link MLP, only to
the walk context inside the alignment loss. Reason: negatives have no
edge feature (negatives are sampled `(u, v_neg)` pairs, no associated
edge exists). Including edge feats in the link MLP would let the
classifier key on "edge feat is present → positive", which is a leak.

## 10. Negative sampling

| Phase | Source | K per positive |
|---|---|---|
| Training | `HistoricalNegativeSampler`: K_hist drawn from each source's reservoir of past destinations (Vitter R, M=32 by default), K_rand uniform over training-destination pool. Default split 50/50. | 10 |
| Validation | `dataset.negative_sampler.query_batch(..., split_mode="val")` (TGB pre-generated, variable K, mix of historical + random per positive) | ~999 on tgbl-wiki |
| Test | `dataset.negative_sampler.query_batch(..., split_mode="test")` | ~999 on tgbl-wiki |

### Causal reservoir

`reservoir.observe(batch.src, batch.tgt)` runs in the post-scoring
block AFTER the link step for batch B. When batch B+1's sampler reads
the reservoir, it sees destinations from events ≤ B only. Batch B's
positives are not in its own reservoir at sampling time. Same strict-
causal guarantee as the Tempest ingest.

### Distribution match with eval

TGB serves a 50/50 historical/random mix of negatives at val/test.
Training on the same mix keeps the link MLP's input distribution
aligned with what eval will present. `hist_neg_ratio=0.5` is the
default; set to 0 to fall back to pure-random training negatives for
ablations.

### False-negative guard

Any historical-negative draw equal to the current positive's target
(`v_neg == v_pos`) is replaced with the random fallback. Empty
reservoir slots (cold-start sources) are likewise replaced. Both
checks are a single vectorised `np.where`.

## 11. TGB integration (the only metric authority)

| Concern | TGB symbol | Where |
|---|---|---|
| Dataset | `tgb.linkproppred.dataset.LinkPropPredDataset(name, root, preprocess=True)` | `data.py::load_tgb` |
| Splits | `dataset.{train,val,test}_mask` | `data.py::load_tgb` |
| Edges | `dataset.full_data["sources" / "destinations" / "timestamps"]` | `data.py::load_tgb` |
| Eval negatives | `dataset.load_val_ns()` / `load_test_ns()`, then `dataset.negative_sampler.query_batch(...)` | `negatives.py::TGBNegativeSampler` |
| Metric scorer | `tgb.linkproppred.evaluate.Evaluator(name).eval({y_pred_pos, y_pred_neg, eval_metric})` | `evaluator.py` |
| Metric name | `dataset.eval_metric` (typically `"mrr"`) | passed through `Loaded.eval_metric` |

**There is no hand-rolled MRR.** A per-positive `tgb_eval.eval(...)`
call computes the metric exactly as the leaderboard does. Pessimistic-
only MRR can disagree with the official Evaluator at score-tie
boundaries; reporting through the official Evaluator is what makes
numbers comparable.

`is_directed` is dataset-specific (heuristic in `data.py::_UNDIRECTED`)
and overridable at the CLI with `--is-directed` / `--no-is-directed`.
TGB itself does not declare directedness — the choice is part of how
we interpret the dataset. On tgbl-wiki specifically, undirected
matches TGB's eval semantics; running with `--is-directed` regressed
test MRR by 0.06 because Tempest's walk arena halves and TGB's eval
negatives don't respect direction.

## 12. Training loop (strict-causal)

```
for epoch in range(num_epochs):
    walk_gen.reset()                  # one Tempest reset per training epoch
    reservoir.reset()                  # one reservoir reset per training epoch

    for batch in create_batches(train_split, target_batch_size):

        # === step 1: walks from pre-ingest Tempest ===
        seeds = unique(batch.src ∪ batch.tgt)
        walks = walk_gen.walks_for_nodes(seeds)

        # === step 2: embedding update ===
        l_align    = alignment_loss(target(seeds), context_walk(walks.nodes, walks.ef), walks, ...)
        l_uniform  = uniformity_loss(target(unique(batch.src ∪ batch.tgt)), ...)
        (l_align + eta * l_uniform).backward()
        embedding_optimizer.step()

        # === step 3: negatives from pre-batch reservoir ===
        neg = neg_sampler.sample(batch)

        # === step 4: link update ===
        logits = link_predictor(target(u), target(v), context(u), context(v))
        link_bce(logits, labels).backward()
        link_optimizer.step()

        # === step 5: POST-SCORING BLOCK (feeds batch B+1) ===
        reservoir.observe(batch.src, batch.tgt)
        walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)   # LAST

@torch.no_grad
def evaluate(split):                  # val and test use the same code path
    for batch in create_batches(split, target_batch_size):
        evaluator.evaluate_batch(batch)            # uses target / context lookups + link MLP
        walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
```

### Batching

`create_batches` groups edges chronologically, keeping all events that
share a timestamp inside a single batch (so the "ingest after scoring"
rule still partitions cleanly). Batches grow until the next timestamp
group would push past `target_batch_size`.

## 13. Feature handling matrix

For any TGB dataset, the four feature regimes are handled uniformly:

|        | nf absent | nf present |
|---|---|---|
| **ef absent** | `target(u) = E_target[u]`<br>`context(u) = E_context[u]`<br>`ctx_walk(u, _) = context(u)` | `target(u) = target_final([E_target[u] ‖ proj_t(nf[u])])`<br>`context(u) = context_final([E_context[u] ‖ proj_c(nf[u])])`<br>`ctx_walk = context(u)` |
| **ef present** | `target(u) = E_target[u]`<br>`context(u) = E_context[u]`<br>`ctx_walk(u, e) = ctx_walk_final([context(u) ‖ proj_e(e)])` | All projections active, all fusion sites active. |

Featureless regime (top-left): zero extra params beyond the dual
embedding tables, the model collapses to pure dual-table SGNS on walks.

Streaming node features: `EmbeddingStore.update_node_feat(new_array)`
overwrites the non-persistent buffer in place; the next lookup picks
up the new values automatically.

## 14. Channel summary

| Channel | Read at | Gradient sink |
|---|---|---|
| `E_target[u]` | seed of alignment; target blocks of link MLP; uniformity | `L_emb` and `L_link` |
| `E_context[u]` | non-seed walk positions of alignment; context blocks of link MLP | `L_emb` and `L_link` |
| `proj_t` (nf, target role) | inside `target(u)` everywhere | `L_emb` and `L_link` |
| `proj_c` (nf, context role) | inside `context(u)` everywhere | `L_emb` and `L_link` |
| `proj_e` (edge feat) | inside `ctx_walk(u, eps)` (walk positions) | `L_align` only |
| `target_final` | inside `target(u)` when nf present | both losses |
| `context_final` | inside `context(u)` when nf present | both losses |
| `context_walk_final` | inside `ctx_walk(u, eps)` when ef present | `L_align` only |
| `LinkPredictor` MLP | scoring positives + negatives | `L_link` only |

Three structural rules:

1. `E_target` is seed-only on alignment; `E_context` is walk-position-
   only. The two tables only meet through the alignment cosine — that
   asymmetry is what makes two tables more than two copies of one.

2. Edge features are context-side and alignment-only. They characterise
   a hop, not a node identity. And they're absent from negatives, so
   they cannot enter the link MLP without leaking.

3. Node features ride inside `target()` and `context()` as residuals —
   every downstream site reads them through a single role-specific
   path. No consumer fuses node features by itself.

## 15. Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `d_emb` | 128 | identity table width |
| `d_hidden_link` | 128 | link MLP hidden dim |
| `max_walk_len` | 20 | walk cap; diagnostic shows ~38% of walks saturate at this on wiki |
| `num_walks_per_node` | 5 | walks per seed |
| `walk_bias` | `ExponentialWeight` | Tempest sampler |
| `target_batch_size` | 200 | timestamp-grouping respected |
| `num_epochs` | 50 | wiki: link BCE still falling at 50 |
| `temporal_decay_exp` (β) | 0.5 | exponent in `(1 + Δt/τ)^(-β)` |
| `alignment_time_scale` | -1 (derive) | `≤ 0` → derive via `(span / L_REF)` with `L_REF=20` fixed |
| `eta_uniform` | 1.0 | weight on uniformity term |
| `uniformity_temperature` (γ) | 2.0 | exponent on squared L2 in uniformity |
| `uniformity_cap` | 20_000 | subsample cap for the all-pairs Gram |
| `num_neg_per_pos` | 10 | training K |
| `hist_neg_ratio` | 0.5 | matches TGB's eval mix |
| `reservoir_size` (M) | 32 | per-source Vitter R reservoir |
| `emb_lr` | 1e-3 | Adam |
| `link_lr` | 1e-3 | Adam |
| `seed` | 42 | numpy + torch |

## 16. Implementation notes

- **Tempest CPU mode.** The laptop has 8 GB VRAM; the walk arena
  collides with PyTorch's allocator if both run on GPU. Walks are
  deterministic regardless of device.
- **Eval row budget.** The Evaluator chunks scoring at ~500K rows per
  pass (scales inversely with `d_emb`) to fit 8 GB VRAM under TGB's
  ~999-negative test batches. Math is identical to the unchunked path.
- **Per-positive eval.** Each positive runs through the TGB Evaluator
  separately rather than batched, since `Evaluator.eval` expects a
  scalar y_pred_pos and an array of y_pred_neg per positive.
- **Walks-at-eval.** At eval the embedding is frozen; only the link MLP
  reads from `target(u) / context(v)`. Walks would still be runnable
  but contribute no gradient — and TGB's streaming protocol forbids
  back-propagation through eval samples anyway.

## 17. What's deliberately NOT in this design

These were tried in earlier (pre-v3) iterations and either failed
honestly or carried a leak:

- **TGN-style per-node memory.** The standard `memory_update_at_start
  = False` mode applies batch positives to memory before scoring —
  classic within-batch leak. A correct version uses a raw-message-
  store and applies stored messages at the START of the next batch.
  Not in v3 yet. If added later it MUST use the raw-message-store
  pattern.
- **Walks-as-transformer-input (encoder over walks at scoring time).**
  Without memory and without walk-supervised embeddings, this
  underperforms (test MRR ~0.011 on wiki in the v0 baseline). The
  signal isn't strong enough to lift a transformer + link MLP + raw
  embeddings simultaneously. A walk encoder feeding INTO the link MLP
  (alongside walk-supervised embeddings) is on the roadmap — the
  current architecture lays the foundation but doesn't include it.
- **Hand-rolled MRR.** Disagrees with the official Evaluator at score-
  tie boundaries. Doesn't match the leaderboard. Banned.
- **Feature-based init.** Initialising `E_target[u]` from `nf[u]` makes
  sense on static-feature datasets but breaks streaming-feature
  ones. The runtime residual path works for both.

## 18. Open issues and roadmap

### Active issues

- **Walk saturation at `max_walk_len = 20`.** Diagnostic on wiki shows
  ~38% of walks pin the cap by end-of-epoch. Bumping to L=50 with
  the corrected `time_scale` (fixed `L_REF=20`) was a wash — same test
  MRR as L=20 within noise. The `1/K` positional decay crushes deep
  walk positions fast: at K=10 the weight is 0.1; at K=49 it's 0.02.
  Longer walks would need either a softer K-decay or a holistic walk
  encoder to actually pay off.
- **Alignment plateaus early.** Loss drops from ~0.90 to ~0.87 in the
  first 5 epochs and barely budges after, even though link BCE keeps
  falling. Possibly an artifact of the cosine bound (1 − cos can't go
  below 0, and the deep-K cells contribute increasingly little).
  Worth instrumenting per-K contribution.
- **Batch size on small datasets.** B=200 outperforms B=500 and B=1000
  on wiki by ~0.02 and ~0.06 test MRR respectively. The per-positive
  gradient dilution at larger batches dominates on a 110K-edge
  dataset. Larger batches will be worth revisiting on
  tgbl-coin / -flight / -comment / -review where there are 10×–100×
  more edges.

### Roadmap (in priority order)

1. **Walks-at-scoring.** Encode walks with a small transformer / GRU,
   feed the seed's pooled hidden state into the link MLP alongside
   the raw `target / context` lookups. The walk-length-sweep result
   already says we don't get more signal from longer walks via the
   current alignment loss alone; a holistic walk encoder is what
   would actually exploit longer walks.
2. **Raw-message-store memory module.** The honest version of TGN
   memory. Should compose multiplicatively with walk-supervised
   embeddings + walks-at-scoring.
3. **Scale-out beyond wiki.** tgbl-coin / -comment / -flight / -review
   with longer walks (Tempest paper conventions: `wpn=10, mwl=80`)
   and revisit larger batches — bigger datasets should be less prone
   to the per-positive gradient dilution that hurts wiki at B=500+.
4. **Honest-protocol baselines** for TGN / DyGFormer / TPNet — re-run
   them under raw-message-store memory and report against our numbers.
   This is the paper's contribution-defining comparison.

## 19. Result so far (tgbl-wiki, strict-causal)

50 epochs, K=5, d=128, undirected (TGB convention), hist_neg_ratio=0.5.
Test reported through TGB's official Evaluator.

| Version | B | L | time_scale | Val MRR | Test MRR |
|---|---|---|---|---|---|
| v3 baseline (within-table link MLP, src-only seeding) | 200 | 20 | 93k | 0.3725 | 0.2884 |
| v3 + init-only node-feat (older variant) | 200 | 20 | 93k | 0.3629 | 0.2942 |
| **v3 + cross-table + union seeding + ef-pad fix (current best)** | **200** | **20** | **93k** | **0.4015** | **0.3313** |
| v3 directed (sanity) | 200 | 20 | 93k | 0.3553 | 0.2735 |
| Per-node mean inter-event time formula (155k) | 200 | 20 | 155k | 0.3980 | 0.3107 |
| Larger batch | 500 | 20 | 155k | 0.3835 | 0.3104 |
| Longer walks + sharper time_scale | 500 | 50 | 93k | 0.3841 | 0.3070 |
| Larger batch + longer walks + buggy ts | 1000 | 50 | 37k | 0.3503 | 0.2690 |
| Larger batch + longer walks + 155k ts | 1000 | 50 | 155k | 0.3574 | 0.2704 |

Three things this sweep settled:

1. **time_scale ≈ 93k is the sweet spot on wiki.** The "mean per-node
   inter-event time" formula (155k) is more principled but worse by
   ~0.02 test MRR. The fixed `L_REF=20` form is now the default.
2. **B=200 is the right batch on wiki.** B=500 costs ~0.02 and B=1000
   costs ~0.06 test MRR. On larger datasets this trade-off will
   likely flip.
3. **L=20 → L=50 is a wash with correct time_scale.** The `1/K`
   positional decay crushes deep walk positions too aggressively for
   longer walks to help the alignment loss. To exploit longer walks
   we need a holistic walk encoder feeding the link MLP — that's
   roadmap item #1.

Leaderboard reference (most carry the TGN memory leak):

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

Current honest walks-supervise-embeddings number is 0.331 on wiki.
The closest leaderboard neighbour is EdgeBank-inf (0.495). Closing
that gap is the immediate next milestone; the levers we haven't yet
pulled (walks-at-scoring, memory, longer walks at correct time-scale)
are exactly what the leaderboard methods rely on.
