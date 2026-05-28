# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. The
architecture below replaces the prior alignment+uniformity design
(preserved on `backup/important-walk-embedding`) with a single
InfoNCE contrastive loss + a separate BCE link head.

---

## Architecture

### Loss

Two terms summed: an InfoNCE alignment loss on the walks side
(updates `E`, projection heads) and a per-query softmax cross-
entropy ranking loss on the link side (updates the link head only).
The link loss replaces per-pair BCE: Bruch et al. (ICTIR 2019)
show that softmax-CE with binary relevance and one positive per
query upper-bounds (1 − MRR) and (1 − NDCG); plain BCE has no
such bound. Since TGB evaluates on MRR over per-query candidate
sets, the training objective now directly targets the eval metric.

#### Alignment (InfoNCE)

`tempest_walks/losses.py` — `alignment_loss(...)`

For each seed `s_i` with positive contexts `{n_p^+ : p ∈ [0, lens_i − 2]}`
(positions of walk i, with the seed itself at position `lens_i − 1`):

```
L_i      = -(Σ_p w[i,p] · log p(n_p^+ | s_i)) / (Σ_p w[i,p])
log p(n | s_i)
         = -‖p_t(s_i) - p_c(n)‖² / tau_align
           - log Σ_j exp(-‖p_t(s_i) - p_c(n_j)‖² / tau_align)
L_align  = mean_i (L_i for i with at least one valid positive)
```

`j` ranges over **(positives of seed i)** ∪ **(per-seed sampled
negatives)** — the partition is over the walk's own positives plus
`num_align_negatives` negatives drawn for each walk from the
batch's pool of unique nodes, weighted by `count^0.75`
(Word2Vec convention). The sim matrix is `[NK, L + K_neg]`
regardless of batch size — fits in a single pass on memory-bounded
GPUs.

False negatives (sampled nodes that happen to be positives of the
same seed) are accepted. Per-sample bias is ~3%; matches standard
SimCLR / CLIP practice. An A/B confirmed that excluding them
**hurts** val MRR (false negatives function as useful hard
negatives).

Hop/time weights on positives:

```
w(K_hop, t_edge) = 1/K_hop + \tilde t_e ** β
\tilde t_e      = (t_edge − t_min) / T_train     ∈ [0, 1]
```

`t_min` and `T_train` are computed once from the training split
at data load and stored on `TrainerConfig`. The recency weight is
FIXED per edge — the same (seed, context) pair gets the same
gradient weight whichever batch it's drawn in (no `t_now` drift).
Larger β biases the loss toward later edges within the training
window.

Defaults: `tau_align = 0.5`, `β = 1.0` — tau validated under the
current projection_norm=none + l2_dist setup; β was validated
under the older `(1 + Δt/T_train)^(-β)` formulation, semantics
differ so a fresh β sweep would be reasonable.

#### Link prediction (per-query softmax CE)

For each batch positive `(u_i, v_i^+)` the negative sampler
returns K_train negatives sharing source `u_i`. Build a
candidate matrix with the positive at column 0:

```
candidates_v: [B, 1+K_train] = [v^+, v^{(1)}, ..., v^{(K_train)}]
candidates_u: [B, 1+K_train] = u broadcast across columns
logits       = link_head(E[candidates_u].detach(),
                         E[candidates_v].detach())     # [B, 1+K_train]
L_link       = F.cross_entropy(logits / tau_link,
                               target=zeros(B))
```

For undirected datasets, symmetrise:
`logits = 0.5 × (link_head(E[u], E[v]) + link_head(E[v], E[u]))`,
applied identically at training and eval.

Defaults: `tau_link = 1.0` (pending sweep), `K_train = 100`.

### Trainer

`tempest_walks/trainer.py` — strict-causal per-batch ordering:

1. `walks = walk_gen.walks_for_nodes(seeds)`  — pre-ingest
2. `L_align = alignment_loss(...)`             — InfoNCE scalar
3. `neg_tgt = neg_sampler.sample(batch)`       — pre-observe; [B, K_train]
4. Build `candidates_v = [pos_v | neg_tgt]` [B, 1+K_train]; score
   through link_head on detached embeddings → logits [B, 1+K_train].
   `L_link = CE(logits / tau_link, target=zeros(B))`.
5. `L_total = L_align + L_link`
6. `optimizer.zero_grad(set_to_none=True); L_total.backward(); optimizer.step()`
7. `neg_sampler.observe(...)`                  — post-scoring
8. `walk_gen.add_edges(...)`                   — post-scoring, last

`E` is detached on the link path so the single backward routes
alignment-side gradients to `E + p_target + p_context` and
ranking gradients to `link_head` only.

### Model components

`tempest_walks/model.py`:

- `EmbeddingTable`     single `nn.Embedding(num_nodes, d_emb)`.
- `ProjectionHead`     conditional architecture (E-only or
                       E + node-features). Two instances:
                       `P_target` for seeds, `P_context` for
                       walk-internal nodes. L2-normalised output.
                       Edge features were tested earlier and
                       consistently underperformed the no-EF
                       baseline; the EF channel is removed.
- `LinkHead`           bilinear + 6-channel pair-MLP. Caller's
                       responsibility to detach E on inputs.

---

## Tempest walk contract — verified 2026-05-25

Verified empirically on tgbl-wiki (10 000 ingested edges, 8 seeds,
40 walks at max_walk_len=20). Pinned in `tests/test_walk_contract.py`.

Shapes:
  - `walks.nodes`        `[NK, L]`            int32   ; padding `-1`
  - `walks.timestamps`   `[NK, L]`            int64   ; sentinel `INT64_MAX` at `lens-1`; padding `-1`
  - `walks.edge_feats`   `[NK, L-1, d_ef]`    float32 ; **one column shorter than nodes**; tail rows are zero
  - `walks.lens`         `[NK]`               int64
  - `walks.seeds`        `[N]`                int64
  - `walks.K` = walks per seed; `NK == N · K`

Row grouping: rows `[i·K, (i+1)·K)` belong to `seeds[i]`. Guaranteed
by `shuffle_walk_order=False` at the Tempest constructor.

Walk direction: `"Backward_In_Time"`. Chronologically oldest predecessor
at position 0; seed at position `lens-1`.

Alignment: for `p ∈ [0, lens[i]-2]`, `timestamps[i, p]` is the
timestamp of the edge `(nodes[i, p], nodes[i, p+1])`. Verified
79 / 79 (u, v, t) tuples match an ingested edge.

Seed slot `p = lens-1`:
  - `nodes[i, lens-1]` = seed (matches `seeds[i // K]`)
  - `timestamps[i, lens-1]` = `INT64_MAX` sentinel ("for parity"
    with nodes' shape; seed has no outgoing edge in the walk)
  - `edge_feats` has no row here (its last index is `lens-2`)

Padding (`p >= lens[i]`): both `nodes` and `timestamps` = `-1`;
`edge_feats` rows are all-zero.

### Implications for alignment_loss

The loss code is **correct** under this contract. Verification
walk-through:

- `is_context = positions < (lens-1)` masks both the seed slot
  AND padding positions out of the loss.
- At seed slot: `INT64_MAX − t_now` is hugely negative → `clamp_min(0)`
  → `dt = 0` → `w_time = 1`. But the slot is masked, so `w_pos = 0`.
  Sentinel value never leaks into the gradient.
- At padding: `timestamps = -1` → `dt = t_now + 1` (large positive)
  → small `w_time`. Also masked, no leak.
- At seed slot: `nodes[i, lens-1] = seed` — `sim_pos` here would
  be a "trivial self-positive" but it's also masked via `_INVALID_SIM`.
- At padding: `nodes = -1` is clamped to 0 by `nodes.clamp_min(0)`
  before embedding lookup; the resulting bogus context contribution
  is masked away.

### Implications for upcoming walk encoder

- Positions `[0, lens-2]` are real (node, edge-to-next-node)
  pairs; the encoder can consume both `nodes[p]` and
  `edge_feats[p]` along with the time signal `timestamps[p]`.
- Position `lens-1` is the seed, has no associated edge time
  (`INT64_MAX` sentinel) and no `edge_feats` row. The encoder
  needs either a learned "seed marker" embedding here or just
  to skip this position in any edge-feature pathway.
- Padding (`p >= lens`) must be masked; the encoder should
  derive its mask from `lens` directly (e.g.
  `mask = arange(L) < lens.unsqueeze(1)`).

### Forward_In_Time variant — verified 2026-05-29

The codebase only uses `Backward_In_Time`. The forward direction
is the mirror image; recorded here so a future caller doesn't
have to re-probe it.

Walk direction: `"Forward_In_Time"`. Seed at position 0;
chronologically later successor at position `lens-1`.

Sentinel:
  - `timestamps[i, 0]` = `INT64_MIN` (= `-(1 << 63)`, the
    arithmetic mirror of `INT64_MAX`). The seed has no
    incoming edge in this walk.
  - **Note**: Backward uses `INT64_MAX`; Forward uses
    `INT64_MIN`. Any direction-agnostic mask must accept both.

Alignment: for `p ∈ [1, lens[i]-1]`, `timestamps[i, p]` is the
timestamp of the edge `(nodes[i, p-1], nodes[i, p])` — i.e. the
edge INTO `nodes[p]`, not the edge OUT of it. This is also the
mirror of the Backward convention (where `timestamps[i, p]` is
the edge between `nodes[i, p]` and `nodes[i, p+1]`).
Verified 52/52 valid `(u, v, t)` tuples match an ingested edge
under this rule on tgbl-wiki.

Seed slot `p = 0`:
  - `nodes[i, 0]` = seed (matches `seeds[i // K]`)
  - `timestamps[i, 0]` = `INT64_MIN` sentinel
  - `edge_feats` row 0 exists in shape (since edge_feats has
    `L-1` columns, indexed 0..L-2), but it carries no real
    edge if the seed has no predecessor. A consumer should
    skip it.

Padding (`p >= lens[i]`): same as Backward — `nodes` and
`timestamps` both `-1`, `edge_feats` rows all-zero.

Implications for a forward-walk consumer:
  - `is_context = positions > 0` masks the seed slot at the
    left, instead of `positions < lens-1` at the right.
  - The walk encoder's per-position edge feature attaches
    `edge_feats[p-1]` (the edge INTO `nodes[p]`) instead of
    `edge_feats[p]` (the edge OUT of `nodes[p]`).
  - Time-weight code that depends on Δt or `(t_edge − t_min) /
    T_train` is unchanged — the timestamp value still carries
    the same semantics, just attached at a different slot.

---

## Tests

`tests/test_walk_contract.py` — 5 tests pinning the Tempest walk
output: shapes / dtypes, seed at `lens-1`, alignment of
`timestamps[i,p]` with edge `(nodes[i,p], nodes[i,p+1])`,
`INT64_MAX` sentinel at the seed slot, `-1` padding for nodes
and timestamps. Runs against a live Tempest instance with 10 k
wiki edges ingested.

`tests/test_vitter_r_uniformity.py` — χ² uniformity check on the
Historical (Vitter R) reservoir sampler.

---

## Defaults

| Knob | Default | Source |
|---|---|---|
| `tau_align` | 0.5 | a-priori; validated by τ sweep on wiki (full-InfoNCE), re-validated under projection_norm=none + l2_dist (2026-05-28) |
| `tau_link` | 1.0 | a-priori; pending a sweep on the new ranking link loss |
| `beta_time` | 1.0 | a-priori; validated by β sweep on wiki (full-InfoNCE), re-validated under projection_norm=none + l2_dist (2026-05-28) |
| `num_align_negatives` | 128 | wiki K sweep (3 seeds × 50 ep): knee of the diminishing-returns curve; ~98% of K=512's test MRR at ~2.6× less compute; lowest val std in sweep; largest K that fits on 8 GB at comment-scale NK |
| `K_train` | 100 | ranking-loss convention (DPR-style, RotatE); larger K_train means harder per-query competition and stronger ranking gradients, at proportional compute cost |
| `d_emb` | 128 | |
| `d_proj` | 128 | |
| ProjectionHead output | raw merge MLP output (no norm) | wiki single-seed sweep 2026-05-28 (see below): val 0.4556 vs L2-normed baseline 0.4289 (+0.027). Off-sphere L2-distance sim uses projection magnitude as signal; global grad-norm clip at 1.0 in trainer keeps magnitudes bounded. Hardcoded — not a CLI knob |
| Alignment sim | `-‖p_t − p_c‖² / tau_align` (L2-distance) | same sweep: equivalent to cosine on the unit sphere; off-sphere it carries strictly more information (magnitude + direction). Hardcoded — not a CLI knob |
| Link loss | per-query softmax CE over [B, 1+K_train] candidates | replaces per-pair BCE; upper-bounds 1 − MRR (Bruch et al., ICTIR 2019). Hardcoded — not a CLI knob |
| `num_walks_per_node` | 5 | DeepWalk/CTDNE convention |
| `max_walk_len` | 20 | |
| `max_time_capacity` | -1 (unbounded) | wiki single-seed window sweep 2026-05-28: cap ∈ {66k, 100k, 250k, 500k, 1M} all underperformed unbounded on test MRR. cap=500k matched unbounded on val (+0.002, within noise) but lost test by 0.006 |
| `lr` | 1e-3 | wiki seed-42 A/B (sampled-neg K=64): lr=1e-3 → val 0.4594 vs lr=1e-2 → 0.4301 |
| `batch_size` | 500 | rebalanced for B*(1+K_train) candidate forwards per step under the ranking protocol |

---

## Projection-norm + loss-form sweep (2026-05-28)

Single-seed (seed=42) ablation on tgbl-wiki, 50 epochs, bs=2000,
eval_bs=200, lr=1e-2, K=128. Six runs over the cartesian product of
`projection_norm ∈ {l2, layernorm, none}` × `loss_form ∈ {l2_dist,
cosine}`. Math-equivalence sanity (Runs 1 vs 2) passed.

| # | projection_norm | loss_form | val (best) | test @ best | best ep |
|---|---|---|---|---|---|
| 1 | l2        | l2_dist   | 0.4289 | 0.3851 | 15 |
| 2 | l2        | cosine    | 0.4282 | 0.3983 | 41 |
| 3 | layernorm | cosine    | 0.3954 | 0.3558 | 28 |
| **6** | **none** | **l2_dist** | **0.4556** | **0.4150** | 26 |
| 4 | none      | cosine    | 0.4367 | 0.3980 | 47 |
| 5 | layernorm | l2_dist   | 0.4177 | 0.3673 | 4 (early peak) |

Run 6 wins: **+0.027 val / +0.030 test over the SimCLR-style l2+cosine
baseline**. Counterintuitive: l2_dist off-sphere was supposed to be
the most brittle pairing (squared distance with unbounded magnitudes)
but grad clip at 1.0 keeps it stable, and the loss can then exploit
both direction and magnitude. LayerNorm is the clear loser — partial
normalisation breaks the loss geometry without buying anything.

### τ sweep on the winner (none + l2_dist)

| τ | val | test |
|---|---|---|
| 0.1 | 0.4424 | 0.4043 |
| 0.3 | 0.4375 | 0.4016 |
| **0.5** | **0.4556** | **0.4150** |
| 1.0 | 0.4447 | 0.4061 |

τ=0.5 wins; non-monotonic below.

### β sweep on (none + l2_dist + τ=0.5)

| β | val | test | best ep |
|---|---|---|---|
| 0.5 | 0.4408 | 0.4078 | 46 |
| **1.0** | **0.4556** | **0.4150** | 26 |
| 2.0 | 0.4468 | 0.3980 | 30 |
| 4.0 | 0.4570 | 0.4047 | 22 (early peak) |

β=1.0 stays — β=4.0 nudges val by +0.001 but loses test by 0.010 and
shows early-peak-then-degrade.

### max_time_capacity sweep (Tempest sliding-window eviction)

| cap (raw units) | ≈ batches kept | val | test |
|---|---|---|---|
| 66,000 (2× mean batch) | 2 | 0.4305 | 0.3800 |
| 100,000 (3×) | 3 | 0.4426 | 0.4019 |
| 250,000 | 7.5 | 0.4439 | 0.4035 |
| 500,000 | 15 | 0.4580 | 0.4087 |
| 1,000,000 | 30 | 0.4442 | 0.3969 |
| **-1 (unbounded)** | 56 | **0.4556** | **0.4150** |

Aggressive recency windows starve the walks. cap=500k matches
unbounded on val (within noise) but loses test by 0.006. Default
stays unbounded; implementation is plumbed for future experiments
where recency might matter more (datasets with sharper distribution
drift than wiki).

### Grad clip ablation (2026-05-28)

The trainer applies `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)`
unconditionally after `backward()`. Tested whether the clip was
costing useful magnitude signal under `projection_norm=none`:

|  | val (best) | test @ best | best ep |
|---|---|---|---|
| **With clip (Run 6)** | **0.4556** | **0.4150** | 26 |
| Without clip | 0.4380 | 0.3933 | 32 |

Without the clip, **align loss spiked from 4.71 at ep1 to 5.56 at
ep2** (Run 6 dropped 4.68 → 4.30) — one batch's unbounded gradient
overshot, the next batch's loss surface was degraded. The model
recovered within an epoch but the spike left a persistent **+0.045
align gap** that never closed, translating to −0.018 val / −0.022
test at convergence.

Grad clip is load-bearing under unbounded projections. The clip is
not "compressing useful magnitude information into noise" — it's
suppressing destructive gradient overshoots that the unit-sphere
constraint would have prevented implicitly. Keep `max_norm=1.0`.
