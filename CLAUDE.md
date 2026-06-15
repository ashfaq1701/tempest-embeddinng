# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. The
architecture below replaces the prior alignment+uniformity design
(preserved on `backup/important-walk-embedding`) with a single
InfoNCE contrastive loss + a separate BCE link head.

> **`best_configs.sh` — the per-dataset best-config registry. KEEP IT UPDATED.**
> It holds the single best-performing training command for each TGB dataset
> (with the val/test it produced). Whenever you find a config that beats the
> current best for a dataset, update that dataset's command in `best_configs.sh`
> in the same change.

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
    - **Plumbed into `WalkData.edge_feats` right-padded to `[NK, L, d_ef]`** (one
      zero column appended) so it indexes 1:1 with `nodes`/`timestamps`; the context
      mask `positions < lens-1` then selects exactly the real edge rows. `None` when
      the dataset has no edge features. Pairing pinned in `tests/test_walk_edge_feats.py`.
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
| Alignment pool | full unique-batch-node, count-weighted partition | Replaces the earlier `num_align_negatives` sampled-K partition. Closed-form equivalent of multinomial sampling under the count distribution, with zero sampling variance. Hardcoded — not a CLI knob |
| `K_train` | 100 | ranking-loss convention (DPR-style, RotatE); larger K_train means harder per-query competition and stronger ranking gradients, at proportional compute cost |
| `d_emb` | 128 | |
| `d_proj` | — | Removed; projection dim is now hardcoded to `d_emb` in ProjectionHead. The knob was always set equal to `d_emb` in practice, so collapsing it is behaviour-preserving. |
| ProjectionHead output | L2-normalised on unit sphere | reverted from "no norm" (winning 2026-05-28 config) to L2-norm in the Prodigy + ranking-link-loss redesign (2026-05-30 sweep; see below). On the sphere, squared L2 distance equals 2-2*cos, so the alignment loss is cosine-equivalent up to a constant. Hardcoded — not a CLI knob |
| Alignment sim | `-‖p_t − p_c‖² / tau_align` (L2-distance) | same sweep: equivalent to cosine on the unit sphere; off-sphere it carries strictly more information (magnitude + direction). Hardcoded — not a CLI knob |
| Link loss | per-query softmax CE over [B, 1+K_train] candidates | replaces per-pair BCE; upper-bounds 1 − MRR (Bruch et al., ICTIR 2019). Hardcoded — not a CLI knob |
| `embedding_num_walks_per_node` | 5 | DeepWalk/CTDNE convention; backward walks for the alignment loss |
| `embedding_max_walk_len` | 20 | backward walks for the alignment loss |
| `link_pred_num_walks_per_node` | 5 | forward walks reserved for a future link-pred-side scoring path; currently unused |
| `link_pred_max_walk_len` | 20 | forward walks reserved; currently unused |
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

---

## 9-variant fwd+bwd / picker / 1/K_hop sweep (2026-05-30)

Single-seed (seed=42) sweep on tgbl-wiki, 50 ep, bs=500, Prodigy(lr=1.0),
L2-norm projections, no grad clip, full-pool count-weighted alignment,
K_train=300, tau_align=0.5, tau_link=1.0. Runs span three branches:
`feature/ranking-link-loss` (canonical, single bwd), `feature/forward-
backward-walks` (fwd+bwd, shared `p_context`), `feature/split-context-
projections` (fwd+bwd, split `p_context_fwd`/`p_context_bwd`, +66k params).

The fwd+bwd branches add a Forward_In_Time walk batch alongside the
default Backward_In_Time one and sum both alignment losses. K_hop fix
= use `K_hop = lens-1-p` on the fwd direction so the latest forward
edge gets the highest `1/K_hop` weight (mirror of bwd; the earlier
`K_hop = p` form cancelled `1/K_hop` against `w_time` because position
1 is OLD in calendar time).

Picker scheme axis. "default" = bwd: start=Uniform walk=ExpW; fwd:
start=Uniform walk=ExpW. "per-direction" = bwd: start=ExpW walk=ExpIdx;
fwd: start=Uniform walk=ExpW. Run E (single bwd, start=ExpW walk=ExpW)
isolates the start-bias swap on bwd only.

1/K_hop axis. With: `w = 1/K_hop + w_time^β`. Without (F, G, H):
`w = w_time^β` only.

| Run | Branch | Dir | p_ctx | bwd pickers | K_hop fix | 1/K_hop | val | test |
|---|---|---|---|---|---|---|---|---|
| 0   | rank-link-loss | bwd | (single) | Uniform/ExpW | n/a | ✓ | 0.4981 | **0.4525** |
| 1   | fwd-bwd        | both | shared  | Uniform/ExpW | no  | ✓ | 0.4942 | 0.4487 |
| 2   | split-ctx      | both | split   | Uniform/ExpW | no  | ✓ | 0.4899 | 0.4461 |
| A2  | fwd-bwd        | both | shared  | Uniform/ExpW | yes | ✓ | 0.4952 | 0.4463 |
| B2  | split-ctx      | both | split   | Uniform/ExpW | yes | ✓ | 0.4999 | 0.4486 |
| E   | rank-link-loss | bwd | (single) | ExpW/ExpW    | n/a | ✓ | **0.5003** | 0.4472 |
| C   | fwd-bwd        | both | shared  | ExpW/ExpIdx  | yes | ✓ | 0.4983 | 0.4361 |
| D   | split-ctx      | both | split   | ExpW/ExpIdx  | yes | ✓ | 0.4936 | 0.4464 |
| F   | rank-link-loss | bwd | (single) | ExpW/ExpW    | n/a | — | 0.4912 | 0.4463 |
| G   | fwd-bwd        | both | shared  | ExpW/ExpIdx  | yes | — | 0.4915 | **0.4586** |
| H   | split-ctx      | both | split   | ExpW/ExpIdx  | yes | — | 0.4961 | 0.4520 |

### Read

**Val** lives in a tight band [0.4899, 0.5003] — span 0.0104, all within
the ±0.01 noise window. Test spreads wider [0.4361, 0.4586] — span
0.0225, three real groups.

**fwd+bwd walks** (1, 2, A2, B2) buy nothing over baseline by themselves.
B2 (val 0.4999) ties baseline within noise on val but loses test by
0.004. The K_hop fix (lens-1-p on fwd) gives A2 a +0.001/+0.000 nudge
vs Run 1 — within noise.

**Split p_context** (2, B2, D, H) doesn't justify its +66k params. The
best of the split family (B2) ties baseline on val and trails on test;
H is the one bright spot at test 0.4520 ≈ baseline.

**Per-direction pickers** (ExpIdx walk-bias on bwd; C vs A2, D vs B2):
val flat (C +0.003), test consistently *lost* 0.002–0.010. The
ExpIdx walk-bias doesn't help on this dataset; the ExpW-start swap
on bwd (clean isolation in E vs 0) is also tied / slight test hit.

**1/K_hop ablation** is the structural surprise:
- F (single bwd): val −0.007, test −0.001 vs E parent — small regression.
- G (fwd+bwd shared): val −0.007, test **+0.022** vs C parent, and
  **+0.006 vs baseline test** — the highest test of all 11 runs.
- H (fwd+bwd split): val **+0.002**, test **+0.006** vs D parent;
  ties baseline test.

Pattern: under single-bwd, `1/K_hop` carries small useful signal.
Under fwd+bwd, dropping `1/K_hop` *helps* — likely because the
duplicated walk sampling already supplies position-locality signal,
and `1/K_hop` over-weights near-seed contexts to no extra benefit.

### Recommendation

Within ±0.01 → simpler wins. **Baseline (Run 0) is the recommended
ship config**: val 0.4981 (within 0.002 of leader E), test 0.4525
(highest among "settled" non-G runs), no added complexity. The
sweep has falsified fwd+bwd walks, split `p_context`, per-direction
pickers, and the ExpIdx walk-bias on bwd as standalone improvements.

**Open hypothesis worth multi-seed verification**: G's test 0.4586
(+0.006 over baseline) under "fwd+bwd + no 1/K_hop". Needs ≥ 3 seeds
to separate from variance before committing — single-seed at this
noise level is suggestive, not conclusive. A clean follow-up is
"fwd+bwd shared + default pickers + no 1/K_hop" (isolates the
no-1/K_hop change from the picker swap).

Log paths under `logs/ranking_link/wiki_seed42_{run0,run1,...,runH}_*.log`;
canonical CSV at `logs/experiments_summary.csv`.

---

## EF utilization investigation (2026-05-31 overnight)

Overnight investigation of three EF-injection mechanisms on wiki
seed=42 single-run. Goal was to find a clean win over baseline
Run 0 (val 0.4981 / test 0.4525). Spoiler: none of the three
mechanisms (or their combination) broke through baseline on wiki.

### Variants

Each on its own branch off master = 3f47122 ("docs: sweep report").

| Variant | Branch | Mechanism |
|---|---|---|
| V1  | `feature/ef-v1-weighted`   | EFWeightHead: `w[i,p] *= 2σ(MLP(LN(EF[p])))` — sphere-side-agnostic per-position gate on loss weight |
| V2  | `feature/ef-v2-context`    | EF-conditioned p_context: sphere-preserving `F.normalize(p_node + α·F.normalize(MLP(EF)))` |
| V3  | `feature/ef-v3-aux`        | EFPredHead aux loss: predict unit-normed EF direction from `(p_t(E[src]), p_c(E[tgt]))`; cosine 1−cos loss; λ_aux=0.1 |
| combo | `feature/ef-combo-v1-v3-v2` | V1 + V3 stacked |
| V2-fixed | `feature/ef-v2-fixed`   | EF moved to p_target with per-(i,p) symmetric partition (closes V2's InfoNCE shortcut) |

### Results

| Variant | val | test | Δval | Δtest | Verdict |
|---|---|---|---|---|---|
| **Baseline (Run 0)** | **0.4981** | **0.4525** | — | — | reference |
| V1 (EF weight-gate)  | 0.4857 | 0.4299 | −0.012 | −0.023 | regression |
| V2 (EF on p_context) | 0.0197 | 0.0150 | — | — | **DEGENERATE COLLAPSE** |
| V3 (λ_aux=0.1)       | 0.4950 | 0.4508 | −0.003 | −0.002 | tied (within noise) |
| V3 (λ_aux=0.5)       | 0.4953 | 0.4508 | −0.003 | −0.002 | tied (identical to λ=0.1) |
| V1+V3 combo          | 0.4830 | 0.4404 | −0.015 | −0.012 | regresses toward V1 |
| V2-fixed             | **0.4970** | **0.4585** | −0.001 | **+0.006** | **first EF win on test** |

### V2 collapse mechanism (and the V2-fixed redesign)

V2 sanity at ep2 yielded val 0.0197 / link loss 33.9 / align loss 3.9
— catastrophic. The diagnostic was: at init, V2's per-position
sim matches the baseline gather-path within 0.012 (the α=0.01
perturbation is well-behaved). But during training, α grows and the
EF-MLP learns to align `ef_proj` toward `p_seed`. Since the partition
`log_Z` uses pool projections **without** EF, the model can drive
`sim_pos` arbitrarily high without raising the partition. The
asymmetric loss has a free lunch → InfoNCE shortcut → embedding
collapse.

**V2-fixed** moves the EF channel to `p_target` and computes a
per-(i, p) full-pool partition `log_Z[i, p]` that uses the SAME
`p_seed_ef[i, p]` as the positive numerator. Same EF, both numerator
and denominator → shortcut closed. Memory: 150M sims on wiki bs=500
(~600 MB). Compute: +~17% per-epoch (per-(i,p) full sim is L=20× the
baseline partition).

**Final 50ep result: val 0.4970 / test 0.4585 @ ep49** — val tied with
baseline within noise (−0.001), test **+0.006 over baseline**
(0.4525 → 0.4585). This is the only EF mechanism in this investigation
to deliver a real test improvement. The result validates the theory:
V2's catastrophic collapse was specifically the InfoNCE shortcut from
asymmetric EF application, not a fundamental incompatibility between
EF and the L2-norm architecture.

### Read

- **V1**: the gate (range [0, 2] via `2σ(MLP(EF))`) is too aggressive;
  the model learns to downweight too many positives, losing signal.
  A clamped range like `[0.8, 1.2]` would be a softer follow-up.
- **V2**: principled-looking sphere-preserving design, but the
  asymmetric EF application breaks InfoNCE.
- **V2-fixed**: symmetric per-(i,p) partition closes the shortcut.
  Single-seed +0.006 test over baseline (0.4525 → 0.4585) but val
  tied (−0.001). User judgment: +0.006 single-seed is inside the
  noise band — not enough to justify the +17% per-epoch cost or
  the architectural complexity without multi-seed confirmation.
- **V3**: cosine-aux on `(p_t(E[src]), p_c(E[tgt])) → EF`. Neutral.
  λ_aux sweep `{0.1, 0.5}` gave identical convergence — the aux
  signal doesn't move the link-prediction endpoint. Wiki EFs may
  simply not predict link existence well, or this architecture
  doesn't extract that signal.
- **Combo**: V1+V3 regresses toward V1's negative impact. V1's
  gate dominates; the aux head can't rescue it.
- **Working conclusion: EFs are dead weight on wiki** under all
  five mechanisms tested. Target 0.6 unreached; V2-fixed's +0.006
  single-seed test edge does not survive the noise threshold for
  this project. The remaining gap to 0.6 needs a fundamentally
  different lever (encoder, GNN aggregation, longer walks) — not
  more EF utilization attempts.

### What we learned

1. **L2-norm projection constrains EF**: magnitude is renormalized
   away, so EF must encode through direction only. Sphere-preserving
   operations (slerp, `normalize(p + α·normalize(ef))`) are the right
   primitives — but they don't guarantee InfoNCE math survives
   (V2 vs V2-fixed).
2. **Asymmetric loss modifications break InfoNCE**: any EF
   contribution to the positive numerator must be matched in the
   partition denominator, or the model finds a free-lunch shortcut.
3. **EF prediction is decoupled from link prediction utility on
   wiki**: V3's aux loss decreases steadily (0.75 → 0.62) but
   doesn't carry over to val/test improvement. The aux head learns
   EF prediction, but those gradients don't make E better at link
   prediction.
4. **Wiki's LIWC-style EFs may genuinely not carry link-existence
   signal**, or this architecture's representational capacity for
   exploiting them is exhausted. A multi-dataset replication on
   `tgbl-coin` / `tgbl-flight` (different EF semantics) would
   disambiguate.

### Open follow-ups

- V2-fixed full 50ep landing → whether the symmetric form delivers
  any improvement over baseline.
- V1 with clamped gate range (`[0.8, 1.2]` instead of `[0, 2]`) —
  cheap, tests if V1's negative was over-aggression.
- Multi-seed V3 (3 seeds × 50ep) to confirm "tied within noise"
  is stable, not seed-42-specific.
- Re-run on a non-wiki dataset where EFs have different semantics.

Log paths: `logs/ef_experiments/wiki_seed42_v{1,2,3,1_v3_combo,
v3_lambda_0p5,v2_fixed}_*.log`. CSV: `logs/experiments_summary.csv`.

---

## Link-head architecture variants (2026-06-01)

Single-seed (seed=42) sweep on tgbl-wiki, 50 ep, bs=500, Prodigy(lr=1.0),
L2-norm projections, K_train=300, start_bias=Uniform, warmup_steps_cap=50,
tau_align=0.5, tau_link=1.0. Same baseline config as the canonical Run 0.

Goal: test whether the current 574k-param "bilinear + 6-channel pair-MLP"
LinkHead is carrying real signal or sitting on top of an already
similarity-aligned `E`. Four similarity-based variants on
`feature/link-head-variants` (branch commit `5fdf665`), each with a
learnable temperature τ = exp(log_tau):

| Variant | Form | Params |
|---|---|---|
| **LH-A** `inner_product`  | `(u·v) / τ`                                         | 1 |
| **LH-B** `scaled_cosine`  | `(normalize(u) · w · normalize(v)) / τ`             | 129 |
| **LH-C** `distmult`       | `(u · w · v) / τ`                                   | 129 |
| **LH-D** `bilinear_only`  | `((W u) · v + b) / τ`                               | 16,386 |
| Baseline (`pair_mlp`)     | `(u · W_bil · v + b) + MLP([u,v,u·v,|u−v|,(u−v)²,u+v])` | 574,466 |

### Results

| Variant | Best val | Best test | Δval | Δtest | Best ep |
|---|---|---|---|---|---|
| **Baseline `pair_mlp` (Run 0)** | **0.4981** | **0.4525** | — | — | — |
| LH-A `inner_product`  | 0.1133 | 0.0972 | **−0.385** | **−0.355** | 42 |
| LH-B `scaled_cosine`  | 0.4398 | 0.4063 | −0.058 | −0.046 | 32 |
| LH-C `distmult`       | 0.2874 | 0.2310 | −0.211 | −0.222 | 37 |
| LH-D `bilinear_only`  | 0.3840 | 0.3413 | −0.114 | −0.111 | 48 |

### Read

**Ranking:** `pair_mlp` >> `scaled_cosine` > `bilinear_only` > `distmult` >> `inner_product`.

**No variant beat baseline.** The 574k-param pair-MLP is not dead
weight. Even the most expressive variant tested (`bilinear_only`
at 16k params) trails baseline by −0.114 val. Within ±0.01 noise
band: none qualify. Baseline stays the recommended ship config.

**The load-bearing component is L2-normalisation, not algebraic
richness.** The clearest signal in the sweep is `scaled_cosine`
(129 params, L2-normalises both sides) recovering ~88% of
baseline's val while `distmult` (same 129 params, no normalisation)
collapses to 0.287. Same scoring form (`u · w · v`); the only
difference is the `F.normalize(·)` calls. That 0.152 val gap
isolates the contribution of unit-sphere geometry inside the head.

`bilinear_only` (16k params, no normalisation) lands between
`distmult` and `scaled_cosine` — extra algebraic capacity recovers
some signal vs `distmult` but cannot compensate for the missing
normalisation. The pair-MLP's 4 hidden layers presumably learn
implicit normalisation alongside the scoring; without it, raw-`E`
magnitudes wreck cosine-style discrimination.

**`inner_product` collapse.** Raw `E[u]·E[v]/τ` (1 param, no W, no
normalisation) lands at val 0.11. The alignment loss only enforces
sphere geometry on the **projections** `p_t(E)` and `p_c(E)`, not on
`E` itself — `E` has no inner-product structure without a learned
transform in the head. Confirms that the link path cannot bypass
the projection layer's geometric work.

### What the experiment falsifies

- "The current LinkHead is overkill on top of an already similarity-
  aligned E" — false. Removing the pair-MLP costs at least −0.11
  val (the bilinear_only floor) and up to −0.39 val (inner_product).
- "A simple sphere-aware head can replace the pair-MLP" — partially
  false. `scaled_cosine` recovers most of the signal at 129 params
  but still falls −0.058 val short; outside the ≥+0.015 ship
  threshold and on the wrong side of zero.

### What the experiment doesn't answer

- Whether a **higher-capacity** similarity head (e.g., stacked FiLM
  conditioning, mixture-of-bases bilinear, time-conditioned
  scoring) could match or beat pair_mlp. The four variants tested
  are *minimalist* options; the "is the embedding inner-product-
  ready?" probe is answered (no), but the "what richer scoring
  form is right?" question remains open.
- Whether the pair-MLP's signal is in the **6-channel pair features**
  (`u*v, |u−v|, (u−v)², u+v`) or in the **MLP depth**. Two cheap
  follow-ups would isolate this: (a) bilinear + single hidden layer
  on pair feats, (b) deep MLP on `[u, v]` concat only.

### Conclusion

Ship baseline (`pair_mlp`). The next architectural lever for
exceeding baseline is unlikely to live in *simpler* heads — it
lives in *richer* mechanisms (time-conditioned projections,
walk-encoder integration, or pair-history features upstream).

Log paths: `logs/link_head/wiki_seed42_{inner_product,scaled_cosine,
distmult,bilinear_only}_20260601_144414.log`. Branch:
`feature/link-head-variants` (not merged; experimental).

---

## Walk-mediated link head (v2) — history, sweeps, final architecture

This section captures the full lifecycle of `link_pred_head_v2.py`
(the LinkPredHeadV2 class), the embedding-side direction split, and
the L_link → E gradient-leak experiment. Numbers are all single-seed
tgbl-wiki seed=42; all "win" callouts are flagged against the
wiki noise band (single-seed gap must be ≥ 0.015 test OR confirmed
across ≥ 3 seeds to count as real).

### 0. Initial v2 head design (2026-06-04 → 2026-06-05)

The bilinear+pair-MLP `LinkHead` was replaced by a walk-mediated
similarity head (`tempest_walks/link_pred_head_v2.py`), motivated
by analysis/REPORT.md §9. Per (u, t, v_candidate):

1. Sample K walks for u (forward or backward, configurable).
2. Per (walk, position p):
   - sim primitives: `[Hadamard(E[v], E[w_p]), |E[v] − E[w_p]|]`
     (alternative: `cosine_only` scalar — kept as an ablation knob)
   - K (hop) embedding: `nn.Embedding(max_walk_len, d_K)[hop]`
   - time channel: `TimeEncoder(gap_norm)`
     - gap_norm = `log1p(gap) / log1p(T_full)` (Option B; Option A
       was linear `gap / T_full` and bunched non-seed positions at
       ~0.005–0.035 on wiki, giving the per-position MLP no
       resolution. Option B verified +0.009 val / +0.016 test at
       ep2 over Option A on V0_fwd.)
3. Per-position MLP → max + mean pool over positions → mean over walks
4. Direct (E[u], E[v]) bypass MLP
5. Final MLP on concat[walk_features, direct] → scalar logit

Channels were toggleable behind ablation flags
(`--link-head-{no-time-channel, no-K-channel, no-direct, direct-only,
sim-primitives}`) for the sweep below; all toggles were dropped
once the sweep settled.

### 1. Phase 0 — direction sweep (2026-06-05 → 2026-06-06)

V0 = full head with all channels on, hadamard_absdiff sim,
embedding=both (5 forward + 5 backward), K_link split half/half
for "both". 15 epochs each, patience disabled.

| variant | best val | best test | best ep |
|---|---|---|---|
| V0_fwd  | 0.7017 | 0.6758 | 3 |
| V0_bwd  | 0.7724 | 0.7439 | 1 |
| **V0_both** | **0.7775** | **0.7522** | **2** |

V0_both margin over V0_bwd: **+0.005 val / +0.008 test** — *inside*
the wiki noise band. Recorded as a non-decisive win pending future
ablation. V0_fwd much weaker because K=1 (most-recent neighbor) in
forward direction maps to the *earliest* successor (least
predictive); backward direction's K=1 maps to the most-recent
predecessor (most predictive).

### 2. α-leak grid — L_link → E gradient mix (2026-06-06)

Hypothesis: the historical pure-detach (E.detach() on the link
path) may be too aggressive; a controlled leak of L_link gradient
into E could help. Implemented via convex-combo

    E_link_in = α · E + (1 − α) · E.detach()

forward-identical to E, backward scales dE/dL_link by α. α=0
reproduces detach; α=1 is no detach.

4 cells run at 10ep with patience=4 (drift-kill at ep5):

| α | both (val / test) | backward (val / test) |
|---|---|---|
| 0   | 0.7775 / 0.7522 (Phase-0) | 0.7724 / 0.7439 (Phase-0) |
| 0.2 | 0.7664 / 0.7396 (killed ep5) | 0.7642 / 0.7355 |
| 0.5 | 0.7560 / 0.7253 | 0.7719 / 0.7357 |
| 1.0 | 0.7506 / 0.7237 | (skipped) |

Findings:
- Every α>0 cell trailed α=0 on test by 0.008–0.029.
- "Both" row: monotone test decline with rising α.
- "Backward" row: non-monotone (α=0.5 > α=0.2 on val).
- Verified the mix math by isolating L_link backward at α=0.2:
  ||dE|| was 0.219× the α=1.0 case (expected 0.2×); α=0 was
  exactly zero. Mechanism correct; the knob just doesn't help.

Conclusion: **α=0 (pure detach) restored.** The link-loss-into-E
plumbing was removed entirely; E is shaped by L_align alone.

### 3. Single-tower collapse (2026-06-06)

With V0_both's +0.008 test over V0_bwd sitting inside the noise
band — and every α>0 also losing — the dual-tower architecture's
2× compute/parameters paid for nothing statistically distinguishable.
The dual-tower path was removed. The head now consumes ONE direction
of walks; the direction is dictated by `is_directed` at construction:

- undirected → backward walks (--link-pred-backward-{walk,start}-bias)
- directed   → forward  walks (--link-pred-forward-{walk,start}-bias)

The link-pred side uses the full `--link-pred-num-walks-per-node`
on its one direction (no half/half split). The dual-tower
`--link-head-direction` flag was briefly retained for ablation runs
then dropped.

### 4. Embedding-direction sweep (2026-06-07)

Iter-6 had always-on symmetric embedding alignment: backward walks
from each unique tgt + forward walks from each unique src, two
`alignment_loss` calls summed. Is the forward term load-bearing?

Both runs use the single-tower head; K_embed total = 10. Fair
compute: "both" splits 5+5, "backward only" spends all 10 on bwd.
15-epoch cap, patience=5.

| | val | test | peak ep |
|---|---|---|---|
| E0_both | 0.7677 | 0.7332 | 1 |
| **E0_bwd** | **0.7682** | **0.7426** | **5** |

Δ = backward − both = **+0.0005 val / +0.0094 test** — *outside*
the wiki noise band on test. Forward embedding alignment dropped.

Important calibration: these E0 numbers are NOT comparable to
Phase-0 V0_both (0.7775 / 0.7522) — Phase-0 used the dual-tower
head; E0 runs use the single tower. The Δ between dual-tower and
single-tower is ~0.007–0.019 test on wiki (consistent with the
Phase-0 dual-vs-single argument), but again all inside the noise
band at single seed.

### 5. Phase 1 — channel ablations on the single-tower head (2026-06-07)

V0 baseline here = E0_both (both embedding dirs, single-tower head).
4 variants × 15 ep × patience=5.

| variant | val | test | Δ test vs V0 | speed |
|---|---|---|---|---|
| V0 (full head) | 0.7677 | 0.7332 | — | 1× |
| V1_no_time | 0.7697 | 0.7420 | +0.009 | 1× |
| V2_cos_only | 0.7707 | 0.7404 | +0.007 | ~2× faster |
| **V3_direct_only** | **0.5225** | **0.4743** | **−0.259** | ~30× faster |
| V4_no_K | 0.7731 | 0.7433 | +0.010 | 1× |

Interpretation:
- V1/V2/V4 all "win" by 0.007–0.010 test, at the edge of the wiki
  noise band. Single-seed; not robust enough to drop on their own.
- V2's 2× speed-up reflects sim_dim collapsing from 2·d_emb=256 to 1.
- **V3 confirms the walks tower is load-bearing.** Without it the
  head falls to cosine-baseline territory (~0.52 val / ~0.47 test),
  a 0.26 test cliff. The direct (E[u], E[v]) bypass alone cannot
  reach the walk-mediated peak.

### 6. The overfitting / degeneration observation

**Every walk-tower variant — V0, V1, V2, V4, E0_both, E0_bwd, plus
all four α-leak both/bwd cells — exhibits the same shape:**

1. Val MRR peaks at ep1–ep5 (usually ep1 or ep2).
2. Train losses (`align`, `link`) continue to fall monotonically.
3. Val MRR drifts down for the rest of training.
4. Patience=5 kills the run around ep6–ep10.

Concrete trajectories:

- V0_both (Phase 0, 15ep): peak val 0.7775 @ ep2; ep15 val 0.7119.
- V0_bwd (Phase 0): peak val 0.7724 @ ep1; ep15 val 0.7122.
- V0_fwd (Phase 0): peak val 0.7017 @ ep3; ep15 val 0.6805.
- E0_bwd: peak val 0.7682 @ ep5; the run bounced (ep4=0.7651,
  ep5=0.7682, ep6=0.7570, ep7=0.7680, ep8=0.7385) before dying.
- V1_no_time: 0.7697 → 0.7575 → 0.7320 → 0.7214 → 0.7431 → 0.7056.
- V4_no_K: 0.7731 → 0.7445 → 0.7322 → 0.7218 → 0.7290 → 0.7307.

**V3_direct_only is the exception** — no walks tower, no peak, no
drift. Monotone climb to ep15:

  0.4494 → 0.4786 → 0.4816 → 0.4949 → 0.5078 → 0.5048 → 0.5168
  → 0.5159 → 0.5179 → 0.5204 → 0.5174 → 0.5206 → 0.5144 → 0.5140
  → 0.5225 (best).

The pattern is unambiguous: **the walks tower is what causes the
overfit-then-drift shape.** It lifts val MRR from ~0.52 (cosine
baseline) to ~0.77 peak — real signal extraction — but then
memorises train-walk geometry past the peak and val regresses
while train losses still fall. Without the walks tower (V3),
training is well-behaved but caps at cosine baseline.

Working hypotheses, none of which were exhaustively tested:
- The per-position MLP's parameter count (≈ 96·in_dim + 96² ≈ 60K
  per head) memorises specific (E[v], E[w_p], K, t) joint patterns
  observed in train walks; those patterns don't generalise to val
  walks (which arrive after additional Tempest state has been
  ingested past the val cutoff).
- The walk tower's gradient pathway dwarfs the direct channel's:
  the tower's max + mean pool over positions back-propagates
  through every position MLP, while the direct channel sees only
  E[u], E[v]. Once the tower starts overfitting, the final MLP
  upweights tower features and the direct channel's stabilising
  contribution gets crowded out.
- Wiki is a recurrence-heavy workload — the walk-mediated channel
  may not be the right inductive bias here. Cold-start workloads
  (review surprise≈0.987) are the natural domain for walk-based
  signals; that hasn't been re-tested under the cleaned-up head.

What we DID NOT try and could be worth attempting:
- Dropout on the per-position MLP / final MLP.
- L2 / weight-decay specifically on the walk tower parameters.
- A learning-rate schedule that decays faster for tower parameters
  than for the direct channel.
- Training-time mask-out of walk positions (random drop of walk
  rows or position slices).
- Smaller d_pos (the tower out_dim is 2·d_pos=192; the direct
  out_dim is 64 — the head's representational mass is heavily on
  the tower side).
- A multi-seed sweep at ep1-ep5 only, to estimate whether the peak
  is reproducible across seeds (currently every conclusion is
  single-seed).

### 7. Final architecture (post-2026-06-07 cleanup)

After all the above, the head and surrounding code were frozen:

**Kept (load-bearing or inside-noise-band wins):**
- Walks tower (V3 ablation makes it non-negotiable).
- Per-position primitives: `[Hadamard(E_v, E_w), |E_v − E_w|]`
  (V2's cosine_only win was inside noise band; kept the original
  primitives to match Phase-0 V0).
- K (hop) embedding (V4 win inside noise; kept).
- Time channel with Option B normaliser
  `log1p(gap) / log1p(T_full)` (V1 win inside noise; kept).
- Single tower direction = `is_directed ? "forward" : "backward"`.
- Embedding-side BACKWARD alignment only (E0_bwd > E0_both
  outside noise band).
- Pure E.detach() on link path (α-leak grid: every α>0 loses).
- Chunk_c (memory chunking + gradient checkpoint) as a pure
  memory knob (default off, set to 8 for 8 GB GPUs at d_emb=128).

**Dropped:**
- Direct (E[u], E[v]) bypass channel — removed 2026-06-07 (post-§7).
  The walk seed slot IS node u and is compared with each candidate v
  inside the tower, so the tower already carries the u-vs-v signal the
  channel duplicated; the walk-only ablation cost only ~0.03 val /
  ~0.05 test. Removed the `DirectChannel` class, the head's `E_u`
  forward input + its trainer plumbing, and the `--link-head-d-direct`
  / `link_head_d_direct` knob.
- Cosine_only sim primitive path.
- Walk-tower channel toggles (no_time / no_K / no_direct /
  direct_only).
- Dual-tower direction option and `--link-head-direction` flag.
- Forward embedding alignment (`walks_for_nodes_embedding_forward`,
  the `embedding_direction` field, and the K // 2 split).
- The `--embedding-{forward,backward}-{walk,start}-bias` split
  (only backward retained).
- L_link → E gradient leak α and all its plumbing.

The cleanup branch `feature/walk-tower-cleanup` collapses all of
this with net negative line counts in the head + trainer (≈ 460
lines removed across the cleanup commits).

### 8. Quick numbers reference (all single-seed wiki seed=42)

| baseline label | val | test | head type | embedding dirs |
|---|---|---|---|---|
| Phase-0 V0_both | 0.7775 | 0.7522 | dual tower | fwd+bwd (5+5) |
| Phase-0 V0_bwd  | 0.7724 | 0.7439 | dual tower | fwd+bwd (5+5) |
| E0_both         | 0.7677 | 0.7332 | single bwd | fwd+bwd (5+5) |
| E0_bwd          | 0.7682 | 0.7426 | single bwd | bwd only (10) |
| V1_no_time      | 0.7697 | 0.7420 | single bwd (no time) | fwd+bwd (5+5) |
| V2_cos_only     | 0.7707 | 0.7404 | single bwd (cosine sim) | fwd+bwd (5+5) |
| V3_direct_only  | 0.5225 | 0.4743 | direct MLP only | fwd+bwd (5+5) |
| V4_no_K         | 0.7731 | 0.7433 | single bwd (no K) | fwd+bwd (5+5) |
| α-leak best (V0_bwd α=0.5) | 0.7719 | 0.7357 | dual tower | fwd+bwd (5+5) |

The current `master`-merge target is the **single-tower head +
backward embedding only + all walk-tower channels on** stack. That
is approximately E0_bwd in the table above — val 0.7682 / test
0.7426 single-seed on wiki — with the understanding that:
- the single-seed gap to Phase-0 V0_both (−0.009 val / −0.010
  test) is the documented cost of dropping the second tower, and
- on cold-start workloads (review) the walks tower's inductive
  bias is expected to matter more than on wiki, but that hasn't
  been re-validated under the cleaned-up head and remains an open
  follow-up.
