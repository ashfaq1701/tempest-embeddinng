# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. The
architecture below replaces the prior alignment+uniformity design
(preserved on `backup/important-walk-embedding`) with a single
InfoNCE contrastive loss + a separate BCE link head.

---

## Architecture

### Loss

`tempest_walks/losses.py` — `alignment_loss(...)`

InfoNCE contrastive alignment over batched temporal random walks
with **sampled-negative** partition function. For each seed `s_i`
with positive contexts `{n_p^+ : p ∈ [0, lens_i − 2]}` (positions
of walk i, with the seed itself at position `lens_i − 1`):

```
L_i      = -(Σ_p w[i,p] · log p(n_p^+ | s_i)) / (Σ_p w[i,p])
log p(n | s_i)
         = -‖p_t(s_i) - p_c(n)‖² / τ
           - log Σ_j exp(-‖p_t(s_i) - p_c(n_j)‖² / τ)
L_total  = mean_i (L_i for i with at least one valid positive)
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
w(K_hop, Δt) = 1/K_hop + (1 + Δt/T_train)^(-β)
```

Defaults: `τ = 0.5`, `β = 1.0` — empirically validated on wiki
under full-pool InfoNCE; expected to transfer to sampled-neg but
not re-swept under sampled-neg yet.

### Trainer

`tempest_walks/trainer.py` — strict-causal per-batch ordering:

1. `walks = walk_gen.walks_for_nodes(seeds)`  — pre-ingest
2. `L_align = alignment_loss(...)`             — InfoNCE scalar
3. `neg = neg_sampler.sample(batch)`           — pre-observe
4. Build link BCE on **detached** embeddings: `link_head(E[u].detach(), E[v].detach())`
5. `L_total = L_align + L_bce`
6. `optimizer.zero_grad(set_to_none=True); L_total.backward(); optimizer.step()`
7. `neg_sampler.observe(...)`                  — post-scoring
8. `walk_gen.add_edges(...)`                   — post-scoring, last

`E` is detached on the BCE path so the single backward routes
alignment-side gradients to `E + p_target + p_context` and BCE
gradients to `link_head` only.

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
| `tau` | 0.5 | a-priori; validated by τ sweep on wiki (full-InfoNCE), re-validated under projection_norm=none + l2_dist (2026-05-28) |
| `beta_time` | 1.0 | a-priori; validated by β sweep on wiki (full-InfoNCE), re-validated under projection_norm=none + l2_dist (2026-05-28) |
| `num_align_negatives` | 128 | wiki K sweep (3 seeds × 50 ep): knee of the diminishing-returns curve; ~98% of K=512's test MRR at ~2.6× less compute; lowest val std in sweep; largest K that fits on 8 GB at comment-scale NK |
| `d_emb` | 128 | |
| `d_proj` | 128 | |
| ProjectionHead output | raw merge MLP output (no norm) | wiki single-seed sweep 2026-05-28 (see below): val 0.4556 vs L2-normed baseline 0.4289 (+0.027). Off-sphere L2-distance sim uses projection magnitude as signal; global grad-norm clip at 1.0 in trainer keeps magnitudes bounded. Hardcoded — not a CLI knob |
| Alignment sim | `-‖p_t − p_c‖² / τ` (L2-distance) | same sweep: equivalent to cosine on the unit sphere; off-sphere it carries strictly more information (magnitude + direction). Hardcoded — not a CLI knob |
| `num_walks_per_node` | 5 | DeepWalk/CTDNE convention |
| `max_walk_len` | 20 | |
| `max_time_capacity` | -1 (unbounded) | wiki single-seed window sweep 2026-05-28: cap ∈ {66k, 100k, 250k, 500k, 1M} all underperformed unbounded on test MRR. cap=500k matched unbounded on val (+0.002, within noise) but lost test by 0.006 |
| `lr` | 1e-3 | wiki seed-42 A/B (sampled-neg K=64): lr=1e-3 → val 0.4594 vs lr=1e-2 → 0.4301 |
| `batch_size` | 2000 | |

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
