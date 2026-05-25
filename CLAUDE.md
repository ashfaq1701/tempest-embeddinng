# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. The
architecture below replaces the prior alignment+uniformity design
(preserved on `backup/important-walk-embedding`) with a single
InfoNCE contrastive loss + a separate BCE link head.

---

# Loss variation — InfoNCE with sampled negatives

Branch: `loss-impl/infonce-sampled-negatives` (from master)
Started: 2026-05-25T15:24:43Z

Replaces in-batch negative pool (M = NK × L entries per seed)
with frequency-weighted sampled negatives from the pool's
unique nodes. Removes the [NK, M] sim matrix that required
chunking; sim matrix becomes [N_seeds, max_positives + num_neg]
and fits in a single pass on 8 GB GPU for all TGB datasets.

Sampling strategy: per batch, build (unique_node, count) over
the pool. Per seed, draw num_negatives = N × walks_per_node ×
max_walk_len samples from a categorical distribution weighted by
count^0.75 (Word2Vec convention). False negatives (sampled nodes
that are positives of the same seed) are accepted — the bias is
small (~3% per sample) and matches standard SimCLR/CLIP practice.

No changes to alignment, positive weighting, or other architecture.
Only the partition function changes from "softmax over all M pool
entries" to "softmax over (positives ∪ sampled_negatives)".

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
  be a "trivial self-positive" (seed vs its own projection) but
  it's also masked via `_INVALID_SIM`.
- At padding: `nodes = -1` is clamped to 0 by `nodes.clamp_min(0)`
  before embedding lookup, so no out-of-range index error. The
  resulting bogus context contribution is masked away.

No fix needed.

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

## Architecture

### Loss

`tempest_walks/losses.py` — `alignment_loss(...)`

InfoNCE contrastive alignment over batched temporal random walks.
For each seed `s_i` with positive contexts `{n_p^+ : p in walk i}`
and a flat pool of all batch contexts (size `M = NK · L`):

```
L_i      = -(Σ_p w[i,p] · log p(n_p^+ | s_i)) / (Σ_p w[i,p])
log p(n | s_i)
         = -‖p_t(s_i) - p_c(n)‖² / τ
           - log Σ_j exp(-‖p_t(s_i) - p_c(n_j)‖² / τ)
L_total  = mean_i (L_i for i with at least one valid positive)
```

`j` ranges over **all valid batch contexts** — every other walk's
positions act as in-batch negatives. The softmax denominator does
the anti-collapse work that Wang-Isola uniformity used to do in the
old architecture, but with task-relevant negatives instead of random
pairs.

Hop/time weights:

```
w(K_hop, Δt) = 1/K_hop + (1 + Δt/T_train)^(-β)
```

Defaults: `τ = 0.5`, `β = 1.0` — empirically validated by τ and β
sweeps on wiki under InfoNCE (single seed 30 epochs at bs=200; the
defaults won both sweeps).

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

## Tests

`tests/test_vitter_r_uniformity.py` — χ² uniformity check on the
Historical (Vitter R) reservoir sampler.

---

## Defaults

| Knob | Default | Source |
|---|---|---|
| `tau` | 0.5 | a-priori; validated by τ sweep on wiki |
| `beta_time` | 1.0 | a-priori; validated by β sweep at τ=0.5 |
| `d_emb` | 128 | |
| `d_proj` | 128 | |
| `num_walks_per_node` | 5 | DeepWalk/CTDNE convention |
| `max_walk_len` | 20 | |
| `lr` | 1e-2 | linear-scaling default at bs=2000 (Goyal 2017) |
| `batch_size` | 2000 | |
