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
