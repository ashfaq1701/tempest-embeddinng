# Tempest walk-supervised temporal link prediction

Walks-supervised temporal link prediction with Tempest. The
architecture below replaces the prior alignment+uniformity design
(preserved on `backup/important-walk-embedding`) with a single
InfoNCE contrastive loss + a separate BCE link head.

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

### Chunked InfoNCE (memory-bounded backward)

`alignment_loss` performs backward **internally**, per chunk, with
`retain_graph=True` on every chunk except the last. The function
returns a **detached scalar** — callers must not call `.backward()`
on it.

Why: a single accumulator `total_loss_sum` would otherwise pin every
chunk's autograd graph until the outer `.backward()`, so peak memory
would scale with NK·M regardless of chunk_size. With per-chunk
backward, each chunk's local intermediates are released by Python
refcounting between iterations, and only the shared upstream graph
(p_target(e_seed), p_context(e_ctx_flat)) is retained across chunks.

Peak memory under this design:
```
fixed (model + Adam + retained projection graph)
+ max over chunks of (one chunk's sim / log_p / w_pos intermediates)
```
which is finally bounded by the `chunk_size` knob.

The auto-sizer in `tempest_walks/utils.py:compute_auto_chunk_size`
picks chunk_size based on free GPU memory, with explicit terms for
the projection-graph retention overhead (scales with NK + M) and
Adam-state overhead (1.5 GB default; override for very large
embeddings).

### Trainer

`tempest_walks/trainer.py` — strict-causal per-batch ordering:

1. `walks = walk_gen.walks_for_nodes(seeds)`  — pre-ingest
2. `optimizer.zero_grad(set_to_none=True)`
3. `L_align = alignment_loss(...)`             — does its own backward
4. `neg = neg_sampler.sample(batch)`           — pre-observe
5. Build link BCE on **detached** embeddings: `link_head(E[u].detach(), E[v].detach())`
6. `L_bce.backward()`                          — grads on link_head only
7. `optimizer.step()`
8. `neg_sampler.observe(...)`                  — post-scoring
9. `walk_gen.add_edges(...)`                   — post-scoring, last

The two backwards touch disjoint parameter sets: alignment owns
`E + p_target + p_context`; BCE owns `link_head` (its E lookups are
detached).

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

`tests/test_chunked_infonce.py` — four tests, all on CUDA when
available:

1. **chunk_vs_full_K1**     chunk sizes {0,1,7,10,32,50,100,NK,500}
                            agree with the chunk=0 reference on
                            loss + grad(target) + grad(context) +
                            grad(E).
2. **chunk_vs_full_Kgt1**   K=4 multi-walk per seed — verifies
                            pool_walk_idx groups by row, not seed.
3. **chunk_vs_full_node_feat**  alternate projection signature.
4. **chunked_matches_naive_reference**
                            both chunked AND non-chunked production
                            paths match an independent triple-loop
                            implementation written from the math
                            spec. The load-bearing test: it doesn't
                            trust either production path on its own.

Tolerance 1e-5 across all four (float32 reorder noise). Run with
`python tests/test_chunked_infonce.py` or
`python -m pytest tests/test_chunked_infonce.py`.

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
| `chunk_size` | 0 | auto-size; override only if you need a fixed value |
| `lr` | 1e-2 | linear-scaling default at bs=2000 (Goyal 2017) |
| `batch_size` | 2000 | |
