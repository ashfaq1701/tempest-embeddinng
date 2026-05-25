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

---

# Experiments (on this branch)

Empirical log of runs on `feature/infonce-experiments`. The
architecture above already absorbs these findings; this section
is the historical evidence.

## Stage A — Wiki 3 seeds × 30 ep, τ=0.5

Per-seed peak val MRR / test MRR (best across 30 epochs):
  seed 42  (best ep 22): val 0.4924 / test 0.4651
  seed 123 (best ep 25): val 0.4934 / test 0.4722
  seed 7   (best ep 23): val 0.4852 / test 0.4636

Mean ± std:
  val  0.4903 ± 0.004
  test 0.4670 ± 0.004

Compare to post-fix C1 (regression alignment + uniformity):
  C1 val 0.3966 ± 0.014 / test 0.3794 ± 0.015 (30 ep, 3 seeds).

Δ vs C1: Δval = +0.094 (+24% relative), Δtest = +0.088 (+23%).
Std collapse: val 0.014 → 0.004 (3.5× tighter).

## Stage 8.5 — Wiki τ sweep (2026-05-25)

Config: tgbl-wiki, bs=200, β=1.0, seed=42, 30 ep, default
scheduled LR (peak 1e-2).

| τ | best ep | val MRR | test MRR | notes |
|---|---|---|---|---|
| 0.1 | 6  | 0.4143 | 0.3593 | very sharp; peaks early then oscillates |
| 0.3 | 8  | 0.3959 | 0.3486 | sharp; peaks mid-run |
| **0.5** | **30** | **0.4315** | **0.4060** | **winner — Stage A default** |
| 0.7 | 30 | 0.4133 | 0.3886 | softer; still climbing at endpoint |
| 1.0 | 30 | 0.3367 | 0.3317 | regression-like; weakest |

Best τ = 0.5. τ ∈ {0.5, 0.7, 1.0} all peak at the last epoch
(30-ep horizon may be cutting them short).
Logs: `logs/t16_stage85_wiki_tau{0.1,0.3,0.5,0.7,1.0}.log`.

## Stage 9 — Wiki β sweep at τ=0.5 (2026-05-25)

Config: tgbl-wiki, bs=200, τ=0.5, seed=42, 30 ep, default
scheduled LR.

| β | best ep | val MRR | test MRR | notes |
|---|---|---|---|---|
| 0.0 | 30 | 0.3835 | 0.3475 | no time decay |
| 0.5 | 30 | 0.4052 | 0.3568 | gentle |
| **1.0** | **30** | **0.4420** | **0.4067** | **winner — Stage A default** |
| 2.0 | 28 | 0.4333 | 0.4085 | aggressive; test marginally beats β=1.0 |
| 4.0 | 12 | 0.3903 | 0.3534 | over-weights recent edges; drifts |

Best β = 1.0. [1.0, 2.0] is a soft optimum on wiki.
Logs: `logs/t16_stage9_wiki_beta{0.0,0.5,1.0,2.0,4.0}.log`.

## Wiki sweeps summary

Best wiki config under InfoNCE + scheduled LR: bs=200, lr=1e-2
(scheduled), **τ=0.5, β=1.0** — same as the a-priori Stage A
defaults. Both sweeps validated them; no knob change warranted.

Several runs hit best at the last (30th) epoch, suggesting a
longer horizon (50+ ep) would clarify whether the soft optimum
is the true peak or whether longer training keeps climbing.

## Stage B' — Review-v2 5 ep × 3 seeds at bs=2000 (RESOLVED, 2026-05-25)

Original outcome (pre-Option B): BLOCKED on 8 GB GPU. Tried bs
∈ {2000, 1000, 500, 200}, all OOM during `alignment_loss`
forward. Diagnosed: chunked InfoNCE retained every chunk's
intermediates until outer `.backward()`, so peak memory scaled
with NK · M regardless of `chunk_size` — the chunking flag was a
no-op for memory. Compounding: `create_batches` doesn't split
timestamp ticks, so batches have a long tail (max 674 edges at
bs_target=200 → NK=5280 → ~14 GB intermediates on worst-case
batch).

Resolved by Option B (per-chunk `.backward()` with
retain_graph=True inside `alignment_loss`) + the M-aware
auto-chunker. Both landed before master merge; this is now the
architectural default.

Three TGB-name bugs were uncovered + fixed along the way and
also live on master:
  - `data: strip -vN suffix before passing dataset name to TGB`
  - `data: canonicalize Loaded.name to the suffix-stripped name`

Empirical sweep results for review-v2 under Option B will fill
this section as they land.
