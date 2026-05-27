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
| `tau` | 0.5 | a-priori; validated by τ sweep on wiki (full-InfoNCE) |
| `beta_time` | 1.0 | a-priori; validated by β sweep on wiki (full-InfoNCE) |
| `num_align_negatives` | 128 | wiki K sweep (3 seeds × 50 ep): knee of the diminishing-returns curve; ~98% of K=512's test MRR at ~2.6× less compute; lowest val std in sweep; largest K that fits on 8 GB at comment-scale NK |
| `d_emb` | 128 | |
| `d_proj` | 128 | |
| `num_walks_per_node` | 5 | DeepWalk/CTDNE convention |
| `max_walk_len` | 20 | |
| `lr` | 1e-3 | wiki seed-42 A/B (sampled-neg K=64): lr=1e-3 → val 0.4594 vs lr=1e-2 → 0.4301 |
| `batch_size` | 2000 | |

---

## Walk encoder ablation on tgbl-wiki (2026-05-27)

Five-stage investigation of the walk encoder design space on
tgbl-wiki (low surprise = 0.108 — most test endpoints have history,
so E[v] alone is informative). **Encoder design intent: produce a
useful h_v on cold-start endpoints where E[v] is near random init.**
This dataset is the wrong workload for the encoder's value; the test
is whether the encoder is at least NEUTRAL when its target audience
(cold nodes) is the minority.

All runs: seed=42 × 50 ep, K=128 sampled negs, walks K=5 L=20.

### Stage A: ATTN encoder + standard link head (clip=0.5)
Replace GRU with single-layer multi-head self-attention (4 heads,
pre-LN, batch_first) over walk edges. Last-valid-edge pooling.
Same `mlp_seed(concat[E[seed], walk_aggregate])` output.
Gradient clipping required (`max_grad_norm=0.5`): at clip=1.0 the
transformer diverged catastrophically (bce → 52303). 1.5M params
(encoder), trains stably with clip.

Best: ep44 val 0.4543 / test 0.4020.

### Stage B: ATTN + exclude_seed
Not run — Stage A landed below GRU encoder; Stage B is a variant
of an already-losing architecture and the data-driven branch was
to test link-head changes instead. The encoder/test code IS in
place (`encoder_exclude_seed` flag; tests pin it to mask E[seed]
everywhere the seed appears in the walk).

### Stage C: GRU encoder + hybrid link head (no bilinear)
HybridLinkHead consumes BOTH `E[v].detach()` and `h_v` for each
side — wider input (2·d_emb), same 6-channel pair-MLP, bilinear
DROPPED (n.Bilinear's weight-gradient at d=256 materialises a
[P, 1, 256, 256] = 5.6 GB intermediate that OOMs the 8 GB GPU).
Tests "augment vs replace at the link head".

Best: ep47 val 0.4612 / test 0.4121.

### Stage D: ATTN encoder + cross-attention link head
Cross-attention over per-seed walk-token banks (h_u/h_v as single
queries against the other side's token banks; concat + MLP score).
Token-bank gather forced bs=1000 (bs=2000 OOMs). Train chunking
at chunk=2000 to keep per-pair scoring under budget.

Stopped at ep15: val 0.064 — 6× below baseline ep15 (~0.36).
Cross-attn head not converging at this scale; killed.

### Stage 0: Baseline (no encoder)
Same config minus encoder. `link_head(E[u].detach(), E[v].detach())`.

Best: ep34 val **0.4933** / test **0.4712**.

### Summary table

All on wiki, bs=2000 (Stage D bs=1000), lr=1e-2, warmup-cap=50
(D=100), seed=42 × 50ep:

| Encoder | Link head | Best val | Best test | Δ baseline val |
|---|---|---|---|---|
| **OFF (baseline)** | standard (E only) | **0.4933** | **0.4712** | — |
| GRU, detached | standard (h_v) | 0.4687 | 0.4187 | −0.025 |
| GRU, detached | hybrid (E + h_v) | 0.4612 | 0.4121 | −0.032 |
| ATTN, clip=0.5 | standard (h_v) | 0.4543 | 0.4020 | −0.039 |
| ATTN, clip=0.5 | cross_attn (tokens) | (killed @ep15, 0.064) | — | severe |

### Findings (Q1–Q5)

**Q1 — encoder helping or hurting?** Hurting on wiki at every config
(−0.025 to −0.039 val). No encoder variant matches, let alone beats,
the baseline.

**Q2 — why?** Wiki surprise is 0.108: ~89% of test endpoints are
historical (E[v] is well-trained by InfoNCE). For these endpoints,
replacing or augmenting E with encoder-derived h_v INJECTS NOISE
because the encoder's BCE-only supervision can't shape h_v as
cleanly as InfoNCE shapes E. The encoder's value is on the cold
~11%, but the win there is dominated by the loss on the warm 89%.

**Q3 — better encoder?** Attention is NOT better than GRU. ATTN
needs grad clipping (clip ≤ 0.5; clip=1.0 diverges). At its best
config, ATTN lands 0.014 val BELOW GRU. The transformer's extra
expressivity has nothing useful to learn from on a dataset where
the encoder shouldn't be the dominant signal.

**Q4 — more info in link head?** Hybrid (E + h) does NOT recover
baseline. The MLP can't fully ignore h's noise — feeding h
alongside E still drags the score down (−0.032 val vs baseline).
The link-head architecture isn't the bottleneck; the encoder's
output quality is.

**Q5 — cross-attention?** No, on wiki at our scale. Cross-attn
between token banks doesn't converge competitively — at ep15
val 0.064 vs baseline 0.376. The head has too many params for
BCE-only supervision to shape from a cold start; even given more
epochs, the gap is too large to close.

### What this DOES NOT say

The encoder's value is **on cold-start nodes**, which tgbl-wiki
barely tests (11% of pairs). The right validation set is **tgbl-
review** (surprise 0.987) where the encoder must produce useful
h_v for endpoints with no training E[v]. That experiment is
deferred. The tonight conclusion: on the warm-pair-dominated wiki,
no encoder variant within the GRU/ATTN/hybrid/cross_attn family
helps — but the family was never designed to help on wiki.

### Code state

All code paths are committed and tested. The encoder is feature-
flagged (`--use-walk-encoder` OFF by default); all heads / encoders
are dispatch-selectable from CLI. Tests pin invariants:
  - tests/test_attention_encoder.py (5)
  - tests/test_hybrid_link_head.py (3)
  - tests/test_cross_attn_link_head.py (5)
  - tests/test_slice_walks.py (4) — shared infra
  - tests/test_walk_contract.py (5) — Tempest contract
  - tests/test_vitter_r_uniformity.py (1)

Logs at `logs/overnight/` (gitignored).
