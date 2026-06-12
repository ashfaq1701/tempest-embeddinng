# Pair-feature integration — attempt 2

Branch: `feature/pair-feature-integration-attempt-2`
Base: `74a6bae` (cross-GRU link-supervised + sphere-E + time + chord + 2-layer GRU).

---

## CAMPAIGN RESULTS (2026-06-12, tgbl-wiki, seed 42, eval-bs 50)

> All runs share the same harness/eval-bs, so deltas are clean vs the in-campaign
> `base` (eval-bs 50 lifts base to 0.7715/0.7362 vs the doc's 0.7345 — smaller eval
> batches = fresher strict-causal state; not comparable to the 0.7345 doc number).

| feature(s) | flags | val | test | Δtest | verdict |
|---|---|---|---|---|---|
| base | — | 0.7715 | 0.7362 | — | reference |
| **#1 recurrence** | `--use-pair-recency` | 0.7857 | 0.7555 | **+0.0193** | win |
| **#1+#2 recurrence+history** | `+ --use-pair-history` | 0.7851 | **0.7581** | **+0.0219** | **WINNER (ship)** |
| #5 ctx (learned co-reach) | `--use-ctx-term` | 0.7680 | 0.7322 | −0.0040 | hurts |
| #1+#5 | | 0.7845 | 0.7553 | +0.0191 | #5 drags #1 |
| #1+#2+#5 | | 0.7842 | 0.7578 | +0.0216 | #5 drags |
| #3 co-reach (exact) alone | `--use-coreach` | 0.7670 | 0.7292 | −0.0070 | hurts |
| #1+#3 | | 0.7839 | 0.7525 | +0.0163 | #3 drags #1 |
| #1+#2+#3 | | 0.7853 | 0.7585 | +0.0223 | #3 = noise (+0.0004) |
| #1+#2+#3 joint pair-MLP | `--use-pair-mlp` | 0.7717 | 0.7484 | +0.0122 | overfits (peak ep4) |

### Findings

1. **Exact recurrence (#1) + history (#2) is the only real win: +0.022 test, smooth
   curve.** The ever-bit + decayed count (#2) adds +0.0026 over #1 alone by
   disambiguating historical negatives. **Ship `--use-pair-features`** (the two were
   collapsed into one flag post-campaign).
2. **Co-reachability is redundant-to-harmful on wiki — both the learned `h[u]·h[v]`
   (#5) and the EXACT walk-derived version (#3).** Alone each *hurts* (−0.004 / −0.007);
   added to recurrence each *drags* it; added to #1+#2 it is pure noise (+0.0004). The
   GRU walk-encoder already extracts the shared-neighbour signal, so an explicit
   co-reach channel only adds variance the model must learn to ignore.
3. **The joint pair-MLP interaction decoder overfits** (peak ep4 then drifts, −0.010
   vs additive) — the conditional "co-reach gated by new-edge" signal isn't there to
   extract; the extra capacity just memorises train pair patterns. Additive wins.
4. **0.83 was not reached (best 0.758 @ bs50, 0.771 @ bs25).** Root cause is the
   OPPOSITE of TPNet's regime: our GRU base is *strong* (0.736), so pair features are
   largely redundant; TPNet's base is *weak* (~0.34), so its pair features are
   load-bearing (→0.84). On a strong walk-encoder base, the headroom for structural
   pair features is small — exact recurrence captures essentially all of it. The
   remaining gap lives in the core encoder/decoder, not in more pair features.

### Multi-seed confirmation (seeds 42, 1, 7; eval-bs 50)

| seed | base test | e2_hist test | Δtest |
|---|---|---|---|
| 42 | 0.7362 | 0.7581 | +0.0219 |
| 1 | 0.7337 | 0.7541 | +0.0204 |
| 7 | 0.7384 | 0.7567 | +0.0183 |

**Δtest = +0.0202 ± 0.0015, Δval = +0.0121 ± 0.0011 across 3 seeds** — tight,
consistent, well clear of the noise band. Recurrence+history is a robust, shippable win.

### Eval-granularity sensitivity (and the fair TPNet comparison)

MRR depends strongly on eval-batch-size: smaller batches refresh the strict-causal
graph state (and the recurrence store / resampled walks) more often, so each query
sees fresher past edges. This is legitimate online eval (no leakage — a batch is
always scored before its own edges are ingested), but it is a *protocol choice* that
must be matched across methods.

| eval-bs | 200 | 100 | 50 | 25 |
|---|---|---|---|---|
| base — test | 0.6932 | 0.7111 | 0.7362 | 0.7513 |
| base — val  | 0.7339 | 0.7538 | 0.7715 | 0.7848 |
| e2_hist — test | 0.7217 | 0.7419 | 0.7581 | 0.7710 |
| e2_hist — val  | 0.7524 | 0.7697 | 0.7851 | **0.7972** |

Two things this settles:
- **`base @ bs=200` (val 0.7339) ≈ the project doc's 0.7345** → the doc used the
  bs=200 default. That is the entire source of the "base looks higher than the doc"
  discrepancy: this campaign ran eval-bs 50.
- **TPNet evaluates tgbl-wiki at `batch_size=20`** (verified in
  `TGB_TPNet/evaluate_link_prediction.py:96` — a wiki/review-specific branch, not the
  bs=200 default). So the protocol-matched comparison is eval-bs ≈ 20–25:
  **our best = e2_hist @ bs=25 = val 0.7972 / test 0.7710 vs TPNet 0.84.** The gap is
  **real (~0.07 test), not an eval artifact** — TPNet already uses fine-grained eval.
  (Our initial bs=50 was *coarser* than TPNet's bs=20, i.e. we were mildly
  under-reporting; the correct protocol for this dataset is bs≈20.)

### Apples-to-apples at TPNet's EXACT batch sizes (train 200 / eval 20)

TPNet's wiki config (verified in `train_link_prediction.py:97` + `:99-102`): **train
`batch_size=200`, val/test `batch_size=20`** (a wiki/review-specific inference shrink).
Re-ran base (master) and e2_hist at that exact protocol, seed 42:

| config | val | test |
|---|---|---|
| base (master, no pair feats) | 0.7928 | 0.7597 |
| **e2_hist (#1+#2)** | **0.8025** | **0.7744** |
| **Δ pair features** | **+0.0097** | **+0.0147** |
| gap to TPNet (reported ~0.84 / ~0.83†) | ~−0.038 | ~−0.056 |

† TPNet's exact reported wiki MRR should be taken from the official TGB leaderboard /
their paper — not asserted here from memory. Their *eval protocol* (bs=20) is verified
from code; the headline figure is cited as "reported ~0.84" pending that confirmation.

Two honest take-aways:
- **At matched protocol our best is val 0.8025 / test 0.7744** — much closer to TPNet
  than the bs=50 numbers suggested. My earlier bs=50 was *coarser* than TPNet's bs=20,
  so it under-reported the model.
- **The pair-feature Δ SHRINKS at finer granularity: +0.0147 test @ bs20 vs +0.0202 @
  bs50.** Coherent: at fine eval granularity the base already sees very fresh causal
  state and captures much of the recency signal implicitly, so the explicit recurrence
  feature has less to add. The win is real at every protocol, but smaller the finer you
  evaluate.

### Bottom line

- **Ship `--use-pair-features`** (exact recurrence + ever-bit/count, one flag):
  +0.020 test, 3-seed confirmed, smooth. Report at eval-bs 20–25 to match TPNet.
  (Post-campaign the two campaign flags `--use-pair-recency`/`--use-pair-history` were
  collapsed into this single `--use-pair-features` flag; the falsified features below
  were removed from the branch.)
- **Do not ship** co-reach (#3/#5) or the pair-MLP — falsified (redundant/overfit).
- **The remaining ~0.07 to TPNet is a core-model gap, not a pair-feature gap.** Closing
  it needs encoder/decoder work (e.g. TPNet's MLP-Mixer backbone + joint
  `MLP([h_u,h_v,f_uv])` decoder over a *link-trained* representation), which is out of
  scope for "pair features on the existing cross-GRU base."
- **Infrastructure delivered & reusable:** `SparseStreamStore` (pandas, O(#keys),
  scales to tgbl-comment), `coreach.py` (scipy), all flag-gated; the store base is
  ready for any future per-node/per-pair streaming feature.

## Where we are

Current best stack on wiki (`tgbl-wiki`, single seed 42):

| component | |
|---|---|
| embedding `E` | unit sphere, `geoopt.ManifoldParameter`, RiemannianAdam |
| encoder | 2-layer GRU over `[E(walk node) ‖ Time2Vec(Δt)]`, output projected to the sphere |
| scoring | symmetric cross **chord** distance `-scale·(‖E[u]-ĥ[v]‖ + ‖E[v]-ĥ[u]‖)` + candidate recency `rec_head(Time2Vec(log1p(t_query - t_last[v])))` |
| supervision | **link loss only** (softmax-CE over `[B, 1+K]`), no alignment, no detach |
| **wiki** | **val 0.7345 / test 0.6926** |

This is a pure cross-embedding model (no walk-cos pooling) and it sits just under
the walk-mediated cos head (~0.74) — with a much simpler, propagation-style
architecture.

## Why retry pair features now

We tried pair features once before (overnight 2026-06-11, branch
`feature/overnight-experiments-june-11`). On the **old** architecture — alignment
loss shaping a detached `E`, a fixed cos-pool link head — explicit pair features
were **redundant with the walk-cos signal** and capped at ~0.757. The exact
recurrence store gave +0.033 but nothing crossed ~0.76. The conclusion then was:
the cap is the extractor/architecture, and TPNet's 0.84 needs its
backbone-trained representation, not bolt-on features.

The architecture has since changed in ways that **invalidate that conclusion** and
make pair features worth a second, serious attempt:

i. **We are now a propagation network.** The GRU propagates a state along the
   walk — structurally the same family as TPNet's representation, not the old
   seed/context contrastive setup.

ii. **The architecture is simpler — no alignment loss.** `E` is shaped purely by
    the link objective. There is no second loss competing for `E`'s geometry, so a
    pair feature added to the decoder is not fighting an alignment prior.

iii. **No detach.** The link loss flows into `E` (and the GRU) directly. A pair
     feature on the scoring path now co-trains the whole representation, instead
     of only tuning a frozen head over a detached `E`.

iv. **Our architecture is now much closer to TPNet.** Link-supervised
    representation + a decoder that consumes node features, no contrastive
    pre-shaping. TPNet adds its RP pairwise feature to exactly this kind of
    decoder.

v. **Headroom is now real, not redundant.** TPNet without pair features reaches
   only ~0.3–0.5 test MRR; *we already reach ~0.69 test without any pair feature.*
   TPNet's pair feature lifts it from ~0.34 to ~0.84 (+0.50). Even a fraction of
   that lift, on top of our already-strong 0.69, should clear 0.83. The earlier
   "features are redundant" finding was specific to the cos-pool extractor that
   already encoded most of the structure; the chord/GRU decoder is a different,
   cleaner injection point.

Verified facts that bound the goal (from this session, read from TPNet source):
- **Eval protocol is identical** to TPNet — official TGB `query_batch` (~999
  dst-negs/positive) + official `Evaluator` MRR. The 0.84↔our-gap is real, not an
  artifact.
- **TPNet's RP is not walk sampling** — it is exact random-feature propagation of
  the time-decayed temporal-walk matrices (matrix powers), JL-compressed. Its pair
  feature is the Gram of multi-hop, time-decayed (u,v) co-reachability. That exact,
  multi-hop, time-decayed structure is the signal to reproduce — and it is allowed
  here as a **local pairwise store**, since the constraint is only that *walks* come
  from Tempest.

## Goals

1. **Pass 0.83 test MRR on wiki** (TPNet ≈ 0.84). Tonight's scope.
2. **Then cross TPNet's margin on every other TGB dataset** (review, coin, comment,
   flight). Not tonight.

## Tonight: pair features for wiki

Integrate pairwise structural features into the chord/GRU decoder. Candidate
signals, cheapest/most-exact first (all maintained as **local strict-causal
stores**, not walks — within the Tempest-only-for-walks constraint):

- **Exact recurrence** (1-hop): has-(u,v)-interacted, count, time-since-last —
  the `ExactPairStore` from the overnight branch (dense `[N,N]` last_ts + count,
  N≈9k on wiki). It is exact and was the strongest signal last time (+0.033 even
  in the wrong architecture).
- **Exact multi-hop time-decayed co-reachability** (the TPNet-RP analog, done
  exactly rather than sampled) — 2-hop common-neighbour with recency, the
  genuinely-untried lever.
- Inject as additional channels into the per-candidate score (concat to the
  decoder input / add a learned pair-term to the logit), co-trained end-to-end
  with `E` and the GRU (no detach).

Success = wiki test MRR ≥ 0.83, holding the smooth-curve / noise discipline
(≥0.015 single-seed or multi-seed confirmation before claiming a win).

---

## TPNet's pair features — full inventory

This is the catalogue of every pairwise signal TPNet's `RandomProjectionModule`
produces, what each one *is* (paper + code), and a plain-language reading. Source
of truth: `TGB_TPNet/models/TPNet.py` (`get_pair_wise_feature`, L119–139;
`get_random_projections`, L108–117; `update`, L70–106) and `modules.py`
(`LinkPredictor_v1.forward`, L97–119).

**One sentence first.** TPNet never stores the pair features explicitly. It
maintains one random-projected vector per node per hop-level, and the pair feature
for `(u,v)` is the **Gram matrix** (all pairwise dot-products) of the stacked
`[u^(0..k), v^(0..k)]` projections — `(2(k+1))²` numbers, `= 36` for wiki's `k=2` —
then `ReLU → log(·+1) → MLP`. By the Johnson–Lindenstrauss guarantee (Theorem 2)
each dot-product `⟨H^(la)_a, H^(lb)_b⟩ ≈ ⟨A^(la)_a, A^(lb)_b⟩`, an inner product of
two **temporal-walk-matrix** rows. So every "feature" below is really one entry (or
a block of entries) of that Gram, and the MLP learns which combinations matter.

### A. The underlying objects (how the features are constructed)

| Name | What it is (paper + code jargon) | Easy explanation |
|---|---|---|
| **Temporal walk matrix** `A^(k)(t)` | The `k`-hop **time-respecting** reachability matrix at time `t`. `A^(k)_{u,v}(t) = Σ_{W: u⇝v, k hops, non-increasing times ≤ t} s(W)`; `A^(0)=I`. Built implicitly, never materialised. | "How many `k`-step temporal walks get from `u` to `v`, counting recent ones more." A recency-weighted count of time-ordered paths. |
| **Time-decay walk score** `s(W) = ∏_i e^{-λ(t − t_i)}` | Per-walk weight: each edge on the walk at time `t_i` contributes `e^{-λ(t−t_i)}`; `λ` = decay rate (`rp_time_decay_weight`). Makes `A^(k)` a *recency-weighted* walk count, and is what lets the relative-time update (L88–90, `*= exp(-λΔt)^i`) avoid the `e^{λt}` overflow. | "Old hops fade, recent hops count full." A walk through stale edges barely registers; a fresh one counts strongly. |
| **Random-feature node rep** `H^(l)(t)` (JL projection) | The maintained per-node vector. The module stores `e^{-λlt}H^(l)(t) = A^(l)(t)P` directly (decayed form), `P ∈ R^{N×dim}` a fixed Gaussian sketch (`random_projections[i]`, `dim = rp_dim`). Theorem 1: `e^{-λlt}H^(l) = A^(l)P` exactly; Theorem 2 (JL): `⟨H^(la)_a, H^(lb)_b⟩ ≈ ⟨A^(la)_a, A^(lb)_b⟩` to `(1±ε)`. `O(1)`-amortised update per edge. | "Each node carries a tiny fingerprint of its whole `l`-hop temporal neighbourhood." Dotting two fingerprints recovers their neighbourhood overlap without touching the full `N×N` matrix. |

### B. The pair feature itself (the Gram and its semantic blocks)

| Name | What it is (paper + code jargon) | Easy explanation |
|---|---|---|
| **Raw pairwise feature** `f̃_{u,v}` | `flatten(F_{u,v} F_{u,v}ᵀ)`, where `F_{u,v} = [H^(0)_u … H^(k)_u, H^(0)_v … H^(k)_v] ∈ R^{2(k+1)×dim}`. In code: `matmul(rp, rp.transpose(1,2)).reshape(B,-1)` (L132). A `(2(k+1))²`-vector (`36`-d for wiki) of all cross-hop, cross-node inner products. | "Take `u`'s and `v`'s fingerprints at every hop level, dot every one against every other." The whole table of overlaps is the feature. |
| **Node-identity / self-norm term** (`A^(0)` block) | Entries `⟨H^(0)_a, H^(0)_b⟩ ≈ ⟨A^(0)_a, A^(0)_b⟩ = δ(a,b)` since `A^(0)=I`: `≈1` on the diagonal, `≈0` for distinct `u,v`. An anchor/normaliser, uninformative alone. | "A node is identical to itself and (mostly) not the other node." Gives the MLP a fixed reference scale. |
| **Direct `l`-hop connectivity** `A^(l)_{u,v}` | Entries `⟨H^(0)_u, H^(l)_v⟩ ≈ A^(l)_{v,u}(t)` (and the mirror): recency-weighted count of `l`-step temporal walks **between `u` and `v`**. `l=1` = recent direct edges (**recurrence**); `l=2` = recency-weighted 2-hop bridging. | "Have `u` and `v` actually interacted lately (`l=1`), or are they one node apart (`l=2`)?" The strongest single link cue. |
| **Co-reachability / shared context** `⟨A^(li)_u, A^(lj)_v⟩`, `li,lj ≥ 1` | Off-diagonal cross-block entries `≈ Σ_w A^(li)_{u,w}(t)·A^(lj)_{v,w}(t)`: recency-weighted count of nodes `w` reachable from `u` in `li` hops **and** from `v` in `lj` hops. `li=lj=1` = time-decayed **common neighbours**; mixed = asymmetric multi-hop overlap. | "How many of the same nodes have `u` and `v` both been touching lately?" Friends-in-common, generalised to multi-hop and weighted by recency. |
| **Self-structure / activity** `⟨A^(li)_w, A^(lj)_w⟩` | The `u·u` and `v·v` diagonal blocks (`li,lj ≥ 1`): a node's own neighbourhood density / return-walk structure, ≈ recency-weighted degree and closed-walk counts. A per-node activity normaliser the MLP can divide by. | "How busy / well-connected is `u` on its own (and `v`)?" Lets the model discount overlap that's high only because a node talks to everyone. |

### C. Processing & where it's used

| Name | What it is (paper + code jargon) | Easy explanation |
|---|---|---|
| **Scale transform** `log(ReLU(·)+1)` | L137–138: clamp negatives (sketch noise / sign) to 0, then `log1p`. Compresses heavy-tailed Gram magnitudes (walk counts span orders of magnitude) into a learnable range. Skipped iff `not_scale`. | "Squash the huge raw counts so a few giant numbers don't dominate." Standard count→log scaling. |
| **Learned pair feature** `f_{u,v} = MLP(log(ReLU(f̃)+1))` | 2-layer MLP (`pair_wise_feature_dim → … → out`, `pair_wise_feature_dim = (2·num_layer+2)² = 36` for `k=2`) projecting the `36` raw overlaps into the model's feature space. The vector the rest of the network consumes. | "Let a small network decide which overlaps matter and mix them." The end product handed to the decoder/backbone. |
| **Injection point 1 — decoder** | `LinkPredictor_v1.forward` (`modules.py` L97–119): `MLP([h_u ‖ h_v ‖ f_{u,v}])`. The pair feature is concatenated with the two node embeddings right before the final link score. | "Glue the pairwise overlap onto the two node vectors and read off the score." Most direct use. |
| **Injection point 2 — backbone** | `TPNetEmbedding.compute_node_temporal_embeddings` (L305–368): for each sampled neighbour `w` of the target pair, the per-neighbour relative encodings `[r_{w|u}, r_{w|v}] = [f_{w,u}, f_{w,v}]` are appended to that neighbour's token before the MLP-Mixer. (Note the latent L361 `masked_fill` no-op bug found in the fidelity pass.) | "Also tell every neighbour how it relates to both endpoints, before mixing." Spreads the pairwise signal through the whole node encoder, not just the final layer. |

### What this means for our reproduction

- The untried lever is **co-reachability with recency** (rows B-4/B-5): 2-hop
  common-neighbour overlap, time-decayed — and crucially we can compute it
  **exactly from Tempest walks** (genuine causal walks) rather than via the JL
  sketch TPNet is forced into. Direct `l=1` connectivity (recurrence) we already
  capture with the recency head; the headroom is the `l≥2` overlap.
- We do **not** need TPNet's `RandomProjectionModule` machinery: the sketch exists
  only because TPNet computes `A^(l)P` analytically over the full graph. We sample
  walks, so we can read these same quantities off walk membership directly (an
  exact edge / 2-hop store keyed on the candidate), sidestepping the JL `ε`-error
  and keeping Tempest as the moat.
- Injection mirrors TPNet's point 1: concat the pair feature to the per-candidate
  decoder input / add a learned pair-term to the logit, co-trained end-to-end with
  `E` and the GRU (no detach).

---

## What Tempest returns (the raw material for our pair features)

Everything we build has to come from one call:
`trw.get_random_walks_and_times_for_nodes(...)` (wrapped in
`tempest_walks/walks.py::walks_for_nodes`). It returns four arrays — **walk
nodes, timestamps, walk lengths, and edge features** — which the wrapper packs
into `WalkData`. Shapes/dtypes are pinned by `tests/test_walk_contract.py` and the
verified contract in `CLAUDE.md`. `N` = #seeds, `K` = walks/seed, `L` =
`max_walk_len`, `NK = N·K`.

| Field | Shape / dtype | What it is | Easy explanation |
|---|---|---|---|
| **nodes** | `[NK, L]` int64; padding `-1` | The node id at each step of each walk. Rows `[i·K, (i+1)·K)` are seed `i`'s `K` walks (`shuffle_walk_order=False` pins the grouping). Backward direction: chronologically **oldest predecessor at position 0**, the **seed at position `lens-1`**. | "For every walk, the list of nodes visited." Each block of `K` rows belongs to one seed; the seed sits at the right end. |
| **timestamps** | `[NK, L]` int64; sentinel `INT64_MAX` at seed slot; padding `-1` | `timestamps[i,p]` = the time of the edge `(nodes[i,p], nodes[i,p+1])` (the hop *out of* position `p`, backward convention). The seed slot `p=lens-1` has no outgoing edge → `INT64_MAX` sentinel. Verified 79/79 `(u,v,t)` tuples match an ingested edge. | "For every hop, when that edge happened." The seed's own slot has a dummy max-time marker because it has no next edge. |
| **lens** | `[NK]` int64 | The true length of each walk (≤ `L`). Positions `p ≥ lens[i]` are padding. The valid mask is `arange(L) < lens.unsqueeze(1)`. | "How long each walk actually is" — walks can be shorter than `L` and the tail is padded. |
| **seeds** | `[N]` int64 | The query nodes we asked walks for; `seeds[i] == nodes[i·K, lens-1]`. | "Which node each block of walks started from." |
| **edge features** | `[NK, L-1, d_ef]` float32 (the 4th return value) | Per-hop edge feature vectors, **one column shorter than `nodes`** (a hop lives between two nodes, so `L-1` hops; tail rows zero). `edge_feats[i,p]` belongs to the edge `(nodes[i,p], nodes[i,p+1])`. **Present only if the graph was ingested with edge features** (wiki has LIWC-style EFs; many TGB datasets do not). | "If the dataset has features on its edges, the feature of each hop." Optional — empty/absent when the data has no edge features. |

Notes that bound what we can do:
- **The current wrapper discards edge features.** `walks_for_nodes` unpacks
  `nodes, ts, lens, _ef` and drops `_ef` (the cross-GRU stack doesn't use EFs;
  prior EF investigations found them dead weight on wiki — see the EF section in
  `CLAUDE.md`). They are still *available* from Tempest if a pair feature wants
  them — re-plumb the 4th return value into `WalkData`.
- **`K = num_walks_per_node`** and **`L = max_walk_len`** are the two knobs that
  set how much neighbourhood each call sees. Multi-hop co-reachability (the TPNet
  analog) is read off `nodes`/`timestamps` directly: shared nodes appearing in
  `u`'s walks and `v`'s walks, weighted by their hop position and timestamp recency.
- **Direction.** The link path uses `"Backward_In_Time"` (most-recent predecessor
  is the most predictive). The `Forward_In_Time` mirror (seed at position 0,
  `INT64_MIN` sentinel, `timestamps[i,p]` = edge *into* `p`) is also verified in
  `CLAUDE.md` if a forward pair feature is ever wanted.

---

## Candidate pair features for tonight

### What our model already covers (the baseline for "gap")

- **`E[u]↔h[v]` + `E[v]↔h[u]` chord** → latent node-context similarity. *Partially*
  carries recurrence (if `v` is a frequent partner of `u`, `v` appears in `u`'s walk
  → `E[v]` lands near pooled `h[u]`) — but pooled-mean over `K` walks/positions
  **dilutes** a single occurrence badly.
- **`rec_head(Time2Vec(log1p(t_query − t_last[v])))`** → candidate *global* recency.
  `t_last[v]` is `v`'s last activity with **anyone**, not with `u` — so pairwise
  recency is **not** in the model.
- **Structurally absent:** no `h[u]↔h[v]` (context-context) term → no explicit
  common-neighbour channel; no pair-specific history; no degree/popularity feature.

### Feature list

`B` = batch sources, `K` = train negs, `N` = #nodes, `NK·L` = total walk slots.
Gain estimates are single-seed-wiki ballparks against the noise discipline
(≥0.015 test or ≥3 seeds to count a win).

| # | Feature | Covered by embedding? | Why / what gap it closes | Fast & vectorizable? | Source | Expected gain (wiki) |
|---|---|---|---|---|---|---|
| 1 | **Exact pairwise recurrence** — `Δt = t_query − last_ts[u,v]` (+ "ever interacted" bit) | **Partially** — walk-membership encodes it but pooled-diluted; recency head is *global to v*, not pairwise | Wiki is recurrence-dominated; "when did `u` last touch `v` specifically" is the single strongest link cue and is exactly TPNet's `A^(1)_{u,v}`. Sharp, exact, pair-specific — closes the dilution gap | **Yes** — gather `store[u, cand]` → `[B,1+K]`, one Time2Vec, add to logit (same shape as existing recency head) | Streaming store, updated in `add_edges` | **Large, +0.04–0.10.** Highest-value single feature; may reach 0.83 alone |
| 2 | **Pairwise frequency** — `count[u,v]` (decayed count of past (u,v) edges) | **No** | Distinguishes a one-off from a habitual partner; recurrence *strength*, not just recency. TPNet gets this from the weighted walk count | Yes — same gather as #1 | Streaming store | Medium, +0.01–0.03 (correlated with #1; stack, don't double-count) |
| 3 | **Time-decayed common neighbours / co-reachability** — `Σ_w e^{−λ(t−t_w)}·1[w∈walks(u)]·1[w∈walks(v)]` | **No** (we never compute `h[u]↔h[v]`) | The TPNet co-reachability block (`⟨A^(li)_u, A^(lj)_v⟩`, li,lj≥1) — "do `u` and `v` share recent neighbours." The untried multi-hop lever, computable **exactly from real Tempest walks** (no JL sketch) | **Yes** — one-hot walk nodes → sparse `[uniq, N]`, weight by recency, gather+dot per (u,cand). Bounded by `NK·L`, not `N²` | **Walks (free — already sampled)** | Medium, +0.01–0.04 |
| 4 | **Temporal Adamic-Adar / resource-allocation** — `Σ_w decay(t_w)/log(deg(w))` over shared `w` | **No** | Down-weights shared *hubs* (a common neighbour everyone talks to is uninformative). Classic, robust; #3 with degree-discount. Strong on heavy-tailed graphs | Yes — #3 weighted by `1/log(deg)` from the degree store | Walks + degree store | Medium, +0.01–0.03 (partly overlaps #3 — A/B which wins) |
| 5 | **Context-context similarity** — add `−scale·‖h[u]−h[v]‖` chord term to the logit | **No** (only E↔h cross-terms exist) | Free structural channel: shared neighbourhoods make `h[u]`, `h[v]` similar — a soft, learned common-neighbour signal with zero new data structures | **Yes** — one extra dot on tensors already in hand | Walks (free) | Small–medium, +0.005–0.02; cheapest to try |
| 6 | **Candidate global recency** — `Δt = t_query − t_last[v]` | **Yes** — this is the current `rec_head` | Already shipped; listed so we don't re-add it | Yes | Already in model | 0 (baseline) |
| 7 | **Node activity / popularity** — decayed `deg(v)`, `deg(u)` | **Partially** — softmax over candidates absorbs some; recency correlates | Popularity prior: busy nodes are likelier endpoints. Cheap normaliser the chord lacks explicitly | Yes — `[N]` gather | Streaming store | Small, +0.005–0.015 |
| 8 | **Preferential attachment** — `deg(u)·deg(v)` | **Partially** (#7 components) | Liben-Nowell PA baseline; one scalar. Marginal once #1/#7 exist | Yes | Streaming store | Small, ≤0.01 |
| 9 | **Jaccard / overlap coefficient** — `#CN / (deg(u)+deg(v)−#CN)` | **No** | Degree-normalised #3; helps when raw CN is degree-confounded | Yes — #3 + #7 | Walks + degree store | Small, +0.005–0.015 (alt normaliser to #4) |

### Ship order tonight

1. **#1 (exact pairwise recurrence)** first and alone. Dominant missing signal on
   wiki, mechanically identical to the existing recency head (Time2Vec → logit term,
   keyed on `[u,v]` instead of `[v]`), near-zero pipeline cost. The overnight
   `ExactPairStore` already proved +0.033 even in the *wrong* architecture. Land it,
   measure — it tells us fast whether 0.83 is in reach.
2. **#5 (`h[u]↔h[v]` term)** — one extra chord on tensors already computed; free A/B.
3. **#3 (time-decayed common neighbours from walks)** — the TPNet co-reachability
   analog from genuine causal walks (our unique angle). Then #4/#9 as normaliser
   variants if #3 helps.
4. Defer **#2/#7/#8** — correlated tail features; add only if #1 leaves a visible
   popularity gap.

### Data structures (fast, vectorizable, no pipeline-cost increase)

- **Streaming store (#1,2,7,8):** on wiki (`N≈9.2k`) a dense `last_ts[N,N]` int64 +
  `count[N,N]` int32 (~1 GB) gathers in one indexed op; updated in the existing
  `add_edges` hook. For scale (coin/comment, millions of nodes) swap to a hash on
  `min(u,v)<<32|max(u,v)` or per-node sorted-neighbour CSR + `searchsorted` — note
  now, build dense tonight.
- **Walk-derived (#3,4,5,9):** we **already sample walks for every unique node**
  (sources + candidates) via the dedup path, so co-reachability is `bincount` /
  sparse-matmul over walk-node one-hots weighted by timestamp recency — bounded by
  `NK·L`, not `N²`, reusing tensors already in the forward pass.

---

## Campaign decision tree (10 h, ~16–18 runs)

**Target:** val ≥ 0.84, test ≥ 0.83 (base: val 0.7345 / test 0.6926).

**The fact that drives ordering:** tgbl-wiki is **~89% repeat edges** (surprise
≈ 0.11), so recurrence is the dominant lever — but TGB's negative sampler injects
**historical negatives** `(u, v')` where `u`–`v'` also have history. A bare
"ever-interacted" bit can't separate the true positive from a historical negative;
you need **Δt + frequency + the embedding** to rank *within* a node's history. Hence
recurrence first, then the history disambiguators, then cold/new-edge
co-reachability — with a repeat-vs-new / historical-negative diagnostic placed
**early**, not at the end.

**Fixed search config:** seed 42, K=5, L=20, 2-layer GRU, chord, RiemannianAdam,
~20 ep / patience 5. Confirm seeds {42, 1, 7}. Wiki ≈ 1 min/epoch → runs ≈ 15–30 min.

```
REFERENCE  base = val 0.7345 / test 0.6926   (target: val ≥0.84, test ≥0.83)
  │        guardrail everywhere: smooth val curve (no ep1-peak-then-drift),
  │        additive logit terms BEFORE any pair-MLP, co-trained (no detach).
  │
  ├─ E1  Feature #1 exact pairwise recurrence
  │      pairwise Δt = t_query − last_ts[u,v]  →  2nd Time2Vec → +logit
  │      (keep the existing global-recency term; this is additive)
  │      ── run E1b in parallel (free, no infra): #5  h[u]↔h[v] chord term ──
  │
  ├─ D1  DIAGNOSTIC (no train, eval E1 ckpt): stratify val MRR by
  │      (a) repeat vs new positive, (b) historical-neg-heavy vs random-neg query.
  │      This decides Stage 2's direction.
  │
  ├── test ≥ 0.83 AND val ≥ 0.84 ───────────────► STAGE 5 (confirm + lock)
  │
  ├── test ∈ [0.78, 0.83)  (recurrence works, gap remains) ──► STAGE 2
  │
  └── test < 0.78  (recurrence underperforms) ──► read D1:
            • repeats already high, NEW edges drag → STAGE 2 (co-reach) directly
            • repeats LOW → injection/scale bug: E1 variants
              (learnable scale, ever-bit, count) before moving on


STAGE 2 — disambiguate history + close the cold/new slice
  ├─ E2  #1 + #2 : add decayed log-count + ever-interacted bit
  │       small MLP on [Δt_feat, log_count, bit] → +logit   (history ranking)
  ├─ E3  #3 time-decayed common-neighbours from the walks we already sample
  │       (TPNet co-reachability, exact, free source) → +logit
  ├─ E4  #4 temporal Adamic-Adar (#3 degree-discounted) — A/B vs E3
  │
  ├── best single Stage-2 add gets test ≥ 0.83 ──► STAGE 5
  └── still short ──► STAGE 3 (stack)


STAGE 3 — stack winners + injection form
  ├─ E5  best-recurrence(E1/E2) + best-co-reach(E3/E4), additive
  ├─ E6  same features joined through ONE small pair-MLP (test interactions)
  │       — watch the overfit cliff; revert to additive if val peaks ep1 & drifts
  ├─ E7  tune decay λ / Time2Vec dims on the E5/E6 winner
  │
  ├── test ≥ 0.83 ──► STAGE 5
  └── plateau < 0.83 ──► STAGE 4


STAGE 4 — popularity cleanup (only if a measurable gap persists)
  ├─ E8  #7 node popularity (decayed deg)      ├─ E9  #9 Jaccard normaliser
  └── pick anything that adds outside noise; else STAGE 5 with best-so-far


STAGE 5 — confirm & generalise
  ├─ E10–E12  best config × seeds {42,1,7}: require mean test ≥0.83 / val ≥0.84
  │           AND smooth curves (not a seed-42 lucky peak)
  └─ budget left? E13+  cross-dataset sanity (review / coin) — stretch goal
```

### Budget allocation

| Stage | Runs | ~Time | Exit gate |
|---|---|---|---|
| E1 (+E1b free) | 1.5 | 0.5 h | recurrence hypothesis |
| D1 diagnostic | 0 train | 0.3 h | picks Stage-2 direction |
| Stage 2 (E2–E4) | 3 | 1.5 h | single-add ≥0.83? |
| Stage 3 (E5–E7) | 3 | 1.5 h | stack ≥0.83? |
| Stage 4 (E8–E9) | 2 | 1.0 h | marginal cleanup |
| Stage 5 (E10–E12) | 3 | 1.5 h | **multi-seed lock** |
| Stretch / reruns | 3–5 | 2–3 h | cross-dataset / re-confirms |

≈ 16 runs, ~9 h, ~1 h slack.

### Guardrails (from the project's hard-won lessons)

- **Additive logit terms first, pair-MLP only at Stage 3.** The walk-tower history
  shows a per-position MLP overfits (val peaks ep1, train loss keeps falling).
  Additive Time2Vec terms don't have that failure mode.
- **Smooth-curve rule, not peak-chasing.** A config that peaks ep1 then drifts loses
  to a monotone one even at a lower peak.
- **Noise discipline.** No single-seed win counts unless test Δ ≥ 0.015; Stage 5
  multi-seed is mandatory before claiming 0.83.
- **Each feature stays co-trained (no detach), keyed correctly**
  (`min(u,v)<<32|max(u,v)` undirected), dense store on wiki — scale path deferred.

### First move

**E1 (exact pairwise recurrence)**, with **E1b (`h[u]↔h[v]`)** riding along free. It
is the decisive test of the dominant lever at near-zero pipeline cost; given the
89%-repeat structure it plausibly lands in the 0.80s by itself, at which point D1
forks the campaign cleanly: remaining gap on the new-edge slice → co-reachability;
gap on history-ranking among negatives → frequency.
