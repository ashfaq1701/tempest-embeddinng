# Pair-feature integration — attempt 2

Branch: `feature/pair-feature-integration-attempt-2`
Base: `74a6bae` (cross-GRU link-supervised + sphere-E + time + chord + 2-layer GRU).

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
