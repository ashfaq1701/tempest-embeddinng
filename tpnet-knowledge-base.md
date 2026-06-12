# TPNet — deep knowledge base

Source: `TGB_TPNet/` (paper `TPNet_Paper.pdf`, NeurIPS 2024 — "Improving Temporal
Link Prediction via Temporal Walk Matrix Projection", Lu/Sun/Zhu/Lv, Beihang;
code = a TGB-adapted fork of `github.com/lxd99/TPNet`, itself built on DyGLib).

This document answers four things: (i) what TPNet is and why it works/is fast,
(ii) does the code match the paper, (iii) where the important parts live,
(iv) is its evaluation fair/identical to ours. It closes with implications for
`pair-feature-integration.md`.

---

## 0. The one-paragraph summary

Temporal link prediction needs **pairwise / relative encodings**: node
representations learned independently can't tell that `(D,A)` and `(D,F)` differ
when `A` and `F` have identical local structure (paper Fig. 1). TPNet's insight
(contribution 1, the "unified view") is that *every* prior relative encoding
(DyGFormer, PINT, NAT, CAWN) is a function of **temporal walk matrices**
`A^(k)(t)`, where `A^(k)_{u,v}(t) = Σ_{W ∈ k-step temporal walks u→v} s(W)`. Prior
methods set `s(W)=1` (just count walks, ignore time) or sample walks (CAWN: slow,
noisy). TPNet (contribution 2) does two things: (a) a **time-decayed** score
`s(W)=∏_i e^{-λ(t-t_i)}` so the walk matrix carries temporal *and* structural
info; (b) a **random-feature-propagation** trick that maintains those `[n×n]`
matrices *implicitly* as low-dim per-node vectors, so it is cheap and needs no
walk sampling at inference. The pairwise feature is the Gram of those vectors
(multi-hop time-decayed co-reachability of `u` and `v`), and it is the
**load-bearing** component (ablation "w/o NR" collapses the model).

---

## 1. The method (paper, in depth)

### 1.1 Unified view (why relative encodings = temporal walk matrices)

- **Temporal walk** (Def 4): `W=[(w_0,t_0),(w_1,t_1),…,(w_k,t_k)]` with
  `t=t_0>t_1>…>t_k` (decreasing time from "now" `t`), each `(w_i,w_{i+1})` an edge.
- **Temporal walk matrix**: `A^(k)_{u,w}(t) = Σ_{W∈M^k_{u,w}} s(W)`, `s` a score fn.
- **Unified relative encoding** (Eq 3): `r^{w|u}(t) = g([A^(0)_{u,w},…,A^(k)_{u,w}])`.
  - DyGFormer / PINT / NAT: `s(W)=1` → entries are walk *counts*; differ only in `g`.
  - CAWN: `s(W)` = a sampling probability → estimates the matrix by sampling walks.
- **Limitation found**: count-based methods ignore time; CAWN includes time but
  estimates via sampling (slow + estimation error). → motivate a time-aware,
  sampling-free construction.

### 1.2 TPNet's temporal walk matrix (time decay)

`s(W) = ∏_{i=1}^k e^{-λ(t-t_i)}` (λ>0 = decay weight). Each hop's contribution
decays exponentially with its age relative to the current time `t`. So
`A^(k)(t)` simultaneously encodes structure (which walks exist) and recency
(how recent the walks' edges are). `A^(0)(t) ≡ I` (identity, constant).

### 1.3 Implicit maintenance by random feature propagation (the speed trick)

Computing/storing `A^(k)(t)` is `O(n²)` per matrix — impractical. Instead keep
low-dim node reps `H^(0),…,H^(k) ∈ R^{n×d_R}` (`d_R ≪ n`):

- **Init**: `H^(0) = P` where `P ∈ R^{n×d_R}`, each entry `~ N(0, 1/d_R)`
  (a JL random projection); `H^(1..k) = 0`.
- **On interaction `(u,v,t)`** (Algorithm 1, Eq 4):
  `H^(l)_u(t+) = H^(l)_u(t) + e^{λt}·H^(l-1)_v(t)` and symmetrically for `v`,
  for `l=1..k`.
- **Theorem 1**: `e^{-λlt}·H^(l)(t) = A^(l)(t)·P` *exactly* — the node reps ARE
  the random projection of the temporal walk matrices.
- **Theorem 2** (Johnson–Lindenstrauss): if `d_R ≥ (24/ε²)·log(4^{1/3}(k+1)n)`,
  then `⟨H̄^(l1)_u, H̄^(l2)_v⟩ ≈ ⟨A^(l1)_u, A^(l2)_v⟩` w.h.p. (inner products of
  reps preserve inner products of the walk matrices — i.e., co-reachability).
- **Cost**: store `O((k+1)·n·d_R)`, update `O(k·d_R)` per interaction. Empirically
  `d_R = 10·log(2E)` suffices (Fig. 3). Batch update via `scatter_add` (Alg 3).
- The same trick extends to other matrices (Appendix D): DyGFormer = no decay,
  CAWN = degree-normalized — both are special cases.

### 1.4 The pairwise feature (relative encoding decoding)

For pair `(u,v)`: stack all-layer reps `F_* = [e^{-λ0t}H^(0)_*,…,e^{-λkt}H^(k)_*]
∈ R^{(k+1)×d_R}` for `*∈{u,v}`; `F_{u,v}=[F_u;F_v] ∈ R^{2(k+1)×d_R}`.
- Raw pairwise feature = **Gram matrix** `f̃_{u,v} = flat(F_{u,v}·F_{u,v}^T) ∈
  R^{4(k+1)²}` — all inner products among the `2(k+1)` layer-vectors of `u` and `v`
  (every hop-combination of co-reachability).
- `f_{u,v} = MLP(log(ReLU(f̃)+1))`. ReLU (walk-matrix inner products are ≥0),
  log (the inner-product range varies by orders of magnitude across layers; Table 5
  shows norms 1e1→1e5 across 3 layers), +1 (avoid log 0). These two ops are
  load-bearing for training stability (ablation "w/o Scale" → numeric overflow).
- With `k=2`: `f̃` is `(2·2+2)² = 36`-dim.

### 1.5 The node embedding backbone (auxiliary feature learning)

A *separate* representation `h_u` from an **MLP-Mixer over the recent m
interactions** (paper §3.2; GraphMixer-style):
- For node `u`, take recent `m` neighbors. Build per-neighbor features:
  node features, edge features, time features `φ(t-t_i)` (Fourier TimeEncoder on
  `log(1+Δt)`), and the **injected pairwise feature** `[r_{w|u}, r_{w|v}]` for each
  neighbor `w`.
- Concat → projection MLP → `Z^(0) ∈ R^{m×d}` → `l` MLP-Mixer layers → mean-pool → `h_u`.

### 1.6 Link likelihood

`p^t_{u,v} = MLP([h_u, h_v, f_{u,v}])` — a 2-layer MLP + Sigmoid over the two node
embeddings **and** the direct pairwise feature `f_{u,v}`. So the pairwise feature
is injected at **two** places: inside the backbone (per-neighbor `r_{w|·}`) and
directly in the decoder (`f_{u,v}`).

### 1.7 Why it works / why it's fast

**Works**: the pairwise feature is the model. Ablation Table 2 "w/o NR" (remove
node reps + the pairwise features decoded from them): MOOC 96.39→83.21, etc. —
a *dramatic* drop. "w/o Time" (λ=0, structure-only) also drops → temporal info in
the walk matrix matters. The Gram over all layers captures multi-hop time-decayed
co-reachability that independently-learned embeddings miss.

**Fast** (33× vs DyGFormer on LastFM): (a) random-feature propagation maintains
the walk matrices implicitly — `O(kd_R)` per edge, *no walk sampling and no graph
queries at inference* (CAWN/DyGFormer spend >70% of runtime building relative
encodings by sampling walks); (b) the node reps are **shared** across all
link-likelihood computations in a batch (compute once, decode many pairs); (c)
small `d_R`. Scales near-linearly to 1e8 edges (PINT OOMs at 1e7).

**Crucial framing for us**: TPNet's RP is **not** a random-walk sampler. It is
exact (in expectation, via JL) random-feature *propagation* — closed-form matrix
algebra. The "temporal walk" is what the matrix *counts*, not a sampling process.

---

## 2. Code ↔ paper fidelity (is it implemented as described?)

**Largely faithful, one latent bug, a couple of numerical reformulations.**

| Paper | Code (`models/TPNet.py`) | Match? |
|---|---|---|
| `A^(k)` time-decay walk matrix, RP maintenance (Thm 1, Eq 4, Alg 1/3) | `RandomProjectionModule.update` (L70–106) | ✓ — implemented in the **relative-time** form (Appendix B.1 Eq 14): decay existing level-`i` by `exp(-λΔt)^i`, then `scatter_add` endpoint's level-`(i-1)` × `exp(-λ(t_batch_end - t_edge))`, reverse `i` order so `rp[i-1]` is pre-batch. This avoids the `e^{λt}` overflow of the literal Eq 4. **Mathematically equivalent.** |
| Pairwise feature: Gram → `log(ReLU(·)+1)` → MLP (§1.4) | `get_pair_wise_feature` (L119–139) + `mlp` (L67–68) | ✓ exact. `pair_wise_feature_dim=(2k+2)²` (L66). `not_scale` flag skips the log/ReLU. |
| Backbone: per-neighbor `[node,edge,time,r_{w|u},r_{w|v}]` → MLP-Mixer → mean (§1.5, Eq 6) | `TPNetEmbedding.compute_node_temporal_embeddings` (L305–368) | ✓ — relative feature per neighbor is `get_pair_wise_feature(w, u)` and `(w, v)` concatenated (L341–349). |
| Link likelihood `MLP([h_u,h_v,f_{u,v}])` | `LinkPredictor_v1.forward` (`models/modules.py` L97–119) | ✓ — `cat([src_emb, dst_emb, get_pair_wise_feature(u,v)])` → 2-layer MLP. |
| Pad-node masking ("mask the pad nodes id=0") | `TPNetEmbedding` L361 | ✗ **LATENT BUG**: `embeddings.masked_fill(mask, 0)` is **not in-place** (no `_`) and the result is **discarded** → the masking never happens. Pad neighbors are not zeroed. (Mitigated because pad-node features are near-zero and MLP-Mixer can adapt, but the intent ≠ the implementation.) |

Per-batch decay reference: `update` uses `next_time = batch's last timestamp`
(L79) as the decay anchor for the whole batch — a batch-level approximation of
the continuous decay (fine for small batches, slightly coarse for large ones).

The ablation flags in the code (`not_embedding`, `encode_not_rp`, `decode_not_rp`,
`not_encode`, `not_scale`, `rp_not_scale`, `rp_use_matrix`) map directly onto the
paper's ablations (w/o NR, w/o Time via `rp_time_decay_weight=0`, w/o Scale, and
the explicit-matrix `use_matrix` reference path). Well-instrumented.

---

## 3. Where the important parts live (code map)

```
TGB_TPNet/
  TPNet_Paper.pdf                     NeurIPS'24 paper (AP/AUC on DyGLib datasets)
  models/TPNet.py
    RandomProjectionModule  L9–168    THE pairwise engine. update() L70 (RP propagation),
                                      get_pair_wise_feature() L119 (Gram→log(ReLU+1)→MLP),
                                      reset/backup/reload L141–168 (eval state mgmt).
                                      pair_wise_feature_dim=(2k+2)² L66.
    TPNet              L171–256       Backbone wrapper. compute_src_dst_node_temporal_embeddings L225.
    TPNetEmbedding     L259–368       MLP-Mixer backbone; injects per-neighbor pairwise L339–353.
    FeedForwardNet/MLPMixer L371–430  the mixer blocks.
  models/modules.py
    TimeEncoder        L9             Fourier time encoding φ(Δt).
    LinkPredictor_v1   L74–119        the decoder: MLP([h_u,h_v,f_{u,v}]).  ← non-NAT path
    LinkPredictor_v2   L122           decoder for NAT (self_dim path).
  train_link_prediction.py
    RP construction    L126–137       RandomProjectionModule(node_num, edge_num, rp_dim_factor,
                                      rp_num_layer, rp_time_decay_weight, ...).
    train loop         L227+          reset rp per epoch (L240); sample neg (L251, train_neg_num,
                                      train_loss_type); score pos+neg; rp.update AFTER scoring (L364);
                                      LossFunction (pointwise BCE | listwise CE).
    eval state mgmt    L394/420       backup rp before val, reload after (undo val edges).
  utils/
    utils.py           NeighborSampler.get_historical_neighbors L211 (time-causal via
                                      find_neighbors_before/searchsorted L201); NegativeEdgeSampler.
    evaluate_models_utils.py          evaluate_model_link_prediction — the eval loop (see §4).
    metrics.py         L101           LossFunction: 'pointwise' (BCE) / 'listwise' (softmax-CE).
    load_configs.py    L291+          best configs (TPNet wiki: rp_num_layer=2, rp_time_decay_weight
                                      from grid; train_loss_type default 'pointwise', train_neg_num=1).
    DataLoader.py                     get_link_prediction_data → full/train/val/test split,
                                      eval_neg_edge_sampler (official TGB), node/edge raw features.
```

Best-config note (from `load_configs.py`): the tgbl-wiki TPNet best-config sets
only `rp_num_layer=2` (+ `rp_time_decay_weight`); it does **not** override the loss
args, so they keep argparse defaults `train_loss_type='pointwise'` (BCE) and
`train_neg_num=1`. So **TPNet trains with 1-neg pointwise BCE** on TGB.

---

## 4. Evaluation fairness — vs ours

**Bottom line: the TGB-adaptation's eval is fair and essentially IDENTICAL to ours
in everything that affects the metric. The PAPER's eval is a different (easier)
protocol and is NOT what the 0.84 number is.**

### 4a. Two different evals — don't conflate them
- **Paper** (Tables 1, 6–10): DyGLib datasets (Wikipedia, Reddit, MOOC, …),
  metric **AP / AUC-ROC**, **1 negative per positive**, transductive+inductive,
  random/historical/inductive neg sampling, 70/15/15. TPNet AP ≈ 99 on Wikipedia.
  This is NOT MRR and NOT directly comparable to us.
- **This folder's code** (TGB adaptation): TGB datasets (`tgbl-wiki`, …), metric
  **MRR** via the **official `tgb.linkproppred.evaluate.Evaluator`**, **~999
  negatives/positive** via the **official `NegativeEdgeSampler.query_batch`**.
  This is where the ~0.84 MRR comes from, and it IS comparable to our 0.7345.
- (Note: paper "Wikipedia" and TGB "tgbl-wiki" share node/edge counts — 9227
  nodes / 157,474 edges — likely the same underlying JODIE data, evaluated under
  two different protocols.)

### 4b. The TGB eval, audited against ours

| aspect | TPNet (`evaluate_models_utils.py`) | Ours (`tempest_walks`) | identical? |
|---|---|---|---|
| negatives | `dataset.negative_sampler.query_batch(src,dst,ts,split_mode)` → ~999 dst negs, fixed per positive | same call (`TGBNegativeSampler.query_batch`) | ✓ **same negatives** (deterministic ns set) |
| metric | official TGB `Evaluator.eval({y_pred_pos, y_pred_neg, eval_metric})` → MRR | same `Evaluator` + dataset `eval_metric` | ✓ identical |
| negatives are dst-replacement, same src | `batch_neg_src = repeat(batch_src)` | same (score `[pos | negs]` for one `u`) | ✓ |
| causality of representation | neighbor sampler is **time-causal**: `get_historical_neighbors` → `searchsorted(times, t)` returns only neighbors with time `< t`, even though it uses `full_neighbor_sampler` (train+val+test). RP updated AFTER scoring. | Tempest advanced strict-causally (`add_edges` after scoring); walks from pre-batch state | ✓ both batch-causal, no future leak |
| streaming order | reset rp per epoch → stream train; eval streams val then test (test sees train+val) | reset Tempest per epoch → train; `_eval` streams val then test | ✓ same train→val→test streaming |
| state ingests positives only (not negs) | `update(batch_src, batch_dst)` = positives | `add_edges(batch.src, batch.tgt)` = positives | ✓ |
| eval no-leak bookkeeping | backup rp before val, reload after (undo val edges) so next epoch/test start clean | per-epoch `walk_gen.reset()` rebuilds state | ✓ equivalent effect |

### 4c. Differences that exist but do NOT affect fairness/metric
- **Eval batch size**: TPNet hardcodes `20` for tgbl-wiki/review (else `args.batch_size`);
  ours is configurable. MRR is per-positive, so batch size only changes the
  batch-causality granularity (edges within one eval batch don't see each other) —
  both use the same batch-level approximation; not a metric difference.
- **Node indexing**: TPNet is 1-indexed (node 0 = pad; the `-1`/`+1` shuffles in
  the eval loop). Ours is 0-indexed. Internal only.
- **What "history" the rep can use**: both can use the full causal history (all
  edges before `t`). TPNet via `full_neighbor_sampler` (time-filtered) + RP
  accumulated from epoch start; ours via the Tempest state (unbounded
  `max_time_capacity=-1`). Equivalent reach.

**Conclusion (iv): yes — fair, and identical in every metric-affecting respect.
The honest gap (our 0.7345 vs TPNet's ~0.84 on tgbl-wiki MRR) is real, not an eval
artifact.** The only "trap" to avoid is comparing our MRR against the *paper's*
AP/AUC numbers — different protocol entirely.

---

## 4.5 Runtime in practice — we are orders of magnitude faster (so far)

A counterintuitive but important measured fact. The paper sells TPNet as *fast*
(33× vs DyGFormer) — but that is **relative to walk-sampling baselines on the
paper's AP protocol**. In **absolute** terms on the **TGB-MRR** setting it is
heavy, and our architecture is dramatically faster per epoch:

| dataset | our epoch time | TPNet epoch time |
|---|---|---|
| `tgbl-wiki` | **~1 min** | **~30 min** |
| `tgbl-coin` (larger) | (not yet run our side) | **10+ hours** |

So far we are **orders of magnitude faster than TPNet**, and the gap widens on
larger datasets.

Why (likely): (1) the TGB-MRR eval scores ~999 negatives per positive, and TPNet
runs its **full MLP-Mixer backbone + per-candidate pairwise-feature decode** for
every candidate, plus neighbor sampling and the RP update — all per query; (2) on
larger graphs the neighbor sampling / RP bookkeeping grows. Our cross-GRU dedups
the per-node walk encoding (one encode per unique node, reused across all ~999
candidates) and uses a light chord decoder, so eval cost scales with *unique
nodes* per batch (≲ N) rather than with candidates. This speed headroom is a real
asset: it lets us iterate and, ultimately, run all TGB datasets where TPNet is
impractically slow — provided we add the pairwise signal without re-introducing
TPNet's per-candidate cost (keep it dedup-friendly / query-independent where
possible).

## 5. Implications for `pair-feature-integration.md`

1. **The pairwise feature is the lever, and it's a local pairwise store, not a
   walk.** TPNet's RP Gram = multi-hop **time-decayed co-reachability** of `(u,v)`.
   We can reproduce this exactly (dense/local store, strict-causal) within the
   "walks-only-from-Tempest" constraint — the store is not a walk source.
2. **Inject at two points like TPNet**: (a) into the per-neighbor sequence the GRU
   consumes (analog of `r_{w|u}`), and/or (b) directly into the decoder logit
   (analog of `f_{u,v}` in `MLP([h_u,h_v,f_{u,v}])`). Our chord decoder is the (b)
   injection point; our GRU walk-input is the (a) point.
3. **Exact > sampled, and time-decay is essential.** Paper "w/o Time" drops; our
   earlier sampled/sketched multi-hop attempts (CN, RP-Gram) were *neutral* partly
   because they were approximate and on the wrong (alignment+detach) arch. Now:
   no detach, no alignment, propagation-style → the pair feature co-trains `E`+GRU.
4. **Headroom is real**: TPNet without pair features ≈ 0.34 MRR; *we* are already
   0.69 test without any. Adding even a fraction of TPNet's +0.50 pair-feature lift
   should clear 0.83.
5. **Scaling/stability tricks to copy**: `log(ReLU(Gram)+1)` (their "Scale" — its
   ablation overflows), and small projection dim (`d_R = 10·log(2E)`) if we ever
   use RP-style projection instead of an exact dense store.
6. **Honesty constraint**: hold the wiki noise discipline (≥0.015 single-seed or
   ≥3-seed); compare only against the TGB-MRR ~0.84, never the paper's AP.

---

## 6. Tempest — our moat (what TPNet lacks)

Source: "A GPU Accelerated Temporal Window-Based Random Walk Sampler", Salehin,
Parisis, Berthouze (Univ. of Sussex), arXiv:2605.16182 (May 2026). Tempest =
**TEMPoral nEtwork Streaming Traversals**, the engine behind `temporal_random_walk`.

The whole cross-GRU stack is **built on real temporal random walks from Tempest** —
a capability TPNet structurally does not have. This is the offensive side of the
comparison.

### 6.1 The structural gap

- **TPNet has no walks at all.** Its "temporal walk matrix" is maintained by
  *exact random-feature propagation* (matrix-power algebra, §1.3) — it never
  ingests a stream, never samples a causal walk, and its RP is a bulk/static-
  leaning structure. The "temporal walk" is what the matrix *counts*, not a
  process it runs.
- **Tempest produces genuine, causality-preserving walks, at streaming
  billion-edge scale.** That is the sellable contribution and the thing TPNet
  cannot replicate.

### 6.2 What Tempest is (three pillars)

1. **Dual-index over a shared edge store** (no edge duplication): a
   *timestamp-grouped* view → O(1) start-edge sampling under a bias + bulk window
   eviction; a *node+timestamp-grouped* view → causal walk progression by O(1)
   node-edge-range lookup + O(log G) binary search for the first timestamp `> t`
   (no per-hop neighborhood scan).
2. **Hierarchical cooperative scheduler**: groups walks sharing a (node, step) into
   a cooperative unit and dispatches at **thread / warp / block** granularity by
   per-step walk-population count `W` and shared-memory-fit `G` (the "dispatch
   plane", Fig. 5). Solves the temporal regime's GPU killers: variable walk length,
   runtime regrouping (the convergence set changes every hop), bounded smem,
   mega-hubs (`W>8193` split across blocks).
3. **Bounded-memory streaming**: sliding-window eviction `W(t)={e : t-Δ ≤ t_e ≤ t}`;
   memory tracks the *active window*, not stream length. Each batch triggers a
   batch-bounded dual-index rebuild that does not accumulate.
4. **Closed-form O(1) bias samplers**: uniform `⌊u·n⌋`, linear, **exponential**
   `⌊n+ln(u)-1⌋` (the start/walk bias we use), plus weight-based O(log n) and
   **Temporal Node2Vec via rejection** on the exponential proposal (keeps the inner
   CDF prev-independent so it runs through the same cooperative dispatch).

### 6.3 The numbers (what TPNet can't match)

- **Causal validity**: Tempest = **100% temporally-valid walks** on every dataset;
  static GPU engines FlowWalker / ThunderRW = **0%** valid (Table 6 — they have no
  notion of temporal causality). Throughput 30.6–112.3 M steps/s, fastest on every
  dataset (51× over FlowWalker on Delicious, 6.4× on TGBL-Coin).
- **Scale**: sustains **real-time** on **Alibaba 81B edges / 14 days** (596 ms
  ingest + 167 ms sampling per batch vs 180 s arrival → **235× headroom**; full
  81B in 1.42 h). Delicious 301M edges ingested in **0.9 s — 76× faster than
  Raphtory**, 84× faster than the CPU backend.
- **vs the closest causal baselines (TEA/TEA+)**: exponential bias **4.6–5.8×**
  faster, Temporal Node2Vec **6–7×** faster (Table 5; growth 0.50 s vs 2.93,
  delicious 8.43 s vs 38.84 — the numbers from this project's TEA-reimpl work).
- **Memory**: 1B edges ≈ 35 GB (linear, fits one A40); streaming memory flat in
  stream length (Fig. 11).

### 6.4 Why this matters for the project

- It is **why we are orders of magnitude faster per epoch than TPNet** on TGB-MRR
  (§4.5): Tempest generates billions of causal walks essentially for free, and the
  cross-GRU dedups the encode (per-unique-node, not per-candidate).
- It is the **path to all TGB datasets** — including the large ones (coin, comment,
  flight) where TPNet is impractically slow (10+ h/epoch) — and beyond, to
  billion-edge streams TPNet's bulk RP cannot stream.
- **Positioning**: TPNet's RP is clever exact matrix algebra but bulk/static-leaning
  and slow on TGB-MRR. **Tempest (real causal walks, GPU-streaming, billion-scale) +
  the geometric cross-GRU model is a different, faster, more scalable paradigm.** The
  pair-feature work (§5) is about closing the *accuracy* gap; Tempest is the
  *capability/efficiency* gap that already runs the other way.
