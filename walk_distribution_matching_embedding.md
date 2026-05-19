# Walk-Distribution-Matched Temporal Embeddings

**Design & Execution Plan (v1.5)**

---

## 1. Thesis

> The walk sampler's distribution is the supervision signal. The alignment loss's job is to make the embedding inner product approximate `log P(walk ends at v | seed = u)` under that distribution. Time decay, distance decay, and recurrence emerge from this matching rather than being hand-coded — the sampler already encodes them.

In one sentence: stop hand-weighting alignment, start contrastive-matching the walk endpoint distribution; the embeddings then encode whatever the sampler concentrates on.

A separate, orthogonal claim that this design also commits to: the link prediction head needs explicit time-since-last-event signals at scoring time. Every leaderboard method has them; the current codebase does not. Adding them is independent of the walk-supervision thesis but composes with it.

---

## 2. Why this is novel

The closest published precedents:

- **CTDNE** (Nguyen et al. 2018) feeds temporal walks to skip-gram with local-context windows over the walk. This design supervises on **walk endpoints under the sampler's distribution**, a different target.
- **TPNet** (Lu et al. NeurIPS 2024) learns from temporal walks via random projections — fixed kernel, no learned matching.
- **DyGFormer** (Yu et al. NeurIPS 2023) learns from neighbor sequences via a transformer — sequence model, no distributional matching.
- **CAWN** (Wang et al. ICLR 2021) uses anonymous walks — structural identity, no temporal weighting.
- **Word2vec / SGNS** (Mikolov 2013; Levy & Goldberg 2014) does this matching for static co-occurrence — no temporal version exists.

**Contribution-defining claim:** *Under a parameterized walk sampler (Tempest), the sampler's distribution is itself the supervision signal for temporal node embeddings. Multiple sampler configurations supervising a shared target embedding produce a multi-view geometry that no fixed-walk-operator method (TPNet, CAWN, DyGFormer) can replicate.*

---

## 3. Why this should hit past 0.5 on wiki and have a real shot at 0.7+

1. **EdgeBank-tw (0.571) is what walk-endpoint-matching learns on wiki.** For short walks under tight recency bias, the walk endpoint distribution ≈ EdgeBank-tw's lookup. So 0.571 is roughly the floor.
2. **For longer walks, walk endpoint distribution encodes structural reach** — multi-hop paths. EdgeBank can't do this; TPNet does it via random projections (fixed kernel); we do it via learned embedding matching (more expressive).
3. **The trajectory diagnostic showed hand-weighted alignment fights the sampler's bias.** Removing the redundancy frees the loss to learn deeper structure.
4. **Time encoding closes the largest input-channel gap to the leaderboard.** TPNet, DyGFormer, TGN, TGAT, GraphMixer all feed Δt features explicitly to the scoring head. The current codebase doesn't.

---

## 4. Architecture

Five components. Each independently ablatable. Each composes with the existing dual-table embedding store and TGB Evaluator integration — no architectural rewrites.

### Component 0: Time encoding at the link MLP

For each scored pair `(u, v, t)`:

```
Δt_u  = t - last_event_time[u]
Δt_v  = t - last_event_time[v]
Δt_uv = t - last_edge_time[u, v]

Φ(Δt) = [cos(ω_1·Δt), sin(ω_1·Δt), ..., cos(ω_k·Δt), sin(ω_k·Δt)] ∈ ℝ^{2k}
```

with `ω_i` learnable, `k = 16` (so `d_time = 32`). This is the Xu et al. 2020 functional time encoding used by TGAT/TGN/DyGFormer.

**Cold-start handling:** `last_event_time[*]` is initialized to 0; `last_edge_time[u, v]` returns 0 for unseen pairs (sentinel: 0 ≤ any real timestamp). Three explicit binary flags:

```
is_cold_start_u  = (last_event_time[u] == 0)
is_cold_start_v  = (last_event_time[v] == 0)
is_cold_start_uv = (last_edge_time[u, v] == 0)
```

The Δt itself is clamped to `time_scale × 100` before passing through Φ — keeps the sinusoid in a regime the learned ω_i has seen during training. The binary bit is what carries the cold-start signal; the clamped Δt is just numerically well-behaved input.

Final concatenated input shape: `phi(u, v, t) ∈ ℝ^{10·d + 3·d_time + 3}`, where the 10 cross-table interaction blocks decompose as 8 from the recency view + 2 from the structural view; see Component 3 for the block-by-block specification.

**Causality:** `last_event_time[u]` and `last_edge_time[u, v]` are maintained in the post-scoring block, updated only after batch B is scored. At scoring time for batch B, they reflect events strictly through B-1.

**Symmetry across positives and negatives:** all three Δt values and the three cold-start flags are computed identically for `(u, v_pos, t)` and `(u, v_neg, t)` — the query timestamp `t` is shared in TGB's eval batch, and each candidate `v` has its own well-defined history regardless of whether the edge exists. No leak.

**Why `last_edge_time[u, v]` is the EdgeBank-tw signal made continuous:** EdgeBank-tw asks "did u and v interact in the last T time units?" A learned link MLP reading `Φ(Δt_uv)` can express that and any other monotonic function of pair recency.

### Component 1: Contrastive walk-endpoint matching (replaces alignment loss)

For each walk `W` seeded at `u`, ending at `v` (deepest valid past node — `nodes[0]` in Tempest's chronological-return convention):

```
score_pos = ⟨target(u), context(v)⟩ / √d
neg ~ K nodes sampled uniformly from training-destination pool
score_neg = ⟨target(u), context(neg)⟩ / √d

L_contrast = -log σ(score_pos) - Σ_neg log σ(-score_neg)
```

This is SGNS, applied to walk endpoints (not walk-internal positions).

**Why endpoint, not all walk positions:** the diagnostic showed `1/K` weighting was crushing deep hops. Walk-internal positions are noisy — path-dependent, not distribution-meaningful. The endpoint of a walk under a bias function *is* the distribution we want to match.

**Why contrastive, not regression against log π:** SGNS converges to shifted-PMI factorization without ever computing PMI explicitly (Levy & Goldberg 2014). Regression against empirical `log π` has high variance; contrastive averages the matching implicitly.

**Phase 2 fallback rules** if endpoint-only regresses against Phase 1.5 (in priority order, both stay inside the distributional framing):
1. **Multiple endpoints per seed** — for each walk, sample K endpoint-like positions (the deepest few in the walk), treat each as an endpoint sample. Strictly more endpoint-distributional samples without changing the objective.
2. **Distance-weighted endpoint sampling** — sample a position along the walk with probability ∝ inverse depth from seed. Softens the endpoint commitment without breaking the framing.

Both fallbacks preserve "the supervision target is the walk distribution"; they just sample from that target more efficiently.

### Component 1.5: Node feature integration

**Status:** Component 1.5 is a refactor of existing infrastructure (preserved from the v3 codebase), in place from Phase 0.5 onward — **not a separately phased component**. No-op when the dataset has no node features; activates automatically otherwise. Wired alongside Component 0.

Three node-feature projections, learned:

```
proj_t   : ℝ^{d_n} → ℝ^d    (node feat → target role; SHARED across views)
proj_c_R : ℝ^{d_n} → ℝ^d    (node feat → recency-context role)
proj_c_S : ℝ^{d_n} → ℝ^d    (node feat → structural-context role)
```

Three fusion layers, learned (one per active site):

```
target_final     : Linear(2d, d)    fuses E_target ‖ proj_t(nf)
context_R_final  : Linear(2d, d)    fuses E_context_R ‖ proj_c_R(nf)
context_S_final  : Linear(2d, d)    fuses E_context_S ‖ proj_c_S(nf)
```

Lookups become role- and view-specific:

```
target(u):
  no nf:    E_target[u]
  with nf:  target_final([E_target[u] ‖ proj_t(nf[u])])

context_R(u):
  no nf:    E_context_R[u]
  with nf:  context_R_final([E_context_R[u] ‖ proj_c_R(nf[u])])

context_S(u):
  no nf:    E_context_S[u]
  with nf:  context_S_final([E_context_S[u] ‖ proj_c_S(nf[u])])
```

**Why one shared `proj_t` but separate `proj_c_R, proj_c_S`:** the multi-view design has multi-view on the *context side only*. There is one `E_target` table shared across views (this is the structural commitment — node target identity is unified), so logically there is one node-feature projection into target geometry. The two context tables encode different geometries (recency vs structural), so each gets its own node-feature projection into its own geometry. The asymmetry is consistent: `target` has one identity table, one projection, one fusion; `context` has two of each.

**Initialization:** `proj_c_R` and `proj_c_S` use identical Xavier init. The supervision signal (walks_R vs walks_S endpoint distributions) drives divergence, not initialization asymmetry. If the two projections evolve identically, that means the walk distributions are too similar — a failure mode the §9 pre-flight catches, not one that should be masked by different init.

**Streaming node features (preserved from current design):** `EmbeddingStore.update_node_feat(new_array)` overwrites a non-persistent buffer in place; the next lookup picks up new values automatically. Critical for datasets where node features evolve over time.

**Feature regime matrix (revised):**

| | nf absent | nf present |
|---|---|---|
| **ef absent** | Plain identity-table lookups; no projections active | All three fusion sites active (target_final, context_R_final, context_S_final) |
| **ef present** | Plain identity-table lookups; edge features ignored at scoring (proj_e not used in this design; see Phase 7/8 in §12) | All three fusion sites active; edge features ignored at scoring (see Phase 7/8 in §12) |

**Reading the matrix:** the columns determine whether node-feature fusion is active; the rows determine whether edge features are used (they are not in this design). All four cells produce a valid, runnable model. tgbl-wiki sits in the bottom-left cell (ef present, nf absent → plain identity-table lookups, edge features ignored). tgbl-coin / tgbl-flight / tgbl-comment / tgbl-review (with node features present per their respective TGB releases) sit in the bottom-right cell.

### Component 2: Two-view walks under different sampler distributions

Run Tempest with `enable_temporal_node2vec=True` and seed walks twice per batch:

- **View R (recency view):** `walk_bias="ExponentialWeight"`, `timescale_bound` tuned per dataset — small (encodes "recent neighbor")
- **View S (structural view):** `walk_bias="TemporalNode2Vec"`, `p=1.0`, `q=0.25` (BFS-like, encodes "locally close")

Two corresponding context tables: `E_context_R` and `E_context_S`. Shared `E_target`. Two alignment losses:

```
L_align = L_contrast(target, context_R, walks_R) + λ_S · L_contrast(target, context_S, walks_S)
```

Default `λ_S = 1.0`, ablatable.

The shared `E_target` is what makes this multi-view rather than two separate models — a node's "target" embedding satisfies both objectives simultaneously, encoding both recency-neighborhoods and structural-neighborhoods.

**Why two views, not three or more:** Tempest paper Figure 7b shows TN2V is ~5% slower than ExpWeight. Two views ≈ 2× walk-generation time, small relative to training. Three views start to matter.

**Why these two biases:** maximally different along the sampler's degrees of freedom. ExponentialWeight is first-order temporal (just edge timestamps). TemporalNode2Vec is second-order (`p, q` modulate based on the previous node).

### Component 3: Link MLP reads both views + time

Final link MLP input:

```
phi(u, v, t) = [
  # Recency view (4 blocks per direction × 2 directions = 8 blocks)
  target(u), context_R(v), target(u) ⊙ context_R(v), |target(u) - context_R(v)|,
  target(v), context_R(u), target(v) ⊙ context_R(u), |target(v) - context_R(u)|,
  # Structural view (2 interaction blocks)
  target(u) ⊙ context_S(v), target(v) ⊙ context_S(u),
  # Time encoding (Component 0)
  Φ(Δt_u), Φ(Δt_v), Φ(Δt_uv),
  # Cold-start sentinel bits
  is_cold_start_u, is_cold_start_v, is_cold_start_uv,
]  ∈ ℝ^{10·d + 3·d_time + 3}
```

LayerNorm → 2-layer GELU MLP → 1 logit. BCE-with-logits.

The "10 cross-table interaction blocks" referenced in Component 0 = 8 recency-view blocks (full 4-block-per-direction pattern) + 2 structural-view blocks (Hadamard interaction terms only, per direction). The structural-view's 2 blocks are intentionally lean — the structural view contributes interaction signal, not standalone vectors.

**No edge features at the scoring head.** Literature audit confirmed every SOTA method keeps edge features out of the scorer to avoid positive/negative asymmetry leak. Edge features are deferred to Phase 7+ as either an auxiliary prediction target (positives only) or a secondary alignment loss with a context_walk path; see §12.

---

## 5. Full math specification

### 5.1 Walk endpoint contrastive loss

```
For each batch B with seeds S = unique(batch.src ∪ batch.tgt):
    walks_V ← walk_gen_V.walks_for_nodes(seeds = S)   for V ∈ {R, S}
    
    For each view V ∈ {R, S}:
        For each walk w (skip if lens < 2):
            u_w = walks_V.nodes[w, walks_V.lens[w] - 1]   # seed
            v_w = walks_V.nodes[w, 0]                      # endpoint
            
            score_pos = ⟨target(u_w), context_V(v_w)⟩ / √d
            Sample K_neg negatives: v_neg ~ Uniform(train_destinations)
            score_neg = ⟨target(u_w), context_V(v_neg)⟩ / √d
            
        L_V = -mean_w[ log σ(score_pos) ] - mean_{w,neg}[ log σ(-score_neg) ]
    
    L_align = L_R + λ_S · L_S
```

`target()` and `context_V()` resolve through Component 1.5's fusion sites when node features are present; otherwise plain table lookups.

### 5.2 Uniformity (kept, on E_target only)

```
S = unique(batch.src ∪ batch.tgt)
t̃_u = target(u) / ‖target(u)‖
L_uniform = log E_{u ≠ v ∈ S} [ exp(-γ · ‖t̃_u - t̃_v‖²) ]
```

### 5.3 Joint loss (Phase 1.5 determines λ_link)

```
L_total = L_align + η · L_uniform + λ_link · L_link
```

If `λ_link = 0`: separate optimizers (current behavior).
If `λ_link > 0`: single optimizer, link BCE backprops into embedding tables and node-feature projections.

### 5.4 Time-state maintenance (post-scoring block)

```
After scoring batch B:
    For each (u, v, t_i) in batch:
        last_event_time[u] ← max(last_event_time[u], t_i)
        last_event_time[v] ← max(last_event_time[v], t_i)
        last_edge_time[u, v] ← max(last_edge_time[u, v], t_i)
```

Use sparse storage (dict / hash) for `last_edge_time`. On large datasets, only seen pairs occupy memory — naturally bounded by unique-edge count. Initialization: `last_event_time[*] = 0`, `last_edge_time[(u,v)] = 0` (lookup default for unseen pairs).

### 5.5 Per-batch strict-causal order

```
1. seeds ← unique(batch.src ∪ batch.tgt)
2. walks_R, walks_S ← walk_gens.walks_for_nodes(seeds)   # PRE-ingest
3. L_emb (or L_total under Phase 1.5) backward → step
4. negs ← neg_sampler.sample(batch)
5. Compute Δt_u, Δt_v, Δt_uv from last_*_time (state ≤ B-1)
   Compute is_cold_start_u, is_cold_start_v, is_cold_start_uv
   L_link.backward (or part of L_total) → step
6. POST-SCORING (in order):
   neg_sampler.observe(batch.src, batch.tgt)
   walk_gen_R.add_edges(...)
   walk_gen_S.add_edges(...)
   update last_event_time, last_edge_time
```

---

## 6. Implementation plan — one day with Claude Code

**Critical starting state.** Phase 0.5 builds on the **master baseline** (~0.33 test MRR; plain dual-table + 8-block cross-table link MLP + walks-as-supervision via alignment + uniformity). This design deliberately drops the overnight session's stack (walk encoder, cross-pair attention, DyG node encoder, NodeMemory, co-occurrence, EdgeBank feature). The implementer must NOT start from `feature/edgebank-feature`, `feature/phase6-*`, or any other overnight feature branch — those add machinery that confounds every ablation in this plan. **Concrete check before starting Phase 0.5:** confirm `git log --oneline master -1` shows the v3 baseline commit, not the overnight session's final commit. Confirm test MRR on a quick training run is in the 0.32–0.34 range, not 0.49.

Eight phases. Each is a single Claude Code session. Total ~12 hours assuming ~1–2 hours per phase plus training. Phases are sequential.

### Phase 0.5: Time encoding ablation (1 hr)

**Goal:** measure how much time encoding alone moves the needle on the master baseline (0.33 test MRR).

Add Component 0 (time encoding + cold-start bits) to the master baseline. Component 1.5 (node-feature infrastructure) is preserved as-is — it's a refactor in place, no-op on wiki. No other changes. Compare against baseline.

**Decision criterion:**
- If +0.10 or more: time encoding is doing major work. Every subsequent phase locks it in.
- If +0.03 to +0.10: composes additively.
- If <+0.03: surprising. Debug `last_edge_time` population first (most likely bug source). Keep anyway if no bug found — should help on other datasets.

**Output:** time encoding baseline; lock-in decision.

### Phase 1: Diagnostic loss-weighting ablation (2 hrs)

**Goal:** confirm or refute that hand-weighting is redundant with the sampler's bias.

Three variants on the Phase 0.5 architecture:
- **A:** current `1/K · (1 + Δt/τ)^(-β)` (control)
- **B:** `1/K` only (drop time decay; sampler does it)
- **C:** uniform `α = 1` over valid walk positions

**Decision criterion:**
- B or C within ±0.01 of A: foundational hypothesis confirmed → proceed with uniform weighting.
- A wins by >0.02: hand-weighting real → keep distance weighting in contrastive design.

**Output:** weighting variant for the base.

### Phase 1.5: Joint training ablation (1 hr)

**Goal:** test whether link BCE should backprop into embedding tables (and node-feature projections, if active).

On Phase 1 winner, sweep `λ_link ∈ {0, 0.1, 0.3, 1.0}`. Single combined loss, single optimizer.

**Decision criterion:**
- `λ_link > 0` wins: lock in best for all downstream.
- `λ_link = 0` wins: keep two optimizers.

**Output:** joint vs separate training decision.

### Phase 2: Walk-endpoint contrastive, single view (2 hrs)

**Goal:** validate Component 1 in isolation.

Replace alignment loss with endpoint-matching contrastive. Single view (`ExponentialWeight`, current `timescale_bound`). Keep Phase 0.5 time encoding, Phase 1 weighting choice, Phase 1.5 joint-training decision.

**Decision criterion:**
- Comparable or better than Phase 1.5: endpoint-matching premise confirmed → proceed.
- Regression >0.05: apply fallback rule 1 (multiple endpoints per seed). If still regression: fallback rule 2 (distance-weighted endpoint).

**Output:** validated contrastive loss formulation.

### Phase 3: `max_time_capacity` window sweep (1 hr)

**Goal:** find wiki's empirical recency horizon.

Configure Tempest with `max_time_capacity ∈ {6h, 1d, 3d, 7d, ∞}` (seconds: 21600, 86400, 259200, 604800, -1). Single view, Phase 2 architecture.

**Decision criterion:** pick window with highest val MRR.

**Output:** per-dataset `max_time_capacity_R` for recency view.

### Phase 4: Two-view multi-bias (3 hrs)

**Goal:** the main bet.

Pre-flight (10 min, before Phase 4 implementation): **measure walk-distribution divergence** under the two biases, stratified by node degree (see §9). If aggregate KL/JS is small but per-bucket divergence (especially low-degree nodes) is meaningful, proceed. If divergence is small across all buckets, Phase 4 is unlikely to pay off — re-evaluate.

Add structural view with `TemporalNode2Vec(p=1.0, q=0.25)`. Add `E_context_S` table + `proj_c_S` + `context_S_final` if node features present. Both views feed into 10-block link MLP. Recency view uses Phase 3's window; structural view uses `max_time_capacity = ∞` initially.

**Decision criterion:**
- +0.05+ over Phase 3: multi-view bet confirmed.
- +0.00 to +0.03: marginal → still proceed, ablate `λ_S` in Phase 5.
- Regression: structural view hurting → ablate `λ_S ∈ {0.1, 0.3, 1.0}` to find right weight.

**Output:** validated multi-view architecture.

### Phase 5: Per-view tuning (2 hrs)

**Goal:** squeeze the design.

One knob at a time, starting from Phase 4 defaults:
- Structural view's `max_time_capacity` ∈ {1d, 3d, 7d, ∞}
- Structural view's `q` ∈ {0.1, 0.25, 0.5, 1.0}
- `λ_S` ∈ {0.3, 1.0, 3.0}

Total ~11 runs (not full grid).

**Output:** final hyperparameters.

### Phase 6: Ablation matrix + error analysis (2 hrs)

**Ablation matrix (9 training runs):**
1. Phase 0 baseline (~0.33 test MRR)
2. Phase 0.5 (+ time encoding)
3. Phase 1 winner (+ loss weighting)
4. Phase 1.5 winner (+ joint training)
5. Phase 2 (+ endpoint contrastive)
6. Phase 3 (+ time-windowed walks)
   7a. **Phase 4 (+ multi-view, `d_emb = 128`)** — the multi-view result. Three embedding tables: `E_target`, `E_context_R`, `E_context_S`; total `3 · 128 · N = 384·N` table parameters.
   7b. **Single-view at `d_emb = 192`** — strict parameter-matched control. Two embedding tables: `E_target`, `E_context`; total `2 · 192 · N = 384·N` table parameters. Recency view only; all `d_emb`-scaled components (link MLP input width, fusion layers if nf present, time-encoding projections that depend on `d_emb`) scale together to `d = 192` — the link MLP input is `8·192 + 3·d_time + 3 = 1635` (single view → 8 cross-table blocks, not 10). All other choices identical to row 7a (Phase 1 weighting, Phase 1.5 joint training, Phase 2 contrastive, Phase 3 windowed walks, Component 0 time encoding).
8. Phase 5 (final, tuned, multi-view + winning hyperparameters)

**Ablation claim:** row 7a > row 7b (multi-view at strictly matched embedding-table parameter count beats single-view). If they tie, contribution is "more parameters helped" — caveat needed, paper claim weakens.

**Note on overfit risk for row 7b:** the overnight session showed `d_emb = 192` overfit on wiki, but that was under the kitchen-sink architecture (walk encoder + DyG node encoder + memory + co-occurrence + EdgeBank feature). The lean v1.4 design (contrastive + time encoding + multi-view, no walk encoder) may not overfit at 192. If 7b does overfit, that's itself a finding worth reporting: "multi-view-128 beats single-view-192 because single-view-192 overfits the same data that multi-view-128 handles via its inductive structure." An optional row 7c at `d_emb = 180` can be added as a robustness check if reviewers want to confirm the 7a > 7b result doesn't depend on 7b's exact width.

**Error analysis (post-hoc, no extra training):** take Phase 5 (final, row 9) model's eval predictions and split MRR by:
- **Positive's last `(u, v)` recency bucket:** never seen, < 1h ago, < 1d ago, < 1w ago, older
- **Source node degree:** low / mid / high (terciles)
- **Whether v is a hub** (top-K most-popular destinations)

This gives concrete claims for the paper. If multi-view wins primarily on the "never-seen-before but structurally close" bucket, the contribution narrative is defensible: "where EdgeBank fails by definition, our multi-view structural supervision learns the connection." If multi-view wins uniformly, the claim is weaker but broader.

**Output:** per-component contribution table + per-bucket error analysis.

---

## 7. Bets, called out

**Bet 0 (Phase 0.5 tests):** Time encoding gives the link MLP signal it currently lacks; closes part of the gap to leaderboard methods. Most likely to pay off — every leaderboard method has it.

**Bet 1 (Phase 1 tests):** Sampler's distribution is informative enough that hand-weighting alignment is redundant.

**Bet 1.5 (Phase 1.5 tests):** Letting link BCE backprop into embeddings helps under contrastive supervision (similar-shape signal to link BCE, so contamination concern is weaker).

**Bet 2 (Phase 2 tests):** Walk endpoint is a meaningful supervision target. Fallback rules apply if not directly.

**Bet 3 (the big one, Phase 4 + Phase 6 row 7a vs 7b tests):** Two walk biases produce qualitatively different geometry; shared `E_target` learning both > single bias with more capacity. Capacity-matched ablation (row 7a vs 7b) is the proper test.

**Bet 4 (Phase 5 tests):** TemporalNode2Vec at `p=1.0, q=0.25` produces meaningfully BFS-like behavior. `q` sweep covers this.

**Bet 5 (cross-dataset, future):** Multi-view recipe generalizes to tgbl-coin, tgbl-flight, tgbl-review, tgbl-comment. Tempest Figure 10(b) is supporting evidence for recency-view component.

---

## 8. Expected outcomes (honest)

| Stage | Expected Test MRR | What it means |
|---|---|---|
| Phase 0 (reference) | 0.33 | Current baseline |
| Phase 0.5 (+ time encoding) | 0.40–0.50 | Closes the time-channel gap |
| Phase 1 (+ loss weighting) | 0.40–0.52 | Marginal but principled |
| Phase 1.5 (+ joint training) | 0.42–0.55 | If positive: composes downstream |
| Phase 2 (+ endpoint contrastive) | 0.45–0.58 | EdgeBank-tw range via learned embeddings |
| Phase 3 (+ windowed walks) | 0.47–0.60 | Recency horizon properly set |
| Phase 4 (+ multi-view) | 0.58–0.72 | Main bet; past TGN territory if landed |
| Phase 5 (tuned) | 0.65–0.78 | Plausibly leaderboard-competitive |

0.82+ remains a stretch goal, not a plan target. The paper's narrative does NOT depend on Phase 5 ≥ 0.72.

---

## 9. Pre-flight checks

### Phase 4 walk-distribution divergence test (mandatory, 10 min before Phase 4)

```python
import numpy as np
from temporal_random_walk import TemporalRandomWalk

def compute_per_seed_js_divergence(walks_R_data, walks_S_data, seeds, num_walks_per_node):
    """For each seed i in seeds, build empirical endpoint distribution
    over node IDs from walks_R_data and walks_S_data (the K walks per seed),
    normalize to probabilities, compute JS divergence between the two.
    
    walks_*_data: tuple (nodes, timestamps, lens, edge_features) as returned
                  by Tempest's get_random_walks_and_times_for_nodes
    Returns: np.ndarray of shape [len(seeds)] with per-seed JS divergence.
    """
    nodes_R, _, lens_R, _ = walks_R_data
    nodes_S, _, lens_S, _ = walks_S_data
    K = num_walks_per_node
    js_per_seed = np.zeros(len(seeds))
    for i, seed in enumerate(seeds):
        # endpoint = nodes[walk_idx, 0] for each of K walks per seed
        ep_R = nodes_R[i*K:(i+1)*K, 0]    # K endpoints under recency bias
        ep_S = nodes_S[i*K:(i+1)*K, 0]    # K endpoints under structural bias
        # empirical distributions over endpoint node ids
        all_ids = np.unique(np.concatenate([ep_R, ep_S]))
        p_R = np.array([np.mean(ep_R == nid) for nid in all_ids])
        p_S = np.array([np.mean(ep_S == nid) for nid in all_ids])
        # JS divergence
        m = 0.5 * (p_R + p_S)
        eps = 1e-12
        kl_R = np.sum(p_R * np.log((p_R + eps) / (m + eps)))
        kl_S = np.sum(p_S * np.log((p_S + eps) / (m + eps)))
        js_per_seed[i] = 0.5 * (kl_R + kl_S)
    return js_per_seed


# After training, with full train edges ingested:
tw = TemporalRandomWalk(
    is_directed=False, use_gpu=False,
    enable_temporal_node2vec=True,
    temporal_node2vec_p=1.0, temporal_node2vec_q=0.25,
)
tw.add_multiple_edges(train_src, train_dst, train_ts)

# Bucket seeds by degree
node_degrees = compute_degrees(tw)
all_nodes = np.array(list(node_degrees.keys()))
sorted_nodes = sorted(all_nodes, key=lambda n: node_degrees[n])
buckets = {
    'low_decile':  sorted_nodes[:len(sorted_nodes) // 10],
    'mid_80pct':   sorted_nodes[len(sorted_nodes) // 10 : -len(sorted_nodes) // 10],
    'high_decile': sorted_nodes[-len(sorted_nodes) // 10:],
}

K = 10
for bucket_name, seeds in buckets.items():
    seeds_arr = np.array(seeds[:1000], dtype=np.int32)   # cap to 1k for speed
    walks_R = tw.get_random_walks_and_times_for_nodes(
        seeds_arr, max_walk_len=20, walk_bias="ExponentialWeight",
        num_walks_per_node=K, walk_direction="Backward_In_Time",
    )
    walks_S = tw.get_random_walks_and_times_for_nodes(
        seeds_arr, max_walk_len=20, walk_bias="TemporalNode2Vec",
        num_walks_per_node=K, walk_direction="Backward_In_Time",
    )
    js_div = compute_per_seed_js_divergence(walks_R, walks_S, seeds_arr, K)
    print(f"{bucket_name}: mean JS = {js_div.mean():.4f}, p50 = {np.median(js_div):.4f}")
```

**Decision rule:**
- If `mid_80pct` mean JS divergence > 0.1: multi-view bet is alive; proceed with Phase 4.
- If `mid_80pct` JS < 0.05 but `low_decile` > 0.1: multi-view helps long tail only; proceed, but expect small absolute MRR gain.
- If all buckets < 0.05: multi-view collapses to one view on this dataset; skip Phase 4, jump to Phase 5 single-view tuning.

**Stratification matters because** high-degree nodes (popular pages on wiki) dominate aggregate divergence — both biases collapse to "recent edges to popular co-edited pages" for hubs. The bet pays off in the long tail. Aggregate divergence hides that.

### Tempest API check (10 min, before Phase 4)

```python
from temporal_random_walk import TemporalRandomWalk
import numpy as np

tw = TemporalRandomWalk(
    is_directed=False, use_gpu=False,
    enable_temporal_node2vec=True,
    temporal_node2vec_p=1.0, temporal_node2vec_q=0.25,
    max_time_capacity=259200,  # 3 days
)
tw.add_multiple_edges(
    np.array([0, 1, 2], dtype=np.int32),
    np.array([1, 2, 3], dtype=np.int32),
    np.array([100, 200, 300], dtype=np.int64),
)

seeds = np.array([1, 2], dtype=np.int32)
walks_exp = tw.get_random_walks_and_times_for_nodes(
    seeds, max_walk_len=5, walk_bias="ExponentialWeight",
    num_walks_per_node=2, walk_direction="Backward_In_Time",
)
walks_tn2v = tw.get_random_walks_and_times_for_nodes(
    seeds, max_walk_len=5, walk_bias="TemporalNode2Vec",
    num_walks_per_node=2, walk_direction="Backward_In_Time",
)
print("OK" if walks_exp and walks_tn2v else "Need two instances")
```

If one instance serves both biases: Component 2 uses one instance with per-call bias.
If TN2V is construction-locked: two instances (Tempest is cheap; either fine).

Confirm `max_time_capacity` is honored:

```python
tw_windowed = TemporalRandomWalk(is_directed=False, use_gpu=False, max_time_capacity=50)
tw_windowed.add_multiple_edges(
    np.array([0, 1, 2, 3], dtype=np.int32),
    np.array([1, 2, 3, 4], dtype=np.int32),
    np.array([10, 20, 100, 110], dtype=np.int64),
)
# At t=110, edges at t=10, 20 should be evicted (110 - 50 = 60)
print("Edge count after window evict:", tw_windowed.get_edge_count())
```

---

## 10. What makes this a paper, not just a leaderboard chase

**The paper's strongest narrative is NOT "we beat 0.82." It is "we identify a within-batch state-update leak shape that inflates the leaderboard, AND we build the first method competitive without it."**

Specific claims, each with an ablation row in Phase 6:

1. **Time encoding closes the largest input-channel gap to the leaderboard** — Phase 0.5 vs Phase 0
2. **Hand-weighted alignment is redundant with a recency-biased sampler** — Phase 1 vs Phase 0.5
3. **Joint training of embeddings + link MLP helps under contrastive supervision** — Phase 1.5 vs Phase 1
4. **Walk-endpoint contrastive is a cleaner supervision target than walk-position alignment** — Phase 2 vs Phase 1.5
5. **Time-windowed walks beat unbounded ones for embedding supervision** — Phase 3 vs Phase 2
6. **Multi-view supervision under different walk biases beats single-view AT MATCHED PARAMETER COUNT** — Phase 6 row 7a vs row 7b
7. **Error-bucket attribution: where does multi-view actually help?** — Phase 6 error analysis (recency × degree × hub bucket breakdown)

**The paper-defining experiment:** honest-protocol re-baselines for TPNet, DyGFormer, TGN under strict-causal regime, compared against this method under the same protocol. If those leaderboard numbers drop under honest protocol and ours holds, the contribution is "leaderboard inflation + first leak-free method" — sharper than absolute MRR. The "we beat 0.82" version doesn't survive a reviewer pointing out wiki is small and saturated by heuristics; the honest-protocol version does.

---

## 11. Strict-causal protocol (non-negotiable, unchanged)

```
1. seeds ← unique(batch.src ∪ batch.tgt)
2. walks_R, walks_S ← walk_gens.walks_for_nodes(seeds)    # PRE-ingest Tempest state
3. L_emb (or L_total) backward → step
4. negs ← neg_sampler.sample(batch)                        # PRE-batch reservoir
5. Compute Δt features + cold-start flags (state ≤ B-1)
   L_link.backward (or part of L_total) → step
6. POST-SCORING BLOCK (feeds batch B+1):
   neg_sampler.observe(batch.src, batch.tgt)
   walk_gen_R.add_edges(...)
   walk_gen_S.add_edges(...)
   update last_event_time, last_edge_time
```

All new state (`last_event_time`, `last_edge_time`) updates only in step 6, strict-causal by construction.

---

## 12. What's deliberately NOT in this design

- **Hand-rolled MRR.** TGB Evaluator only.
- **TGN-style memory.** Roadmap. If added, must use raw-message-store pattern.
- **Walk encoder feeding link MLP.** Overnight session got +0.098, but this design replaces supervision rather than adds decoder. If validated, walk encoding is orthogonal extension afterward.
- **Edge features at link MLP.** Literature audit confirmed.
- **Edge features anywhere in the new design.** Two paths if needed in future work: **Phase 7** (auxiliary prediction target — predict `e_uv` from `(target(u), context_R(v), Φ(Δt_uv))` for positives only; edge features supervise embeddings indirectly without entering the scorer); or **Phase 8** (reintroduce a per-hop context_walk path as a secondary alignment loss alongside endpoint contrastive). Neither is in the main plan.
- **Feature-based init for E_target.** Streaming-feature datasets break it.
- **Time encoding inside walks (TGAT-style per-hop).** Could add later. Current design uses time encoding only at the scoring head; the alignment loss gets temporal signal through the sampler bias.

---

## 12.1 Implementation watch-list (not doc fixes; track during implementation)

**Cold-start bits and LayerNorm.** The 3 binary cold-start bits go into the link MLP's input alongside the 10·d + 3·d_time wide blocks (total input dim 10·128 + 3·32 + 3 = 1379). LayerNorm normalizes across input dim; the 3 binary positions have low variance relative to the d-wide blocks within a single sample, so after normalization their signal may be heavily attenuated. The first Linear can re-amplify them via its weights, but the model has to learn that the bit positions are signal-bearing.

This is not pre-emptively a problem — modern Linear+LayerNorm stacks generally recover such signals through training. But if Phase 0.5 underperforms the +0.10 expectation and the cold-start bits look implicated (e.g., the bits' contribution to the first Linear's output is small), mitigation is: separate the cold-start bits onto a parallel small MLP (e.g., 3 → 16 → 16) and concat at the LAST hidden layer of the link MLP, skipping the input LayerNorm. Don't pre-emptively complicate — implement plainly first, debug if Phase 0.5 is off-expectation.

---

## 13. Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `d_emb` | 128 | overnight confirmed 192 overfits on prior architecture; revisit under leaner design |
| `d_n` | dataset-specific | node feature dim if present; 0 disables node-feature path |
| `d_hidden_link` | 128 | link MLP hidden dim |
| `d_time` | 32 | time encoding dim (k=16 frequencies) |
| `max_walk_len` | 20 | 52% pin cap end-of-epoch under previous; Phase 5 may sweep |
| `num_walks_per_node` | 5 | per view |
| `target_batch_size` | 200 | overnight confirmed B=1000 regresses |
| `num_epochs` | 50 | matches current convergence |
| `K_neg_walk` | 5 | negatives per walk endpoint in contrastive |
| `max_time_capacity_R` | TBD Phase 3 | recency view window |
| `max_time_capacity_S` | -1 (∞) | structural view: full history |
| `tn2v_p` | 1.0 | return parameter |
| `tn2v_q` | 0.25 | in-out parameter (BFS-like) |
| `λ_S` | 1.0 | structural view weight |
| `λ_link` | TBD Phase 1.5 | joint training weight |
| `η_uniform` | 1.0 | uniformity weight |
| `γ_uniform` | 2.0 | uniformity sharpness |
| `uniformity_cap` | 20_000 | all-pairs Gram cap |
| `num_neg_per_pos` | 10 | training K for link BCE |
| `hist_neg_ratio` | 0.5 | matches TGB eval mix |
| `reservoir_size` | 32 | per-source Vitter R reservoir |
| `cold_start_dt_clamp` | `time_scale × 100` | clamp Δt to this before passing through Φ |
| `cold_start_sentinel_init` | 0 | initialization for `last_event_time` and lookup default for unseen pairs in `last_edge_time` |
| `emb_lr` | 1e-3 | Adam |
| `link_lr` | 1e-3 | Adam |
| `seed` | 42 | numpy + torch |

---

## 14. Leaderboard reference (May 2026)

| Method | tgbl-wiki-v2 Test MRR |
|---|---|
| Random | 0.0075 |
| EdgeBank-inf | 0.495 |
| EdgeBank-tw | 0.571 |
| GraphMixer | 0.594 |
| DyRep | 0.665 |
| TGN | 0.690 |
| CAWN | 0.711 |
| TNCN | 0.718 |
| DyGMamba | 0.739 |
| DyGFormer | 0.798 |
| HyperEvent | 0.810 |
| Heuristic(LocalGlobal) | 0.821 |
| **TPNet (#1)** | **0.827** |
| **This design (target)** | **0.65–0.78** |

Per May 2026 audit, top performers appear leak-free under within-batch state-update analysis. Honest protocol comparison is fair. Stretch goal: leaderboard-competitive (0.82+); plan target: convincingly past TGN territory under honest protocol; floor: clean ablation matrix with attributable per-component gains regardless of absolute MRR.

---

*Document version: v1.5, May 2026. Fixes from v1.4: row 7b clarifies that `d_emb`-scaled components (link MLP input, fusion layers, time-encoding projections) scale together to `d = 192`, removing an "all else identical" ambiguity that could mislead an implementer about which dimensions change with `d_emb`. Fixes from v1.3 (preserved): row 7b parameter-matching math corrected (`d_emb = 192` for strict table-parameter match, not 180; old "2 × 128 × N ≈ 1 × 180 × N" was off by 40%), Phase 6 row numbering uses 7a/7b/8 instead of conflicting 7/8/9, §12.1 implementation watch-list added for cold-start bits and LayerNorm. Companion to Tempest paper (Salehin et al., arXiv:2605.16182).*
