# Phase S Group A3 Amendment — Loss Function Search

> **ARCHIVAL as of v2.3 (2026-05-19).** Load-bearing content integrated
> into `walk_distribution_matching_embedding_v2.md` §4.7. This file is
> preserved for the full reasoning trail (per-loss analysis, deferred
> candidates, ablation discipline) that doesn't fit in the main plan.
> The v2.md is the source of truth; this is the reasoning record.

**Document type:** Amendment to `walk_distribution_matched_design_v2.2.md`
**Status:** ARCHIVAL (integrated into v2.3 §4.7).
**Created:** after A2 single-seed result (alignment-on 0.7083 vs A2-off 0.7092 on seed 42) and the loss-function research deliverable.
**Reads alongside:** v2.2 §3 (anchor validation, confirmed at 0.7070 ± 0.0016), v2.2 §4 (Phase S frame), v2.2 §6 (architecture lock).

---

## 1. Why this amendment exists

Phase S Group A3 was originally specified in v2.2 §4.1 as three variants of supervision target:

- Current per-position alignment (the A1 winner)
- Endpoint contrastive (SGNS on walk endpoints)
- Multi-endpoint contrastive (sample K endpoint-like positions per walk)

These three are all variations within a narrow family — inner-product similarity with positional sampling. The Phase 0.5 diagnostic (2-epoch 0.7070 → 50-epoch 0.4269; cross-table column norms grow 5× while test MRR drops 0.28) and the Group A2 seed-42 result (alignment-on 0.7083 ties alignment-off 0.7092 within anchor std 0.0016) indicate that the current alignment+uniformity loss form may not be the right *family* for this task — not just the wrong *weighting* (A1) or *target position* (A3 original).

A loss-function research deliverable evaluated 8 candidates across contrastive, distributional, sequence, time-aware, and regularization families. Three primary candidates are recommended for Phase S Group A3 from that research. **One additional candidate — a diagnostic-informed custom regularizer (§4.4 norm-brake) — is added on top, derived directly from the Phase 0.5 column-norm finding rather than from prior literature.** This amendment specifies all of them with the implementation detail needed for one-shot execution.

**Honest provenance of the cells in this amendment:** A3.1 (InfoNCE), A3.2 (triplet), and A3.3 (SGNS) are standard published losses, applied to our setting with the existing v3 positional weight. The hyperparameter defaults are literature-defaults, not custom-tuned. §4.4 norm-brake is the only loss component in this amendment that is custom-derived from the diagnostic — see §4.4 for its rationale and the explicit caveat that it has no published precedent in this form.

**What is NOT in this amendment (and why).** An earlier draft (v1.1) included an EdgeBank-as-teacher distillation auxiliary. That was removed in v1.2: distilling a known heuristic's predictions into the embeddings dilutes the contribution claim from "we found a loss that produces useful walks-supervised embeddings" to "we trained embeddings to encode EdgeBank." For paper purposes, the search must isolate what the loss family alone contributes — independent of heuristic-supervised priors. If post-Phase-S analysis shows a need for distillation-style auxiliaries, they belong in a follow-up paper, not this one.

The amendment does NOT change any other part of v2.2:
- Anchor validation result (0.7070 ± 0.0016) stays as the floor for §4.4 success criterion
- Group A1 (within-family weighting), A2 (alignment on/off), C (joint training), D (regularization), E (link MLP head structure) are unchanged
- Locked architecture in v2.2 §6 is unchanged
- Strict-causal protocol in v2.2 §7 is unchanged
- Compute-graph deduplication guidance in v2.2 §4.6 is unchanged

---

## 2. Context that drives the cell choices (preserve across sessions)

These facts are load-bearing for the cell design. If a future session loses context, these are the things to re-read first.

**Empirical anchors:**

- NeurTWs (Jin et al., NeurIPS 2022) ran a literal "BCE vs multi-negative contrastive" ablation on continuous-time dynamic graphs. Every per-dataset row in their Table 4 Ablation 5 deteriorated under BCE. This is the strongest direct empirical signal that multi-negative contrastive supervision beats binary supervision on this kind of data.
- TPNet (rank 1 on tgbl-wiki-v2 at Test MRR 0.827) uses vanilla BCE per its DyGLib base. Loss form is NOT the bottleneck for absolute leaderboard performance on wiki; architecture is (temporal walk matrix projection).
- The TGB leaderboard numbers (TPNet 0.827, Heuristic-LocalGlobal 0.821, HyperEvent 0.810, DyGFormer 0.798, CAWN 0.711, TGN 0.396) are 1-vs-all on tgbl-wiki-v2. Our 0.7070 baseline is on ~1000-negative protocol. These are NOT directly comparable; do not chase absolute leaderboard rank.
- Triplet loss has a literal self-limiting gradient (∇L = 0 once positive clears margin). 2025 head-to-head study (Zeng arXiv:2510.02161v2): triplet retains 2.4× more intra-class variance than contrastive on synthetic; beats contrastive at retrieval r@1 on CIFAR-10 (0.9192 vs 0.8433), CARS196 (0.2982 vs 0.2542), CUB-200 (0.3421 vs 0.3154).
- Word2vec SGNS has the Levy & Goldberg (NIPS 2014) shifted-PMI factorization guarantee. This is the only candidate with a *principled* early-stop signal (reconstruction error against empirical walk PMI matrix) independent of val MRR.

**Goal-shaping facts:**

- The goal of the loss-function search is to: (1) eliminate the over-training cliff so 50-epoch test MRR doesn't fall below the 2-epoch value; (2) generalize across tgbl-wiki, tgbl-coin, tgbl-flight, tgbl-comment, tgbl-review. The goal is NOT to push wiki past 0.71 by loss change alone — wiki's 99.1% uv-cold-start rate and EdgeBank-tw 0.571 floor mean Component 0's signal dominates wiki regardless of embedding quality.
- Wiki is expected to tie the anchor (~0.7070) across loss families, NOT to be lifted by loss-form change. If a primary loss beats the anchor on wiki by > 0.005 reproducibly, that's a surprise worth investigating. The locked production loss should be chosen primarily on cliff behavior and cross-dataset transferability, NOT on wiki peak.
- On tgbl-coin and tgbl-flight (where recurrence is less dominant), the loss change is expected to matter more. NeurTWs reports its biggest gains on networks where motif recurrence is structurally distinct (CollegeMsg, Enron) — directly analogous to coin (transaction recurrence) and flight (route recurrence). Expected lift over BCE: +0.03 to +0.08 MRR.

**Diagnostic facts (resolved, do not re-investigate):**

- Cold-start bits are not LayerNorm-washed; the 3 binary bits get amplified to 1.78× cross-table mean in the 50-ep model.
- Seed plumbing is healthy (init-divergence check passed; init genuinely varies, the bit-tight loss-trajectory reproduction across seeds is a real property of the loss surface at 2 epochs).
- The 0.7070 ± 0.0016 anchor is reproducible across seeds {42, 7, 13}.

---

## 3. What this amendment changes in v2.2 §4.1

Replace Group A3 in v2.2 §4.1 with the new specification:

**Group A3 (REVISED) — Supervision objective / loss family.**

When alignment loss is active (A2-on), what loss family should supervise the walks?

- **A3.1 — Multi-positive InfoNCE with positional weighting** (drops uniformity entirely)
- **A3.2 — Triplet/margin loss with semi-hard mining** (drops uniformity entirely)
- **A3.3 — SGNS / Skip-gram with negative sampling** (drops uniformity entirely; sigmoid saturation replaces it)

Plus one auxiliary loss (orthogonal to A3 primary, composable with any):

- **A3.x_normbrake — Custom diagnostic-derived norm-brake regularizer** (custom; see §4.4 for full rationale)

The original v2.2 A3 cells (per-position alignment / endpoint contrastive / multi-endpoint contrastive) are subsumed:
- "Endpoint contrastive" is A3.1 with `walk_window=1` (single positive per walk = endpoint). The new A3.1 generalizes by including all walk positions as multi-positive with the existing positional weight.
- "Multi-endpoint contrastive" is a hyperparameter range within A3.1 (varying which positions count as positive).
- "Per-position alignment" is the v3 baseline already covered by Group A1; only its weighting (A1.1/A1.2/A1.3) is searchable.

Group A1, A2, C, D, E specifications and all decision rules from v2.2 §4 are preserved unchanged.

---

## 4. Specifications of the three primary loss cells and one custom regularizer

### 4.1 A3.1 — Multi-positive InfoNCE with positional weighting

**Description.** For each walk W returned chronologically by Tempest (nodes[0] = deepest past, nodes[L-1] = seed), the anchor is `target(seed)` and every earlier walk position is a multi-positive context with weight `w(i) = 1/K · (1 + Δt_{u, n_i}/τ_pos)^(-β)` (the existing positional weight from the v3 alignment loss). Negatives are in-batch other anchors' contexts plus uniform random destinations.

**Formula:**

```
For each walk W = (n_0, n_1, ..., n_{L-1}) with seed u = n_{L-1}:
  For i = 0 to L-2:
    pos_score = target(u) · context(n_i) / τ_contrastive
    neg_scores = [target(u) · context(v_j) / τ_contrastive for v_j in negatives]
    
    L_i = -w(i) · log(exp(pos_score) / (exp(pos_score) + sum(exp(neg_scores))))
  
  L_walk = mean over i

L_A3.1 = mean over walks
```

**Critical implementation details:**

1. **Delete `L_uniform` entirely** when running A3.1. Set `η_uniform = 0`. InfoNCE's denominator subsumes the uniformity term; running both creates double-counting and unstable optimization.
2. **Use in-batch negatives plus uniform-random negatives.** In-batch: take the other anchors' context embeddings from the same batch (free; reuses already-computed embeddings). Uniform: sample from training-destination pool (same pool as link BCE).
3. **Temperature `τ_contrastive`** is a new hyperparameter; defaults below.
4. **Positional weight `w(i)`** uses the existing `1/K · (1 + Δt/τ)^(-β)` formula from `config.A1_weighting` — match whatever the Group A1 winner is.

**Hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `τ_contrastive` | 0.1 | Try 0.07 and 0.2 as ablations if time permits |
| `β_temporal` | match Group A1 winner | Same as existing |
| `τ_pos` | median walk-step Δt for the dataset | Compute once from training walks |
| `num_neg_in_batch` | 256 | Limited by batch size; use min(B-1, 256) |
| `num_neg_uniform` | 256 | Sample from training-destination pool |
| `walk_window` | use Tempest default (full walk) | All earlier walk positions are positives |
| `η_uniform` | **0 (mandatory)** | Delete this term |

**Implementation complexity:** Low. The existing v3 alignment loop iterates over walk positions with the same positional weight. Replace the per-position alignment-style pull with a log-softmax over (1 positive context, N negatives). The negative sampling reuses the existing link-BCE negative sampler. Estimated diff: ~30 lines.

**Failure modes to watch:**

- False-negative collisions: walks often revisit the same destination node (heavy-tail of wiki page popularity). If many in-batch negatives happen to actually be valid neighbors of the anchor, this inflates the loss. The mitigation is unigram^0.75 sampling for the uniform-random negatives (same distribution as SGNS).
- Temperature mis-tuning: `τ < 0.05` produces gradient explosion on hard negatives; `τ > 0.5` collapses to uniform. Stick to `τ = 0.1` unless you observe instability.
- **Do NOT enable hard-negative mining.** Bipartite graphs like wiki amplify collapse under hard mining.

**Expected outcome on tgbl-wiki:** Test MRR 0.700 ± 0.020. Cliff reduced from −0.28 to roughly −0.05 to −0.10 over 50 epochs.

**Expected outcome on tgbl-coin/flight (future work):** +0.03 to +0.08 over BCE-only baseline.

---

### 4.2 A3.2 — Triplet/margin loss with semi-hard mining

**Description.** For each walk W, sample triplets (anchor `target(seed)`, positive `context(walk-internal position sampled UNIFORMLY)`, negative `context(uniform random destination)`). Compute the cosine-margin hinge **and multiply it by the positional/temporal weight `w(p)`** at the sampled position. This makes Δt enter the loss as a multiplicative weight on the per-triplet hinge — identical to how InfoNCE and SGNS use `w_pos` — so the three primaries are mutually consistent in their timestamp use. (Earlier draft used Categorical(w(i)) sampling with an unweighted hinge; the audit in §4.7 of v2.3 showed that throws away Δt after the multinomial. The current spec is the fix.) This is the only candidate with literal self-limiting gradient — the textbook fix for the over-training cliff.

**Formula:**

```
For each walk W = (n_0, n_1, ..., n_{L-1}) with seed u = n_{L-1}:
  Sample position p ~ Uniform({i : 0 ≤ i ≤ lens − 2 ∧ position i is valid})
  Sample negative q ~ uniform from training-destination pool

  w_p     = (1 / depth(p)) · (1 + Δt(p) / time_scale)^(-β)   # same w(i) family used by A1 / A3.1 / A3.3
  pos_sim = cosine(target(u), context(p))
  neg_sim = cosine(target(u), context(q))

  L_triplet_walk = w_p · max(0, m − pos_sim + neg_sim)         # hinge weighted by Δt-decay

L_A3.2 = Σ_walks (keep · L_triplet_walk) / Σ_walks (keep · w_p)   # mean over kept triplets, weight-normalised
```

where `keep` is the semi-hard mining mask defined below.

**Semi-hard mining (mandatory):** within each batch, after computing all triplets, restrict gradient backprop to triplets where `pos_sim - neg_sim < m` (negative is inside the margin band) AND `neg_sim < pos_sim` (negative not closer than positive). This keeps the loss focused on the boundary triplets and avoids the collapse mode of hard mining.

**Critical implementation details:**

1. **Use cosine similarity** with `margin = 0.5`, not raw dot product. Raw dot product on 128-d Xavier-init embeddings has unbounded magnitude scale, making margin selection brittle across training. Cosine normalizes to [-1, 1] and `m = 0.5` is the literature default.
2. **Delete `L_uniform` entirely.** Margin loss provides no uniformity pressure; rely on weight decay for L2 norm control.
3. **Add weight decay** `1e-4` on `E_target, E_context` (this is also a Group D hyperparameter; pick the same value if D has been tuned).
4. **Sample positive `p` UNIFORMLY from valid walk positions; weight the hinge by `w(p)`.** Δt enters the loss the same way as InfoNCE / SGNS — as a per-pair loss-weight multiplier, NOT as a categorical sampling probability. This avoids the "Δt discarded after multinomial" failure mode of the earlier draft and keeps the three primaries directly comparable.
5. **Sample negative `q` uniformly from training destinations** (same pool as link BCE negative sampler).

**Hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `margin m` | 0.5 (cosine) | 1.0 if you must use normalized dot product; 5.0 only if raw dot product (NOT recommended) |
| `weight_decay_emb` | 1e-4 | Mandatory; supplies the norm control that uniformity used to provide |
| `mining` | "semi-hard" | NOT "hard"; hard mining collapses bipartite graphs |
| `triplets_per_walk` | 1 | One triplet per walk; over-sampling causes spurious gradient correlation |
| `η_uniform` | **0 (mandatory)** | Delete this term |

**Implementation complexity:** Low. ~20 lines: replace the alignment loop with cosine-similarity-based margin computation, plus a mask for semi-hard mining.

**Failure modes to watch:**

- Hard-negative collapse: if you accidentally enable hard mining (negative closer than positive), the embedding space collapses to a low-dimensional manifold. Stick to semi-hard.
- Margin too large (`m > 1.5`): loss is rarely zero, no self-limiting behavior, embeddings drift unbounded.
- Margin too small (`m < 0.2`): loss is always zero, no learning signal.
- Easy-triplet drift: even at zero loss, satisfied triplets' effective margins drift slowly under SGD over many epochs (arXiv:2603.26389). This is orders of magnitude slower than InfoNCE/uniformity-driven drift but eventually shows up at 100+ epochs. Not a concern for our 50-epoch ceiling.

**Expected outcome on tgbl-wiki:** Test MRR 0.690 ± 0.020 (slightly below A3.1 peak). **The cliff is largely eliminated:** probable 50-epoch test MRR within ±0.01 of 2-epoch test MRR. This is the candidate to choose if the success criterion is "no cliff" rather than "peak performance."

**Expected outcome on tgbl-coin/flight:** +0.02 to +0.05 over BCE. Margin loss preserves rank structure on long-tailed degree distributions, which coin and flight have.

---

### 4.3 A3.3 — SGNS (Skip-gram with negative sampling)

**Description.** The classical word2vec loss applied to walk-context pairs, with negatives drawn from the empirical unigram^0.75 distribution. The Levy & Goldberg (NIPS 2014) result gives a principled stopping criterion: at convergence, `target · context^T` factorizes the shifted PMI matrix `M_{u,v} = log(P(v|u)/P(v)) - log(k)`. Stop when the Frobenius distance between current and empirical PMI plateaus.

**Formula:**

```
For each (anchor u, walk-internal position v with weight w(i)):
  pos_score = sigmoid(target(u) · context(v))
  
  Sample k negatives v_- ~ P_n where P_n(x) ∝ d_x^{0.75}
  neg_scores = sigmoid(-target(u) · context(v_-))   # note the minus sign
  
  L_SGNS_per_pair = -log(pos_score) - sum_{v_-} log(neg_scores)
  L_SGNS = mean over walks/positions, weighted by w(i)
```

**Critical implementation details:**

1. **Delete `L_uniform` entirely.** Sigmoid saturation provides natural norm-bounding; running uniformity on top creates redundant pressure.
2. **Use the unigram^0.75 distribution** for negatives, NOT uniform sampling. Compute node degrees over training edges once, raise to 0.75 power, normalize, cache. This is the Mikolov 2013 default and is the empirical regime where the Levy & Goldberg factorization result applies.
3. **Use `BCEWithLogitsLoss` for numerical stability** (PyTorch built-in). Don't compute `log(sigmoid(x))` directly.
4. **Learning rate schedule** matters more than for InfoNCE/triplet: linear decay from `0.025` to `1e-3` over the first 5 epochs (Mikolov schedule), then constant at `1e-3`. If you don't implement the schedule, expect early-epoch divergence on negative gradients.
5. **Optional but recommended: subsampling frequent positives.** Mikolov's `t = 1e-5` formula: discard each (u, v) walk-pair with probability `1 - sqrt(t / f(v))` where `f(v)` is the relative frequency of v in walks. Counteracts wiki's heavy-tailed page popularity. If skipped, expect inflated target embeddings for frequent editors.

**Hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `k` (num negatives) | 5 | Classical default; up to 15 for small datasets |
| `P_n exponent` | 0.75 | Mikolov default; critical for the factorization result |
| `subsampling_t` | 1e-5 | Mikolov's frequent-positive discard parameter |
| `lr schedule` | linear decay 0.025 → 1e-3 over 5 epochs | Otherwise expect instability |
| `η_uniform` | **0 (mandatory)** | Delete this term |
| `walk_window` | Tempest default | All positions weighted by `w(i)` |

**Implementation complexity:** Low. ~15 lines once `BCEWithLogitsLoss` and the unigram-cache are wired. The lr schedule is the only non-trivial bit.

**Failure modes to watch:**

- Heavy-tail bias: without subsampling, frequent destinations get inflated `context` norms. Subsampling fixes this; weight decay does NOT (because the gradient is asymmetric in frequency).
- Anisotropy: SGNS does NOT enforce uniformity on the hypersphere. The embeddings will live in a non-isotropic ellipsoid. This is fine for inner-product downstream MLP but breaks cosine-similarity probes if you want to diagnostic-check.
- Wrong negative distribution: uniform sampling instead of unigram^0.75 silently degrades quality and breaks the factorization interpretation.

**Expected outcome on tgbl-wiki:** Test MRR 0.695 ± 0.020. Cliff: −0.05 over 50 epochs (better than current −0.28, comparable to A3.1).

**Expected outcome on tgbl-coin/flight:** +0.04 to +0.07 over current alignment baseline. SGNS historically transfers extremely well across domains.

**Principled stopping bonus:** during training, periodically (every 5 epochs) compute the Frobenius distance between `target · context^T` and the empirical shifted PMI matrix `log(P(v|u)/P(v)) - log(k)` from a sliding window of 10k walks. When this distance plateaus (relative change < 1% over 5 epochs), the embedding factorization has converged and you can stop regardless of val MRR. This is a cross-dataset-portable signal that does not require dataset-specific patience tuning.

---

### 4.4 A3.x_normbrake — Custom norm-brake regularizer (auxiliary, diagnostic-derived)

**Provenance disclosure.** This is the only loss component in the amendment that is NOT from prior literature. It is derived directly from the Phase 0.5 diagnostic finding that cross-table column norms grow 5× from 2-ep (mean ~0.36) to 50-ep (mean ~1.985) while test MRR drops from 0.7070 to 0.4269. Standard L2 weight decay penalizes norm growth uniformly across training; this regularizer is *empirically calibrated* to brake norm growth at the specific threshold where the diagnostic showed the cliff begins. It composes with any primary loss (A3.1/A3.2/A3.3) and with the distillation auxiliary (§5). No published precedent in this exact form, though the underlying idea (column-norm-bounded regularization) is well-known in matrix factorization literature.

**Description.** A hinge-style penalty that activates only when the per-column L2 norm of `E_target` or `E_context` exceeds a threshold derived from the empirical "safe" norm regime (the column-norm distribution at the best-val checkpoint of the anchor run). Below the threshold, the regularizer contributes zero gradient. Above the threshold, the regularizer scales quadratically with the excess.

**Formula:**

```
For each embedding table E ∈ {E_target, E_context}:
  col_norms_E = ||E[:, j]||_2 for j in 0..d-1   # shape [d]
  
  excess_E = max(0, col_norms_E - threshold_norm)
  
  L_normbrake_E = mean over j of (excess_E[j])^2

L_A3.x_normbrake = L_normbrake_target + L_normbrake_context

L_total = L_primary + λ_distill · L_distill + λ_normbrake · L_normbrake
```

The `threshold_norm` is calibrated once from the anchor run: take the mean column norm of the best-val-MRR checkpoint of the anchor (which corresponds to test 0.7070), multiply by `1.5` to give headroom for legitimate norm growth, and lock that as the threshold for all Phase S cells. For the anchor at epoch 2 with column norm ≈ 0.36, `threshold_norm = 1.5 × 0.36 = 0.54`. Verify this empirically before locking — read the best-val checkpoint's column-norm distribution directly.

**Critical implementation details:**

1. **Calibrate `threshold_norm` from the anchor run, not from heuristics.** Load the best-val checkpoint from the anchor validation, compute per-column L2 norms of both `E_target` and `E_context`, take the joint mean, multiply by 1.5. Lock that value for all Phase S cells. If different cells need different thresholds, that itself is a finding worth reporting (it means the loss family fundamentally changes the embedding-norm regime, not just the rate of growth).
2. **Apply per-column, not per-row or per-matrix.** Per-row (per-node) penalizes high-degree nodes and is dataset-biased. Per-matrix (Frobenius) doesn't distinguish "one column blew up" from "all columns slightly grew" — the diagnostic showed the cliff happens via uniform growth, but per-column gives a cleaner self-limiting gradient.
3. **The penalty is one-sided (`max(0, ...)`)** — gradient is zero below threshold. This is the self-limiting property: when norms are in the "safe" regime, the regularizer contributes nothing and the primary loss + auxiliary distill drive learning normally.
4. **`λ_normbrake = 0.1`** as default. If column norms still grow past threshold during training, raise to 0.5 or 1.0. If column norms stay well below threshold (and test MRR is high), the regularizer is unnecessary — drop it.
5. **Compose with any primary.** A3.x_normbrake is a strict regularizer; it does NOT replace the primary alignment-side loss, and it does NOT conflict with `L_uniform` if you choose to keep that for some reason (though A3.1/A3.2/A3.3 already drop `L_uniform`, so normbrake replaces uniformity functionally as the norm-controlling term).

**Hyperparameters:**

| Parameter | Default | Notes |
|---|---|---|
| `threshold_norm` | 1.5 × anchor mean col norm | Calibrate ONCE from anchor checkpoint, then lock |
| `λ_normbrake` | 0.1 | Try 0.5 if norms still grow; drop if norms stay below threshold |
| `apply_per` | column | NOT per-row; NOT Frobenius |
| `tables` | both E_target and E_context | Same threshold for both |

**Implementation complexity:** Low. ~10 lines: compute per-column L2 norm, apply hinge, square, sum across columns and tables, multiply by λ. The threshold calibration is a one-time precomputation from the anchor checkpoint.

**Why this might work where standard regularization didn't.**

- **Weight decay (L2) penalizes ALL embeddings equally regardless of whether they're already in a good regime.** Once an embedding has "the right" norm, weight decay still pulls it down, fighting the alignment loss's pull upward. Result: a steady-state norm that depends on the ratio of weight-decay strength to alignment gradient magnitude, which is task- and dataset-dependent and hard to tune.
- **The norm-brake is dormant below threshold and active above.** Embeddings can reach their natural-task-driven norm without resistance. The penalty only fires when norms cross into the empirically-measured "cliff regime."
- **The threshold is empirically grounded.** It's not a hyperparameter guessed from theory; it's the boundary observed in the actual diagnostic. This is in the spirit of v2.2's design philosophy ("anchor validation produces measurements, not commitments to current architecture").

**Failure modes to watch:**

- **Threshold too tight:** if the anchor's epoch-2 column norm was itself in a bad regime (which is *possible* — the anchor scored 0.7070 but we don't know if that's a local maximum of the loss surface or a global one), then capping at 1.5× of that may starve embeddings of capacity. Detect this by checking whether `L_normbrake` stays near zero throughout training (good) vs always-active (threshold too tight).
- **Wrong calibration:** if the anchor checkpoint's col-norm distribution is heavy-tailed (a few columns much larger than the mean), the mean is the wrong calibration target — use the 75th percentile instead. Verify by reading the anchor distribution before locking the threshold.
- **Interaction with weight decay:** if Group D winner included weight decay > 1e-4, normbrake's threshold may be effectively redundant with weight-decay's pressure. Test by ablating WD vs normbrake separately.
- **Composability with distillation:** if distillation pushes embeddings toward EdgeBank-bit-encoding norms that exceed the threshold, normbrake will fight distillation. Watch for `L_normbrake` rising sharply when distill is enabled — that's the signal to either raise the threshold or lower `λ_distill`.

**Expected outcome on tgbl-wiki:** test MRR 0.700 ± 0.020 paired with any primary; **cliff substantially mitigated** because the regularizer activates exactly when the diagnostic predicts it should. If A3.1+normbrake outperforms A3.1-alone by > 0.005 with cliff < −0.05 over 50 epochs, that's the paper-novel finding: "empirically-calibrated norm-brake regularization fixes the over-training cliff on walks-supervised temporal embeddings without requiring early stopping or loss-family change."

**Expected outcome on tgbl-coin/flight:** unclear without testing. The threshold may need to be re-calibrated per dataset (it's a per-dataset measurement, not a per-architecture constant). If normbrake transfers cross-dataset with per-dataset threshold calibration, that itself is a transferable technique worth reporting.

**Honest caveat (read before claiming this as a paper contribution).** This is an exploratory regularizer. It is not derived from a theoretical framework; it's derived from an empirical observation. There is a real risk that:

1. It works on wiki but doesn't transfer (the cliff threshold is dataset-specific in a way that doesn't generalize)
2. It works but only because it's effectively replicating weight decay with a different parameterization (in which case it's a minor variation, not a contribution)
3. It works but the same effect could be achieved with `λ_uniform → 0` + careful weight decay (in which case the contribution dissolves)

To distinguish (1)/(2)/(3): the ablation matrix in §6.3 must include:
- (a) Primary alone, no normbrake, no WD
- (b) Primary + standard weight decay 1e-4, no normbrake
- (c) Primary + normbrake, no WD
- (d) Primary + normbrake + WD

If only (c) and (d) have clean cliff behavior, the normbrake is doing something WD doesn't. If (b) is comparable to (c), the contribution dissolves into "tune weight decay."

---

## 5. Execution plan

### 5.1 Cell order

Six primary cells, then top-N multi-seed validation. Estimated ~1.5 hours total per single-seed pass, ~1.5 hours additional for multi-seed.

**Cells 1–3 (primary loss baselines, no auxiliary — the core loss-family comparison):**

- **Cell 1:** A3.1 InfoNCE, alignment-on, NO normbrake, seed 42
- **Cell 2:** A3.2 triplet, alignment-on, NO normbrake, seed 42
- **Cell 3:** A3.3 SGNS, alignment-on, NO normbrake, seed 42

**Cells 4–6 (primary + normbrake auxiliary; tests the custom regularizer):**

- **Cell 4:** A3.1 InfoNCE + A3.x_normbrake, seed 42
- **Cell 5:** A3.2 triplet + A3.x_normbrake, seed 42
- **Cell 6:** A3.3 SGNS + A3.x_normbrake, seed 42

**Cells 7–N (multi-seed validation):**

- Seeds 7 and 13 on the top 2 configurations from Cells 1–6 (4 runs total, or 6 if budget permits the top 3)

**Anchor calibration (one-time, BEFORE Cell 4):**

- Run a one-off "anchor checkpoint loader" script that reads the seed-42 anchor's best-val-MRR checkpoint, computes per-column L2 norms of `E_target` and `E_context`, takes the joint mean (or 75th percentile if the distribution is heavy-tailed; check both), multiplies by 1.5, and saves the result to `config.normbrake_threshold`. Lock this value for all Cells 4–N. This is a precomputation, not a training run; ~30 seconds.

### 5.2 Default tiebreaker if budget pressure forces a single starting cell

**Start with Cell 2 (A3.2 triplet, no auxiliary).** Triplet has the strongest theoretical fit for the over-training cliff via its literal self-limiting gradient, AND the cleanest comparison against the existing alignment+uniformity baseline (single change: loss family). If Cell 2 produces clean cliff behavior + peak test MRR within anchor noise, that's evidence the loss form matters for cliff-avoidance — proceed to Cell 5 (triplet + normbrake) to test whether the custom regularizer adds further cliff stability on top.

**Second-default tiebreaker:** if Cell 2 shows a cliff or regresses peak MRR meaningfully, run Cell 1 (InfoNCE) next — it's the candidate with the strongest published precedent on this kind of data (NeurTWs ablation).

### 5.3 Per-cell reporting

For each cell, the comparison matrix entry must include:

1. **Best epoch** (per patience=5 early stopping)
2. **Best val MRR**
3. **Best test MRR**
4. **Per-epoch test MRR trajectory** (every epoch logged, not just the best)
5. **Final cross-table column norm** (compare to 1.985 from the 50-epoch alignment+uniformity run; if any cell holds column norms < 2× growth across full training, that's the cliff-fix signal)
6. **Per-epoch column-norm trajectory** (required for all cells, especially normbrake cells; tracks whether the brake activates)
7. **Per-epoch `L_normbrake` value** (for normbrake cells only; should be near zero if the brake is functioning as designed)
8. **Wall time and walltime-per-epoch** (so budget can be projected for multi-seed)
9. **Hyperparameter values used** (τ, m, k, λ_normbrake, threshold_norm — for reproducibility)

### 5.4 Logging convention

Log files per cell: `runs/phase_s_A3<primary>_<auxiliaries>_seed<N>_<timestamp>.{log,json}`

Example: `runs/phase_s_A3triplet_normbrake_seed42_20260519_180000.log`

JSON should include the 9 fields above plus full per-epoch metrics.

---

## 6. Decision rules

### 6.1 After Cells 1–3 (primary alone)

**Decision criterion A — clear winner:**
- If any of A3.1/A3.2/A3.3 beats A2-off baseline (0.7092) by > 0.0016 on seed 42: that's the primary loss candidate. Multi-seed validate immediately on seeds 7 and 13. If multi-seed mean > 0.7092 ± 0.0016 with std < 0.02, that's the locked primary.

**Decision criterion B — all tie within noise:**
- If all three tie A2-off within ±0.0016 on peak MRR: the loss form is NOT a peak-MRR bottleneck on wiki. This is the EXPECTED outcome per §2 goal-shaping facts. Pick the cell with the cleanest cliff shape (smallest decline from peak over 4 epochs post-stop) as the locked primary. Proceed to Cells 4–6 to test whether normbrake further stabilizes that primary.

**Decision criterion C — a cell shows no peak by epoch 10:**
- If any cell shows test MRR still rising at epoch 10+ without a peak: that loss is genuinely fixing the over-training problem. Let that cell run to epoch 30 or until patience=5 triggers. This is the most important signal for cross-dataset deployment — flag it immediately and run multi-seed validation on it. Skip Cells 4–6 for this primary if its no-auxiliary form already fixes the cliff.

**Decision criterion D — all three regress below A2-off (0.7092):**
- This means the loss-family change is actively harmful on wiki. Two possibilities: (i) implementation bug — verify before proceeding; (ii) the current alignment+uniformity formulation is actually the right family for this task and the search itself was misframed. If verified bug-free, this is a real paper finding ("alignment+uniformity is the right loss family for walks-supervised temporal embeddings on memorization-saturated datasets") — proceed to Cells 4–6 anyway to see if normbrake rescues anything.

### 6.2 After Cells 4–6 (primary + normbrake)

**Decision criterion E — normbrake fixes the cliff:**
- If any primary+normbrake cell holds column norms below `1.5 × threshold_norm` across all 50 epochs AND test MRR at epoch 50 is within 0.02 of best-val test MRR: the regularizer is functioning as designed. This is the paper-novel finding ("empirically-calibrated norm-brake regularization eliminates the over-training cliff"). Multi-seed validate immediately.

**Decision criterion F — normbrake is dormant:**
- If `L_normbrake` stays near zero throughout training (the brake never activates), it means the primary loss family was already self-limiting enough. Report `L_normbrake` trajectory in the comparison matrix but drop normbrake from the locked production config. No paper finding, no harm done.

**Decision criterion G — normbrake is always-active (threshold too tight):**
- If `L_normbrake` stays > 0.5 throughout training (the brake fights primary gradient continuously): threshold is too tight. Either (a) re-calibrate threshold from a *post-Cell-1* checkpoint instead of the anchor (the loss family has changed the natural norm regime), or (b) raise `1.5×` multiplier to `2.5×` and re-run.

**Decision criterion H — normbrake hurts:**
- If normbrake regresses test MRR by > 0.005: the regularizer is starving embeddings of capacity. Try raising `1.5×` to `2.5×` once. If still hurts, drop normbrake and note "calibrated norm-brake hurts on wiki under this primary loss" — informative.

**Decision criterion I — normbrake duplicates weight decay:**
- If Cell N+normbrake performs equivalently to Cell N+weight-decay (which Group D may have tested separately): the contribution dissolves. Report as "normbrake replicates the effect of properly-tuned weight decay" — still useful methodology note but not a novel contribution.

To support criterion I, run the §4.4 ablation matrix at minimum on the winning primary:
- (a) Primary alone, no normbrake, no WD
- (b) Primary + WD only (1e-4)
- (c) Primary + normbrake only
- (d) Primary + normbrake + WD

These 4 sub-cells are the empirical test for whether normbrake is novel or merely a reparameterization of WD. Skip if (c) alone is decisively worse than (a) — it means normbrake isn't doing anything that needs comparing against WD.

### 6.3 Stop conditions (don't burn the rest of the budget)

- **Stop loss search if any of Cells 1–6 gives test MRR ≥ 0.7090 with a clean cliff (≤ 0.02 decline from peak across 50 epochs).** That's strong evidence for the production loss; multi-seed validate immediately and proceed to Group E.
- **Stop loss search if all 6 cells regress below A2-off (0.7092) by > 0.005.** Lock in A2-off (no walks supervision) + Group E and write up the result. The thesis "walk-distribution-matching is the right supervision" does not survive this outcome on wiki, but may survive on other datasets.
- **Do NOT stop just because peaks tie within noise.** The cliff shape is the more important signal per §2.

---

## 7. Compute-graph deduplication (carry-forward from v2.2 §4.6)

The compute-graph deduplication guidance still applies to these new cells.

Under Group E option E.2 (Component-0-only head; embeddings unread at scoring), all primary A3 losses produce mathematically equivalent gradient flow at the link MLP: the embeddings never contribute to the link BCE gradient, regardless of which alignment loss trains them. Only the alignment loss itself differs, but its effect on link MLP output is zero by construction.

**Practical implication for primaries:** if at execution time Group E has already moved to E.2 (Component-0-only head), DO NOT run cells 1–3 as specified — they collapse to a single experiment. Instead:

1. Roll back to E.1 (8-block cross-table head) for the A3 cell run; OR
2. Skip the A3 search entirely (under E.2, walks-supervision is read-irrelevant to scoring, so the loss form doesn't matter for downstream link prediction); OR
3. Pivot to evaluating the A3 losses by *intrinsic* embedding quality (PMI reconstruction error for SGNS; walk-distribution KL for InfoNCE; satisfied-triplet ratio for margin) without going through the link MLP — useful as a paper deliverable but not for production.

**Practical implication for normbrake (§4.4):** the normbrake regularizer acts directly on the embedding tables, not on the link MLP path. Under E.2, the embeddings still receive normbrake gradient even though they're unread by the link MLP — but this is gradient with no downstream effect on scoring. So under E.2, Cells 4–6 (primary + normbrake) ALSO collapse to a single experiment, with the same three options as cells 1–3 above.

**Check the agent's current Group E state before launching A3 cells.** If E.2 has been locked, this whole amendment is moot for production purposes and should be re-scoped per the three options above.

---

## 8. Losses explicitly deferred (not in scope for Phase S)

The research evaluated 8 total candidates. Four are excluded from Phase S:

- **EdgeBank-as-teacher distillation (formerly §5 in v1.1):** removed from this amendment in v1.2 to avoid diluting the contribution claim. Distilling a known heuristic into the embeddings makes the paper narrative "we trained embeddings to encode EdgeBank" rather than "we found a loss family that produces useful walks-supervised embeddings." Belongs in a follow-up paper if needed, not this one.

- **Decoupled Contrastive Loss (DCL):** marginal expected gain over A3.1 InfoNCE; theoretical concern that it could WORSEN the cliff (the negative-positive coupling DCL removes is what's protecting easy positives from inflating norms). Defer to a follow-up investigation only if A3.1 NPC behavior is empirically diagnosed as the cliff cause.

- **Barlow-Twins / VICReg on dual tables:** inductive-bias mismatch — these methods are designed for augmented views of the *same* underlying object, while `target(u)` and `context(u)` are fundamentally different roles (source vs destination). Forcing the cross-correlation matrix toward identity may actively destroy the role asymmetry that the dual-table architecture is designed to capture. Defer; explore as a small-λ auxiliary in a separate workstream if column-norm growth remains a problem after Group D regularization is tuned.

- **Hawkes-process log-likelihood (HTNE-style):** highest expected upside on tgbl-flight specifically (where periodic temporal patterns dominate) but implementation cost is medium-high (~150 lines, history buffer required) and NO TGB-leaderboard precedent exists. Pilot it as a separate workstream after Phase S settles a locked production architecture.

Other candidates considered and ruled out (not even deferred):
- Supervised contrastive (Khosla et al. 2020): needs class labels; no natural class structure on TGB nodes.
- BYOL / SimSiam / asymmetric self-supervised: same augmented-views mismatch as Barlow-Twins, worse because no negatives anchor the geometry against EdgeBank.
- Mutual-information bounds (DGI / JSD-MI): NeurTWs empirically dominates them on continuous-time graphs.
- Masked walk modeling (BERT-style): burns walk budget on reconstructing within-walk nodes rather than predicting future interactions; misaligned with the task.
- BPR (Bayesian Personalized Ranking): functionally a softer margin loss with no margin; A3.2 dominates it.

---

## 9. Watch-list (cross-cutting concerns)

These are things to monitor during execution; they are not cell-specific.

- **Compute-graph deduplication:** check Group E state before launching cells (see §7).
- **Follow-the-surprise principle from v2.2 §4.6:** if any cell shows qualitatively different training dynamics (a genuine plateau where val MRR holds for 10+ epochs without declining), investigate that cell first regardless of its peak number. That IS the cliff-fix signal — most important deliverable for cross-dataset generalization.
- **Phase S budget:** the new A3 specification is within the v2.2 §4.3 12-hour budget. Six cells at ~15 min each = 1.5 hours; multi-seed validation adds ~1.5 hours; total ~3 hours, well under budget.
- **Don't expand prematurely.** Resist the temptation to add Decoupled CL, Barlow-Twins, Hawkes losses, or any auxiliary supervision (including the dropped distillation) back into Phase S even if Cells 1–6 are clear-cut. The 3 primary + 1 custom auxiliary specification is the deliberate result of research and amendment; expanding mid-flight reintroduces the "fixed prescribed sweep" failure mode that v2.0 → v2.1 → v2.2 was designed to escape.
- **Cliff diagnostic per cell:** in addition to test MRR, log cross-table column-norm trajectory across epochs. A "clean cliff" cell has column-norm growth < 2× across full training; the current alignment+uniformity baseline has 5× growth (the cliff-cause signature).
- **Normbrake activity monitoring:** for normbrake cells (4–6), log per-epoch `L_normbrake` value. The brake is functioning as designed if `L_normbrake` stays at zero for most of training and only activates briefly near the cliff edge. If always-zero across all epochs, threshold is too loose (or primary is self-limiting and brake is redundant). If always-active, threshold is too tight (see decision criterion G).
- **PMI early-stop signal for A3.3 (SGNS):** if the agent implements the Frobenius-distance-to-PMI tracking, this is a paper deliverable on its own ("the only published TGB method with an analytical convergence criterion"). Treat it as a side-deliverable; don't gate Phase S on it.
- **Normbrake ablation discipline:** if any normbrake cell is a candidate for v2.3 lock-in, run the §4.4 4-way ablation (no-aux vs WD-only vs normbrake-only vs both) on that primary BEFORE locking, to distinguish "novel regularizer" from "reparameterized WD." This is paper-integrity-critical.

---

## 10. Reporting back to v2.2

After loss search completes, the design document evolves:

- **v2.2 + this amendment (current):** snapshot of the search-frame state including the loss specification AND the custom norm-brake regularizer.
- **v2.3 (post-Phase-S):** locked configuration. Single primary loss + (optionally) normbrake auxiliary specified. The loss search is done; v2.3 commits.
- **v2.4 (post-P1/P2/P3):** ablation matrix with the locked loss.
- **v2.5 (post-P4):** honest-protocol re-baseline comparison.

The amendment becomes archival once v2.3 lands. It exists to preserve the *reasoning* behind the cell choices, not just the choices themselves — so a future session investigating "why didn't we try X" can find the answer in §8 (deferred) or §4.4 (custom regularizer rationale).

**Paper-integrity note on §4.4:** if the normbrake regularizer is adopted into v2.3 as a production component, the v2.3 → paper writeup MUST include the §4.4 4-way ablation (no-aux vs WD-only vs normbrake-only vs both). Without that ablation, the contribution claim is unsupported. This is on the implementer to remember; don't let it slip into v2.4 without it.

---

*Document version: 1.3, post v2.3 audit. Companion to walk_distribution_matching_embedding_v2.md (v2.3). ARCHIVAL — load-bearing spec lives in v2.3 §4.7; this file preserves the full per-loss reasoning trail. Changes from v1.2: §4.2 (A3.2 Triplet) — fixed the Δt-discard hole. Earlier draft sampled positive `p ~ Categorical(w(i))` and used an unweighted hinge, which means Δt only biased WHICH position got sampled and was discarded from the loss after the multinomial. v1.3 switches to UNIFORM sampling of `p` and multiplies the per-walk hinge by `w(p)` — so Δt enters the triplet loss the same way it does in InfoNCE / SGNS (multiplicative weight on the per-pair loss term), making the three primaries directly comparable in the comparison matrix. Description, formula block, critical-implementation item 4 all updated. No other primary changes. Changes from v1.1 → v1.2 (preserved): removed EdgeBank-as-teacher distillation; reduced execution plan from 10 to 6 cells; renumbered sections; updated §8 to list distillation as deferred-not-in-scope. Changes from v1.0 → v1.1 (preserved): added §4.4 A3.x_normbrake as custom diagnostic-derived regularizer; added decision criteria for normbrake outcomes; updated §7 deduplication for normbrake-under-E.2. Sources: NeurTWs (Jin et al., NeurIPS 2022), Wang & Isola (ICML 2020), Levy & Goldberg (NIPS 2014), Yeh et al. (ECCV 2022 DCL), Zeng (arXiv:2510.02161v2 triplet vs contrastive), Mikolov et al. (NIPS 2013), Zuo et al. (KDD 2018 HTNE), Tan et al. (ICLR 2024 Kernel-InfoNCE), and the TGB leaderboard (January 2026 snapshot). The §4.4 norm-brake regularizer is not from a published source; it is derived directly from Phase 0.5 diagnostic data in this project.*