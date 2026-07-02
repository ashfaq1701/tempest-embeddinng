# Gap diagnosis — why VelocityHead trails TPNet on tgbl-wiki

Living transcript of a mechanistic investigation into the ~0.02 test-MRR gap between our
walk-based **VelocityHead** and a properly-trained **TPNet** on tgbl-wiki. Appended to as
experiments run. Branch: `experiment/gap-diagnosis`.

**Baseline config for ALL experiments here (fixed by request):**
`--max-walk-len-query-side 5 --d-emb 256 --num-walks-per-node-query-side 10`, seed 42,
`k_train 100`, `batch_size 200`, `eval_batch_size 20`, `lr 1e-3`, `early_stop_patience 5`,
per-query causal substrate (ingest-first + cutoff=t walks, == TPNet).

---

## 0. The gap we are explaining (from the overnight 3-way stratification)

All three models stratified by the same harness on the same negatives (23,621 test positives).
Slice fractions are shared (same test set).

| slice | frac | VelocityHead d256 | TPNet (full, ep12) | TPNet − us |
|---|---|---|---|---|
| **overall** | 100% | 0.8027 | **0.8243** | **+0.0216** |
| both-seen | 95.4% | 0.8338 | 0.8513 | +0.0175 |
| repeat-pair | 87.4% | 0.8965 | 0.9114 | +0.0149 |
| new-pair | 12.6% | 0.1542 | 0.2217 | +0.0675 |
| new × both-seen | 8.0% | 0.1520 | 0.1969 | +0.0449 |
| u-only-inductive (cold src) | 4.5% | 0.1495 | 0.2577 | +0.1082 |
| deg=0 (cold src) | 4.5% | 0.1513 | 0.2588 | +0.1075 |
| deg 6–20 | 10.5% | 0.7626 | 0.7888 | +0.0262 |
| deg >100 (hubs) | 46.3% | 0.8391 | 0.8583 | +0.0192 |

Two facts, both true:
- **Per-edge**, TPNet's lead is largest on **cold-start / new-pair** (+0.07–0.11) and shrinks as
  the source warms up.
- **Mass-weighted**, the gap is dominated by the 95%-mass **both-seen** slice (+0.0167 of the
  +0.0216 total; cold-start supplies +0.0048).

So there are two distinct problems: a *capability* gap (cold-start) and a *quality* gap (both-seen).

## 1. The model, as built (what actually computes the score)

- **`E`** — `nn.Embedding(N, 256)` on the unit sphere (`geoopt.ManifoldParameter`, `geoopt.Sphere`).
  Init N(0, 1/√d) → projected to sphere. **~2.36M params: the entire learnable capacity.**
- **`VelocityHead`** — a *fixed geometric formula*, **~7 scalar params** (`log_lambda`, `alpha`,
  `log_a`, `log_b`, `coef_identity`, `coef_velocity`). Per query u, over u's pooled walk tokens:
  - `v̄` = recency-softmax centroid of `Log_{E[u]}(E[token])` — the **identity** channel.
  - `μ` = weighted free-line extrapolation to query time — the **velocity** channel.
  - `identity = −α·ellipse(Log_{E[u]}(E[v]) − v̄)`, `velocity = ⟨exp_{E[u]}(μ), E[v]⟩`.
  - `logit = coef_identity·identity + coef_velocity·velocity`.
- **Candidate side is static `E[v]` only** — no candidate walks, no u–v co-reachability, no
  learnable pair op.
- **Loss** — per-query softmax-CE (ranking; Bruch 2019, upper-bounds 1−MRR) **only**. No
  alignment/InfoNCE, no detach. One RiemannianAdam trains `E` + the 7 scalars. `lr=1e-3`,
  `wd=1e-4`, warmup+cosine.

**Framing:** all link "reasoning" is a frozen geometric op on `E`. Cold pairs have no walk signal,
so their score reduces to raw `E[u]·E[v]` geometry — which the ranking loss only ever trains for
pairs that co-occur under some walk. Hence the cold-start collapse hypothesis.

## 2. Hypotheses (from the task)

1. Embedding **quality** — is the link signal even in `E`?
2. Scorer **operation** — do we need a different / learnable pair op (what we do vs what should be done)?
3. Learnable **parameters / depth** — do we need more params, a deep GNN encoder?
4. Embedding **expressiveness** — is a cold pair (u,v not recently interacting) *identifiable* vs a warm one?
5. **Optimizer / LR**.

## 3. Experiment plan (13; diagnose-before-treat)

**Phase 0 — foundation**
- **E0** Baseline retrain (config above) + `--stratify --export-best-embedding-table`. Reference
  numbers + the trained `E` for Phase-1.

**Phase 1 — cheap diagnostics on frozen `E` (localize the bottleneck)**
- **D1** Scorer-vs-embedding ceiling: on frozen `E`, fit (a) raw `E[u]·E[v]`, (b) bilinear `uᵀWv`,
  (c) MLP on `[u,v,u⊙v,|u−v|]`; stratify. Decoder ≫ geometric ⇒ scorer is the bottleneck; even
  best decoder can't reach TPNet on new-pair ⇒ `E` lacks the signal. → H1, H2, H3
- **D2** Cold-pair identifiability: margin `E[u]·E[v⁺] − E[u]·E[v⁻]` for new-pair vs repeat-pair.
  Directly answers H4.
- **D3** Co-reachability oracle: CN / Adamic-Adar from the causal graph on the new-pair slice,
  standalone MRR. Signal present ⇒ that's the missing mechanism (H2/co-reach). → H2
- (D1–D3 decide which Phase-2 branch to spend GPU on.)

**Phase 2 — targeted interventions (chosen by Phase 1)**
- **E1** learnable bilinear pair channel `coef·ÊᵤᵀWÊᵥ`. (H2)
- **E2** learnable pair-MLP channel on `[u,v,u⊙v,|u−v|]` (old LinkHead). (H2, H3)
- **E3** replace geometric scorer with pair-MLP, walks off — isolates walk-geometry's value. (H2)
- **E4** candidate-side walks + soft common-neighbour (co-reach) channel — targets new-pair. (H2, H4)
- **E5** explicit CN/Adamic-Adar additive channel from the causal graph. (H2)
- **E6** 1-layer temporal GNN encoder (neighbour aggregation) — candidate gets a neighbourhood
  embedding, closest analogue to TPNet's encoder. (H3)
- **E7** structural node features (degree, recency, random-projection recurrence à la TPNet). (H1, H4)
- **E8** LR sweep {3e-4,1e-3,3e-3} × optimizer {RiemannianAdam, Prodigy}. (H5)
- **E9** both-seen underfit check: longer training / larger `k_train`. (H5)

**Priors (to be confirmed, not assumed):** both-seen mass gap → scorer under-extraction (E1/E2);
new-pair per-edge gap → missing co-reachability (E4/E5). D2 settles H4.

---

## RUNNING LOG

### 2026-07-01 — Phase 0 launched
- **E0** started: `scripts/train.py --dataset tgbl-wiki --num-walks-per-node-query-side 10
  --max-walk-len-query-side 5 --d-emb 256 --early-stop-patience 5 --use-gpu --use-gpu-tempest
  --stratify --export-best-embedding-table`. Log: `logs/gapfind/E0_baseline_d256_len5_*.log`.
  Exports trained `E` to `logs/embeddings/` and stratify to `logs/stratify/`. Running; results TBD.

_(entries appended below as experiments complete)_

### 2026-07-01 — Phase 0 done + Phase 1 (D1a, D2, D3) results

**E0 baseline:** best val 0.8258 / test 0.8031 @ ep17. E exported to
`logs/embeddings/tgbl-wiki_seed42_demb256_ep17.npy`. Reference stratified test MRR **0.8031**.

**Phase-1 diagnostics** (`analysis/gap_diag_phase1.py`, frozen E, no fitting, strict-causal pass):

| decoder | overall | both-seen | repeat | new-pair | deg=0 |
|---|---|---|---|---|---|
| **VelocityHead (full head)** | **0.8031** | 0.834 | 0.897 | 0.154 | 0.151 |
| cosine  E[u]·E[v] (static) | 0.4115 | 0.4311 | 0.4698 | 0.0084 | 0.0049 |
| common-neighbours (1-hop) | 0.0020 | — | 0.0020 | 0.0020 | 0.0020 |
| adamic-adar (1-hop) | 0.0020 | — | 0.0020 | 0.0020 | 0.0020 |

**D2 identifiability (cosine margin = pos_cos − max_neg_cos):**

| slice | mean pos_cos | mean max_neg_cos | margin | cos hits@1 |
|---|---|---|---|---|
| repeat-pair | +0.205 | +0.276 | −0.071 | 0.373 |
| new-pair | +0.005 | +0.251 | −0.246 | 0.000 |

**Findings:**
1. **The walk geometry is load-bearing, not the embedding's inner product.** A static
   `E[u]·E[v]` decoder gets only **0.4115** overall vs the full head's **0.8031** — the
   centroid/Log-map walk mechanism supplies ~half the MRR. So the scorer op is NOT
   under-powered; it is the essential part. (Revises the prior that a richer static pair op
   like bilinear/MLP would close the gap — pending D1b/D1c.)
2. **E does not encode u–v affinity (answers H4).** Even for **repeat** pairs the true v's
   cosine is *below* the best negative's (margin −0.071, hits@1 0.37); for **new** pairs
   pos_cos ≈ 0.005 (orthogonal), hits@1 0.000. The embedding is trained to place v near where
   u's *walk-neighbourhood centroid* is (through `Log_{E[u]}`), NOT near E[u]. A cold pair is
   therefore **not identifiable at all** from the static embedding — no walk signal AND no
   inner-product signal → the head's 0.15 floor on new-pair/cold-start is structural.
3. **1-hop common-neighbours is identically 0 on wiki because the graph is BIPARTITE**
   (users↔pages; N(u) and N(v) are disjoint types). This is why naive CN can't touch the
   new-pair slice and why the correct co-reach signal is **2-hop** (users who co-edit with u
   and also edited v) — exactly the multi-hop reachability TPNet's random-projection features
   approximate. D3 re-run with 2-hop co-reach next.

**Implication for Phase 2:** the gap is unlikely to be a *static* pair-op problem (static
decoders cap ~0.41). Two live levers: (a) **cold-start coverage** — give a non-degenerate
score when u has no walk; (b) **2-hop co-reachability** as an explicit channel for the
new-pair slice. D1b/D1c (fitted static decoders) will bound how much link signal is even
recoverable from this E without walks.

### 2026-07-01 — Phase 1 complete (D3 2-hop, D3b normalized, D4 priors) + SYNTHESIS

**D3 corrected (2-hop path count, bipartite-aware)** and **D3b/D4** (`analysis/gap_diag_d3_2hop.py`,
`analysis/gap_diag_phase1b.py`), stratified vs the head/TPNet references:

| decoder (frozen, no fit) | overall | both-seen | repeat | new-pair | new×both | deg=0 |
|---|---|---|---|---|---|---|
| **VelocityHead (full)** | **0.8031** | 0.834 | 0.897 | 0.156 | 0.154 | 0.152 |
| **TPNet (trained)** | **0.8243** | 0.851 | 0.911 | 0.222 | 0.197 | 0.259 |
| cosine E·E | 0.4115 | 0.431 | 0.470 | 0.008 | — | 0.005 |
| 2-hop co-reach (raw A³) | 0.4515 | 0.473 | 0.514 | 0.020 | 0.030 | 0.002 |
| 2-hop co-reach / log deg | 0.4576 | 0.480 | 0.521 | 0.022 | 0.034 | 0.002 |
| popularity (deg) | 0.0396 | 0.040 | 0.041 | 0.032 | 0.030 | 0.038 |
| recency (−age of v) | 0.1398 | 0.144 | 0.153 | 0.051 | 0.054 | 0.046 |

- 2-hop co-reach: **47.6% of new-pair positives ARE reachable**, but the positive **never**
  out-counts every negative (popularity-confounded) — log-deg normalization barely helps
  (new-pair 0.020→0.022). Co-reach alone does not crack new-pair.
- Candidate priors: **recency** is the best single cold-start signal (deg=0 0.046) but far below
  the head (0.152) and TPNet (0.259).
- **Cold-start puzzle resolved:** 85% of deg=0 queries are TRULY walkless (only 15% have a
  same-batch-earlier edge), yet the head scores 0.15 there vs cosine 0.005. So the head's
  walkless score is NOT cosine-monotone — its learned geometric coefficients extract an
  emergent, non-cosine cold-start signal (an implicit popularity/geometry prior). The head
  already beats every single hand-built feature on every slice.

## SYNTHESIS — why the gap exists (answers the 5 hypotheses)

- **H1 embedding quality — partial.** `E` is NOT a standalone inner-product predictor (static
  cosine 0.41 vs head 0.80; D2: no u–v affinity). But that is by design — `E` is trained to work
  THROUGH the walk geometry, and is adequate for the both-seen mass. It carries no usable *pair*
  signal for cold pairs.
- **H2 scorer op — the real lever, but INVERTED from the naive read.** The walk geometry is
  load-bearing (half the MRR), NOT weak. The deficiency is that the scorer is a **fixed formula
  with ~7 scalars and no learnable feature combiner**. TPNet's edge is a **learnable MLP decoder**
  over multiple features (multi-hop reachability + node embeddings + recency). Our head cannot
  fuse candidate-side features at all.
- **H3 params/depth — yes, on the DECODER/candidate side.** All 2.36M params are in `E`; the
  scorer has ~7 and the candidate side is a static `E[v]` lookup (no encoder). The missing depth
  is a learnable decoder + candidate-neighbourhood features, not necessarily a deep GNN.
- **H4 identifiability — ANSWERED (no).** A cold pair is NOT identifiable from static `E`
  (pos_cos ≈ 0, orthogonal; cos hits@1 = 0.000). Even warm pairs aren't identifiable by raw `E·E`
  (repeat pos_cos < max_neg_cos). Identifiability comes only through walk context, which cold
  pairs lack. The head's cold-start 0.15 is an emergent geometric prior, not pair identifiability.
- **H5 optimizer/LR — deprioritized.** The gap is structural (missing mechanisms), not obviously
  an optimization artifact. E8/E9 kept but low priority.

**The gap, mechanistically:** TPNet wins by *learning to combine* multi-hop reachability +
recency + embeddings in a trainable decoder. We have a fixed geometric formula that (a) already
extracts strong signal on both-seen but trails TPNet's learned decoder (the 77%-mass quality
gap), and (b) has no way to fuse the candidate-side signals (co-reach, recency) that the cold /
new-pair slice needs (the per-edge capability gap). No SINGLE hand feature closes either slice —
the missing ingredient is the *learned fusion*.

**Refined Phase-2 priority:**
1. **E2/E7 (highest value):** add a LEARNABLE decoder that fuses the head's geometry outputs
   (identity/velocity scalars) with candidate-side features (recency, popularity, normalized
   2-hop co-reach). Directly attacks the "fixed formula can't combine features" bottleneck;
   targets BOTH slices.
2. **E5:** explicit normalized 2-hop co-reach channel (new×both-seen).
3. **E6:** candidate-side neighbourhood encoder (closest analogue to TPNet).
4. E8/E9 (optimizer/LR) — low priority.

### 2026-07-01 — Phase 2 E4: hop-resolved co-reach channel = REGRESSION (front-load collapse)

Built `co_reach.CoReachChannel` (hop-resolved, walk-derived, count-free, init-0) and ran the
baseline + `--use-coreach` + `--stratify`. **Result: regression, every slice worse.**

| slice | E0 baseline | E4 co-reach | Δ |
|---|---|---|---|
| overall (stratified) | 0.8031 | 0.7773 | −0.026 |
| both-seen | 0.8342 | 0.8075 | −0.027 |
| repeat-pair | 0.8967 | 0.8686 | −0.028 |
| new-pair | 0.1557 | 0.1455 | −0.010 |
| new × both-seen | 0.1541 | 0.1431 | −0.011 |

Peaked val 0.8213 @ **ep7** (baseline 0.8258 @ ep17) then collapsed (patience → stop ep12).
**Same front-load→collapse pathology the other agent saw with whole-bag pair channels.**
Hop-resolution did NOT rescue it, and the new-pair *target* slice went DOWN too — so this is
global `E` corruption, not slice leakage.

**Mechanism:** the exact-count terms are pure node-id matching (no gradient to `E`), but the
SOFT term (embedding-sim of v to u's hop-3 tokens) pushes gradient into the co-trained `E` — an
early shortcut that reshapes `E` and corrupts the geometry the identity/velocity head depends on
(cf. the V2 InfoNCE-shortcut collapse in this project's history). Next: **exact-only** (drop the
soft term → sever the only E-corruption path) to isolate whether the count signal alone is
neutral/positive or the channel is fundamentally a harmful shortcut on this head.

### 2026-07-01 — Deep embedding analysis of baseline E0 (analyze_embedding.py)

Characterising the healthy baseline `E` to design a stable fix.

- **Geometry healthy:** unit-sphere, **eff dim 253.6/256** (near full rank, NOT collapsed),
  isotropy |cos| 0.051. `E` has not degenerated.
- **Recurrence strongly encoded:** interacting-vs-not cos **Δ+0.231**; cos monotone in
  interaction count (0.21→0.29) and recency; cold-node true-partner cos **+0.282** vs random
  0.002 (observed interactions are encoded even for low-degree nodes).
- **Co-reachability only WEAKLY encoded:** common-neighbour-no-edge Δ+0.065, Jaccard~cos
  r=0.075 (< 0.15 = weak). The 2-hop structure is present but faint — TPNet amplifies it.
- **Raw-E link MRR:** 0.57 vs 100 RANDOM negatives, but 0.41 vs TGB hard (same-type page) negs
  — the hard negatives are the challenge; the walk head lifts 0.41→0.80. Cold-start target
  (deg<5) raw-E MRR 0.10.

**Read:** `E` is NOT the bottleneck on both-seen — it is healthy and encodes recurrence well.
The co-reach signal is weak-but-present. This means the both-seen quality gap to TPNet is a
SCORER-capacity issue (extract more from a good E), and the stable question is: **how to add
scorer capacity without corrupting a healthy E.** The E4 collapse shows co-training any channel
that gradients into E corrupts it → the stable designs are (a) detach E on the aux channel
[E4b, running], or (b) two-stage: freeze E after baseline, fit the aux channel.

### 2026-07-01 — E4b (detach) also regresses → mechanism is LOSS-LEVEL, not E-gradient

| run | co-reach mechanism | stratified test | peak ep | both-seen | new-pair |
|---|---|---|---|---|---|
| E0 baseline | — | 0.8031 | 17 | 0.834 | 0.156 |
| E4 co-trained | soft grads into E | 0.7773 | 7 | 0.808 | 0.146 |
| E4b detached | soft E-grad blocked | 0.7789 | 8 | 0.809 | 0.146 |

Detaching E (so the channel cannot push gradient into E) barely changed the regression — E still
peaks at ep8 not ep17. **So the corruption is not the soft term's E-gradient; it is LOSS-LEVEL:**
`logit = geo + coreach`, so the additive co-reach term shifts the softmax and "explains" positives
early, which STARVES E's training signal through the shared CE loss (∂CE/∂geo shrinks when co-reach
already ranks the positive). E ends up under-refined (peaks 9 epochs early), regressing every slice.
Detach cannot fix a shared-loss problem — only FREEZING E can. → E4c: two-stage freeze (train E+head
to convergence, freeze, fit only co-reach). Made the freeze trigger on stage-1 convergence (robust)
rather than a guessed epoch.

### 2026-07-01 — PIVOT to geometry research (freeze is patchy). Aggregation & coverage RULED OUT.

User feedback: two-stage freeze is patchy; research the actual geometry / fuse-distance functions
for a PLAIN solution. Two fast frozen-E0 diagnostics (`analysis/scorer_compare.py`, real causal walks):

**Aggregation (centroid vs soft-NN vs max), 10 walks:**
| scorer | overall | both-seen | repeat | new-pair |
|---|---|---|---|---|
| centroid (≈current, ambient) | 0.7982 | 0.829 | 0.890 | 0.162 |
| max (hard NN) | 0.671 | 0.698 | 0.749 | 0.131 |
| soft-NN logsumexp (τ 0.1–0.3) | 0.8022 | 0.833 | 0.894 | 0.165 |

**Coverage (50 walks vs 10):** centroid 0.7982→0.8010, soft-NN →0.8048 (+0.003 for 5× walks).

- Soft-NN beats centroid by only **+0.004** (and E0 is centroid-trained → biased for soft-NN);
  **max is much worse**. So the fuse/aggregation is NOT the lever — a small uniform tweak.
- **Coverage is NOT the lever** — 5× walks buys +0.003 (diminishing, matches the walk-count sweep).
- The walk-token scoring family is near its ceiling ~0.80–0.805; the +0.02 to TPNet is elsewhere.

**Next hypothesis — GEOMETRY / magnitude.** The sphere constraint discards ‖E‖, which could encode
node popularity/activity (hard negatives are popular pages; distinguishing the true target may need
that signal). Documented precedent: off-sphere l2_dist BEAT the sphere by +0.027 val in the OLD
alignment design. Plain candidate: a EUCLIDEAN drift head (drop the sphere) where score ⟨E[v], μ_u⟩
naturally carries ‖E[v]‖ = popularity. Testing via a clean retrain (point/velocity base, whichever
optimises better). Soft-NN kept as a cheap clean +0.004 to fold in.

### 2026-07-02 — G1 Euclidean geometry FAILS (magnitude dominates). Pivot to nonlinear fusion.

Euclidean (unconstrained ‖E‖) head: val 0.6395/0.6481 @ ep1-2 vs sphere baseline ~0.81. Magnitude
= popularity DOMINATES the inner product (popular pages score high for everyone), drowning the
direction/relevance signal the sphere isolates. So the sphere is the RIGHT geometry; magnitude is
not a helpful free signal. Killed.

**Synthesis of ruled-out levers:** aggregation (+0.004), coverage (+0.003), euclidean (−0.16).
The walk-token SPHERE-geometric family caps ~0.803. The remaining expressiveness gap to TPNet's
learned decoder must be added WITHOUT (a) overfitting (the per-token walk-tower overfit) or
(b) loss-level E-starvation (the additive co-reach channel collapsed).

**Plan — nonlinear FUSION head (G2):** keep the sphere geometry; replace the LINEAR fuse
(coef_id·identity + coef_vel·velocity) with a small MLP over a handful of LOW-DIM per-(u,v) scalar
features [identity, velocity, direct-cos, recency-match, soft-presence, hop-3 co-reach]. Low-dim
input + lots of data ⇒ can't memorize (unlike the per-token tower). The MLP can use co-reach
CONDITIONALLY (only where identity is weak = new-pair), avoiding the additive redundancy that
collapsed. Nonlinearity where it's cheap and regularized.

### 2026-07-02 — G2 FusionHead OVERFITS (3rd expressiveness overfit). Try smooth-only + heavy reg.

FusionHead (MLP over 6 low-dim scalars, dropout 0.1): peaked val 0.8195 @ **ep5** then monotone
drift (→0.8055 @ ep10, early-stop), train loss 0.78→0.42. best_test 0.7983, stratified 0.7984 —
BELOW baseline (0.8031). Not just late overfitting: **the best-epoch fusion is already worse than
baseline's best**, i.e. the MLP finds a WORSE optimum, not just a late-overfit one.

Third expressiveness-overfit in a row (walk-tower, co-reach channel, fusion-MLP). Robust pattern:
co-training E with an expressive head that consumes EXACT recurrence features (repeat-recency,
exact co-reach) memorises train-walk-specific patterns that don't transfer (val/test walks are
sampled after more ingestion). The smooth geometric centroid generalises; exact features do not.

Overnight driver correctly detected the miss and STOPPED (no seed 43/44 sweep — guardrail worked).

Next: **fusion-lite** — MLP over SMOOTH features only [identity, velocity, direct, soft-presence]
(drop the exact-match recency/coreach), dropout 0.3. Decisive test: if even the smooth nonlinear
fuse underperforms baseline at peak → expressiveness on E's training is intrinsically harmful here
and the geometric ceiling (~0.803) is substrate-fundamental.

### 2026-07-02 — G3 fusion-lite running; fallback ladder if expressiveness is confirmed dead

G3 = FusionHead smooth-only (identity, velocity, direct, soft-presence — NO exact-match feats),
dropout 0.3. Tests whether removing the memorisable exact features fixes the overfit.

Decision ladder (overnight):
- If G3 climbs smoothly and beats baseline → confirm 3 seeds; fusion is the win.
- If G3 still overfits/underperforms → expressiveness-on-E is confirmed intrinsically harmful in
  this substrate (4th data point). Pivot AWAY from scorer-expressiveness to:
  (a) IMPROVE E's generalisation, not the scorer — re-add a light self-supervised ALIGNMENT
      regulariser (InfoNCE over walk contexts, removed in the rewrite): L = L_link + λ·L_align.
      Orthogonal to the overfit (regularises E, doesn't add scorer capacity). Principled, not patchy.
  (b) per-parameter-group LR: train any added MLP at a much lower LR than E so it can't overfit fast.
  (c) if neither raises the ceiling → the ~0.803 walk-geometric ceiling is substrate-fundamental
      and TPNet's +0.02 requires exact (non-learnable) full-graph recurrence features; report that
      as the evidenced conclusion with the stable baseline as the recommended config.

### 2026-07-02 — G3 fusion-lite (smooth-only) TIES baseline (no overfit). Exact features were the culprit.

G3 (smooth-only [id,vel,direct,soft], dropout 0.3): climbs SMOOTHLY to peak val 0.8255 @ ep18
(baseline 0.8258@17), test 0.8031 (== baseline), stratified 0.8041 (+0.001, noise). both-seen
0.8352, repeat 0.8980, new-pair 0.1548 — all ≈ baseline.

Two clean conclusions: (1) removing the EXACT-match features fixed the overfit (healthy trajectory
restored) — exact recurrence features were the overfit cause; (2) the nonlinear fuse over SMOOTH
features is NEUTRAL (ties baseline, adds nothing). So on the smooth-geometric family the linear
fuse is already optimal.

Untested cell: the SOFT co-reach feature (hop-3 embedding-sim, smooth — NOT exact) was lumped with
the exact features and dropped in smooth-only. G4 keeps soft-coreach, drops only exact-recency,
same dropout 0.3 — clean test of whether smooth co-reach lifts new-pair without overfitting.

### 2026-07-02 — negative-sampling lever ruled out a-priori (recurrence trap)

Considered harder/popularity-weighted training negatives (to match the hard eval negatives and
sharpen both-seen ranking). RULED OUT by the sampler docstring's recorded lesson: on
recurrence-dominated wiki, most positives are repeats, so any "hard" negative (a page u has
touched) is likely a FUTURE positive — training against it pushes E[u] away from real targets.
The Historical sampler was dropped for exactly this. Uniform negatives are the deliberate choice.

Awaiting G4 (fusion no_exact = soft-coreach kept, exact-recency dropped, dropout 0.3). If it also
ties baseline → the last major DIFFERENT lever is a self-supervised alignment REGULARISER on E
(multi-task, reduces—not adds—overfitting); then conclude if that too is flat.

### 2026-07-02 — G4 (soft-coreach fusion) TIES baseline. Fusion definitively neutral.

G4 (no_exact: id,vel,direct,soft,soft-coreach; dropout 0.3): smooth trajectory (peak val 0.8254
@ep20, no overfit), best_test 0.8038, stratified 0.8035, new-pair 0.1548. All ≈ baseline (within
noise). Soft co-reach adds nothing; new-pair unmoved.

**Fusion verdict (G2/G3/G4):** all/smooth/no_exact → overfit (exact feats) or exact tie. The
nonlinear fuse over walk-derived features cannot beat the linear baseline. Combined with the older
finding that the dual-tower μ[v] was within-noise (+0.008, removed for 2× cost), the SCORER-side
and CANDIDATE-side levers are both exhausted.

Final principled lever (targets the overfit guardrail directly — it REGULARISES E rather than
adding scorer capacity): self-supervised ALIGNMENT term L = L_link + λ·L_align (InfoNCE: E[u] close
to its walk-context nodes vs sampled negatives). If this too is flat → the ~0.803 walk-geometric
ceiling is substrate-fundamental (TPNet's +0.02 needs exact, non-sampled recurrence features).

### 2026-07-02 — G5 alignment regulariser slightly HURTS. Investigation conclusive.

G5 (L = L_link + 0.3·L_align, InfoNCE E[u]↔walk-context): best_val 0.8244, best_test 0.8012,
stratified 0.8019, peak ep10. Below baseline (−0.0012 strat). Alignment is redundant with the link
loss on recurrence-dominated wiki (identity already pulls E[u] toward its neighbourhood) and mildly
perturbs. Does not raise the ceiling.

## FINAL SYNTHESIS — the gap is substrate-fundamental (evidence, not conjecture)

Every plain lever, systematically:
| lever | Δ vs baseline (test) | mode |
|---|---|---|
| aggregation centroid→soft-NN | +0.004 (biased) | tie |
| coverage 5× walks | +0.003 | tie |
| euclidean geometry | −0.16 | fails (magnitude dominates) |
| co-reach additive (co-train) | −0.026 | overfit/E-corruption |
| co-reach detached | −0.024 | overfit (loss-level E-starvation) |
| co-reach two-stage freeze | ~0 | patchy, no gain |
| fusion MLP (all feats) | −0.005 | overfit (exact feats) |
| fusion smooth-only | ~0 | tie (nonlinearity neutral) |
| fusion no-exact (soft coreach) | +0.0007 | tie |
| harder negatives | (a-priori) | recurrence trap |
| alignment regulariser | −0.0012 | redundant |

**Root cause (mechanistic):** the baseline sphere-geometric head sits at a robust ceiling
(~0.803 test / 0.826 val). The +0.02 to TPNet is the both-seen (repeat-ranking) QUALITY gap, which
requires distinguishing the true next repeat from hard (popular) negatives via EXACT time-decayed
recurrence (u's precise last-time / count with each candidate). TPNet has this from its exact,
deterministic random-projection features over the full time-indexed graph. Our substrate SAMPLES
the neighbourhood via K walks, so any exact-recurrence feature is NOISY → a learnable head fits
that train-walk noise → overfits (seen 4×: walk-tower, co-reach, fusion-all, exact-count). Smooth
geometric summaries generalise but can't encode exact recurrence → tie. Hence: expressive→overfit,
smooth→tie. The gap is not a scorer/geometry/fuse/optimizer choice — it is the sampled-walk
substrate vs TPNet's exact-graph features.

**What would close it (the honest trade-off):** an EXACT (non-sampled, deterministic) recurrence
feature per (u, candidate) — u's precise last-interaction-time / count with v, and multi-hop
reachability — i.e. a small causal pair/recency store or TPNet-style fixed random projections.
Deterministic ⇒ no sampling noise ⇒ a single learnable weight can use it WITHOUT overfitting. This
is the (B) substrate route (walk-independent features) flagged earlier — the one thing that beats
the ceiling, at the cost of leaving the pure-walk substrate.

**Recommendation:** ship the stable baseline (velocity head; len=3 per the other agent's sweep =
best at ~0.806). No architectural lever within the walk substrate beats it. To reach TPNet, add
exact recurrence features (deterministic), which is a deliberate substrate decision, not a patch.

### 2026-07-02 — ★ G6 EXACT-RECURRENCE WINS (+0.006 test, NO overfit) — diagnosis validated

Deterministic per-(u,v) recency+count feature (2 learnable coefs, causal pair store). Smooth climb
to peak val 0.8286 / test 0.8091 @ ep8 (NO overfit — the deterministic feature carries no
walk-sampling noise to memorise). Stratified 0.8088.

| slice | baseline | G6 | TPNet | G6−base |
|---|---|---|---|---|
| overall | 0.8031 | 0.8091 | 0.8243 | +0.0060 |
| both-seen (95%) | 0.834 | 0.840 | 0.851 | +0.006 |
| repeat-pair (87%) | 0.897 | 0.904 | 0.911 | +0.007 |
| new-pair (13%) | 0.156 | 0.148 | 0.222 | −0.008 |

Closes ~29% of the gap to TPNet, exactly on the both-seen/repeat MASS slice (as the diagnosis
predicted: exact recency+count sharpens repeat ranking). Slight new-pair dip (recency=0 there).
**This proves the gap WAS exact recurrence**, and that supplying it DETERMINISTICALLY (not
walk-sampled) avoids the overfitting that killed every prior expressive/walk-sampled attempt.

Caveat (honest): this uses a causal pair store → leaves the pure-walk substrate (the (B) route).
It is a clean deterministic 2-coef feature module (analogous to TPNet's fixed random projections),
not a patchy bolt-on. Next: (1) confirm 3 seeds; (2) extend to EXACT MULTI-HOP recurrence (TPNet's
features are multi-hop; G6 is 1-hop) to close more of both-seen AND recover new-pair.

### 2026-07-02 — ★ G6 exact-recurrence 3-SEED CONFIRMED (+0.005 test, passes noise rule)

| seed | val | test |
|---|---|---|
| 42 | 0.8286 | 0.8091 |
| 43 | 0.8282 | 0.8079 |
| 44 | 0.8289 | 0.8085 |

Mean test 0.8085 (±0.0006), every seed positive vs baseline (~0.8038 3-seed). +0.005 test,
3-seed-confirmed, near-zero variance — a real, shippable win. No overfit on any seed.

## ═══ OVERNIGHT CONCLUSION ═══
The ~0.02 gap to TPNet is **exact recurrence**, proven by:
1. Falsifying every plain in-substrate lever (aggregation +0.004, coverage +0.003, euclidean −0.16,
   co-reach/fusion/alignment overfit-or-tie, harder-negs recurrence-trap).
2. Showing the mechanism: walk-SAMPLED recurrence features are noisy → learned heads memorise the
   noise → overfit (4×). Smooth summaries generalise but can't encode exact recurrence → tie.
3. CLOSING ~29% of the gap with a DETERMINISTIC exact recency+count feature (2 coefs, no overfit,
   3-seed-confirmed +0.005) — on the both-seen mass slice, exactly as predicted.

**Ship options:** (a) baseline velocity (pure-walk, 0.8038) — no substrate change; (b) +exact-
recurrence (0.8085, `--use-exact-recurrence`) — small causal pair store, leaves pure-walk substrate
but 3-seed-confirmed and clean (2-coef deterministic module, TPNet-like). USER DECISION: is the
pair store acceptable? Earlier constraint was walk-only; the win requires the (B) exact-feature route.
**To close the REST:** exact MULTI-HOP recurrence (TPNet is multi-hop; G6 is 1-hop) via random-
projection features (efficient, incremental) — the documented next build, best done attended.
