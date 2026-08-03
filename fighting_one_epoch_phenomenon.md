# Fighting the One-Epoch Phenomenon

A running lab notebook of the effort to stop tgbl-review's validation MRR from
peaking at epoch 1 and collapsing thereafter. **Every prompt and action from
2026-08-02 onward is transcribed here**, newest entries appended at the bottom.

---

## What the "one-epoch phenomenon" is

Validation metric peaks at epoch ~1, then degrades monotonically while the
**training loss keeps falling** — the model memorises training-specific
structure that does not transfer. Named and characterised for deep CTR /
recommendation models by **Zhang et al., CIKM'22, "Towards Understanding the
Overfitting Phenomenon of Deep Click-Through Rate Models" (arXiv:2209.06053)**.
It is not hyperbolic-specific; it shows up broadly in over-parameterised models
trained on sparse, high-cardinality categorical/interaction data (exactly our
regime: 352k nodes, one embedding row each, sparse temporal interactions).

## Our setup (context for every experiment below)

- **Model:** Tempest per-query causal walks + Poincaré-ball tangent-pool head
  (`NeighborhoodProjection`), scorer = MLP over geodesic distances + node/edge
  features. Per-query softmax-CE ranking loss. Two LR groups (`manifold_lr` for
  E, `model_lr` for head).
- **Dataset in focus:** tgbl-review (3.41M train edges, low recurrence).
- **Review-optimal knobs:** `manifold_lr 1e-3`, `dropout 0`, `K_train 10`,
  `batch-size 1000 / eval-batch-size 1000`, `early-stop-patience 5`.
- **Baseline curve:** every uniform-negative config peaks at epoch 1
  (~0.28 val) and collapses. Best clean uniform result this line: ~0.2835 val
  / 0.2719 test.
- **The ONE thing that ever changed the curve:** the **alignment loss** — a
  non-exhausting auxiliary gradient on E (a genuinely different objective, not
  another variation on the ranking loss). It makes review *climb* for several
  epochs and wins test (~0.29), but is a ~40GB memory hog (two-sided [S,P]
  matrix) needing a strictly-solo GPU.

## What has already been tried (all FAILED to change the curve shape)

| Lever | Result | Curve |
|---|---|---|
| LR (manifold/head 1e-4…1e-3), K (10/50), dropout (0/0.2/0.5), boundary prior, weight decay | peak ep1, no climb ≥6 epochs | unchanged |
| **Historical negatives** (MixedNegativeSampler, ratio 0.2, reservoir 64, k=50) | val 0.2792→0.2777→0.2222 | peak ep1, same collapse |
| **Remove 2 self-displacement distances** d(E,P) from scorer (6→4 dists, hist 0.2, k=50) | val 0.2829→0.2830→0.2126 | peak shifted to ep2, then same cliff |

**Interpretation:** the collapse is a *memorisation* failure of the head + E
(train loss keeps falling by fitting non-transferable structure). Anything that
is "just a different negative distribution" or "a head-capacity tweak" peaks and
collapses. Only a different *objective* (alignment) has moved it.

---

## Log

### 2026-08-02 — Establish this notebook

**Prompt:** "Okay maintain a separate fighting_one_epoch_xxx.md document. Every
action / prompt from now on will be transcribed to this document. I didn't use
the proper term, use the correct name."

**Action:** Created `fighting_one_epoch_phenomenon.md` (this file) — corrected
"xxx" → "phenomenon" (the proper term is the *one-epoch phenomenon*, Zhang et
al. CIKM'22). Seeded it with the definition, our setup, and the already-tried
table above. Currently on branch `feature/link-trained-e` (clean uniform line,
6-distance scorer, no hist negatives) after reverting the self-displacement
ablation. Not committed (working doc).

### 2026-08-02 — Retrieve the "why we removed alignment" thread

**Prompt:** "i) You are the one who mentioned REMOVING alignment loss, saying
it's trained for a separate task, not directly transferrable for wiki. ii) You
said this to improve wiki after doing analysis, you were pretty confident. Can
you go back in the thread and find it?"

**Action:** Searched the full session transcript. Found the three-step thread
(all 2026-08-01):
- **Recommendation (00:39):** diagnosed the wiki head-plateau; argued E was
  `.detach()`-ed and trained *only* by the alignment loss — "a **proxy**
  (cluster-by-walk-neighbourhood), **not the link objective**." Proposed
  "option 4": drop alignment, train E directly on the link loss.
- **Implementation (00:51):** commits `e23972e4` (deleted `alignment_loss.py`)
  + `81f9bced` (head reads E live, no detach) on `feature/link-trained-e`.
- **Confident payoff on wiki (01:08):** at ep1, link-trained E beat the
  alignment head's whole-run peak by **+0.019 val / +0.029 test**; commP fell
  0.62→0.42, confirming alignment had *over-clustered* E toward a proxy
  community structure that wasn't fully link-relevant.

**Key nuance for THIS fight:** that removal was **wiki-specific**. On wiki,
Louvain community is *anti-correlated* with MRR, so shedding the community proxy
helped. On **review**, community structure *is* the cold-start signal — which is
why alignment is the one lever that makes review climb. **Same mechanism,
opposite sign across datasets.** So "alignment is a separate/non-transferable
task" was true *for wiki*; on review it is precisely the transferable signal.
This dataset-mirror is the crux of the one-epoch problem here.

### 2026-08-02 — Goal reframed: ONE solution for wiki + review + coin

**Prompt:** "We are not creating a review only model. We need to find one
solution that works well on wiki, review and coin, all three (which will also
likely cover more datasets)."

**Framing established:** the collapse (head memorises; train loss ↓ while val
peaks early) is the *shared* enemy. The mirror-image knobs (wiki: slow-E/K100/
dropout; review: fast-E/K10/no-dropout) are all proxies for ONE axis — *how
much/fast should E acquire structure*. A single solution must **remove the need
to pick** on that axis, not pick a better point. Candidate agnostic levers:
(1) kill head memorisation via head-LR (proven to stop wiki's collapse; NEVER
tried low on review/coin); (2) replace the E-LR constant with a warmup→decay
schedule (fast E early for cold-start, slow late to stop over-clustering);
(3) self-limiting E auxiliary. Protocol: every candidate runs on wiki+review+
coin together, accepted only if it regresses none. Coin's curve is unmeasured
this session (the untested middle).

### 2026-08-03 — review at slow-E (1e-4): FALSIFIES the unified-slow-E hypothesis

**Prompt:** "Run review with manifold lr 1e-4 now."

**Action:** review, K=10, fixed-weight head, **manifold_lr 1e-4** (only change from
the fast-E baseline; model_lr 1e-3, dropout 0). Log:
`review_k10_fixedweight_mlr1e-4_seed42.log`.

**RESULT — slow-E does NOT fix review; strictly worse:**

| ep | val (slow 1e-4) | val (fast 1e-3) | commP (slow) |
|---|---|---|---|
| 1 | 0.2273 | 0.2850 | ×1.2 |
| 2 | 0.1407 | 0.2492 | ×1.2 |
| 3 | 0.1142 | 0.2083 | ×1.3 |

Slow-E gives a LOWER peak (0.2273 vs 0.2850) AND still collapses (harder). commP
pinned at the NULL (×1.2→×1.3) while val craters and train loss falls
(1.33→0.96) — E isn't clustering at all, yet val collapses. Definitive: review's
collapse is memorisation of non-transferable structure, ORTHOGONAL to E-speed.

**UNIFIED-SLOW-E HYPOTHESIS FALSIFIED:**
- wiki: slow-E fixes it (its collapse = over-clustering drift, commP ×16 → throttled).
- review: slow-E worse (its collapse = memorisation, commP at null → E-speed irrelevant).
- No single manifold_lr serves both. wiki wants slow-E; review wants fast-E (peak)
  and collapses at ANY E-speed. Different mechanisms → no shared E-LR fix.
The only lever that ever changed REVIEW's curve is the alignment loss (a different
objective) — review's problem is what E is optimised TOWARD, not how fast.
[Open: coin at 1e-4 — does coin pattern with wiki (clustering) or review (memorisation)?]

### 2026-08-03 — tgbl-wiki trajectory (K=100): CLIMBS then gently drifts — different class

**Prompt(s):** "Did you run wiki too?" → "run wiki with K=100" → "launch a
parallel run with 1e-4 manifold lr". (Also corrected an over-eager ep1 call:
initially mislabeled wiki-1e-3 ep1 val 0.03 as "broken/failed to learn" — it was
just a SLOW START; wiki climbed from ep2.)

**Runs (fixed-weight head, K=100, dropout 0, bs 1000/eval-bs 50, seed 42):**
- 1e-3 (fast E): `logs/oneepoch/wiki_k100_fixedweight_seed42.log`
- 1e-4 (slow E, parallel): `..._mlr1e-4_seed42.log`

**wiki 1e-3 (fast E):**

| ep | val | commP |
|---|---|---|
| 1 | 0.0298 | ×3.4 |
| 2 | 0.2694 | ×9.1 |
| 3 | 0.6329 | ×12.6 |
| 4 | **0.7891** (peak) | ×13.8 |
| 5 | 0.7827 (pat 1/5) | ×14.3 |
| 6 | 0.7566 (pat 2/5) | ×14.9 |

**wiki 1e-4 (slow E) — FULL curve, plateaus (no drift):**

| ep | val | commP |
|---|---|---|
| 3 | 0.3679 | ×2.2 |
| 4 | 0.7364 | ×3.0 |
| 5 | 0.8125 | ×3.6 |
| 6 | 0.8155 | ×4.4 |
| 7 | 0.8161 | ×5.1 |
| 8 | 0.8161 | ×5.9 |
| 9 | **0.8164** | ×6.8 |

**WIKI E-SPEED VERDICT (decisive):** slow-E wins on BOTH axes —
- fast E (1e-3): peak 0.7891 @ep4 → DRIFTS to 0.7371, commP ×13.8→×16.5.
- slow E (1e-4): climbs to **0.8164** → STABLE PLATEAU (no drift), commP only ×6.8.
Slow-E = +0.027 higher peak AND no over-cluster drift. On wiki the E learning
rate IS the mechanism: fast E over-clusters (×16) and drifts; slow E organises
gently (×6.8) to a higher, stable plateau.

**Sharpened unified question:** slow-E turns wiki's drift into a stable plateau.
Does slow-E (1e-4) likewise turn review/coin's ep1 CLIFF into a climb/plateau?
That single experiment (review+coin at manifold_lr 1e-4, still untested) decides
whether 1e-4 is the one regime for all three. [NEXT]

**KEY FINDING — wiki is a DIFFERENT failure class from review/coin:**

| dataset | climbs to peak at | post-peak | severity |
|---|---|---|---|
| review | ep1 | −0.057 / 1ep | sharp cliff |
| coin | ep1 | −0.168 / 1ep | sharpest cliff |
| wiki | **ep4** (4-epoch climb) | −0.033 / 2ep | **gentle drift** |

Wiki needs several epochs to BUILD E structure (val↑ as commP↑ together), hits a
high peak, then slowly over-clusters (commP keeps climbing ×13.8→×14.9 while val
gently drifts). Review/coin fit everything in ep1 then crater as continued
training memorises non-transferable structure.

**Consequence for the unified fix:** "slow-E to stop over-clustering" is a WIKI
remedy (wiki over-clusters past ep4). It won't address review/coin's ep1 cliff,
which is memorisation of non-transferable structure, not a clustering drift. So a
single manifold_lr can't serve both regimes — ALSO note: the SAME fast-E (1e-3)
that peaks review/coin lets wiki climb fine (wiki was NOT broken at 1e-3, just
slow-starting). Caveat retired: everything earlier this session ran at 1e-3;
slow-E (1e-4) on review/coin is still UNTESTED and remains the open experiment.

### 2026-08-03 — tgbl-coin trajectory (K=20): collapses too, HARDEST of the three

**Prompt:** "Run tgbl coin to know its trajectory. Does it face similar issue?
Use K=20."

**Action:** tgbl-coin, K=20, fixed-weight head (committed f90c83a5), review-
comparable knobs (manifold_lr 1e-3, dropout 0, bs 1000 / eval-bs 2000, patience
5, seed 42). 22.8M edges, ~38 min/epoch. Log:
`logs/oneepoch/coin_k20_fixedweight_seed42.log`.

**RESULT — yes, coin collapses, most severely of all three:**

| epoch | val | commP | train link |
|---|---|---|---|
| 1 | **0.6129** (peak) | 0.394 (×3.7) | 0.5294 |
| 2 | 0.4452 (patience 1/5) | 0.458 (×4.3) | 0.2819 |
| 3 | 0.4320 (patience 2/5) | 0.473 (×4.5) | 0.2153 |

- Peak ep1 then −0.168 at ep2 (−27% relative) — the sharpest collapse of any
  dataset. Train loss halved (0.53→0.28→0.22).
- **commP climbs monotonically (×3.7→×4.5) as val collapses** — like WIKI. Coin
  has real community structure (Louvain Q=0.544, commP ×3.7), unlike review (~null).

**THE MAP IS NOW COMPLETE — one-epoch collapse is UNIVERSAL across wiki/review/coin:**

| dataset | commP | community structure | collapse | commP-while-collapsing |
|---|---|---|---|---|
| wiki | ×13–19 | strong | yes | climbs |
| coin | ×3.7→×4.5 | real (Q=0.544) | yes (hardest) | climbs |
| review | ×1.3 (~null) | ~none | yes | wobbles at null |

**Through-line:** all three collapse, all track the TRAIN-LOSS crash =
memorisation. Whether memorisation ALSO manifests as clustering (wiki, coin) or
not (review) is a dataset property, NOT the mechanism. → One E-side throttle
(E-LR decay/schedule or freeze-E) is the lever that could address all three at
once. Run killed after ep3 (trajectory established).

### 2026-08-03 — Fixed recency/hop pooling weights (remove learned weight_net)

**Prompt:** "Remove TimeEncoder, remove weight_net, do the same log addition
[log_weight = -(log1p(age)+log1p(hop-1))] as the alignment loss. Make sure lower
age & lower hop get higher weight — will the negative do the job? Edit and run
cleanly (don't commit yet), I'm hopeful."

**Action (clean, on `feature/link-trained-e`):** deleted `TimeEncoder` class +
dead numpy import; removed learned `weight_net` + `time_encoder`;
`_token_weight_logits` is now the FIXED prior `-(log1p(age)+log1p(hop-1))`
(t2v_dim kept in signature for CLI compat, unused). Head params 1974→1618.
**Direction verified numerically:** softmax weights close(age1,hop2)=0.81 >
mid(age5,hop3)=0.18 > far(age50,hop6)=0.01 — the leading minus makes
recent+closest the argmax. ✓

**RESULT — best peak, same collapse:**

| epoch | val | commP | train link |
|---|---|---|---|
| 1 | **0.2850** (best ep1 of session) | 0.143 (×1.3) | 1.20 |
| 2 | 0.2492 (patience 1/5) | 0.158 (×1.4) | 1.08 |
| 3 | 0.1768 (patience 2/5) | 0.150 (×1.3) | 0.73 |

- Fixed prior is a slightly BETTER scorer than the learned weight_net (ep1 0.2850
  > v_combo 0.2835) — worth keeping. But the collapse is untouched.
- **commP went DOWN at ep3 (0.158→0.150) while val collapsed** — final nail in the
  "over-clustering" coffin. The collapse tracks the TRAIN-LOSS crash
  (1.20→1.08→0.73), i.e. pure memorisation; commP is a near-null bystander.

**HEAD-SIDE SEARCH CLOSED.** Tried on review: 6-dist / 4-dist / 2-cross /
2-same-type / bare-linear / ±tau / learned-vs-fixed weighting — EVERY variant
peaks then collapses, tracking train-loss. No head/pooling/distance/tau lever
changes the shape. → Fix must be E-optimisation (E-LR decay/schedule, freeze-E)
or objective (alignment). Committing the fixed-weight refactor (clean win on
peak + simpler head), per user.

### 2026-08-02 (interim no-tau probes, brief)
- 6-dist, tau removed: 0.2668→0.1976 (collapse a full epoch EARLIER than with-tau;
  tau was a mild stabiliser absorbing distance inflation).
- 2-cross, tau removed / 2-same-type ±tau: all peak-then-collapse (killed early).

### 2026-08-02 — CORRECTION: "over-clustering" was a wiki import; review is memorisation

**Prompt:** "How can x6 or x7 and x1.3 both be over clustering? Are we missing
something?"

**The catch (valid):** commP is relative to the per-dataset null. wiki null
~0.032 → commP ×13–19 (genuine strong clustering). **review null 0.113 → commP
×1.0–×1.3 = essentially AT the null.** So on review E is NOT meaningfully
clustering. Calling review's ×1.3 "over-clustering" was WRONG — I pattern-matched
the wiki narrative onto review without checking magnitudes.

**Corrected picture:**
- The true shared invariant is the one-epoch DEFINITION: **train loss ↓ while val
  peaks-then-drops = memorisation.** On review this is stark (train link
  1.24→1.09→0.84, earlier runs →~0.23) while val cliffs.
- commP is NOT the mechanism — it's a dataset-specific *readout*: wiki overfitting
  takes a community-clustering shape (commP ×19); review overfitting takes some
  OTHER geometric form (fitting specific pairs / walk geometry) that Louvain
  purity doesn't capture (commP stays at null).
- So "throttle E's clustering" is wiki-logic; on review there's little clustering
  to throttle. The head-invariance result STILL holds (driver is E-side) but the
  driver is **E memorisation**, not E over-clustering.

**What we're missing:** commP is the wrong probe for review (near-null,
uninformative). Need (a) train-loss vs val-MRR on one axis, (b) a seen-vs-unseen
split to see WHAT E overfits when it isn't community structure. Fix family
unchanged (E-LR decay / freeze-E / E-reg) but reframed as ANTI-MEMORISATION, not
anti-clustering — the E-LR schedule helps both datasets for two different reasons.

### 2026-08-02 — TEMP TEST: 2 cross distances only (channel ablation)

**Prompt:** "Now just do two distances [d(E[u],P[v]), d(P[u],E[v])]. Remove 4
others. See if this fixes the issue."

**Hypothesis:** keep the real MLP head + features (so performance is preserved,
unlike the bare-linear probe); drop the 4 non-cross channels (identity d(E,E),
overlap d(P,P), both self-disp). Do those channels drive the over-clustering?

**Action (uncommitted, on `feature/link-trained-e`, over the reverted 6-dist
head):** distance stack → 2 cross terms only; `score_in = 2 + 2*d_nf + 2*F`; MLP
head unchanged (head params 1974→1846). Same review knobs (K=10, manifold_lr
1e-3, bs 1000, patience 5). Log: `logs/oneepoch/review_2crossdist_seed42.log`.

**RESULT — does NOT fix it (collapses, peak shifted to ep2):**

| epoch | val | commP | train link |
|---|---|---|---|
| 1 | 0.2787 | 0.118 (×1.0) | 1.24 |
| 2 | **0.2796** (peak, new best) | 0.136 (×1.2) | 1.09 |
| 3 | 0.2083 (patience 1/5) | 0.153 (×1.3) | 0.84 |

- Performance preserved (0.28 level, unlike bare-linear's 0.11) — the MLP+features
  carry it. commP started at the NULL (×1.0) — dropping d(E,E)/d(P,P) delayed E's
  clustering by ~1 epoch — but it still climbed (×1.0→×1.2→×1.3) and val still
  cliff-collapsed at ep3 (−0.071, steeper than the 6-dist −0.056).

**CROSS-VARIANT SUMMARY (review, K=10) — the collapse is UNIVERSAL:**

| variant | curve | commP | verdict |
|---|---|---|---|
| 6-dist MLP (baseline) | .2792→.2777→.2222 | ↑ | collapses |
| 4-dist (no self-disp) | .2829→.2830→.2126 | ↑ | collapses |
| 2-cross-dist MLP | .2787→.2796→.2083 | ×1.0→↑ | collapses |
| bare linear (6-dist) | .1012→.1077→.0979 | ↑ | collapses |

**Every variant: commP↑ monotonic, train-loss↓, val peaks-then-collapses.** Head
capacity / distance-channel choice only moves the peak epoch and cliff steepness,
never the collapse. → CONFIRMED: no head-side lever works. Fix must throttle E's
trajectory (E-LR decay/schedule, freeze-E-after-N, anti-clustering/spread prior).
Run killed after ep3. model.py holds the temp 2-cross-dist edit (uncommitted).

### 2026-08-02 — TEMP TEST: bare linear distance-only scorer (capacity ablation)

**Prompt:** "Let's verify first, just score dists. Don't concat any features.
Use just (6, 1) scorer. Remove GELU, dropout and 2 layers. This is just a temp
test, so no need to make it clean or commit. See if this fixes anything."

**Hypothesis:** if the collapse is head *memorisation*, a near-capacity-free
head can't memorise → the curve should stop peaking-at-1. This is the cleanest
possible test of the "head capacity is the culprit" theory.

**Action (uncommitted, on `feature/link-trained-e`):** scorer →
`nn.Linear(6, 1)` (7 params, was 225); dropped GELU / Dropout / hidden layer;
`feats = distances` only (nf / nbhd_feat no longer concatenated — still computed
but unused). Launched tgbl-review, K=10, manifold_lr 1e-3, dropout 0 (now inert),
bs 1000/1000, patience 5, seed 42, no hist. Log:
`logs/oneepoch/review_lineardist6_seed42.log`. Watching the ep1→ep3 curve —
does the collapse (baseline 0.2792→0.2777→0.2222) soften or persist?

**RESULT — decisive, and it REDIRECTS the fight to E:**

| epoch | val | commP |
|---|---|---|
| 1 | 0.1012 | 0.131 (×1.2) |
| 2 | **0.1077** (peak) | 0.149 (×1.3) |
| 3 | 0.0979 (patience 1/5) | 0.166 (×1.5) |

- Absolute MRR cratered (0.11 vs MLP's 0.28) — stripping features + nonlinearity
  removed the head's predictive power. Useless as a model; pure mechanism probe.
- **The collapse STILL happens with a 7-param head.** Head capacity controls the
  *sharpness*, not the *existence*: MLP peaks ep1 → cliff (−0.056@ep3); linear
  peaks ep2 → gentle slope (−0.010@ep3). Gutting the head delayed + softened the
  collapse but did not remove it.
- **The tell: commP climbs monotonically (0.131→0.149→0.166) even with a head
  that CANNOT memorise (7 params), while val declines.** So the drifting thing is
  **E** — it keeps over-clustering past the generalising point, eroding val
  *regardless of head capacity*. Matches the 07-29 finding ("drift is in E's
  trajectory itself"); now confirmed to survive a near-zero-capacity head.

**CONCLUSION:** the one-epoch collapse on review is driven primarily by **E
over-clustering**, NOT head memorisation. The head only modulates how violent
the drop is. → The unified fix must target **E's trajectory** (E-LR warmup→decay,
E regularisation, freeze-E-after-N, spread prior), not the head. The E-LR
schedule idea gains weight: it directly throttles the over-clustering and could
unify wiki-slow-E / review-fast-E into one temporal schedule. Run killed after
ep3 (shape decided). model.py still holds the temp bare-linear scorer
(uncommitted) — revert before the next real experiment.

---

### 2026-08-03 — On the record: HARD (historical) negatives were tried and did NOT help

Noting this explicitly because it's easy to forget it was already tested. Early
this session we tried **hard negatives** — TGB-style **historical negatives**:
per positive `(u, v, t)`, draw a fraction of the K training negatives from the
destinations `u` has previously linked to (destinations seen in `u`'s own past =
genuinely hard negatives, since the model has to distinguish them from the true
next target), with the remainder uniform-random.

**How we implemented it (on branch `feature/hist-negatives`, commits
`4398944b` → `0c491445` → `0fc7de72`):**
- `HistoricalReservoir` — a per-source reservoir of past destinations maintained
  by **Vitter's Algorithm R** (uniform reservoir sample, O(V·M) memory, not
  O(t·V·V)). Strictly causal: `observe(src, dst)` is called POST-scoring (so at
  score time the reservoir holds only strictly-earlier edges), and `reset()` per
  epoch.
- `MixedNegativeSampler` — one abstraction that takes `hist_ratio`, splits K into
  `K_hist` (drawn from the reservoir) + `K_rand` (uniform), backfills invalid /
  cold-source historical slots with random, and returns the combined `[B, K]`.
  hist_ratio = 0 → pure uniform (safe default).

**How we ran it:** tgbl-review, `--hist-neg-ratio 0.2 --reservoir-size 64`,
K=50, manifold_lr 1e-3, dropout 0, bs 1000/1000, patience 5, seed 42.

**Result — same one-epoch collapse, hard negatives changed nothing:**

| variant | ep1 | ep2 | ep3 |
|---|---|---|---|
| hist 0.2 (6-dist scorer) | 0.2792 | 0.2777 | **0.2222** |
| hist 0.2 + no-self-disp (4-dist) | 0.2829 | 0.2830 | **0.2126** |

Both peak early and fall off the same cliff as uniform negatives.

**Why it didn't help (consistent with the E-side diagnosis):** hard negatives
change *which* negatives the head ranks against, but the collapse is E
memorising training-specific structure that doesn't transfer (train loss keeps
crashing regardless). A harder negative distribution doesn't stop that — it
just gives the head a different (arguably more systematic, thus easier-to-
memorise) target. Hard negatives are a **head/loss-side** lever, and by now we
have strong evidence (head ablations, ±tau, learned-vs-fixed weighting,
E-speed) that the collapse is **not** on the head/loss side. So: **hard
negatives are ruled out as a fix — already tried, no effect on the curve.**
