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
