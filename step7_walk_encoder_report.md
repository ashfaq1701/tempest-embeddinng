# Step 7 walk encoder report — wiki seed 42

**Branch:** `experiment/add-source-walk-embedding` (off `locked-v2`).
**Cells:** W_off, W_gru (K=5), W_gru_k1 (K=1). All 50-ep wiki seed 42 with `--log-debug`. Locked-v2 base config (alignment + normbrake + WD_link + hist_neg_ratio=0.5 + Component 0).

## Final results

| Cell | Best val (ep) | Best test (ep) | ep 50 val | Drop peak→50 | Train/ep | Eval/ep |
|---|---|---|---|---|---|---|
| **W_off** (encoder OFF) | 0.7453 (ep 9) | **0.7096** (ep 9) | 0.7261 | **-0.019** | ~17s | ~50s |
| **W_gru** (K=5) | 0.7452 (ep 9) | 0.7080 (ep 9) | 0.7404 | **-0.005** | ~20s | ~57s |
| **W_gru_k1** (K=1) | **0.7453** (ep 10) | 0.7089 (ep 10) | **0.7431** | **-0.002** | ~17s | ~57s |

**Headline:** All three within CUDA noise on peak (Δ best test ±0.0016 = anchor std). Encoder ON variants produce **3–10× smoother long-training trajectories**.

## Diagnostic trajectories (ep 1 → ep 50)

| Metric | W_off | W_gru K=5 | W_gru_k1 K=1 |
|---|---|---|---|
| col_norm (clamped target 3.87) | 2.08 → 3.91 | 2.08 → 3.91 | 2.09 → 3.91 |
| L_normbrake | 0 → 0.0054 | 0 → 0.0052 | 0 → **0.0061** (highest) |
| link_w_norm | 0.19 → **0.17** | 0.18 → 0.13 | 0.18 → **0.11** (most constrained) |
| grad_E_target | 0.28 → 0.09 | 0.28 → 0.09 | **0.33 → 0.11** (richer signal) |
| grad_E_context | 0.005 → 0.0001 | 0.005 → 0.0001 | **0.012 → 0.0001** (richer early) |

**Mechanism observations:**

- All three: col_norm clamped + normbrake active. Cliff-fix infrastructure works under encoder ON.
- K=1 has **15% higher per-batch gradient on E_target** (richer learning signal — fewer walks averaging away the seed's pull) AND **2× higher grad_E_context early** (BCE-through-encoder reaches E_context more strongly).
- link_w_norm flatter in encoder-ON cells (0.13 / 0.11) than encoder-OFF (0.17). The encoder absorbs some of what the link MLP was over-fitting to → less weight runaway.
- grad_E_context collapses universally to ~0.0001 by ep 7 — same as the locked-v2 baseline diagnostic (this is Stage 2 Lesson 17's universal context-side grad collapse). Encoder doesn't fix this fundamental.

## §14 decision rule mapping (NOT applied — reporting only)

Per v2.4 §14:

- **Rule A** — W_gru beats W_off by > 0.005: **NO** (Δ -0.0016 K=5, -0.0007 K=1)
- **Rule B** — W_gru ties W_off within anchor std (0.0016): **YES** (both within)
- **Rule C** — W_gru loses by > 0.005: **NO**
- **Rule D** — W_gru_k1 matches W_gru: **YES** (Δ +0.0009 K=1 over K=5)
- **Cliff-shape bonus** — W_gru smoother long-training: **YES** (drop -0.005 / -0.002 vs W_off -0.019; 4–10× smoother)

**Outcome under §14: Scenario B + D + cliff-shape bonus.** Per the spec, this is the "could be lockable on smoothness even at peak parity" regime. **Decision deferred per your instruction.**

## K=1 vs K=5 specifically

| Metric | K=5 | K=1 | Verdict |
|---|---|---|---|
| Best val | 0.7452 | **0.7453** | K=1 +0.0001 (noise) |
| Best test | 0.7080 | **0.7089** | K=1 +0.0009 (noise) |
| ep 50 val | 0.7404 | **0.7431** | K=1 +0.0027 (smoother) |
| Drop peak→50 | -0.005 | **-0.002** | K=1 smoother |
| Train/ep | 19.9s | **16.8s** | K=1 ~15% faster |
| Eval/ep | 57s | 57s | tied |
| Walks-at-eval cost | K=5 per source | **K=1 per source** | K=1 5× cheaper |

**Per §14 Rule D, K=1 matches K=5** within CUDA noise. K=1 is cheaper everywhere (training time, scoring-time walks, eval-time per-source sampling).

## Pre-registered prediction reconciliation

§14 pre-registered (2026-05-20):
- 50% **B** (W_gru ties W_off) — **CONFIRMED** ✓
- 30% A (wins > 0.005) — did not occur
- 15% C (loses > 0.005) — did not occur
- 5% D (K=1 sufficient) — **CONFIRMED** ✓

Both predicted outcomes landed. Cliff-shape bonus is a *positive surprise* relative to the pre-registration: the encoder's BCE-through-the-encoder gradient flow into E tables (per the §14 gradient-flow story) makes the long-training trajectory materially smoother without paying a peak-MRR cost.

## What this tells us

1. **The static-table-feeds-encoder pattern (as currently designed) doesn't unlock new peak MRR on wiki.** Component 0 + cross-table + cliff fix already capture most of the temporal signal walks would add. The encoder's contribution is regularization-shaped (smoother trajectory), not signal-shaped (higher peak).

2. **K=1 suffices.** Five-fold reduction in scoring-time walk sampling without any MRR cost. This is the cheap win regardless of whether the encoder lock decision favors it.

3. **The cross-domain literature predicts wiki saturation here.** Wiki's surprise index is 0.108 — the regime where transductive ID embeddings work and walks add little. Review (surprise index 0.987) is the discriminator dataset per v2.4 §16.

4. **All three cells preserve cliff-fix mechanism.** col_norm clamped at 3.91, L_normbrake active, link_w_norm constrained. The encoder doesn't break locked-v2's stability — it AMPLIFIES it (lower link_w_norm with encoder ON).

## Next-step options (NO auto-execute)

Per the user-prioritized plan + v2.4 §16:

**Option α** (recommended per Phase 1 in earlier prioritization):
- Phase 1 — sanity check on this branch (freeze E_target/E_context, retrain encoder+link, 50-ep wiki). ~2 hr including impl. Disambiguates whether tables carry signal or are dead weight.
- Then per the sanity check outcome: prioritize Exp A or Exp B.

**Option β** (if user prefers to commit to Exp B directly per the lit-review prior):
- Skip sanity check; implement Exp B (walks-only with CAWN hitting count + attention pool) directly. ~3 hr impl. Then 50-ep wiki + 6-ep review.

**Option γ** (if user wants to lock K=1 walk encoder NOW as locked-v3):
- Multi-seed K=1 encoder cell on seeds 7, 13. ~3 hr. Then merge experiment branch to master + tag locked-v3.
- Then start §16 experiments off locked-v3 instead of locked-v2.

**My recommendation:** Option α (sanity check first). Cheapest decisive datapoint for the bigger Hybrid-vs-Walks-only decision in §16.

## Files

- `/tmp/W_off.log` — encoder OFF integration check
- `/tmp/W_gru.log` — encoder ON K=5
- `/tmp/W_gru_k1.log` — encoder ON K=1
- Branch HEAD: `f48b0f9` (doc sync from master)
- Code: `tempest_walks/walk_encoder.py`, plus integrations in `trainer.py` / `evaluator.py` / `config.py` / `scripts/train.py`

**Awaiting your direction before any further action.**
