# Single-table ablation result (v2.4 §13 — Step 6 outcome)

**Date:** 2026-05-21 11:04
**Branch:** `experiment/embedding-table-variations` (off master at `9972ff8`)
**Cell:** `1T_asym` (single E + P_src + P_tgt projections)

## Setup

- wiki seed 42, 50 epochs, no early stop, `--log-debug`
- Pre-launch warmup: col_norm @ ep 2 = 2.9147 → calibrated threshold = 1.5 × 2.9147 = **4.37**
- Threshold differs from dual-table (3.87) by **+12.9%** — within the 20% "noteworthy" band per §13.
- All other knobs match Gate B locked-config (lambda_normbrake=0.1, weight_decay_link=1e-4, lambda_link=0, hist_neg_ratio=0.5, head_mode=cross_table).

## Result

| Metric | 1T_asym (single) | Gate B (dual) | Δ |
|---|---|---|---|
| Best val (peak) | **0.7453 (ep 3)** | 0.7449 (ep 11) | +0.0004 |
| Best test (peak) | 0.7090 (ep 3) | **0.7096 (ep 6)** | -0.0006 |
| ep 50 val | 0.7105 | 0.7335 | **-0.023** |
| Drop from peak | **-0.035** | -0.011 | 3.2× worse |
| col_norm at ep 50 | 4.39 (clamped at 4.37) | 3.91 (clamped at 3.87) | normbrake works |
| L_normbrake | 0.0010 (mild) | 0.0054 (active) | weaker brake |
| Param count | 1.21M (E 1.18M + P_src 16K + P_tgt 16K) | 2.36M | ~half |

## Decision: Outcome C (lock dual-table)

Per §13 decision rules, peak gap is in **band B** (loses < 0.005 → simpler-wins-ties → lock single). **However**, the cliff-shape regression is material and outside CUDA noise:

- ep 50 val gap: -0.023 (vs anchor std 0.0016 — 14× outside)
- Peak-to-ep-50 drop: 3.2× worse than dual

The user's stated decision rule (2026-05-20) explicitly prioritizes smooth val MRR curves: *"We will try to make the loss curve going down smoothly and MRR curve going up smoothly. We will iterate until this is achieved."*

Dual-table delivers cleanly (drop -0.011); single-table doesn't (drop -0.035). The 50% param savings doesn't justify the smoothness regression.

**§13 cliff-shape bonus** (smoother cliff would lock single-table even on a tie) — applies in the OPPOSITE direction here: dual is smoother, so dual wins.

## Mechanism

In 1T_asym, the shared E table must simultaneously serve "source" and "target" roles for every node. The asymmetry is offloaded to two Linear(d, d) projections (P_src, P_tgt), each with d² = 16K params on wiki (d=128). This is limited capacity vs the 2.36M params dual-table allocates to role asymmetry.

Early epochs: P_src/P_tgt suffice because alignment loss has just begun to specialize roles.

Later epochs: alignment loss continues pulling E in conflicting directions (E[seed]-as-source vs E[walk-neighbor]-as-target), and the projections can't fully decorrelate. Result: E drifts in an "average" direction that hurts the link MLP's cross-table reads.

The diagnostic: `grad_E_target == grad_E_ctx` throughout (since they share parameters) — the gradient signal is the SUM of both pulls, not specialized to either role.

## Lock

- Master STAYS on dual-table architecture (Gate B verified).
- `experiment/embedding-table-variations` branch preserved as PORT-FLAG paper-ablation reproducibility artifact.
- master tagged: `locked-v2 = dual-table architecture confirmed`.

## Multi-seed deferred

Per §13 rule "Loses by > 0.005: dual-table stays as locked production. Document outcome in §13 and in port_plan.md as 'tested, rejected.' Keep single-table code as PORT-FLAG for paper ablation completeness." — applied here on cliff-shape grounds rather than peak-MRR grounds.

No multi-seed run needed: the cliff-shape signal is unambiguous (3.2× worse drop is far outside CUDA-noise band).
