# tgbl-wiki test-set MRR stratification — candidate-conditioned attention head

**Model:** candidate-conditioned `SourceWalkAttnHead` + pair features, sphere E.
seed 42, d_emb 128, TPNet protocol (train bs 200
/ eval bs 20), best epoch 4.

- **test MRR (this stratified run): 0.7814** over 23,621 positives
- training-run best: val 0.7998 / test 0.7816 (walk-noise vs above)
- **TPNet ref: test 0.827 / val 0.842  →  test gap ≈ +0.0456**

## 1. Transductivity (endpoint seen in any prior edge)
| stratum | count | frac | mean_rr | hits@1 | hits@10 | contrib |
|---|---|---|---|---|---|---|
| both-seen | 22,522 | 0.953 | 0.8183 | 0.762 | 0.903 | 0.7802 |
| u-only-inductive | 1,066 | 0.045 | 0.0271 | 0.003 | 0.068 | 0.0012 |
| v-only-inductive | 26 | 0.001 | 0.0053 | 0.000 | 0.000 | 0.0000 |
| both-inductive | 7 | 0.000 | 0.0028 | 0.000 | 0.000 | 0.0000 |

## 2. Pair recurrence
| stratum | count | frac | mean_rr | hits@1 | hits@10 | contrib |
|---|---|---|---|---|---|---|
| repeat-pair | 20,634 | 0.874 | 0.8916 | 0.831 | 0.983 | 0.7788 |
| new-pair | 2,987 | 0.126 | 0.0206 | 0.001 | 0.044 | 0.0026 |

## 3. Source-degree buckets (u cumulative interactions)
| stratum | count | frac | mean_rr | hits@1 | hits@10 | contrib |
|---|---|---|---|---|---|---|
| deg=0 | 1,073 | 0.045 | 0.0269 | 0.003 | 0.067 | 0.0012 |
| deg=1 | 512 | 0.022 | 0.5812 | 0.568 | 0.600 | 0.0126 |
| deg 2-5 | 1,213 | 0.051 | 0.6883 | 0.658 | 0.724 | 0.0353 |
| deg 6-20 | 2,486 | 0.105 | 0.7418 | 0.698 | 0.797 | 0.0781 |
| deg 21-100 | 7,393 | 0.313 | 0.8670 | 0.823 | 0.925 | 0.2713 |
| deg >100 | 10,944 | 0.463 | 0.8263 | 0.753 | 0.944 | 0.3828 |

## 4. Cross-tab: pair-recurrence × transductivity (decisive)
| stratum | count | frac | mean_rr | hits@1 | hits@10 | contrib |
|---|---|---|---|---|---|---|
| repeat x both-seen | 20,634 | 0.874 | 0.8916 | 0.831 | 0.983 | 0.7788 |
| repeat x u-only-ind | 0 | 0.000 | 0.0000 | 0.000 | 0.000 | 0.0000 |
| repeat x v-only-ind | 0 | 0.000 | 0.0000 | 0.000 | 0.000 | 0.0000 |
| repeat x both-ind | 0 | 0.000 | 0.0000 | 0.000 | 0.000 | 0.0000 |
| new x both-seen | 1,888 | 0.080 | 0.0173 | 0.000 | 0.032 | 0.0014 |
| new x u-only-ind | 1,066 | 0.045 | 0.0271 | 0.003 | 0.068 | 0.0012 |
| new x v-only-ind | 26 | 0.001 | 0.0053 | 0.000 | 0.000 | 0.0000 |
| new x both-ind | 7 | 0.000 | 0.0028 | 0.000 | 0.000 | 0.0000 |

## Headroom sizing — Δ overall-MRR if the weak stratum's mean_rr were lifted

| stratum | frac | mean_rr | →0.30 | →0.60 | →both-seen (0.818) |
|---|---|---|---|---|---|
| new-pair | 0.126 | 0.0206 | +0.0353 | +0.0733 | +0.1009 |
| new x both-seen | 0.080 | 0.0173 | +0.0226 | +0.0466 | +0.0640 |
| u-only-inductive | 0.045 | 0.0271 | +0.0123 | +0.0259 | +0.0357 |
| new x u-only-ind | 0.045 | 0.0271 | +0.0123 | +0.0259 | +0.0357 |
| v-only-inductive | 0.001 | 0.0053 | +0.0003 | +0.0007 | +0.0009 |
| new x v-only-ind | 0.001 | 0.0053 | +0.0003 | +0.0007 | +0.0009 |
| both-inductive | 0.000 | 0.0028 | +0.0001 | +0.0002 | +0.0002 |
| new x both-ind | 0.000 | 0.0028 | +0.0001 | +0.0002 | +0.0002 |
