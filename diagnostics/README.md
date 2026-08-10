# Weighted-midpoint collapse diagnosis (P == E[seed])

**Finding:** in the centroid-token-cross head, with **plain `log1p(age)`** pooling weights
`-(log1p(age) + log1p(hop-1))`, the geoopt weighted gyro-midpoint `P_x` collapses onto the seed
embedding `E[seed]` on **every** training batch of `tgbl-review`, i.e. `P_u == E[u]` and `P_v == E[v]`
to float precision. Cause: review edge ages are ~1e7, so the seed slot (age 0 → logit 0) beats every
context slot (age ~1e7 → logit ~-16) by ~e^16 in the softmax; the midpoint is then ~1.0·E[seed].

## Evidence (real batches, this branch)
- `midpoint_collapse_batches.txt` — per-batch `[MIDDIAG]` summary for train batches 1,50,…,300.
  Every batch: `d(P_u,E_u) mean ~4e-6, max ~1e-4`; same for `P_v`.
- `midpoint_collapse_dump.pt` — one full batch (batch 300): per-query `d(P,E[seed])`, seed weights,
  and `P_u`/`E_u` (first 4 dims) side by side. First query `P_u == E_u` bit-identical to 6 dp.

Two collapse mechanisms (both give `P = E[seed]`):
1. normal query — age-0 seed slots dominate the softmax (`seed_w ≈ 1.0`);
2. cold query — the cold-bag guard leaves only the seed slot valid (weight exactly 1.0). The
   `seed_w` metric reads 0 here because `seed_mask` doesn't tag the guard-inserted seed, but the
   collapse is total (`d(P,E[seed]) ≈ 0`). Use `d(P,E[seed])`, not `seed_w`, as the definitive metric.

## Reproduce
```
MID_DIAG_DUMP_AT=300 MID_DIAG_PATH=diagnostics/midpoint_collapse_dump.pt \
PYTHONUNBUFFERED=1 python -u scripts/train_link_property_prediction.py \
  --dataset tgbl-review --use-gpu --use-gpu-tempest --d-emb 64 --k-train 20 \
  --num-walks-per-node 5 --max-walk-len 5 --walk-bias ExponentialWeight --start-bias ExponentialWeight \
  --lr 1e-4 --num-epochs 1 --batch-size 1000 --eval-batch-size 1000 --seed 42
```
Instrumentation: `LinkPredHead._diag_midpoint_collapse` in `link_property_prediction/model.py`
(TEMPORARY, this diagnostic branch only). The iterated log `log1p(log1p(age))` on `master`/the
ctc branch was the fix — it de-collapses the midpoint (seed_w ~0.7, d(P,E) ~0.35).
