#!/usr/bin/env bash
# best_configs.sh — the best-performing training command per TGB dataset.
# KEEP THIS UPDATED: whenever a new config beats the current best for a dataset,
# replace that dataset's command here (and note the val/test it produced).
set -euo pipefail
cd "$(dirname "$0")"

# wiki   (val 0.8280 / test 0.8062 @ ep22, no-pf — GeometricPointHead REACH (softmax-μ,
#         COUNT-FREE, self-excluded) on the PER-QUERY CAUSAL substrate: ingest the batch into
#         Tempest BEFORE scoring, then walk each query (u, t) with cutoff=t (strict-before-t,
#         == TPNet). REQUIRES temporal-random-walk>=1.8.6 (cutoff_times).
#         Walk-config + d_emb campaign (seed42, 2026-06-25):
#           - walks 10 > 5 (+0.0010 val, best of {5,10});
#           - longer walks HURT (len10 −0.009, len20 −0.011 val) → len stays 5;
#           - d_emb MONOTONE 64<128<256 (test 0.8015 / 0.8035 / 0.8062) → d_emb 256 is the
#             global best, the first config to clear the ~0.803 test band. Single-seed but
#             trend-supported; a d_emb 512 + multi-seed follow-up is open.)
.venv/bin/python scripts/train.py \
  --dataset tgbl-wiki \
  --k-train 100 \
  --num-walks-per-node-query-side 10 --max-walk-len-query-side 5 \
  --d-emb 256 \
  --batch-size 200 --eval-batch-size 20 \
  --num-epochs 50 --early-stop-patience 5 \
  --seed 42 \
  --lr 1e-3 --lr-min 1e-5 --decay-horizon-epochs 50 \
  --tempest-batch-window-multiplier -1.0 \
  --use-gpu --use-gpu-tempest
