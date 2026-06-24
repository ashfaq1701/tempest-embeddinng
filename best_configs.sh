#!/usr/bin/env bash
# best_configs.sh — the best-performing training command per TGB dataset.
# KEEP THIS UPDATED: whenever a new config beats the current best for a dataset,
# replace that dataset's command here (and note the val/test it produced).
set -euo pipefail
cd "$(dirname "$0")"

# wiki   (val 0.8262 / test 0.8027 @ ep17, no-pf — GeometricPointHead REACH on the
#         PER-QUERY CAUSAL substrate: ingest the batch into Tempest BEFORE scoring, then
#         walk each query (u, t) with cutoff=t (strict-before-t, == TPNet). Same head as
#         old reach (val 0.8046 / test 0.7779); +0.025 test from the substrate alone.
#         REQUIRES temporal-random-walk>=1.8.6 (cutoff_times). Smooth monotone curve.)
.venv/bin/python scripts/train.py \
  --dataset tgbl-wiki \
  --k-train 100 \
  --num-walks-per-node-query-side 5 --max-walk-len-query-side 5 \
  --d-emb 128 \
  --batch-size 200 --eval-batch-size 20 \
  --num-epochs 50 --early-stop-patience 5 \
  --seed 42 \
  --lr 1e-3 --lr-min 1e-5 --decay-horizon-epochs 50 \
  --tempest-batch-window-multiplier -1.0 \
  --use-gpu --use-gpu-tempest
