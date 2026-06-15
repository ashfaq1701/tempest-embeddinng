#!/usr/bin/env bash
# best_configs.sh — the best-performing training command per TGB dataset.
# KEEP THIS UPDATED: whenever a new config beats the current best for a dataset,
# replace that dataset's command here (and note the val/test it produced).
set -euo pipefail
cd "$(dirname "$0")"

# wiki   (val ~0.816 / test ~0.794 — GeometricVelocityPerWalkAvgHead: per-walk
#         trajectory fit, averaged, ellipse along the mean motion + learnable channel
#         coeffs, pair features, K_train=300, unbounded. NOTE: still carries the OLD
#         motion-frame ellipse; the heading-frame "ellipse point fix" that made the
#         retired Point head win wiki no-pf is not yet ported — TODO.)
.venv/bin/python scripts/train.py \
  --dataset tgbl-wiki \
  --use-pair-features \
  --k-train 300 \
  --num-walks-per-node 10 --max-walk-len 20 \
  --d-emb 128 \
  --batch-size 200 --eval-batch-size 20 \
  --num-epochs 50 --early-stop-patience 5 \
  --seed 42 \
  --lr 1e-3 --lr-min 1e-5 --decay-horizon-epochs 50 \
  --tempest-batch-window-multiplier -1.0 \
  --use-gpu --use-gpu-tempest
