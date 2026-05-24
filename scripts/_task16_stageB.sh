#!/usr/bin/env bash
# Task 16 Stage B — InfoNCE on review, 3 seeds × 5 ep, tau=0.5.
# Review is ~5M edges, ~25K batches/epoch. Under C1+two-head SUM,
# review collapsed (val 0.0196 stuck). InfoNCE shouldn't have that
# failure mode — its softmax denominator does anti-collapse over
# task-relevant negatives rather than relying on a separate
# uniformity term that the old architecture's hop count overwhelmed.
# Expectation: review trains stably under InfoNCE.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

for SEED in 42 123 7; do
  LOG="logs/t16_review_seed${SEED}.log"
  echo "===== t16 review seed ${SEED} → ${LOG} ====="
  # batch-size 50 (default 200) to fit InfoNCE's [NK, NK*L] sim
  # matrix on 8 GB GPU alongside review's 45M-param E table.
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-review \
      --use-gpu \
      --num-epochs 5 \
      --seed "${SEED}" \
      --skip-final-full-eval \
      --tau 0.5 \
      --batch-size 50 \
      2>&1 | tee "${LOG}"
  echo "===== t16 review seed ${SEED} done ====="
done
echo "=== t16 stage B complete ==="
