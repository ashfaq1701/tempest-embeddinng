#!/usr/bin/env bash
# Task 15 Stage 8.2 — 100ep × 3 seeds wiki default with diagnostics.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

for SEED in 42 123 7; do
  LOG="logs/t15_8_2_seed${SEED}.log"
  echo "===== 8.2 default seed ${SEED} → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-wiki \
      --use-gpu \
      --num-epochs 100 \
      --seed "${SEED}" \
      --skip-final-full-eval \
      --config-tag default-100ep \
      2>&1 | tee "${LOG}"
  echo "===== 8.2 default seed ${SEED} done ====="
done
echo "=== 8.2 complete ==="
