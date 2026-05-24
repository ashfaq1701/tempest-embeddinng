#!/usr/bin/env bash
# Task 16 Stage A — InfoNCE on wiki, 3 seeds × 30 ep, tau=0.5.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

for SEED in 42 123 7; do
  LOG="logs/t16_wiki_seed${SEED}.log"
  echo "===== t16 wiki seed ${SEED} → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-wiki \
      --use-gpu \
      --num-epochs 30 \
      --seed "${SEED}" \
      --skip-final-full-eval \
      --tau 0.5 \
      2>&1 | tee "${LOG}"
  echo "===== t16 wiki seed ${SEED} done ====="
done
echo "=== t16 stage A complete ==="
