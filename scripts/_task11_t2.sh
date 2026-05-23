#!/usr/bin/env bash
# Task 11 T2 — V0 + V3 on review (the TGB v1 bare name; data is v2).
# 3 seeds × 15 epochs per config.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

run_one () {
  local TAG="$1"; local SEED="$2"; shift 2
  local LOG="logs/t11_t2_${TAG}_seed${SEED}.log"
  echo "===== t2 ${TAG} seed ${SEED} → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-review \
      --use-gpu \
      --num-epochs 15 \
      --seed "${SEED}" \
      --skip-final-full-eval \
      "$@" \
      2>&1 | tee "${LOG}"
  echo "===== t2 ${TAG} seed ${SEED} done ====="
}

for SEED in 42 123 7; do
  run_one v0 "${SEED}" --force-no-ef
done
for SEED in 42 123 7; do
  run_one v3 "${SEED}" --ef-low-dim 16
done
echo "=== t2 complete ==="
