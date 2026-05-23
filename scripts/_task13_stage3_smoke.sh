#!/usr/bin/env bash
# Task 13 Stage 3 smoke: 1-ep each of asym, sym_shared, sym_two.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

run_one () {
  local TAG="$1"; shift
  local LOG="logs/t13_smoke_${TAG}.log"
  echo "===== SMOKE ${TAG} → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-wiki \
      --use-gpu \
      --num-epochs 1 \
      --seed 42 \
      --skip-final-full-eval \
      "$@" \
      2>&1 | tee "${LOG}"
  echo "===== SMOKE ${TAG} done ====="
}

run_one asym --ef-variant asym
run_one sym_shared --ef-variant sym_shared
run_one sym_two --ef-variant sym_two
echo "=== Stage 3 smoke complete ==="
