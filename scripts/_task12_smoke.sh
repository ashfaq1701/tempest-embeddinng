#!/usr/bin/env bash
# Task 12 smoke tests — 1 epoch each of C1, C2, C3, C4 on wiki seed 42.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

run_one () {
  local TAG="$1"; shift
  local LOG="logs/t12_smoke_${TAG}.log"
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

run_one c1 --force-no-ef
run_one c2
run_one c3 --no-ef-on-context --ef-on-target
run_one c4 --ef-on-target
echo "=== smoke complete ==="
