#!/usr/bin/env bash
# Task 14 Stage 2 smoke: V2 (master default) + V3 (--force-no-ef).
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

run_one () {
  local TAG="$1"; shift
  local LOG="logs/t14_stage2_${TAG}.log"
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

run_one v2_master_default
run_one v3_force_no_ef --force-no-ef
echo "=== Stage 2 smoke complete ==="
