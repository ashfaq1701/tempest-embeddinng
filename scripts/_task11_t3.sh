#!/usr/bin/env bash
# Task 11 T3 — V0 + V3 at 30 epochs, seed 42 only, wiki.
# Sequential because both need the GPU.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

run_one () {
  local TAG="$1"; shift
  local LOG="logs/t11_t3_${TAG}_30ep_seed42.log"
  echo "===== t3 ${TAG} 30ep seed 42 → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-wiki \
      --use-gpu \
      --num-epochs 30 \
      --seed 42 \
      --skip-final-full-eval \
      "$@" \
      2>&1 | tee "${LOG}"
  echo "===== t3 ${TAG} done ====="
}

run_one v0 --force-no-ef
run_one v3 --ef-low-dim 16
echo "=== t3 complete ==="
