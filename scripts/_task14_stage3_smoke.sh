#!/usr/bin/env bash
# Task 14 Stage 3 smoke: 1-ep each of 14a (modulator) and 14b (aux loss).
# Both run on C1 base: --no-ef-in-projection (suppresses EF in p_context)
# while keeping d_edge_feat available for the new aux modules.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

run_one () {
  local TAG="$1"; shift
  local LOG="logs/t14_smoke_${TAG}.log"
  echo "===== SMOKE ${TAG} → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-wiki \
      --use-gpu \
      --num-epochs 1 \
      --seed 42 \
      --skip-final-full-eval \
      --no-ef-in-projection \
      "$@" \
      2>&1 | tee "${LOG}"
  echo "===== SMOKE ${TAG} done ====="
}

run_one 14a --ef-modulate-weight
run_one 14b --ef-aux-lambda 0.1
echo "=== Stage 3 smoke complete ==="
