#!/usr/bin/env bash
# Task 10 variant runner — 3 seeds × 15 epochs, log per-seed.
# Usage:  _task10_variant.sh <variant_tag> <extra_flag1> [extra_flag2 ...]
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <variant_tag> [extra train.py flags...]" >&2
  exit 2
fi
TAG="$1"; shift
EXTRA_FLAGS=("$@")

for SEED in 42 123 7; do
  LOG="logs/t10_${TAG}_seed${SEED}.log"
  echo "===== ${TAG} seed ${SEED} → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-wiki \
      --use-gpu \
      --num-epochs 15 \
      --seed "${SEED}" \
      --skip-final-full-eval \
      "${EXTRA_FLAGS[@]}" \
      2>&1 | tee "${LOG}"
  echo "===== ${TAG} seed ${SEED} done ====="
done
echo "=== ${TAG} variant complete ==="
