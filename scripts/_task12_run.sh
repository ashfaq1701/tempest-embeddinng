#!/usr/bin/env bash
# Task 12 — 4 configs × 3 seeds × 30 epochs on wiki.
# Usage: _task12_run.sh <config_id>
#   config_id ∈ {c1, c2, c3, c4}
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {c1|c2|c3|c4|c5}" >&2
  exit 2
fi
CONF="$1"
case "${CONF}" in
  c1) FLAGS=(--force-no-ef) ;;
  c2) FLAGS=() ;;
  c3) FLAGS=(--no-ef-on-context --ef-on-target) ;;
  c4) FLAGS=(--ef-on-target) ;;
  c5) FLAGS=(--ef-symmetric) ;;
  *)  echo "unknown config: ${CONF}" >&2; exit 2 ;;
esac

for SEED in 42 123 7; do
  LOG="logs/t12_${CONF}_seed${SEED}.log"
  echo "===== ${CONF} seed ${SEED} → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-wiki \
      --use-gpu \
      --num-epochs 30 \
      --seed "${SEED}" \
      --skip-final-full-eval \
      "${FLAGS[@]}" \
      2>&1 | tee "${LOG}"
  echo "===== ${CONF} seed ${SEED} done ====="
done
echo "=== ${CONF} complete ==="
