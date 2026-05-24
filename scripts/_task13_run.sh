#!/usr/bin/env bash
# Task 13 — 3 seeds × 30 epochs for one Variant 4 architecture.
# Usage: _task13_run.sh <variant>
#   variant ∈ {asym, sym_shared, sym_two}
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {asym|sym_shared|sym_two}" >&2
  exit 2
fi
VARIANT="$1"
case "${VARIANT}" in
  asym|sym_shared|sym_two) ;;
  *) echo "unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

for SEED in 42 123 7; do
  LOG="logs/t13_${VARIANT}_seed${SEED}.log"
  echo "===== ${VARIANT} seed ${SEED} → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset tgbl-wiki \
      --use-gpu \
      --num-epochs 30 \
      --seed "${SEED}" \
      --skip-final-full-eval \
      --ef-variant "${VARIANT}" \
      2>&1 | tee "${LOG}"
  echo "===== ${VARIANT} seed ${SEED} done ====="
done
echo "=== ${VARIANT} complete ==="
