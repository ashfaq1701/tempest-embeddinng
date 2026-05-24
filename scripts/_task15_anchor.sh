#!/usr/bin/env bash
# Task 15 Stage 7 — anchor baseline (3 seeds × 5 epochs) on one dataset.
# Usage: _task15_anchor.sh <dataset>
#   dataset ∈ {tgbl-wiki, tgbl-review}
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {tgbl-wiki|tgbl-review}" >&2
  exit 2
fi
DATASET="$1"
case "${DATASET}" in
  tgbl-wiki)   TAG="wiki" ;;
  tgbl-review) TAG="review" ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 2 ;;
esac

for SEED in 42 123 7; do
  LOG="logs/t15_anchor_${TAG}_seed${SEED}.log"
  echo "===== anchor ${TAG} seed ${SEED} → ${LOG} ====="
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u scripts/train.py \
      --dataset "${DATASET}" \
      --use-gpu \
      --num-epochs 5 \
      --seed "${SEED}" \
      --skip-final-full-eval \
      2>&1 | tee "${LOG}"
  echo "===== anchor ${TAG} seed ${SEED} done ====="
done
echo "=== anchor ${TAG} complete ==="
