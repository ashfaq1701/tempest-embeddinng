#!/usr/bin/env bash
# Task 7 anchor: 5 epochs × 3 seeds on tgbl-wiki and tgbl-review.
# Sequential. Logs to logs/anchor_<dataset>_seed<S>.log.
set -euo pipefail

PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

# Use the bare TGB names — in TGB 1.x they ARE the v2 data per
# DATA_VERSION_DICT: {'tgbl-wiki': 2, 'tgbl-review': 2, ...}. The
# "-v2" suffix is rejected by the loader.

for DATASET in tgbl-wiki tgbl-review; do
  for SEED in 42 123 7; do
    LOG="logs/anchor_${DATASET}_seed${SEED}.log"
    echo "===== ${DATASET} seed ${SEED} → ${LOG} ====="
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" -u scripts/train.py \
      --dataset "${DATASET}" \
      --use-gpu \
      --num-epochs 5 \
      --seed "${SEED}" \
      --skip-final-full-eval \
      2>&1 | tee "${LOG}"
    echo "===== ${DATASET} seed ${SEED} done ====="
  done
done

echo "=== All 6 anchor runs complete ==="
