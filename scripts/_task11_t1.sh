#!/usr/bin/env bash
# Task 11 T1 — bottleneck dimension sweep on wiki.
# d in {8, 32, 64}; d=16 reused from Task 10.
# 3 seeds × 15 epochs per d.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
HERE=/home/ms2420/CLionProjects/tempest-walk-embedding-new
cd "${HERE}"
mkdir -p logs

for D in 8 32 64; do
  for SEED in 42 123 7; do
    LOG="logs/t11_t1_d${D}_seed${SEED}.log"
    echo "===== t1 d=${D} seed ${SEED} → ${LOG} ====="
    PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${PY}" -u scripts/train.py \
        --dataset tgbl-wiki \
        --use-gpu \
        --num-epochs 15 \
        --seed "${SEED}" \
        --skip-final-full-eval \
        --ef-low-dim "${D}" \
        2>&1 | tee "${LOG}"
    echo "===== t1 d=${D} seed ${SEED} done ====="
  done
done
echo "=== t1 complete ==="
