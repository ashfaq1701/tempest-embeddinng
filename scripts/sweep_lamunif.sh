#!/usr/bin/env bash
# lam_unif sweep for the propagation-alignment + uniformity loss.
# Sequential (one 8 GB GPU); each epoch line is tagged and appended to
# sweep_progress.log so a single monitor can follow the whole sweep.
set -u
cd /home/ms2420/CLionProjects/tempest-walk-embedding-new
mkdir -p logs/sweep_lamunif
SWEEP=logs/sweep_lamunif/sweep_progress.log
: > "$SWEEP"

for LAM in 0.1 0.3 1.0 3.0 10.0; do
  echo "===== lam_unif=${LAM} =====" >> "$SWEEP"
  LOG="logs/sweep_lamunif/lam_${LAM}.log"
  PYTHONUNBUFFERED=1 stdbuf -oL .venv/bin/python -u scripts/train.py \
      --dataset tgbl-wiki --use-gpu --use-gpu-tempest \
      --num-epochs 20 --early-stop-patience 5 --seed 42 \
      --lam-unif "${LAM}" --unif-t 2.0 2>&1 \
    | stdbuf -oL tee "$LOG" \
    | stdbuf -oL grep -E "epoch |best|Traceback|Error|nan|NaN|collapse" \
    | stdbuf -oL sed "s/^/[lam=${LAM}] /" >> "$SWEEP"
  echo "[lam=${LAM}] DONE" >> "$SWEEP"
done
echo "===== SWEEP COMPLETE =====" >> "$SWEEP"
