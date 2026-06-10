#!/usr/bin/env bash
# Diagnose the diag-metric overfit. Single seed (42), 30 ep, |a| logging on.
# Three discriminating runs:
#   repro  : original diag (H2/H3 trajectory baseline)
#   nonorm : diag - ||a.E_v||^2 term  (surgical H2 test, Step 4)
#   lowlr  : diag, head LR x1/3       (H1 LR-control test, Step 3)
set -u
cd /home/ms2420/CLionProjects/tempest-walk-embedding-new
PY=./.venv/bin/python
OUT=logs/iterations/diag_diagnose
mkdir -p "$OUT"
PROG="$OUT/_progress.log"; : > "$PROG"

COMMON="--dataset tgbl-wiki --d-emb 128 --tau-align 0.5 --tau-link 1.0 --n-negatives-per-positive 100 \
  --gamma-recency 0.4 --embedding-num-walks-per-node 10 --embedding-max-walk-len 20 \
  --link-pred-max-walk-len 20 --emb-lr 1e-3 --weight-decay 1e-4 --batch-size 500 \
  --eval-batch-size 200 --num-epochs 30 --early-stop-patience 0 --use-gpu --use-gpu-tempest \
  --seed 42 --link-metric diag"

run () {  # label, extra-args...
  local label="$1"; shift
  local log="$OUT/${label}.log"
  echo "[$(date '+%F %T')] START ${label}" | tee -a "$PROG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train.py $COMMON "$@" > "$log" 2>&1
  local rc=$? best
  best=$(grep -E "best_val_mrr|best_test_mrr" "$log" | tr '\n' ' ')
  echo "[$(date '+%F %T')] END   ${label} (exit ${rc})  ${best}" | tee -a "$PROG"
}

run repro
run nonorm --metric-no-norm
run lowlr  --head-lr-mult 0.33
echo "[$(date '+%F %T')] DIAG DIAGNOSE DONE" | tee -a "$PROG"
