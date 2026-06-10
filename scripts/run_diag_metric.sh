#!/usr/bin/env bash
# Arm-1 (diagonal learned metric) vs cos baseline, matched budget.
# cos is re-run on this branch (not just reused from master) to control for
# cross-run nondeterminism. 30 ep, 2 seeds, v1-baseline config. No set -e.
set -u
cd /home/ms2420/CLionProjects/tempest-walk-embedding-new
PY=./.venv/bin/python
OUT=logs/iterations/diag_metric
mkdir -p "$OUT"
PROG="$OUT/_progress.log"; : > "$PROG"

COMMON="--dataset tgbl-wiki --d-emb 128 --tau-align 0.5 --tau-link 1.0 --n-negatives-per-positive 100 \
  --gamma-recency 0.4 --embedding-num-walks-per-node 10 --embedding-max-walk-len 20 \
  --link-pred-max-walk-len 20 --emb-lr 1e-3 --weight-decay 1e-4 --batch-size 500 \
  --eval-batch-size 200 --num-epochs 30 --early-stop-patience 0 --use-gpu --use-gpu-tempest"

run () {  # label, seed, extra-args...
  local label="$1" seed="$2"; shift 2
  local log="$OUT/${label}_s${seed}.log"
  echo "[$(date '+%F %T')] START ${label} seed${seed}" | tee -a "$PROG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train.py $COMMON --seed "$seed" "$@" > "$log" 2>&1
  local rc=$? best
  best=$(grep -E "best_val_mrr|best_test_mrr" "$log" | tr '\n' ' ')
  echo "[$(date '+%F %T')] END   ${label} seed${seed} (exit ${rc})  ${best}" | tee -a "$PROG"
}

for seed in 42 43; do run cos  "$seed" --link-metric cos;  done
for seed in 42 43; do run diag "$seed" --link-metric diag; done
echo "[$(date '+%F %T')] DIAG METRIC DONE" | tee -a "$PROG"
