#!/usr/bin/env bash
# Step 5: isotropy-anchored diag. Does pulling a->1 recover parity with cos
# (fixable) or does it still lose (metric capacity not the lever)? seed 42,
# 30 ep, |a| logging on, three penalty strengths.
set -u
cd /home/ms2420/CLionProjects/tempest-walk-embedding-new
PY=./.venv/bin/python
OUT=logs/iterations/diag_diagnose
PROG="$OUT/_progress.log"

COMMON="--dataset tgbl-wiki --d-emb 128 --tau-align 0.5 --tau-link 1.0 --n-negatives-per-positive 100 \
  --gamma-recency 0.4 --embedding-num-walks-per-node 10 --embedding-max-walk-len 20 \
  --link-pred-max-walk-len 20 --emb-lr 1e-3 --weight-decay 1e-4 --batch-size 500 \
  --eval-batch-size 200 --num-epochs 30 --early-stop-patience 0 --use-gpu --use-gpu-tempest \
  --seed 42 --link-metric diag"

run () {  # lambda
  local lam="$1"; local label="anchor_${lam}"
  local log="$OUT/${label}.log"
  echo "[$(date '+%F %T')] START ${label}" | tee -a "$PROG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train.py $COMMON --metric-aniso-l2 "$lam" > "$log" 2>&1
  local rc=$? best
  best=$(grep -E "best_val_mrr|best_test_mrr" "$log" | tr '\n' ' ')
  echo "[$(date '+%F %T')] END   ${label} (exit ${rc})  ${best}" | tee -a "$PROG"
}

run 0.01
run 0.1
run 1.0
echo "[$(date '+%F %T')] DIAG ANCHOR DONE" | tee -a "$PROG"
