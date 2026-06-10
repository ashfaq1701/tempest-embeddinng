#!/usr/bin/env bash
# Shuffle gate, Step 1. Runs ONLY full-shuffled x{42,43}; v1 and full
# baselines are code-identical to logs/iterations/kslot_sweep (reused).
# Config matches the kslot-sweep full-head control EXACTLY (10 ep, eval-bs
# 200) so full_shuffled is directly comparable. No set -e: one crash does
# not stop the other seed.
set -u
cd /home/ms2420/CLionProjects/tempest-walk-embedding-new
PY=./.venv/bin/python
OUT=logs/iterations/shuffle_gate
mkdir -p "$OUT"
PROG="$OUT/_progress.log"; : > "$PROG"

COMMON="--dataset tgbl-wiki --d-emb 128 --tau-align 0.5 --tau-link 1.0 --n-negatives-per-positive 100 \
  --gamma-recency 0.4 --embedding-num-walks-per-node 10 --embedding-max-walk-len 20 \
  --link-pred-max-walk-len 20 --link-head-d-K 16 --link-head-d-pos 96 --link-head-chunk-c 8 \
  --emb-lr 1e-3 --weight-decay 1e-4 --batch-size 500 --eval-batch-size 200 \
  --num-epochs 10 --early-stop-patience 0 --use-gpu --use-gpu-tempest \
  --readout full --shuffle-walk-positions"

for seed in 42 43; do
  log="$OUT/full_shuffled_s${seed}.log"
  echo "[$(date '+%F %T')] START full_shuffled seed${seed}" | tee -a "$PROG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train.py $COMMON --seed "$seed" > "$log" 2>&1
  rc=$?
  best=$(grep -E "best_val_mrr|best_test_mrr" "$log" | tr '\n' ' ')
  echo "[$(date '+%F %T')] END   full_shuffled seed${seed} (exit ${rc})  ${best}" | tee -a "$PROG"
done
echo "[$(date '+%F %T')] SHUFFLE GATE DONE" | tee -a "$PROG"
