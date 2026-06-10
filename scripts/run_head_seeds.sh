#!/usr/bin/env bash
# Run one --readout head over seeds {42,43} at the v1-baseline config, so the
# result compares directly to v1 (0.6901) and the full ceiling (0.7040).
# Usage: run_head_seeds.sh <readout>   (e.g. linrec | ordpool)
# No set -e: one seed crashing does not stop the other.
set -u
READOUT="${1:?usage: run_head_seeds.sh <readout>}"
cd /home/ms2420/CLionProjects/tempest-walk-embedding-new
PY=./.venv/bin/python
OUT=logs/iterations/cheaper_cc
mkdir -p "$OUT"
PROG="$OUT/_progress.log"

COMMON="--dataset tgbl-wiki --d-emb 128 --tau-align 0.5 --tau-link 1.0 --n-negatives-per-positive 100 \
  --gamma-recency 0.4 --embedding-num-walks-per-node 10 --embedding-max-walk-len 20 \
  --link-pred-max-walk-len 20 --link-head-d-K 16 --link-head-d-pos 96 --link-head-chunk-c 8 \
  --emb-lr 1e-3 --weight-decay 1e-4 --batch-size 500 --eval-batch-size 200 \
  --num-epochs 15 --early-stop-patience 0 --use-gpu --use-gpu-tempest"

for seed in 42 43; do
  log="$OUT/${READOUT}_s${seed}.log"
  echo "[$(date '+%F %T')] START ${READOUT} seed${seed}" | tee -a "$PROG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train.py $COMMON --seed "$seed" \
    --readout "$READOUT" > "$log" 2>&1
  rc=$?
  best=$(grep -E "best_val_mrr|best_test_mrr" "$log" | tr '\n' ' ')
  echo "[$(date '+%F %T')] END   ${READOUT} seed${seed} (exit ${rc})  ${best}" | tee -a "$PROG"
done
echo "[$(date '+%F %T')] ${READOUT} DONE" | tee -a "$PROG"
