#!/usr/bin/env bash
# ordpool confirm at 30 epochs (15-ep run was epoch-limited: best-val landed
# at the last epoch on both seeds). Same v1-baseline config otherwise. Logs to
# a separate dir so the 15-ep results stay intact. No set -e.
set -u
cd /home/ms2420/CLionProjects/tempest-walk-embedding-new
PY=./.venv/bin/python
OUT=logs/iterations/cheaper_cc_30ep
mkdir -p "$OUT"
PROG="$OUT/_progress.log"; : > "$PROG"

COMMON="--dataset tgbl-wiki --d-emb 128 --tau-align 0.5 --tau-link 1.0 --n-negatives-per-positive 100 \
  --gamma-recency 0.4 --embedding-num-walks-per-node 10 --embedding-max-walk-len 20 \
  --link-pred-max-walk-len 20 --link-head-d-K 16 --link-head-d-pos 96 --link-head-chunk-c 8 \
  --emb-lr 1e-3 --weight-decay 1e-4 --batch-size 500 --eval-batch-size 200 \
  --num-epochs 30 --early-stop-patience 0 --use-gpu --use-gpu-tempest --readout ordpool"

for seed in 42 43; do
  log="$OUT/ordpool30_s${seed}.log"
  echo "[$(date '+%F %T')] START ordpool30 seed${seed}" | tee -a "$PROG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train.py $COMMON --seed "$seed" > "$log" 2>&1
  rc=$?
  best=$(grep -E "best_val_mrr|best_test_mrr" "$log" | tr '\n' ' ')
  echo "[$(date '+%F %T')] END   ordpool30 seed${seed} (exit ${rc})  ${best}" | tee -a "$PROG"
done
echo "[$(date '+%F %T')] ORDPOOL30 DONE" | tee -a "$PROG"
