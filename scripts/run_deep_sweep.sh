#!/usr/bin/env bash
# Deep on-sphere head depth sweep {2,3,4}, seed 42, 30 ep, vs cos 0.7023.
# Identity-init == cos (verified), so depth is the sole axis. No set -e.
set -u
cd /home/ms2420/CLionProjects/tempest-walk-embedding-new
PY=./.venv/bin/python
OUT=logs/iterations/deep_sphere
mkdir -p "$OUT"; PROG="$OUT/_progress.log"; : > "$PROG"
COMMON="--dataset tgbl-wiki --d-emb 128 --tau-align 0.5 --tau-link 1.0 --n-negatives-per-positive 100 \
  --gamma-recency 0.4 --embedding-num-walks-per-node 10 --embedding-max-walk-len 20 \
  --link-pred-max-walk-len 20 --emb-lr 1e-3 --weight-decay 1e-4 --batch-size 500 \
  --eval-batch-size 200 --num-epochs 30 --early-stop-patience 0 --use-gpu --use-gpu-tempest \
  --seed 42 --readout deep_sphere"
for dep in 2 3 4; do
  log="$OUT/depth${dep}_s42.log"
  echo "[$(date '+%F %T')] START depth${dep}" | tee -a "$PROG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train.py $COMMON --depth "$dep" > "$log" 2>&1
  rc=$?; best=$(grep -E "best_val_mrr|best_test_mrr" "$log" | tr '\n' ' ')
  echo "[$(date '+%F %T')] END   depth${dep} (exit ${rc})  ${best}" | tee -a "$PROG"
done
echo "[$(date '+%F %T')] DEEP SWEEP DONE" | tee -a "$PROG"
