#!/bin/bash
# Patent: pop bias ON + weight decay 1e-6.
# wd=1e-4 COLLAPSED the embedding table to the vertex (|E|mean=1.000 -> hyperbolic radius 0)
# and drove geo_temp negative (-8.1 then -15.7), which inverts the distance term. Killed at ep2.
# geoopt decays EVERY parameter, so 1e-4 was mild for the head and destructive for a 145M-param
# embedding table. 1e-6 is 100x weaker: health check at epoch 1 is |E|mean > 1.02 and geo_temp > 0.
# Rationale: the overfitting surface is the 1.84M-param per-node pop table (~5 train edges
# per node), which memorises the val distribution while test rots. L2 pulls it toward zero.
# Note geoopt applies wd to EVERY parameter, so the embedding table and pooler are decayed too.
# Reference peaks: pop-bias baseline 0.1809 | ball 0.2175 | no-pop-bias 0.2361 | LB#1 0.2460
set -u
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1
ROOT=$WD/logs/patent_fix; DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
LOG=$ROOT/wd1e-4_split/Patent.log; mkdir -p "$(dirname "$LOG")"
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 20 --early-stop-patience 5 \
--use-pop-bias --weight-decay 1e-4 --use-gpu --use-gpu-tempest"
{ echo "# ARM=wd1e-4_split  pop bias ON + wd 1e-4 on pop_bias+pooler ONLY (manifold exempt)"; echo "# branch=$BRANCH commit=$SHA";
  echo "# refs: baseline 0.1809 | ball 0.2175 | nopop 0.2361 | LB#1 0.2460";
  echo "# started=$(date '+%F %T')"; echo "# cmd: $CMD"; echo; } > "$LOG"
echo "[$(date '+%F %T')] START wd1e-4_split (pop bias ON, wd 1e-4 on pop_bias+pooler only, 20 ep)" >> "$DRIVER"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1; RC=$?
TRACE=$(grep -oP 'test \K[0-9.]+' "$LOG" | tr '\n' ' ')
PEAK=$(printf '%s\n' $TRACE | sort -rn | head -1)
echo "[$(date '+%F %T')] DONE  wd1e-4_split rc=$RC peak=${PEAK:-NONE} trace: $TRACE" >> "$DRIVER"
