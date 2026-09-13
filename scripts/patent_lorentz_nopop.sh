#!/bin/bash
# Patent, NO pop bias, on branch feature/lorentz (geoopt arccosh distance, no seed plumbing).
# Controlled counterpart to logs/patent_truepeak/ which ran the same config on
# feature/lorentz-stable-dist: the ONLY difference is the geodesic distance implementation.
#
# Reference peaks: Lorentz+pop 0.1809 | ball+pop 0.2175 | stable-dist nopop 0.2347 / 0.2361
#                  all-time record 0.2362 | LB #1 JODIE 0.2460
set -u
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1
ROOT=$WD/logs/patent_lorentz_nopop; mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log; LOG=$ROOT/Patent.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --lr 1e-3 --early-stop-patience 5 --use-gpu --use-gpu-tempest"
{
  echo "# Patent, NO pop bias, branch feature/lorentz (geoopt arccosh dist)"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
  echo "# NOTE: driver scripts/patent_lorentz_nopop.sh is UNTRACKED; SHA covers trained code only"
  echo "# no --num-epochs -> branch default 50; patience 5 decides the stop"
  echo "# counterpart: logs/patent_truepeak/ (same config, stable dist) peak 0.2347 @ep12"
  echo "# refs: Lorentz+pop 0.1809 | ball+pop 0.2175 | record 0.2362 | LB 0.2460"
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
echo "[$(date '+%F %T')] START Patent nopop on $BRANCH @ $SHA" >> "$DRIVER"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1; RC=$?
TRACE=$(grep -oP 'test \K[0-9.]+' "$LOG" | tr '\n' ' ')
PEAK=$(printf '%s\n' $TRACE | sort -rn | head -1)
SUM=$(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')
echo "[$(date '+%F %T')] DONE rc=$RC peak=${PEAK:-NONE} $SUM" >> "$DRIVER"
echo "[$(date '+%F %T')] trace: $TRACE" >> "$DRIVER"
