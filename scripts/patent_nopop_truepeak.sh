#!/bin/bash
# Patent, NO pop bias, on feature/lorentz-stable-dist (stable geodesic distance is now
# unconditional -- no flag). Purpose: find the TRUE peak.
#
# Last night's equivalent run reached test 0.2361 @ep12 but was CUT at ep13 while only at
# patience 1/5, so that number is a floor. Cap raised to 30 so early stopping, not the cap,
# decides where it ends.
#
# Reference peaks: Lorentz+pop 0.1809 | ball+pop 0.2175 | cut nopop run 0.2361
#                  all-time record 0.2362 | LB #1 JODIE 0.2460
set -u
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1
ROOT=$WD/logs/patent_truepeak; mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
LOG=$ROOT/Patent.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --lr 1e-3 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# Patent TRUE PEAK: no pop bias, stable Lorentz dist (unconditional)"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
  echo "# NOTE: driver scripts/patent_nopop_truepeak.sh is UNTRACKED; SHA covers trained code only"
  echo "# no --num-epochs passed -> branch default 50; patience 5 decides the stop"
  echo "# refs: Lorentz+pop 0.1809 | ball+pop 0.2175 | cut nopop 0.2361 | record 0.2362 | LB 0.2460"
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
echo "[$(date '+%F %T')] START Patent nopop truepeak (branch $BRANCH @ $SHA)" >> "$DRIVER"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1; RC=$?
TRACE=$(grep -oP 'test \K[0-9.]+' "$LOG" | tr '\n' ' ')
PEAK=$(printf '%s\n' $TRACE | sort -rn | head -1)
SUM=$(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')
echo "[$(date '+%F %T')] DONE rc=$RC peak=${PEAK:-NONE} $SUM" >> "$DRIVER"
echo "[$(date '+%F %T')] trace: $TRACE" >> "$DRIVER"
