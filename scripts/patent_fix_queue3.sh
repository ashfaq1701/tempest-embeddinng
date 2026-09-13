#!/bin/bash
# PATENT FIX -- queue 3 (main event). Waits for the 5-epoch nopop control to finish.
#
# nopop @ep3 = 0.2121 and climbing (+.033,+.042): already past the ball's ep3 0.2010 and past
# the pop-bias baseline's PEAK 0.1809. val rises slowly while test climbs fast -- the healthy
# shape the 0.2362 record had, and the inverse of the failing baseline.
#
# Open question is where it PEAKS (ball 0.2174 @ep11, record 0.2362 @ep12, LB#1 0.2460), which
# needs a long run. stable_nopop runs first at 20 epochs: its first 5 epochs are directly
# comparable to the nopop control (isolating the numerics fix) AND its full length answers the
# peak question -- strictly more information than repeating nopop. If it trails nopop badly by
# ep5 it gets killed and nopop_long takes over.
set -u
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1
ROOT=$WD/logs/patent_fix; DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
while pgrep -f "train_link_property_prediction.py --data-suite" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] queue3 starting (commit $SHA)" >> "$DRIVER"

ARMS=(
  "stable_nopop_long|20|--stable-lorentz-dist"
  "nopop_k10|8|--k-train 10"
)
for entry in "${ARMS[@]}"; do
  IFS='|' read -r LABEL EP EXTRA <<< "$entry"
  LOG=$ROOT/$LABEL/Patent.log; mkdir -p "$(dirname "$LOG")"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs $EP --early-stop-patience 5 \
$EXTRA --use-gpu --use-gpu-tempest"
  { echo "# ARM=$LABEL epochs=$EP extra='$EXTRA'"; echo "# branch=$BRANCH commit=$SHA";
    echo "# started=$(date '+%F %T')"; echo "# cmd: $CMD"; echo; } > "$LOG"
  echo "[$(date '+%F %T')] START $LABEL (epochs=$EP) extra='$EXTRA'" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1; RC=$?
  TRACE=$(grep -oP 'test \K[0-9.]+' "$LOG" | tr '\n' ' ')
  PEAK=$(printf '%s\n' $TRACE | sort -rn | head -1)
  echo "[$(date '+%F %T')] DONE  $LABEL rc=$RC peak=${PEAK:-NONE} trace: $TRACE" >> "$DRIVER"
done
echo "[$(date '+%F %T')] QUEUE3 COMPLETE" >> "$DRIVER"
