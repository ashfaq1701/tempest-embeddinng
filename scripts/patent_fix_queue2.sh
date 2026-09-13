#!/bin/bash
# PATENT FIX -- queue 2, reprioritised after the float32 numerics finding.
# Waits for the in-flight arm (nopop) to finish, then runs serially.
#
# Why this order changed: the leading cause of Lorentz < ball on Patent is now believed to be
# float32 precision loss in geoopt's arccosh distance (4.8e-2 relative error at d=1e-3, i.e.
# exactly the close pairs that decide MRR). --stable-lorentz-dist fixes that (31,000x better).
# Weight decay is DEPRIORITISED because it shrinks hyperbolic radius, pushing MORE pairs into
# the ill-conditioned regime -- it should make the problem worse, not better. wd1e-4 is kept
# last so the suggestion still gets tested if time allows.
set -u
WD=/its/home/ms2420/tempest-embeddinng
PY=$WD/venv/bin/python
cd "$WD" || exit 1
EPOCHS="${EPOCHS:-5}"
ROOT=$WD/logs/patent_fix
DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)

# wait for the GPU to free (arm 1 still running)
while pgrep -f "train_link_property_prediction.py --data-suite" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] queue2 starting (commit $SHA)" >> "$DRIVER"

# Reordered after the archive sweep: across EVERY Patent log ever committed, the only run
# above 0.22 is the no-pop-bias record (0.2362, head params 2); every pop-bias run caps at
# ~0.217 (ball) / ~0.208 (euclidean). Pop bias is the dominant lever, so the numerics fix is
# most informative measured ON TOP of nopop: stable_nopop vs arm1 nopop isolates it exactly.
ARMS=(
  "stable_nopop|--stable-lorentz-dist"
  "nopop_k10|--k-train 10"
  "stable|--use-pop-bias --stable-lorentz-dist"
  "wd1e-4|--use-pop-bias --weight-decay 1e-4"
)

for entry in "${ARMS[@]}"; do
  IFS='|' read -r LABEL EXTRA <<< "$entry"
  LOG=$ROOT/$LABEL/Patent.log
  mkdir -p "$(dirname "$LOG")"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs $EPOCHS --early-stop-patience 5 \
$EXTRA --use-gpu --use-gpu-tempest"
  { echo "# ARM=$LABEL  extra='$EXTRA'"; echo "# branch=$BRANCH commit=$SHA epochs=$EPOCHS";
    echo "# started=$(date '+%F %T')"; echo "# cmd: $CMD"; echo; } > "$LOG"
  echo "[$(date '+%F %T')] START $LABEL  extra='$EXTRA'" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  RC=$?
  HEADP=$(grep -m1 'head params' "$LOG" | tr -s ' ')
  TRACE=$(grep -oP 'test \K[0-9.]+' "$LOG" | tr '\n' ' ')
  PEAK=$(printf '%s\n' $TRACE | sort -rn | head -1)
  echo "[$(date '+%F %T')] DONE  $LABEL rc=$RC peak=${PEAK:-NONE} |$HEADP| trace: $TRACE" >> "$DRIVER"
done
echo "[$(date '+%F %T')] QUEUE2 COMPLETE" >> "$DRIVER"
