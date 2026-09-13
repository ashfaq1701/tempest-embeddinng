#!/bin/bash
# PATENT FIX EXPEDITION -- screening arms, SERIAL (GPU is 95-100% utilised; parallel adds
# contention, not throughput). Branch feature/lorentz.
#
# Baseline to beat (same branch, same cell, no change):
#   peak test 0.1809 @ep3, then DECLINES: ep4 0.1748, ep5 0.1597, ep6 0.1593
#   while val CLIMBS 0.1242 -> 0.1779. Classic overfit; val is scored on uniform random
#   negatives, test on TGB-Seq's shipped test_ns, so val can improve while test rots.
# Ball reference (poincare, same cell): ep3 0.2010, ep6 0.2059, PEAK 0.2174 @~ep11.
# LB #1 (JODIE) = 0.2460. Project record on Patent = 0.2362 (head params: 2, i.e. NO pop
# bias, K=10) -- that record is the prior behind the nopop arms.
#
# 5 epochs per arm: enough to separate "climbing past 0.19" from "peaked at 0.18 and falling".
set -u
WD=/its/home/ms2420/tempest-embeddinng
PY=$WD/venv/bin/python
cd "$WD" || exit 1
EPOCHS="${EPOCHS:-5}"
ROOT=$WD/logs/patent_fix
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)

# label | extra args   (base config: d64 k5 lr1e-3 seed42, Patent is NON-bipartite)
ARMS=(
  "nopop|"
  "wd1e-4|--use-pop-bias --weight-decay 1e-4"
  "wd1e-2|--use-pop-bias --weight-decay 1e-2"
  "nopop_k10|--k-train 10"
)

{
  echo "# PATENT FIX SCREEN  branch=$BRANCH commit=$SHA  epochs/arm=$EPOCHS"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
  echo "# NOTE: driver scripts/patent_fix_screen.sh is UNTRACKED; SHA covers trained code only"
  echo "# baseline: peak 0.1809 @ep3 then falls to 0.1593 @ep6 (val rises to 0.1779)"
  echo "# ball peak 0.2174 | LB#1 JODIE 0.2460 | project record 0.2362 (no pop bias, K=10)"
  echo "# started=$(date '+%F %T')"
  echo
} > "$DRIVER"

for entry in "${ARMS[@]}"; do
  IFS='|' read -r LABEL EXTRA <<< "$entry"
  LOG=$ROOT/$LABEL/Patent.log
  mkdir -p "$(dirname "$LOG")"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs $EPOCHS --early-stop-patience 5 \
$EXTRA --use-gpu --use-gpu-tempest"
  {
    echo "# ARM=$LABEL  extra='${EXTRA:-(none, and NO --use-pop-bias)}'"
    echo "# branch=$BRANCH commit=$SHA  epochs=$EPOCHS"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  echo "[$(date '+%F %T')] START $LABEL  extra='${EXTRA:-none}'" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  RC=$?
  HEADP=$(grep -m1 'head params' "$LOG" | tr -s ' ')
  PEAK=$($PY - "$LOG" <<'PYEOF'
import re,sys
t=open(sys.argv[1]).read()
v=[(float(m.group(2)),int(m.group(1))) for m in re.finditer(r'epoch (\d+)/\d+.*?test ([0-9.]+)',t)]
print(f"peak={max(v)[0]:.4f}@ep{max(v)[1]}  last={v[-1][0]:.4f}" if v else "peak=NONE")
PYEOF
)
  TRACE=$(grep -oP 'test \K[0-9.]+' "$LOG" | tr '\n' ' ')
  echo "[$(date '+%F %T')] DONE  $LABEL rc=$RC  $PEAK  |$HEADP| test-trace: $TRACE" >> "$DRIVER"
done
echo "[$(date '+%F %T')] SCREEN COMPLETE" >> "$DRIVER"
