#!/bin/bash
# Embedding-dimension sweep for the paper. Master head (pure geodesic + pooling
# temperature, 2 head params). One GPU -> strictly sequential.
#
# Order is cheap-first BY DATASET so each dataset's d-curve completes before the
# next one starts: a partial sweep yields whole curves, not scattered cells.
# Overnight (~15h) covers GoogleLocal + YouTube + ML-20M; it then continues into
# Patent and Flickr rather than idling. Kill the driver PID to stop after the
# current cell.
#
# RESUMABLE: a cell whose log already contains "=== Final results ===" is skipped,
# so relaunching after a kill continues where it stopped.

PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-wt-mainline
OUT=$WD/experiment_logs/d-sweep
DRIVER=$OUT/DRIVER.log

DIMS=(8 16 32 64 128 256)
ORDER=(GoogleLocal YouTube ML-20M Patent Flickr)
declare -A BIP=( [GoogleLocal]=1 [ML-20M]=1 [YouTube]=0 [Flickr]=0 [Patent]=0 )

log() { echo "[$(date '+%F %T')] $*" >> "$DRIVER"; }

running() {
  ps -eo args --no-headers | grep 'train_link_property_prediction.py' | grep -qv grep
}

cd "$WD" || exit 1
SHA=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

log "=== d-sweep START  branch=$BRANCH commit=$SHA  dims=${DIMS[*]}  order=${ORDER[*]}"
while running; do log "waiting for an in-flight run to finish"; sleep 120; done

for DS in "${ORDER[@]}"; do
  FLAG=""; [ "${BIP[$DS]}" = "1" ] && FLAG="--is-bipartite"
  for D in "${DIMS[@]}"; do
    LOG=$OUT/$DS/d-$D.log
    if [ -f "$LOG" ] && grep -q '=== Final results ===' "$LOG"; then
      log "SKIP  $DS d=$D (already complete)"
      continue
    fi
    log "START $DS d=$D $FLAG"
    # Provenance header, then the run appends to it.
    {
      echo "# d-sweep cell: dataset=$DS d_emb=$D"
      echo "# branch=$BRANCH commit=$SHA"
      echo "# started=$(date '+%F %T')"
      echo "# cmd: $PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq \\"
      echo "#        --dataset $DS --d-emb $D --num-epochs 50 --early-stop-patience 3 \\"
      echo "#        --use-gpu --use-gpu-tempest $FLAG"
      echo
    } > "$LOG"
    PYTHONUNBUFFERED=1 $PY -u scripts/train_link_property_prediction.py \
      --data-suite tgb-seq --dataset "$DS" --d-emb "$D" \
      --num-epochs 50 --early-stop-patience 3 \
      --use-gpu --use-gpu-tempest $FLAG >> "$LOG" 2>&1
    RC=$?
    RES=$(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')
    log "DONE  $DS d=$D rc=$RC  $RES"
  done
  log "---- $DS curve complete ----"
done
log "=== d-sweep COMPLETE ==="
