#!/bin/bash
# Chained EXACT replicate of the euclidean Patent run, launched once the 8-dataset
# euclidean sweep reaches SWEEP COMPLETE. Same seed (42), same branch, same flags --
# a determinism check, not a new configuration. Per CLAUDE.md, same-seed/same-machine
# reproduction lands within ~0.0002, so a materially different number here means a
# code or config change, not variance.
#
# Writes to logs/ only. NOT auto-archived: experiment_logs/ is for curated results and
# patent/3.log is already taken by the run this replicates.
set -u
WD=/its/home/ms2420/tempest-embeddinng
PY=$WD/venv/bin/python
cd "$WD" || exit 1

ROOT=$WD/logs/geometry_euclidean/d64_k5_lr1e-3
DRIVER=$ROOT/DRIVER.log
CHAIN=$ROOT/CHAIN.log
LOG=$ROOT/Patent/run_3_rerun/Patent.log
mkdir -p "$(dirname "$LOG")"

echo "[$(date '+%F %T')] chain armed; waiting for SWEEP COMPLETE" >> "$CHAIN"
until grep -q "SWEEP COMPLETE" "$DRIVER" 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] sweep complete; starting Patent rerun" >> "$CHAIN"

BRANCH=$(git rev-parse --abbrev-ref HEAD); SHA=$(git rev-parse --short HEAD)
if [ "$BRANCH" != "feature/euclidean" ]; then
  echo "[$(date '+%F %T')] ABORT: on branch $BRANCH, expected feature/euclidean" >> "$CHAIN"
  echo "[$(date '+%F %T')] CHAIN ABORTED" >> "$CHAIN"
  exit 1
fi

CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 100 --early-stop-patience 5 --use-pop-bias \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=Patent (10.8M edges, NON-bipartite)"
  echo "# d_emb=64 k_train=5 lr=1e-3 seed=42  patience 5  cap 100  --use-pop-bias"
  echo "# experiment=geometry_euclidean cell=d64_k5_lr1e-3 tag=run_3_rerun"
  echo "# EXACT replicate of tag=run_3 (same seed, same branch, same machine): a determinism"
  echo "# check. run_3 gave stopped_at_epoch 14, best_val 0.1281, best_test 0.1225 (max test"
  echo "# 0.1344 at ep6). Expect this to land within ~0.0002 of that."
  echo "# geometry=euclidean  branch=$BRANCH commit=$SHA"
  echo "# NOTE: driver scripts/chain_patent_rerun.sh is UNTRACKED; SHA covers the trained code only"
  echo "# NOT archived to experiment_logs/ -- patent/3.log holds the run this replicates."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"

PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
RC=$?
SUMMARY=$(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')
echo "[$(date '+%F %T')] Patent rerun rc=$RC  $SUMMARY" >> "$CHAIN"
echo "[$(date '+%F %T')] CHAIN COMPLETE" >> "$CHAIN"
