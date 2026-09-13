#!/bin/bash
# Standalone Patent run at seed 40, in PARALLEL with the main sweep's seed-1 Patent run.
# Writes to its own log dir and does NOT archive to experiment_logs -- the winner gets
# archived deliberately, not automatically.
set -u
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1
ROOT=$WD/logs/patent_seed40; mkdir -p "$ROOT"
LOG=$ROOT/Patent.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --num-walks-per-node 10 --lr 1e-3 --seed 40 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{ echo "# Patent, seed 40, parallel arm (main sweep runs seed 1 concurrently)"
  echo "# branch=$BRANCH commit=$SHA"
  echo "# NOT auto-archived"
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"; echo; } > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
