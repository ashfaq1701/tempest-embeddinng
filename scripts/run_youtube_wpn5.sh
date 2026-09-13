#!/bin/bash
# YouTube at num_walks_per_node=5 (vs 10 in the geometries sweep), k_train default 5.
# logs/ only -- NOT auto-archived; this is a comparison arm.
set -u
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1
SEED=42
LOG=$WD/logs/youtube_wpn/d64_k5/wpn5/YouTube.log
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --num-walks-per-node 5 --lr 1e-3 --seed $SEED --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (3.29M edges, NON-bipartite)"
  echo "# experiment=youtube_wpn  cell=d64_k5  tag=wpn5   ARM: num_walks_per_node=5 (vs 10 in the sweep)"
  echo "# geometry=lorentz  seed=$SEED"
  echo "# d_emb=64 k_train=5 num_walks_per_node=5 lr=1e-3 patience 5 cap 100; no popularity channel"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
  echo "# driven by scripts/run_youtube_wpn5.sh (UNTRACKED); NOT auto-archived (comparison arm)"
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
