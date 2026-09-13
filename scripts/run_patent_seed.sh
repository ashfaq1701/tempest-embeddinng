#!/bin/bash
# Patent at a given seed. d64, k_train 5, wpn 5 (the current default and the better of the
# two wpn arms measured on Patent), lr 1e-3, patience 5.
# Patent is NON-bipartite -- no --is-bipartite flag.
# logs/ only, NOT auto-archived.
#   usage: scripts/run_patent_seed.sh <seed>
set -u
SEED="${1:?usage: run_patent_seed.sh <seed>}"
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1
LOG=$WD/logs/patent_seeds/d64_k5/seed${SEED}_wpn5/Patent.log
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --num-walks-per-node 5 --lr 1e-3 --seed $SEED --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=Patent (10.8M edges, NON-bipartite)"
  echo "# experiment=patent_seeds  cell=d64_k5  tag=seed${SEED}_wpn5"
  echo "# geometry=lorentz  seed=$SEED"
  echo "# d_emb=64 k_train=5 num_walks_per_node=5 lr=1e-3 patience 5 cap 100; no popularity channel"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
  echo "# driven by scripts/run_patent_seed.sh (UNTRACKED); NOT auto-archived"
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
