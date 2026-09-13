#!/bin/bash
# ML-20M, seed 3. Matches the archived replicates' config (d64, k_train 5, wpn 10, lr 1e-3,
# patience 5) so it is directly comparable; only the seed differs.
# ML-20M is BIPARTITE -- the flag changes the negative-candidate pool.
# logs/ only, NOT auto-archived: 1/2/3.log are seeds 44/43/42, so seed 3 is not a slot.
set -u
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1
SEED=3
LOG=$WD/logs/ml20m_seeds/d64_k5/seed3_wpn5/ML-20M.log
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset ML-20M \
--d-emb 64 --k-train 5 --num-walks-per-node 5 --lr 1e-3 --seed $SEED --early-stop-patience 5 \
--is-bipartite --use-gpu --use-gpu-tempest"
{
  echo "# dataset=ML-20M (14.5M edges, BIPARTITE -- flag: --is-bipartite)"
  echo "# experiment=ml20m_seeds  cell=d64_k5  tag=seed3_wpn5"
  echo "# geometry=lorentz  seed=$SEED   (archived replicates 1/2/3 are seeds 44/43/42)"
  echo "# d_emb=64 k_train=5 num_walks_per_node=5 lr=1e-3 patience 5 cap 100; no popularity channel"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
  echo "# driven by scripts/run_ml20m_seed3.sh (UNTRACKED); NOT auto-archived"
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
