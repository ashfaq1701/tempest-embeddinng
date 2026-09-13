#!/bin/bash
# WikiLink, d=64 K=5, on feature/simple-head-8:
#   3-feature NN pooler (softmax(MLP([-(age/mnia), -pos, rad])), 8*3 hidden, random init)
#   + LINEAR geo_temp on its own optimizer group at --lr-temperature.
#
# WARNING: this script belongs to branch feature/simple-head-8 and CANNOT be reproduced
#   from feature/pooler-rest-lr-split. Two reasons:
#     1. --lr-temperature no longer exists. On this branch geo_temp shares --lr-network with
#        the pooler, so it cannot be given a rate of its own independent of the pooler.
#     2. The head itself differs (8*N_FEAT hidden there, fixed 32 here).
#   The flags below were mechanically renamed to keep the script parseable; they do NOT
#   reproduce the original run. `git checkout feature/simple-head-8` to re-run it for real.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/simple_head_8/d64_k5/run_1/WikiLink.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset WikiLink \
--d-emb 64 --k-train 5 --lr-embedding 1e-3 --lr-network 1e-2 --num-epochs 50 --early-stop-patience 3 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=WikiLink (NON-bipartite) d_emb=64 k_train=5 lr=1e-3 lr_temperature=1e-2  NO pop bias"
  echo "# experiment=simple_head_8 cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA alone does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR nn.Parameter init 1.0"
  echo "#       pooling = NN: softmax(MLP([-(age/mnia), -pos, rad])), hidden 8*3, random init"
  echo "#       122 head params (121 pooler + 1 geo_temp)"
  echo "#       optimizer split: geo_temp at lr 1e-2, pooler+E at lr 1e-3 (commit 4f9cce16)"
  echo "# WikiLink references (d=64, K=5, lr=1e-3, seed 42, no pop), fixed parameter-free pooler:"
  echo "#   1-param pooling temp + linear geo_temp : 0.5828 @ ep7  (converged)"
  echo "#   NO pooling temp      + linear geo_temp : 0.5430 @ ep7  (converged, stop ep10)"
  echo "#   NO pooling temp      + EXP geo_temp    : 0.6108 @ ep8  (val-peaked, best 1-param)"
  echo "#   LB #1 TGN = 0.6294.  Our record 0.7904 @ ep44 (much larger head)."
  echo "# TWO variables move vs the 0.5430 arm: the NN pooler REPLACES the fixed pooling rule,"
  echo "#   and geo_temp now steps 10x faster. A win here does not attribute to either alone."
  echo "# WATCH: WikiLink baselines converge by ep7-10, so per YouTube the faster temperature"
  echo "#   slew has little tail to truncate here -- expect it to help or be neutral, unlike"
  echo "#   YouTube where fast scale arrival cost -0.018."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
