#!/bin/bash
# WikiLink, d=64 K=5, on feature/3-feat-nn-pooler:
#   3-feature NN pooler softmax(MLP([-(age/mnia), -pos, rad])) + LINEAR geo_temp,
#   single optimizer group at --lr (NO split temperature lr on this branch).
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/3feat_nn_pooler/d64_k5/run_1/WikiLink.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset WikiLink \
--d-emb 64 --k-train 5 --lr-embedding 1e-3 --lr-network 1e-3 --num-epochs 50 --early-stop-patience 3 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=WikiLink (NON-bipartite) d_emb=64 k_train=5 lr=1e-3  NO pop bias"
  echo "# experiment=3feat_nn_pooler cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA alone does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR nn.Parameter init 1.0"
  echo "#       pooling = NN: softmax(MLP([-(age/mnia), -pos, rad])), hidden 8*3, random init"
  echo "#       122 head params (121 pooler + 1 geo_temp), ONE optimizer group at lr 1e-3"
  echo "# This branch forks from 1e5c0d4e and carries the pooler ONLY -- it does NOT include"
  echo "#   the split temperature lr (4f9cce16). So vs the 0.5430 reference exactly ONE"
  echo "#   variable moves: the fixed parameter-free pooling rule -> the 3-feature MLP."
  echo "# WikiLink references (d=64, K=5, lr=1e-3, seed 42, no pop), fixed pooler:"
  echo "#   1-param pooling temp + linear geo_temp : 0.5828 @ ep7  (converged)"
  echo "#   NO pooling temp      + linear geo_temp : 0.5430 @ ep7  (converged, stop ep10)  <-- THE BASELINE"
  echo "#   NO pooling temp      + EXP geo_temp    : 0.6108 @ ep8  (val-peaked, best 1-param)"
  echo "#   LB #1 TGN = 0.6294.  Our record 0.7904 @ ep44 (much larger head)."
  echo "# Prior arm on feature/simple-head-8 (same pooler + geo_temp at lr 1e-2), KILLED at ep1:"
  echo "#   ep1 val 0.3967 test 0.3832, geo_temp 18.039."
  echo "#   Baseline ep1 references: linear geo_temp 0.3867 (geo_temp 11.70), exp 0.4469 (18.18)."
  echo "#   Here geo_temp steps at 1e-3, so expect a SLOWER scale climb than that 18.039."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
