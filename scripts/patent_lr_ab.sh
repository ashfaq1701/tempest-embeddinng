#!/bin/bash
# Run one Patent arm of the fast-lr-group A/B, from a named source tree.
#
#   usage: scripts/patent_lr_ab.sh <tree> <tag>
#   e.g.   scripts/patent_lr_ab.sh /mnt/nfs2/inf/ms2420/tempest-embeddinng A_temp_only
#          scripts/patent_lr_ab.sh /mnt/nfs2/inf/ms2420/tempest-varB       B_non_embedding
#
# Each arm is a plain code state -- no variant flag. A = temperature alone in the 1e-2
# group; B = every non-embedding param (temperature + NN pooler). Logs always land in the
# MAIN repo's logs/ tree per CLAUDE.md, under logs/lr_group_ab/d64_k5/<tag>/Patent.log.
# Patience is raised to 8: the whole point is to see past the warmup grind that stopped
# Patent at epoch 1 under patience 3.

TREE=${1:?source tree required}
TAG=${2:?tag required}
WD=/mnt/nfs2/inf/ms2420/tempest-embeddinng
source $WD/scripts/env.sh >/dev/null 2>&1
OUT=$WD/logs/lr_group_ab/d64_k5/$TAG
mkdir -p "$OUT"
LOG=$OUT/Patent.log

cd "$TREE" || exit 1
BRANCH=$(git rev-parse --abbrev-ref HEAD); SHA=$(git rev-parse --short HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | awk '{print $2}' | tr '\n' ' ')
{
  echo "# A/B arm: dataset=Patent experiment=lr_group_ab cell=d64_k5 tag=$TAG"
  echo "#   d_emb=64 k_train=5 lr=1e-3 lr_temperature=1e-2 patience=8 seed=42 NO pop-bias"
  echo "#   tree=$TREE branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# UNCOMMITTED tracked changes on top of $SHA: $DIRTY"
  echo "#   fast-lr group: $([ "$TAG" = A_temp_only ] && echo 'geo_temp only' || echo 'geo_temp + NN pooler (all non-E)')"
  echo "# started=$(date '+%F %T')"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 stdbuf -oL -eL $PY -u scripts/train_link_property_prediction.py \
  --data-suite tgb-seq --dataset Patent --data-root $WD/datasets \
  --d-emb 64 --k-train 5 --lr 1e-3 \
  --num-epochs 50 --early-stop-patience 8 \
  --use-gpu --use-gpu-tempest >> "$LOG" 2>&1
echo "[$(date '+%F %T')] $TAG rc=$? $(grep -E 'best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$OUT/../AB.log"
