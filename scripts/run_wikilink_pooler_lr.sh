#!/bin/bash
# WikiLink d=64 K=5, settled 3-feat NN pooler, POOLER-ONLY fast lr.
#
# WARNING: this script belongs to branch feature/pooler-rest-lr-split (aad0ce91) and CANNOT
#   be reproduced from feature/lr-e-net-split. There is no longer any way to give the pooler
#   a fast lr while keeping geo_temp slow: --lr-network covers BOTH. The flags below were
#   mechanically renamed to keep it parseable, but --lr-embedding 1e-3 --lr-network 1e-2 on
#   this branch is arm B (temperature AND pooler fast), NOT arm C.
#   `git checkout feature/pooler-rest-lr-split` to run arm C for real.
#   It was launched once on aad0ce91 and killed during epoch 1: zero epochs, no result.
# No local edits -- HEAD fully describes this run (drivers are untracked by design, 50653d00).
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/wikilink_pooler_lr/d64_k5/run_1/WikiLink.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset WikiLink \
--d-emb 64 --k-train 5 --lr-embedding 1e-3 --lr-network 1e-2 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=WikiLink (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 lr_pooler=1e-2  NO pop bias  seed=42"
  echo "# experiment=wikilink_pooler_lr cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA alone does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([rec, pos, rad])), 3 -> 32 -> 1"
  echo "#       162 head params (161 pooler + 1 geo_temp)"
  echo "# groups (verified on this commit): lr_pooler=1e-2 holds the four bag_weights tensors ONLY;"
  echo "#       lr=1e-3 holds E.weight, geo_temp, pop_bias.weight."
  echo "#"
  echo "# PURPOSE: third arm of a three-way that separates WHICH fast parameter mattered."
  echo "#   Same head, data, seed, d, K in all three. Only the lr partition differs."
  echo "#"
  echo "#   arm                       geo_temp   pooler    result"
  echo "#   A wikilink_peak    19eb15f3   1e-3     1e-3     0.6313 @ep11, stop ep14 (patience 3)"
  echo "#   B wikilink_2group  303dd012   1e-2     1e-2     KILLED @ep5, behind A by ~0.009 at ep4"
  echo "#   C this run         aad0ce91   1e-3     1e-2     ?"
  echo "#"
  echo "# THE HYPOTHESIS THIS TESTS (aad0ce91's commit message): in arm B a fast geo_temp supplied"
  echo "#   the logit scale directly, so E stayed compact and the ceiling was capped. Arm B's four"
  echo "#   epochs are consistent with that and are the reason this arm exists:"
  echo "#     geo_temp ep1-4  B: 21.14 18.50 16.07 14.22   vs  A: 12.50 15.45 13.23 11.42  (B ~+2.8)"
  echo "#     |E|mean  ep1-4  B: 0.206 0.311 0.396 0.470   vs  A: 0.248 0.365 0.451 0.526  (B ~-0.05)"
  echo "#     link     ep1-4  B: 0.965 0.503 0.351 0.276   vs  A: 1.197 0.574 0.381 0.303  (B LOWER)"
  echo "#     test     ep1-4  B: 0.475 0.550 0.582 0.600   vs  A: 0.426 0.560 0.593 0.609  (B behind 3/4)"
  echo "#   B fit better and ranked worse at every epoch after the first. C keeps the pooler fast"
  echo "#   but hands the scale back to E."
  echo "#"
  echo "# WHAT EACH OUTCOME MEANS:"
  echo "#   C > A  -> the pooler genuinely needed the faster rate; B's loss was the temperature."
  echo "#   C ~ A  -> the pooler lr is neutral on WikiLink; B's deficit was ENTIRELY the temperature."
  echo "#   C < A  -> a fast pooler hurts here on its own, and B's deficit was not the temperature"
  echo "#             at all -- in which case aad0ce91's rationale does not hold on this dataset."
  echo "#   Note A ran at patience 3 and C at patience 5, so C may run longer for the same curve;"
  echo "#   compare peaks epoch-for-epoch, not stop epochs."
  echo "#"
  echo "# WATCH: (a) |E|mean vs A epoch-for-epoch -- the hypothesis predicts C tracks A, not B;"
  echo "#   (b) whether the peak clears A's 0.6313; (c) the ep1 lead again -- B led by +0.049 at ep1"
  echo "#   and was behind by ep2, so DO NOT call this run before ep4."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~30 min/epoch (25 min train + 5 min eval); A took 7.9 h for 14 epochs."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
