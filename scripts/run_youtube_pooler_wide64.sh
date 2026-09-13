#!/bin/bash
# YouTube d=64 K=5 with a WIDER NN pooler: 3 -> 64 -> 1 (321 pooler params) via --pooler-hidden 64.
# No local edits -- the width knob is committed, so the SHA fully describes this run.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/pooler_wide64/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr-embedding 1e-3 --lr-network 1e-3 --pooler-hidden 64 --num-epochs 50 --early-stop-patience 3 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5 lr=1e-3  NO pop bias  seed=42"
  echo "# experiment=pooler_wide64 cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA alone does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0, ONE optimizer group"
  echo "#       pooling = softmax(MLP([rec, pos, rad])), 3 -> 64 -> 1  (WIDTH under test)"
  echo "#       322 head params (321 pooler + 1 geo_temp)  vs 162 at width 32"
  echo "# Third capacity probe of the same pooler. The two already measured:"
  echo "#   3 -> 32 -> 1       161 par  ckpt 0.5551 (ep27)  MAX 0.5605 (ep18)  escape ep13  30 ep"
  echo "#   3 -> 32 -> 32 -> 1 1217 par ckpt 0.5538 (ep25)  MAX 0.5556 (ep20)  escape ep16  28 ep"
  echo "# DEPTH did not pay: -0.0049 max for 7.6x the params, and train loss was marginally HIGHER"
  echo "#   at every epoch -- the capacity was inert, not overfitting. Width is the other axis."
  echo "# PRIOR: expect no gain. If depth-2's 1056 extra params did nothing, width's 160 extra are"
  echo "#   unlikely to do more; the pooler is 161 par against 25.8M in E, and the PARAMETER-FREE"
  echo "#   pooling rule still beats every learned pooler here (0.5677 K=5 / 0.5756 K=10), which"
  echo "#   says the pooler is not the bottleneck at all. A clear win would falsify that reading."
  echo "# Width 64 draws DIFFERENT RNG than width 32, so a gap under ~0.01 is not separable from"
  echo "#   init luck -- the same caveat that governed the feature ablation."
  echo "# LB #1 YouTube = GraphMixer 0.5887."
  echo "# WATCH: escape epoch (13 at width 32, 16 at depth 2) and the plateau it fires from."
  echo "#   Every change tried so far that DELAYED the escape lost: dev, rad+dev, depth 2."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
