#!/bin/bash
# YouTube d=64, LOG-parameterised geo_temp, floored ladder: K x depth, three arms.
# Sequential (one GPU). Then the caller switches to the clean log-temp branch for WikiLink.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
ROOT=$WD/logs/logtemp_k_depth
cd "$WD" || exit 1
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
{
  echo "# DRIVER logtemp_k_depth  arms: k10_d1, k5_d3, k10_d3"
  echo "# branch=$BRANCH commit=$SHA  started=$(date '+%F %T')"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code"
} > "$DRIVER"

run_arm () {
  K=$1; L=$2; TAG=$3; PARAMS=$4
  LOG=$ROOT/$TAG/YouTube.log
  mkdir -p "$(dirname "$LOG")"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train $K --pooler-hidden-layers $L --lr 1e-3 --num-epochs 50 \
--early-stop-patience 5 --use-gpu --use-gpu-tempest"
  {
    echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=$K  lr=1e-3 (SINGLE group)  NO pop bias  seed=42"
    echo "# experiment=logtemp_k_depth cell=d64 tag=$TAG   pooler depth $L ($PARAMS head params)"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
    echo "# head: score = geo_temp * (-d_H), geo_temp = exp(geo_temp_raw) LOG-parameterised, init 1.0"
    echo "#       pooling = softmax(MLP([TimeEncoding | pos_emb | rad])), width 32, depth $L"
    echo "#       TimeEncoding ladder floored at 2.5 * ts_quantum (no sub-Nyquist frequencies)"
    echo "#"
    echo "# PURPOSE: the two untried levers on top of the log-temp result, and their combination."
    echo "#   K: the fixed-pooler head gained +0.0079 from K=5->10 on YouTube (0.5677 -> 0.5756)."
    echo "#      NO YouTube run with a LEARNED pooler has ever used K=10 -- every one is K=5."
    echo "#   depth 3: the best learned-pooler number ever seen here (0.5645) came from depth 3,"
    echo "#      and that arm was killed while still improving, so 0.5645 is a floor not a peak."
    echo "#   The two are orthogonal (training signal vs pooler capacity), hence the 2x2 corner."
    echo "#"
    echo "# THE NUMBER TO BEAT IS NOT OUR OWN. logs/simple_head_4_notemp/YouTube = 0.5752 with a"
    echo "#   TWO-parameter head and PARAMETER-FREE pooling: softmax(-age/mnia - (pos-1)), no"
    echo "#   learned pooler at all. Our 762-param learned pooler at 0.5609 is -0.0143 behind it."
    echo "#   So the target order is: 0.5609 (log-temp K=5 d1) -> 0.5677 (fixed rule K=5)"
    echo "#   -> 0.5752 (best in repo) -> 0.5887 (LB #1 GraphMixer)."
    echo "#"
    echo "# BASELINES, all d=64 seed 42, identical except where named:"
    echo "#   log temp,    K=5 depth 1 : 0.5609 ckpt/max, stop ep25, escape ep11-12   <- THE CONTROL"
    echo "#   linear temp, K=5 depth 1 : 0.5457, stop 50 (cap)"
    echo "#   linear temp, aliased ladder: 0.5487, stop ep42"
    echo "#   fixed-rule pooling, K=5  : 0.5677  |  K=10: 0.5756"
    echo "#   2-param head, fixed pool : 0.5752 @ep17"
    echo "#"
    echo "# WATCH: (a) whether K=10's +0.0079 transfers to a learned pooler or is absorbed;"
    echo "#   (b) whether depth 3 reproduces its >depth 1 result now that it runs to convergence;"
    echo "#   (c) geo_temp -- the control overshot to 79 then settled to 73, so ~44 from the linear"
    echo "#   runs was never the optimum; (d) escape epoch vs ep11-12; (e) val/test drift."
    echo "# Record BOTH test@val-checkpoint AND max test observed."
    echo "# K=5 is ~100 s/epoch; K=10 is roughly 1.5-2x that."
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  echo "[$(date '+%F %T')] START $TAG (K=$K depth=$L, $PARAMS head params)" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%F %T')] DONE rc=$RC  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
  echo "[$(date '+%F %T')] DONE $TAG rc=$RC  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$DRIVER"
}

run_arm 10 1 k10_d1 762
run_arm  5 3 k5_d3  2874
run_arm 10 3 k10_d3 2874
echo "[$(date '+%F %T')] SWEEP COMPLETE" >> "$DRIVER"
