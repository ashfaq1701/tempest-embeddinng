#!/bin/bash
# YouTube d=64 K=5, encoded pooler, single lr, floored ladder, LOG-parameterised geo_temp.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/encoded_logtemp/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE param group)  NO pop bias  seed=42"
  echo "# experiment=encoded_logtemp cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([TimeEncoding(age) | pos_embedding | rad])), 762 head params"
  echo "#       geo_temp = exp(geo_temp_raw), raw init 0.0 -> geo_temp starts at 1.0 as before"
  echo "#       encoder dims: time_dim=16 pos_dim=4 hidden_dim=32 (defaults)"
  echo "#"
  echo "# THE ONE VARIABLE: geo_temp is LOG-parameterised -- geo_temp = exp(geo_temp_raw)."
  echo "#   Same init (1.0), same param count (762), same everything else as logs/encoded_tsquantum:"
  echo "#   c6d84d99 is 7c7c112a plus this one edit. Seed, d, K, lr, patience, encoder dims, and"
  echo "#   the ts_quantum ladder floor are all identical."
  echo "#"
  echo "# WHY: Adam steps a parameter by ~lr regardless of gradient magnitude, so a LINEAR geo_temp"
  echo "#   slews ADDITIVELY and the steps needed scale with the target. Under exp() the step is"
  echo "#   MULTIPLICATIVE -- a constant 0.100% per step at lr=1e-3, at any current value:"
  echo "#     target  44 (YouTube optimum): linear  43,000 steps | log  3,784 steps"
  echo "#     target 170 (Patent optimum) : linear 169,000 steps | log  5,136 steps"
  echo "#   At 2,731 steps/epoch that is ~15.7 epochs vs ~1.4 to reach YouTube's optimum."
  echo "#"
  echo "# THE PREDICTION, and it is falsifiable: the matched linear run crawled geo_temp 4.3 (ep1)"
  echo "#   -> 15.2 (ep6) -> 29.1 (ep18) -> 44.4 (ep42), and its escape fired at ep19-20 -- roughly"
  echo "#   when the temperature finally arrived. If the escape IS the temperature arriving, this"
  echo "#   run should reach geo_temp ~40 within 2 epochs and escape FAR earlier, near ep2-4."
  echo "#   If it escapes at ep19-20 anyway, the escape is NOT about the temperature and the"
  echo "#   explore-then-escape shape has some other cause worth finding."
  echo "#"
  echo "# PRIOR EVIDENCE (WikiLink d=64 K=5, fixed pooler, one lr group):"
  echo "#   linear geo_temp 0.5430 @ep7   |   EXP geo_temp 0.6108 @ep8   (+0.068)"
  echo "#"
  echo "# BASELINES (identical config except where noted):"
  echo "#   floored ladder, linear temp (7c7c112a): ckpt 0.5456  max 0.5457  escape ep19-20  50 ep"
  echo "#   aliased ladder, linear temp (b86d2080): ckpt 0.5487  max 0.5487  escape ep20     stop 42"
  echo "#"
  echo "# OTHER YOUTUBE REFERENCES (d=64 K=5 seed 42):"
  echo "#   raw-scalar pooler, 1 lr group : 0.5551 ckpt, 0.5605 max, escape ep14, stop ep27"
  echo "#   raw-scalar pooler, 2 lr groups: 0.5564 ckpt, 0.5585 max, escape ep7,  stop ep12"
  echo "#   BEST EVER learned pooler here : 0.5645 (depth 3, mnia, 2 groups; arm was cut short)"
  echo "#   parameter-free pooling rule   : 0.5677 (K=5), 0.5756 (K=10)   <- the bar to clear"
  echo "#   LB #1 GraphMixer              : 0.5887"
  echo "#   The learned pooler has NEVER beaten the parameter-free rule on YouTube."
  echo "#"
  echo "# WATCH: (a) geo_temp per epoch -- expect ~40 by ep2, vs ep30 in the linear run;"
  echo "#   (b) escape epoch vs ep19-20; (c) plateau vs 0.5457, then 0.5605, then 0.5677;"
  echo "#   (d) whether geo_temp OVERSHOOTS -- multiplicative steps cut both ways, and a runaway"
  echo "#   temperature would show as |E|mean staying small while geo_temp climbs past ~50."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~100 s/epoch; the baseline ran 47 epochs in ~80 min."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
