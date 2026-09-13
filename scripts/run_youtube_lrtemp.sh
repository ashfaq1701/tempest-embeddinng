#!/bin/bash
# YouTube d=64 K=5, encoded pooler, floored ladder, LINEAR geo_temp on its OWN lr group.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/encoded_lrtemp/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --lr-temp 1e-2 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 lr_temp=1e-2  NO pop bias  seed=42"
  echo "# experiment=encoded_lrtemp cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H), geo_temp LINEAR init 1.0, 762 head params"
  echo "#       pooling = softmax(MLP([TimeEncoding | pos_emb | rad])), width 32, depth 1"
  echo "#       TimeEncoding ladder floored at 2.5 * ts_quantum"
  echo "# lr groups (verified on this commit): lr_temp=1e-2 holds geo_temp ALONE;"
  echo "#       lr=1e-3 holds E, pop_bias (when on), and all four pooler tensors."
  echo "#"
  echo "# THE ONE VARIABLE vs logs/encoded_single_lr (0.5487) and the floored-ladder control"
  echo "#   (0.5457): geo_temp gets its own 10x lr. Seed, d, K, patience, encoder dims and the"
  echo "#   ladder floor are identical."
  echo "#"
  echo "# THIS IS THE SECOND OF TWO FIXES FOR THE SAME PROBLEM. Adam steps a parameter by ~lr"
  echo "#   regardless of gradient size, so a LINEAR geo_temp slews additively and the steps"
  echo "#   needed scale with its target. Two ways out:"
  echo "#     (a) log-parameterise it  -> feature/encoded-nn-pooler-log-temp, MEASURED 0.5609"
  echo "#     (b) give it its own lr   -> THIS RUN"
  echo "#   They should NOT be combined: multiplicative steps at a 10x rate would overshoot."
  echo "#"
  echo "# MEASURED CAUTION -- this shape has failed before. In the Patent A/B on the raw-scalar"
  echo "#   pooler, temperature-alone-on-a-fast-group STALLED at epoch 3 with an identical"
  echo "#   geo_temp trajectory and was abandoned; widening the fast group to include the pooler"
  echo "#   is what took Patent 0.1630 -> 0.2633. Different pooler, different dataset, so it is"
  echo "#   worth re-measuring here -- but a null or negative result would be a REPRODUCTION of"
  echo "#   that earlier finding, not a surprise."
  echo "#"
  echo "# BASELINES, all YouTube d=64 K=5 seed 42, identical except where named:"
  echo "#   encoded, single lr, floored ladder  : 0.5457  stop 50 (cap)   <- THE CONTROL"
  echo "#   encoded, single lr, aliased ladder  : 0.5487  stop ep42"
  echo "#   encoded, LOG temp, floored ladder   : 0.5609  stop ep25  escape ep11-12  <- fix (a)"
  echo "#   encoded, two-term score w.[-d,r_u*r_v]: 0.5324 @ep31 (killed, still rising)"
  echo "#   fixed-rule pooling, 2-param head    : 0.5752  stop ep17   <- best in repo"
  echo "#   LB #1 GraphMixer                    : 0.5887"
  echo "#"
  echo "# WATCH: (a) geo_temp per epoch -- the control crawled 4.3/15.2/29.1/44.4 at ep1/6/18/42;"
  echo "#   at 10x it should arrive in ~2 epochs. If it arrives fast and the run STILL stalls,"
  echo "#   that reproduces the Patent finding that a fast temperature alone is not enough;"
  echo "#   (b) escape epoch vs ep19-20 (control) and ep11-12 (log temp);"
  echo "#   (c) whether geo_temp OVERSHOOTS -- log temp went to 79 and settled at 73, so the"
  echo "#   ~44 the linear runs reached was where they ran out of epochs, not an optimum;"
  echo "#   (d) plateau vs 0.5457, then 0.5609;  (e) val/test drift."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~100 s/epoch; recent YouTube runs went 25-50 epochs, so budget 45-90 min."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
