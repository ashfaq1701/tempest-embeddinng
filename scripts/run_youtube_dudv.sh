#!/bin/bash
# YouTube d=64 K=5, encoded pooler, single lr, floored ladder, TWO-TERM score head.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/encoded_dudv/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE group)  NO pop bias  seed=42"
  echo "# experiment=encoded_dudv cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = w[0]*(-d_H(P_u,P_v)) + w[1]*(r_u * r_v),  w init [1.0, 1.0], 763 head params"
  echo "#       r_x = d_H(0, P_x), the hyperbolic radius of x's pooled centroid"
  echo "#       pooling = softmax(MLP([TimeEncoding | pos_emb | rad])), width 32, depth 1"
  echo "#       geo_temp is GONE -- w[0] is now the distance scale. TimeEncoding ladder floored"
  echo "#       at 2.5 * ts_quantum. Temperature is LINEAR here (not the log branch)."
  echo "#"
  echo "# THE ONE VARIABLE vs logs/encoded_single_lr (0.5487): the score function. Everything else"
  echo "#   -- seed, d, K, lr, patience, encoder dims, ladder floor, pooler -- is identical."
  echo "#"
  echo "# WHY: logs/simple_head_4_notemp/YouTube = 0.5752, the BEST YouTube number in the repo,"
  echo "#   using this two-term score with a TWO-parameter head and PARAMETER-FREE pooling."
  echo "#   Our 762-param learned pooler tops out at 0.5609 (log temp) / 0.5487 (linear temp),"
  echo "#   i.e. -0.0143 behind a rule with no parameters. This run asks whether the two-term"
  echo "#   score and the learned pooler are ADDITIVE, or whether the score function was doing"
  echo "#   all the work in that reference and the learned pooler is a net cost either way."
  echo "#"
  echo "# SIGN WATCH -- the single most informative number in this run is w[1]."
  echo "#   The reference negates the whole bracket and converged to w=[30.2, 8.9], i.e. BOTH"
  echo "#   terms penalising. Here the radius product enters UNNEGATED and starts at +1.0, i.e."
  echo "#   REWARDING far-out centroid pairs. So to reproduce the reference's behaviour w[1] must"
  echo "#   cross zero and go negative. If it does, that is the model finding the reference"
  echo "#   configuration, not a failure. If it stays positive and the run does well, the two"
  echo "#   sign conventions are genuinely different solutions and both work."
  echo "#   Also watch w[0]: the reference drove it to ~30 additively over 17 epochs; our linear"
  echo "#   geo_temp runs reached ~44 by ep42 and the log-temp run overshot to 79."
  echo "#"
  echo "# BASELINES, all YouTube d=64 K=5 seed 42, identical except where named:"
  echo "#   encoded pooler, linear temp, floored ladder : 0.5457  stop 50 (cap)   <- THE CONTROL"
  echo "#   encoded pooler, linear temp, aliased ladder : 0.5487  stop ep42"
  echo "#   encoded pooler, LOG temp,    floored ladder : 0.5609  stop ep25   <- our best"
  echo "#   fixed-rule pooling, 2-param head           : 0.5752  stop ep17   <- the number to beat"
  echo "#   fixed-rule pooling, K=5 / K=10             : 0.5677 / 0.5756"
  echo "#   LB #1 GraphMixer                           : 0.5887"
  echo "#"
  echo "# WATCH: (a) w[1] sign and magnitude, per the above; (b) w[0] trajectory vs ~30/~44/~79;"
  echo "#   (c) escape epoch -- the linear-temp control escaped ep19-20, the log-temp run ep11-12,"
  echo "#   and the 2-param reference showed NO escape, it converged smoothly by ep17;"
  echo "#   (d) whether it clears 0.5609, then 0.5752;  (e) val/test drift, 0.0000 on recent runs."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~100 s/epoch; recent YouTube runs went 25-50 epochs, so budget 45-90 min."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
