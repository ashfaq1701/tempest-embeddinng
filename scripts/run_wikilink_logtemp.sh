#!/bin/bash
# WikiLink d=64 K=5, encoded pooler, single lr, floored ladder, LOG-parameterised geo_temp.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/wikilink_logtemp/d64_k5/run_1/WikiLink.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset WikiLink \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=WikiLink (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE group)  NO pop bias  seed=42"
  echo "# experiment=wikilink_logtemp cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H), geo_temp = exp(geo_temp_raw) LOG-parameterised, init 1.0"
  echo "#       pooling = softmax(MLP([TimeEncoding | pos_emb | rad])), width 32, depth 1, 762 head params"
  echo "#       TimeEncoding ladder floored at 2.5 * ts_quantum -- on WikiLink this fires:"
  echo "#       2/8 frequencies were below Nyquist before the floor (86,400 s daily grid)."
  echo "#"
  echo "# PURPOSE: carry the log-temperature change to the dataset where it should pay MORE."
  echo "#   On YouTube it was worth +0.0152 (0.5457 -> 0.5609), the largest single effect measured."
  echo "#   The mechanism is that Adam slews a LINEAR geo_temp additively, so steps-to-optimum"
  echo "#   scale with the optimum; under exp() they scale with its log. WikiLink's optimum is"
  echo "#   larger than YouTube's, so the linear parameterisation should be hurting more here."
  echo "#   Prior direct evidence on WikiLink (fixed pooler, one lr group, d=64 K=5, seed 42):"
  echo "#     linear geo_temp 0.5430 @ep7   |   EXP geo_temp 0.6108 @ep8   (+0.068)"
  echo "#   That +0.068 is 4.5x what log temp bought on YouTube. If it transfers to the ENCODED"
  echo "#   pooler, this run should land well clear of the references below."
  echo "#"
  echo "# WIKILINK REFERENCES (d=64, K=5, seed 42, no pop bias):"
  echo "#   NN pooler [rec,pos,rad], 1 lr group, linear temp : 0.6313 @ep11, stop ep14"
  echo "#     -- that arm BEAT LB #1 TGN (0.6294) and the fixed-pooler baseline (0.5430) by +0.088"
  echo "#   fixed pooler, no pooling temp, linear geo_temp   : 0.5430 @ep7  (converged, stop ep10)"
  echo "#   fixed pooler, 1-param pooling temp, linear temp  : 0.5828 @ep7"
  echo "#   fixed pooler, no pooling temp, EXP geo_temp      : 0.6108 @ep8"
  echo "#   LB #1 TGN = 0.6294.   Our record 0.7904 @ep44 (a much larger head)."
  echo "#"
  echo "# NOTE the head differs from the 0.6313 run: that used raw [rec,pos,rad] scalars with mnia"
  echo "#   scaling and a LINEAR temperature. This run uses the TimeEncoding + positional embedding"
  echo "#   pooler (762 params vs 162) AND the log temperature AND the Nyquist ladder floor. So a"
  echo "#   difference against 0.6313 is not attributable to any single one of those."
  echo "#"
  echo "# WATCH: (a) geo_temp -- on YouTube it overshot to 79 then settled to 73, and the linear"
  echo "#   runs' ~44 turned out to be where they RAN OUT OF EPOCHS, not an optimum. Expect a much"
  echo "#   larger value here and record where it settles;  (b) escape epoch -- WikiLink has never"
  echo "#   shown YouTube's explore-then-escape shape, it decelerates smoothly from ep1, so a jump"
  echo "#   here would itself be news;  (c) whether it clears 0.6313, then LB #1 0.6294;"
  echo "#   (d) val/test drift -- the 0.6313 run drifted 0.0000, the YouTube log-temp run 0.0000."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~30 min/epoch (25 train + 5 eval). The 0.6313 run took 7.9 h for 14 epochs; patience is"
  echo "#   5 here vs 3 there, so budget 8-14 h."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
