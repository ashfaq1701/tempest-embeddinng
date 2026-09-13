#!/bin/bash
# WikiLink d=64 K=5 on feature/pooler-lr: NN pooler in its own param group at 1e-2.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/wikilink_pooler_lr/d64_k5/run_1/WikiLink.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset WikiLink \
--d-emb 64 --k-train 5 --lr 1e-3 --lr-pooler 1e-2 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=WikiLink (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 lr_pooler=1e-2  NO pop bias  seed=42"
  echo "# experiment=wikilink_pooler_lr cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H), geo_temp LINEAR init 1.0, 162 head params"
  echo "#       pooling = softmax(MLP([log1p(age) | raw pos | rad])), width 32, depth 1"
  echo "#       nothing standardised; no dataset-derived constant in the pooler"
  echo "# lr groups (f3f1e9b7): lr_pooler=1e-2 holds the four bag_weights tensors ALONE;"
  echo "#       lr=1e-3 holds E, geo_temp, and pop_bias when the channel is on."
  echo "#"
  echo "# THE ONE VARIABLE vs master: the pooler gets its own 10x lr. Same seed, d, K, patience,"
  echo "#   features, score and temperature parameterisation."
  echo "#"
  echo "# WIKILINK REFERENCES (d=64, K=5, seed 42, no pop bias):"
  echo "#   master features [log1p(age),pos,rad], single lr : 0.6282 @ep6 -- KILLED still rising,"
  echo "#     a FLOOR not a peak. Was ahead of the mnia arm at every epoch (+0.0046 at ep6)."
  echo "#   NN pooler [rec,pos,rad] mnia scale, single lr   : 0.6313 @ep11, stop ep14, 162 par"
  echo "#   fixed pooler, no pooling temp, linear geo_temp  : 0.5430 @ep7 (converged)"
  echo "#   fixed pooler, 1-param pooling temp             : 0.5828 @ep7"
  echo "#   fixed pooler, EXP geo_temp                     : 0.6108 @ep8"
  echo "#   encoded pooler + LOG temp                      : 0.7014 @ep6 -- also KILLED while"
  echo "#     rising, and on a different branch (log temp), so NOT comparable to this run."
  echo "#   LB #1 TGN = 0.6294.   Our record 0.7904 @ep44 (a much larger head)."
  echo "#"
  echo "# PRIOR EVIDENCE ON THIS EXACT SPLIT, and it is mixed:"
  echo "#   Patent, pooler-fast alone on the RAW-SCALAR pooler : 0.1644 -- did NOT work there."
  echo "#   Patent, temp+pooler BOTH fast                      : 0.2633, the record. Neither half"
  echo "#     alone worked (temp alone 0.0620 / 0.1681; pooler alone 0.1644) -- a real interaction."
  echo "#   So this run tests the half that failed on Patent, on a different dataset and on a"
  echo "#   different pooler (log1p features, whose input scale is already balanced)."
  echo "#"
  echo "# WHY IT MIGHT STILL PAY HERE: the mechanism the split addresses is a pooler whose"
  echo "#   first-layer weights must travel before its features are usable. On master's features"
  echo "#   that imbalance is already fixed -- log1p(age) and pos are both order 1 -- so a null"
  echo "#   result is entirely plausible and would be worth recording as such."
  echo "#"
  echo "# WATCH: (a) whether it clears 0.6282 (the same-features single-lr floor), then 0.6313,"
  echo "#   then LB #1 0.6294;  (b) geo_temp -- on WikiLink it RISES then RETREATS (13.9 -> 15.4"
  echo "#   -> 13.1 -> 11.3 -> 10.0 -> 9.0 on the single-lr arm) while |E|mean expands, the"
  echo "#   opposite of YouTube. A fast pooler should not change that; if it does, note it;"
  echo "#   (c) no escape -- WikiLink decelerates smoothly from ep1, so a jump would be news;"
  echo "#   (d) val/test drift."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~30 min/epoch (25 train + 5 eval). Prior WikiLink runs: 7.9 h for 14 epochs. Patience 5"
  echo "#   here, so budget 8-14 h -- this is the overnight run."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
