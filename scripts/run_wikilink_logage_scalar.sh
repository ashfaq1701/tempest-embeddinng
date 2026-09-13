#!/bin/bash
# WikiLink d=64 K=5: pooler on [log1p(age) | raw pos | rad], no standardisation, single lr.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/wikilink_logage_scalar/d64_k5/run_1/WikiLink.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset WikiLink \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=WikiLink (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE group)  NO pop bias  seed=42"
  echo "# experiment=wikilink_logage_scalar cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H), geo_temp LINEAR init 1.0, 162 head params"
  echo "#       pooling = softmax(MLP([log1p(age) | raw pos 1..5 | rad])), width 32, depth 1"
  echo "#       NOTHING is standardised. log1p is a fixed function of the age alone, so a given"
  echo "#       age maps to the same value in every batch and on every dataset -- no batch-"
  echo "#       dependent or dataset-derived quantity enters the pooler at all."
  echo "#"
  echo "# WHY THIS CONFIG: per-batch standardisation was dropped because it destroys ABSOLUTE"
  echo "#   anchoring -- it rescales each batch to mean 0 / std 1, so a candidate with a fresh"
  echo "#   history and one with a decade-old history become indistinguishable. On a dataset whose"
  echo "#   source bags are almost all cold, the candidate bag's absolute staleness is the only"
  echo "#   temporal signal in the score. That evidence is from another agent's PATENT runs; those"
  echo "#   logs are NOT on this machine and are unverified here."
  echo "#"
  echo "# WHAT IT COSTS ON YOUTUBE, measured here as a clean 2x2 (max test):"
  echo "#                        standardised   not standardised"
  echo "#     one-hot position      0.5793          0.5617"
  echo "#     raw scalar position   0.5787          0.5625"
  echo "#   -> standardisation is worth +0.0176 / +0.0168 (rows agree to 0.0008);"
  echo "#      position encoding is worth -0.0006 / +0.0008, i.e. nothing, at 128 fewer params."
  echo "#   This run therefore carries a KNOWN ~0.017 handicap on a YouTube-like dataset. The"
  echo "#   question is whether WikiLink behaves like YouTube (richly occupied both sides, so"
  echo "#   relative ordering suffices) or like Patent (cold sources, so anchoring is the signal)."
  echo "#"
  echo "# WIKILINK REFERENCES (d=64, K=5, seed 42, no pop bias):"
  echo "#   NN pooler [rec,pos,rad] mnia scale, linear temp : 0.6313 @ep11, stop ep14, 162 par"
  echo "#     -- that arm BEAT LB #1 TGN (0.6294) and the fixed-pooler baseline (0.5430) by +0.088"
  echo "#   fixed pooler, no pooling temp, linear geo_temp  : 0.5430 @ep7 (converged, stop ep10)"
  echo "#   fixed pooler, 1-param pooling temp, linear temp : 0.5828 @ep7"
  echo "#   fixed pooler, no pooling temp, EXP geo_temp     : 0.6108 @ep8"
  echo "#   encoded pooler + LOG temp                       : 0.7014 @ep6 -- KILLED while STILL"
  echo "#     RISING, so that is a FLOOR not a peak; it had already beaten LB #1 by +0.072"
  echo "#   LB #1 TGN = 0.6294.   Our record 0.7904 @ep44 (a much larger head)."
  echo "#"
  echo "# NOTE the temperature here is LINEAR on a single lr, not the log parameterisation that"
  echo "#   produced 0.7014. So a shortfall against that number is not attributable to the pooler"
  echo "#   alone. The like-for-like reference at 162 params is the 0.6313 run, whose only"
  echo "#   difference is the age feature: age/mnia there, log1p(age) here."
  echo "#"
  echo "# WATCH: (a) whether it clears 0.6313, then LB #1 0.6294;  (b) escape shape -- WikiLink"
  echo "#   has NEVER shown YouTube's explore-then-escape, it decelerates smoothly from ep1, so a"
  echo "#   jump here would itself be news;  (c) geo_temp -- linear single-group arms crawl toward"
  echo "#   ~44 on YouTube; WikiLink's optimum is unmeasured on this head;  (d) val/test drift."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~30 min/epoch (25 train + 5 eval). Prior WikiLink runs took 7.9 h for 14 epochs;"
  echo "#   patience is 5 here, so budget 8-14 h."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
