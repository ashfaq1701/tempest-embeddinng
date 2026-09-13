#!/bin/bash
# WikiLink d=64 K=5, settled 3-feat NN pooler, under the NEW two-group optimiser.
# No local edits -- HEAD fully describes this run (drivers are untracked by design, 50653d00).
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/wikilink_2group/d64_k5/run_1/WikiLink.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset WikiLink \
--d-emb 64 --k-train 5 --lr-embedding 1e-3 --lr-network 1e-2 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=WikiLink (NON-bipartite) d_emb=64 k_train=5  lr_e=1e-3 lr_network=1e-2  NO pop bias  seed=42"
  echo "# experiment=wikilink_2group cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA alone does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([rec, pos, rad])), 3 -> 32 -> 1"
  echo "#       162 head params (161 pooler + 1 geo_temp)"
  echo "#"
  echo "# PURPOSE: re-run wikilink_peak under the TWO-GROUP optimiser (303dd012). Same head, same"
  echo "#   data, same seed, same d/K -- the ONLY changes are the optimiser split and the fuse:"
  echo "#     lr_network 1e-2 : geo_temp + all four pooler tensors (was 1e-3, shared with E)"
  echo "#     lr_e       1e-3 : the embedding table alone (unchanged)"
  echo "#     patience   5    : was 3 (303dd012; clears late winners that arrive after 4 misses)"
  echo "#"
  echo "# THE RUN THIS REPLACES -- logs/wikilink_peak/d64_k5/run_1, commit 19eb15f3, ONE group at 1e-3:"
  echo "#   ep1 0.4353 -> ep11 val 0.6400 test 0.6313 (best) -> stop ep14.  Drift 0.0000."
  echo "#   geo_temp: 12.50 -> 15.45 -> 13.23 -> 11.42 -> ... -> 6.58 @ep11 (still falling at stop)"
  echo "#   |E|mean climbed 0.248 -> 0.849 across those 11 epochs."
  echo "#   Beat the fixed-pooler baseline 0.5430 by +0.088 and LB #1 TGN 0.6294 by +0.0019."
  echo "#"
  echo "# WHAT THE SPLIT DID ELSEWHERE: on Patent it was decisive -- 0.1630 -> 0.2633 -- but the"
  echo "#   measured decomposition matters: temperature alone did NOT fix Patent (stalled at ep3,"
  echo "#   identical geo_temp trajectory); the POOLER riding the fast group is what moved it."
  echo "#   Patent was a rescue case: best val at ep1, run dead at ep4 in the warmup grind."
  echo "#   WikiLink is NOT that case -- it already converges smoothly over 11 epochs with val"
  echo "#   rising monotonically and no early-stop pathology to rescue. So there is no reason to"
  echo "#   expect a Patent-sized gain here, and a REGRESSION is a live outcome: a 10x pooler lr"
  echo "#   can just as well overshoot a pooling rule that was already finding its way."
  echo "#"
  echo "# WATCH: (a) does the peak beat 0.6313, and at which epoch -- a faster pooler should reach"
  echo "#   its plateau in FEWER than 11 epochs even if the height is unchanged; (b) geo_temp's"
  echo "#   trajectory -- at 1e-2 it should settle ~10x sooner, so compare where it lands, not just"
  echo "#   how fast; (c) val/test divergence -- the patience-3 run drifted 0.0000, but patience 5"
  echo "#   gives two extra epochs of val flicker to walk the checkpoint away from the true peak."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~30 min/epoch (25 min train + 5 min eval): the 14-epoch predecessor took 7.9 h."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
