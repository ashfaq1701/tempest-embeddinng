#!/bin/bash
# WikiLink d=64 K=5 on feature/split-lr: manifold/net lr split, no popularity channel.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/wikilink_split_lr/d64_k5/run_1/WikiLink.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset WikiLink \
--d-emb 64 --k-train 5 --lr-manifold 1e-3 --lr-net 5e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=WikiLink (NON-bipartite) d_emb=64 k_train=5  lr_manifold=1e-3 lr_net=5e-3  seed=42"
  echo "# experiment=wikilink_split_lr cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u,P_v)), purely geometric -- the popularity channel is"
  echo "#       REMOVED on this branch, not merely disabled. 162 head params."
  echo "#       pooling = softmax(MLP([log1p(age) | raw pos | rad])), width 32, depth 1"
  echo "# lr groups: lr_manifold=1e-3 holds every geoopt.ManifoldParameter, i.e. E alone;"
  echo "#            lr_net=5e-3 holds geo_temp AND the four pooler tensors. Verified on this commit."
  echo "#"
  echo "# TWO VARIABLES vs master, so this is not a clean single-factor A/B:"
  echo "#   (1) the lr split itself, and (2) lr_net=5e-3 covering the temperature as well as the"
  echo "#   pooler. Read a gain as 'this configuration works', not as 'the split works'."
  echo "#"
  echo "# WIKILINK REFERENCES (d=64 K=5 seed 42, same [log1p(age),pos,rad] features unless noted):"
  echo "#   pooler @1e-2, geo_temp SLOW    : 0.6654 @ep12 -- FLOOR, killed ep14 at patience 2/5."
  echo "#     Best on record for this head. Beat LB #1 TGN by +0.036. THE NUMBER TO BEAT."
  echo "#   mnia-scale pooler, single lr   : 0.6313 @ep11 (converged, stop ep14), 162 par"
  echo "#   LB #1 TGN                      : 0.6294"
  echo "#   same features, single lr 1e-3  : 0.6282 @ep6 -- also a floor, killed while rising"
  echo "#   fixed pooler, EXP geo_temp     : 0.6108 @ep8"
  echo "#   same features, single lr 3e-3  : 0.6006 @ep3 -- CONVERGED and worse; see below"
  echo "#   our record, much larger head   : 0.7904 @ep44"
  echo "#"
  echo "# THE SPECIFIC RISK, and it is why lr_net is 5e-3 rather than 1e-2. On WikiLink geo_temp"
  echo "#   RISES THEN RETREATS as the geometry expands -- 15.4 / 16.7 / 13.6 / 11.8 over ep1-4 on"
  echo "#   the 0.6654 arm. An earlier arm that put the temperature on a fast group forced it to"
  echo "#   21.1 at ep1 and scored 0.5999 at ep4 against 0.6375 for the same features with only the"
  echo "#   pooler fast. With two groups by construction there is no way to slow geo_temp without"
  echo "#   also slowing the pooler, so 5e-3 is a hedge between those two measured points."
  echo "#   IF geo_temp CLIMBS PAST ~18 IN THE FIRST TWO EPOCHS, that is the known failure mode."
  echo "#"
  echo "# AND THE OTHER FAILURE MODE, measured 2026-09-01 (logs/wikilink_lr3e3): raising the MANIFOLD rate to 3e-3 drove"
  echo "#   |E|mean to 0.470/0.714/0.845/0.919/0.958 by ep5 -- saturating the Poincare boundary at"
  echo "#   1.0 -- after which TRAINING LOSS ROSE (0.3389 -> 0.3339 -> 0.3469) and the run peaked"
  echo "#   at 0.6006. lr_manifold stays at 1e-3 here for exactly that reason. Watch |E|mean: the"
  echo "#   0.6654 arm reached 0.888 only by ep14 and was still improving."
  echo "#"
  echo "# WATCH: (a) geo_temp in ep1-2, per above;  (b) |E|mean vs 0.242/0.351/0.432/0.503 on the"
  echo "#   0.6654 arm at ep1-4;  (c) whether it clears 0.6282, then 0.6313/0.6294, then 0.6654;"
  echo "#   (d) no escape -- WikiLink decelerates smoothly from ep1, so a jump would be news;"
  echo "#   (e) val/test drift."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~30 min/epoch (25 train + 5 eval). Prior runs: 7.9 h for 14 epochs. Budget 8-14 h."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
