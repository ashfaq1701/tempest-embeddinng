#!/bin/bash
# YouTube d=64 K=5, ENCODED NN pooler (TimeEncoding + positional embedding), SINGLE lr.
# No local edits -- HEAD fully describes this run (drivers are untracked by design, 50653d00).
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/encoded_single_lr/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE param group)  NO pop bias  seed=42"
  echo "# experiment=encoded_single_lr cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([TimeEncoding(age), pos_embedding, rad]))"
  echo "#       encoder dims: time_dim=16 pos_dim=4 hidden_dim=32 (defaults)"
  echo "# optimiser: ONE RiemannianAdam group at --lr. E, geo_temp and the pooler all step at 1e-3."
  echo "#"
  echo "# PURPOSE: first run of the encoded pooler, and the first with the lr split REMOVED."
  echo "#   TWO changes move at once against every prior YouTube number: the raw scalars rec/pos"
  echo "#   are replaced by learned encodings, AND the two-group optimiser is collapsed to one."
  echo "#   So this run cannot attribute an effect to either change on its own -- it establishes"
  echo "#   where the combination lands. If it disappoints, the split arm is the cheap next test."
  echo "#"
  echo "# WHAT THE SINGLE GROUP COST LAST TIME IT WAS MEASURED (raw-scalar pooler):"
  echo "#   Patent d=64 k=5 : 0.1630 single group, DIED at ep1 -- val peaked before the distance"
  echo "#                     scale had slewed. Two groups took it to 0.2633."
  echo "#   YouTube d=64 k=5: roughly neutral on quality, mostly speed. Two groups escaped at ep7"
  echo "#                     and peaked 0.5585; one group escaped at ep14 and peaked 0.5605."
  echo "#   YouTube is therefore the SAFE dataset for a single-group run; Patent is the one that"
  echo "#   collapsed. Do not generalise this run to Patent."
  echo "#"
  echo "# YOUTUBE REFERENCES (d=64 K=5 seed 42, raw-scalar pooler):"
  echo "#   1 group, depth 1, mnia scale : 0.5551 ckpt, 0.5605 max, escape ep14, stop ep27"
  echo "#   2 groups, depth 1, mnia      : 0.5564 ckpt, 0.5585 max, escape ep7,  stop ep12"
  echo "#   2 groups, depth 1, med-age   : 0.5511 ckpt, 0.5521 max, escape ep6,  stop ep15"
  echo "#   2 groups, depth 5, med-age   : 0.5573 ckpt, 0.5596 max,              stop ep14"
  echo "#   BEST EVER learned pooler here: 0.5645 (depth 3, mnia, 2 groups; arm was cut short)"
  echo "#   parameter-free pooling rule  : 0.5677 (K=5), 0.5756 (K=10)   <- the bar to clear"
  echo "#   LB #1 GraphMixer             : 0.5887"
  echo "#   The learned pooler has NEVER beaten the parameter-free rule on YouTube."
  echo "#"
  echo "# WATCH: (a) ESCAPE EPOCH -- with one group expect ~ep14 (the slow-pooler signature), not"
  echo "#   ep7; an early escape would mean the encoders removed the need for the fast lr;"
  echo "#   (b) geo_temp and |E|mean -- two-group runs drove geo_temp to ~85 with |E|mean ~0.15,"
  echo "#   one-group runs sat at ~35 with |E|mean ~0.25; which pattern appears says which"
  echo "#   parameter is supplying the logit scale;  (c) whether it clears 0.5645, then 0.5677."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~100 s/epoch; a slow-escape run went 27 epochs, so budget ~45-60 min."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
