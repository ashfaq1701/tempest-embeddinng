#!/bin/bash
# YouTube d=64 K=5: radius-product score, NO learned temperature.
#   s(u,v) = -rad(P_u) * rad(P_v) * d_H(P_u, P_v)
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/radius_product/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (single group)  seed=42"
  echo "# experiment=radius_product cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = -rad(P_u) * rad(P_v) * d_H(P_u,P_v)   <- NO learned scalar anywhere"
  echo "#       161 head params: the pooler MLP alone (geo_temp deleted)"
  echo "#       pooling = softmax(MLP([log1p(age) | raw pos | rad])), width 32, unchanged"
  echo "#       radii in the SCORE are not detached (the pooler's rad feature still is)"
  echo "#"
  echo "# ONE VARIABLE vs master (04ab5ab9): geo_temp * (-d) becomes -rad_u * rad_v * d."
  echo "#   Everything else -- pooler, features, walks, optimiser, lr -- is identical."
  echo "#"
  echo "# WHAT THE CHANGE DOES. geo_temp was ONE GLOBAL scalar. The radius product is a"
  echo "#   per-PAIR adaptive temperature: pairs sitting far from the origin get their"
  echo "#   distance amplified, pairs near the origin get it damped. It also gives r_v a"
  echo "#   second role -- a high-radius CANDIDATE is penalised against every query, which"
  echo "#   is an inverse popularity term read off the geometry instead of learned per node."
  echo "#   (The learned per-node popularity table was deleted in 04ab5ab9.)"
  echo "#"
  echo "# THE PREDICTION THIS RUN TESTS, and it is a scale argument:"
  echo "#   rad = 2*artanh(|x|), and E inits at irange 1e-3, so rad ~1e-3 and the product"
  echo "#   ~1e-6. MEASURED on a synthetic forward at init: scores span -1.1e-8 to -2.0e-9."
  echo "#   The logits are FLAT and the softmax is uniform, so epoch 1 starts from no signal."
  echo "#   geo_temp started at 1.0 and climbed to 35-80 on YouTube -- the old head opened"
  echo "#   about three orders of magnitude sharper."
  echo "#   To match YouTube's measured optimum of ~35-40 this head needs rad ~5.9 PER SIDE,"
  echo "#   i.e. |x| ~ 0.995 -- hard against the boundary. Prior YouTube runs converged at"
  echo "#   |E|mean ~0.27 (rad 0.55, product 0.31) which is ~100x too flat."
  echo "#   SO: either E is driven to the boundary, or the head never gets sharp enough."
  echo "#   |E|mean and |E|max ARE the experiment here -- read them before the MRR."
  echo "#"
  echo "# COMPETING GRADIENT, worth watching for: shrinking r_u flattens a whole query row"
  echo "#   (r_u multiplies every candidate), while the CE wants the positive raised above"
  echo "#   the negatives. Growing radii sharpens but also inflates the positive's own"
  echo "#   penalty. It is not obvious a priori which way that resolves."
  echo "#"
  echo "# YOUTUBE REFERENCES (d=64 K=5 seed 42, no pop bias, from logs/*/YouTube.log):"
  echo "#   logage_zscore                                : 0.5793"
  echo "#   rawpos_std                                   : 0.5787"
  echo "#   [log1p(age),pos,rad] = THIS POOLER, geo_temp : 0.5625  <- THE BASELINE"
  echo "#   encoded pooler + log geo_temp                : 0.5609"
  echo "#   parameter-free pooling rule, K=10            : 0.5756"
  echo "#   LB #1 GraphMixer                             : 0.5887  (unbeaten by any no-pop head)"
  echo "#"
  echo "# WATCH: (a) record BOTH test@val-checkpoint AND max test -- val/test drift on this"
  echo "#   pipeline has walked reported numbers down by 0.005 and flipped verdicts;"
  echo "#   (b) a flat epoch 1 is EXPECTED here and is not by itself a failure -- the"
  echo "#   comparable arms take 13-18 epochs to escape ~0.36 and then jump ~+0.15;"
  echo "#   (c) |E|max -> 1.0 means E is on the boundary: check for NaN/inf, not just MRR;"
  echo "#   (d) no geo_temp field in the epoch line is expected -- the probe reads it via"
  echo "#   hasattr and degrades to nothing."
  echo "# ~100 s/epoch (70 train + 28 eval). 50 epochs is ~85 min; comparable arms ran 17-47."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
