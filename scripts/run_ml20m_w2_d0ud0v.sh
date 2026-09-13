#!/bin/bash
# ML-20M d=64 K=5 seed 5: two-term head  score = w[0]*(-d_H(P_u,P_v)) + w[1]*(r_u*r_v).
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/w2_scorer_d0ud0v/d64_k5/run_1/ML-20M.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset ML-20M \
--d-emb 64 --k-train 5 --num-walks-per-node 5 --lr 1e-3 --seed 5 --early-stop-patience 5 \
--is-bipartite --use-gpu --use-gpu-tempest"
{
  echo "# dataset=ML-20M (14.5M edges, BIPARTITE) d_emb=64 k_train=5 lr=1e-3 seed=5 patience 5 cap 100"
  echo "# experiment=w2_scorer_d0ud0v cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = w[0]*(-d_H(P_u,P_v)) + w[1]*(r_u*r_v), r_x = dist0(P_x),  w init [1.0, 1.0]."
  echo "# vs the d0v sibling (feature/improvement-variant-d0v, 43dfe512): the ONLY difference is"
  echo "#   the second term, dist0(P_v) -> dist0(P_u)*dist0(P_v). That sibling peaked val 0.2941"
  echo "#   @ep8 / max test 0.2550 @ep4, and decayed as w[1]/w[0] fell 0.978 -> 0.769."
  echo "# vs the baseline experiment_logs/geometries/lorentz/ml-20m/3.log (commit 07dcc1e6,"
  echo "#   best_test 0.2429 @ep7, stop ep12): the score function. Seed, d, K, lr, patience,"
  echo "#   pooler, negatives and optimiser are identical. At w=[1,0] this IS the baseline."
  echo "# SIGN WATCH: the radius PRODUCT enters UNNEGATED at +1.0, i.e. rewarding far-out candidates."
  echo "#   9adb9e07 is EXACTLY this head and converged to w=[30.2, 8.9] -- BOTH terms penalising."
  echo "#   So w[1] crossing zero is the model finding that configuration, not a failure."
  echo "# WATCH: w[0] (took geo_temp's role -- baseline crawled 17.7 -> 18.1 over ep11-12),"
  echo "#   r_mean/r_max (baseline 0.388/6.04 @ep11), and whether the ep7 val peak moves."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
$CMD >> "$LOG" 2>&1
echo "# finished=$(date '+%F %T')" >> "$LOG"
