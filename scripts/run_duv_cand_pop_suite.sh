#!/bin/bash
# All 8 TGB-Seq datasets, d=64 K=5 seed 5, sequential on the one A6000.
# head: score = mix([-d_H, log1p(cand_popularity)]), mix = Linear(2,1,bias=False), w init [1,0]
# loss: softmax cross-entropy (unchanged from master)
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
OUT=$WD/logs/duv_cand_pop/d64_k5/run_1
cd "$WD" || exit 1
mkdir -p "$OUT"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)

# CHEAPEST FIRST, so the four fast datasets give a read in under 4h instead of the
# suite order burying them behind Patent. Per-epoch cost and reference epoch count
# measured from experiment_logs/geometries/lorentz/*/3.log (commit 07dcc1e6):
#   YouTube      1.9 min/ep x 24 = 0.8h    cumulative  0.8h
#   GoogleLocal  1.2 min/ep x 44 = 0.9h                1.7h
#   ML-20M       4.7 min/ep x 12 = 0.9h                2.6h
#   Flickr       3.5 min/ep x 18 = 1.0h                3.6h   <- 4 datasets by here
#   Taobao      26.4 min/ep x  9 = 4.0h                7.6h
#   Patent      21.2 min/ep x 18 = 6.4h               14.0h
#   Yelp        32.2 min/ep x 19 = 10.2h              24.2h
#   WikiLink    43.9 min/ep x 20 = 14.6h              38.8h
# ~39h total, and those epoch counts are LOWER bounds: d0u*d0v ran Taobao to 20 epochs
# where the baseline quit at 9, and run length has been the recurring variable.
for SPEC in "YouTube:" "GoogleLocal:--is-bipartite" "ML-20M:--is-bipartite" "Flickr:" \
            "Taobao:--is-bipartite" "Patent:" "Yelp:--is-bipartite" "WikiLink:"; do
  DS="${SPEC%%:*}"; BIP="${SPEC##*:}"
  LOG="$OUT/$DS.log"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $DS \
--d-emb 64 --k-train 5 --num-walks-per-node 5 --lr 1e-3 --seed 5 --early-stop-patience 5 \
$BIP --use-gpu --use-gpu-tempest"
  {
    echo "# dataset=$DS  d_emb=64 k_train=5 lr=1e-3 seed=5 patience 5 cap 100"
    echo "# experiment=duv_cand_pop cell=d64_k5 tag=run_1  (chained; see DRIVER.log)"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
    echo "# head: score = mix([-d_H, log1p(cand_popularity)]), Linear(2,1,bias=False), w init [1,0]."
    echo "#       163 head params. LOSS: softmax CE, unchanged from master."
    echo "#"
    echo "# WHAT IT IS. The master baseline plus ONE extra scalar feature: how many edges the"
    echo "#   candidate had strictly before the query cutoff. w init [1,0] means the score STARTS"
    echo "#   as exactly the baseline -d_H (geo_temp pinned at 1) with the popularity channel"
    echo "#   contributing nothing, so w[1] has to earn its weight rather than being granted it."
    echo "#"
    echo "# log1p NOT log. get_candidate_popularity returns 0 for a node with no prior edge and"
    echo "#   cold candidates are common; log(0) = -inf would NaN the first batch. log1p maps"
    echo "#   0 -> 0, which is also the right inert value for a cold node."
    echo "#"
    echo "# THE RISK, STATED UP FRONT. w[1] is inert at init in VALUE but not in GRADIENT:"
    echo "#   measured d(CE)/dw = [1.97e-04, 7.65e-01] at init, so the popularity weight carries"
    echo "#   ~3900x the gradient of the distance weight and will move immediately. That is NOT"
    echo "#   the earned-range profile of d0_u*d0_v, whose gradient was genuinely ~0 (2.3e-07)."
    echo "#   Every head on this suite that CAPTURED the score in the first epoch or two finished"
    echo "#   badly -- d0v captured at ep1 with w[0] -> -30.6 and scored 0.4567 against the"
    echo "#   baseline 0.5626. WATCH w[1] AT EPOCH 1. If it jumps and w[0] collapses, this is"
    echo "#   that failure mode and the run can be called early."
    echo "#"
    echo "# PRIOR ON POPULARITY. TGB-Seq test negatives are UNIFORM over the destination pool,"
    echo "#   popularity-blind: measured corr(log10 train-popularity, log10 test-negative-freq)"
    echo "#   = +0.0137 on ML-20M, rare pool nodes averaging 1340.2 appearances against popular"
    echo "#   ones at 1340.6. A popularity channel therefore cannot help by exploiting the"
    echo "#   negative sampler -- it can only help if popularity genuinely predicts the POSITIVE."
    echo "#   A per-node popularity bias table was also part of the withdrawn historical-best"
    echo "#   configs that CLAUDE.md records as unstable. Treat a large early gain with suspicion."
    echo "#"
    echo "# SEED-5 BASELINE (commit 07dcc1e6), the number this must beat, val / test / stop-ep:"
    echo "#   GoogleLocal 0.6818 / 0.6535 / 44     YouTube  0.6467 / 0.5626 / 19"
    echo "#   Flickr      0.6581 / 0.6251 / 13     Patent   0.1262 / 0.2263 / 13"
    echo "#   ML-20M      0.2826 / 0.2429 / 7      Taobao   0.5536 / 0.5313 / 4"
    echo "#   Yelp        0.6633 / 0.6204 / 14     WikiLink 0.6682 / 0.6592 / 15"
    echo "#"
    echo "# WATCH: (a) w[1] at ep1-2, see THE RISK above;  (b) w[0] -- if it collapses toward 0"
    echo "#   the geometry has been abandoned for the popularity shortcut;  (c) PATENT especially"
    echo "#   -- its walks are structurally dead at length 2 (mean 2.20, only 0.3% reach 5), so"
    echo "#   the bag is the seed plus one neighbour and a cheap non-geometric feature has the"
    echo "#   most room to take over there;  (d) VAL/TEST DRIFT -- on Patent val is a POOR"
    echo "#   selector (baseline best val 0.1262 -> test 0.2263, while a recent arm reached a"
    echo "#   HIGHER val 0.1276 at a LOWER test 0.2023). Record peak test AND test@val-ckpt."
    echo "#"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  echo "[$(date '+%F %T')] START $DS" >> "$OUT/DRIVER.log"
  $CMD >> "$LOG" 2>&1
  RC=$?
  echo "# finished=$(date '+%F %T')  rc=$RC" >> "$LOG"
  echo "[$(date '+%F %T')] END   $DS rc=$RC" >> "$OUT/DRIVER.log"
done
echo "[$(date '+%F %T')] SUITE COMPLETE" >> "$OUT/DRIVER.log"
