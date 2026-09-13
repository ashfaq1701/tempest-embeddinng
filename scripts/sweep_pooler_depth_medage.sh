#!/bin/bash
# YouTube d=64 K=5: NN-pooler DEPTH sweep at constant width 32, pooler on the fast lr,
# with the CALIBRATED recency scale (median context walk-token age) instead of mnia.
# Sequential (one GPU). Depths 1, 2, 3, 5.  Override: DEPTHS="3 5" ./scripts/sweep_pooler_depth_medage.sh
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
ROOT=$WD/logs/pooler_depth_sweep_medage/d64_k5
DEPTHS=${DEPTHS:-"1 2 3 5"}
cd "$WD" || exit 1
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)

{
  echo "# DRIVER pooler_depth_sweep_medage d64_k5  depths='$DEPTHS'"
  echo "# branch=$BRANCH commit=$SHA  started=$(date '+%F %T')"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code"
} > "$DRIVER"

for L in $DEPTHS; do
  LOG=$ROOT/L$L/YouTube.log
  mkdir -p "$(dirname "$LOG")"
  case $L in
    1) POOL=161  ;; 2) POOL=1217 ;; 3) POOL=2273 ;; 5) POOL=4385 ;; *) POOL="?" ;;
  esac
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr-embedding 1e-3 --lr-network 1e-2 --pooler-hidden 32 \
--pooler-hidden-layers $L --num-epochs 50 --early-stop-patience 5 --use-gpu --use-gpu-tempest"
  {
    echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr_embedding=1e-3 lr_network=1e-2  NO pop bias  seed=42"
    echo "# experiment=pooler_depth_sweep_medage cell=d64_k5 tag=L$L"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
    echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0"
    echo "#       pooling = softmax(MLP([rec, pos, rad])), width 32, DEPTH $L  ($POOL pooler params)"
    echo "#       rec = -(age / recency_scale), recency_scale CALIBRATED from the ingested graph"
    echo "#       (median context walk-token age) -- NOT mnia. The scale is printed at calibration."
    echo "# lr groups: lr_embedding=1e-3 holds E (+pop_bias when on); lr_network=1e-2 holds"
    echo "#            geo_temp AND every pooler tensor."
    echo "#"
    echo "# PURPOSE: re-run the depth sweep with the calibrated recency scale underneath."
    echo "#"
    echo "# EXPECT THE SCALE CHANGE TO BE NEAR-NEUTRAL HERE. 0c55f548's own measurement puts"
    echo "#   YouTube's within-bag std of rec at 1.48 against the ~1.4 that pos contributes --"
    echo "#   i.e. ALREADY well scaled under mnia. The datasets the recalibration was built for"
    echo "#   are GoogleLocal (0.13, 11x squashed), Yelp (0.50, 3x) and Flickr (5.30, 4x inflated)."
    echo "#   So a large move here would be a surprise worth investigating, not a confirmation."
    echo "#   CAVEAT from 0c55f548: the scale now depends on the walk config, so a sweep over"
    echo "#   num_walks_per_node / max_walk_len / the biases also moves it. mnia did not."
    echo "#"
    echo "# PRIOR SWEEP, same depths, same lrs, MNIA scale (commit 23de173b, logs/pooler_depth_sweep/):"
    echo "#   L1  161 params : max test 0.5585  ckpt 0.5564  escape ep7   stop ep12"
    echo "#   L2 1217 params : max test 0.5543  ckpt 0.5543               stop ep25"
    echo "#   L3 2273 params : max test 0.5645 @ep15 -- PARTIAL, killed while still improving,"
    echo "#                    so 0.5645 is a FLOOR on that arm, not its peak"
    echo "#   L5 4385 params : never ran"
    echo "#   Note L3 > L1 > L2 -- not monotone in depth, and the L3-L1 gap (0.0060) sits inside"
    echo "#   the ~0.01 band where different-RNG arms are not separable from init luck."
    echo "#"
    echo "# OTHER YOUTUBE REFERENCES (d=64 K=5 seed 42):"
    echo "#   depth 1, SLOW pooler lr 1e-3 : 0.5551 ckpt, 0.5605 max, escape ep14, stop ep27"
    echo "#   parameter-free pooling rule  : 0.5677 (K=5), 0.5756 (K=10)   <- the bar to clear"
    echo "#   LB #1 GraphMixer             : 0.5887"
    echo "#   The learned pooler has NEVER beaten the parameter-free rule on YouTube."
    echo "#"
    echo "# WATCH: (a) the printed recency_scale vs mnia -- record both, it is the whole change;"
    echo "#   (b) ESCAPE TIMING, not warmup height -- YouTube grinds then jumps ~+0.15, and the"
    echo "#   feature ablation showed warmup ordering INVERTS, so do not call an arm early;"
    echo "#   (c) whether L3 reproduces its >L1 result -- one partial arm is not a finding;"
    echo "#   (d) val/test drift between the true peak and the val-selected checkpoint."
    echo "# Record BOTH test@val-checkpoint AND max test observed."
    echo "# ~100 s/epoch; arms ran 17-30 epochs last time, so ~30-60 min each."
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  echo "[$(date '+%F %T')] START depth=$L ($POOL pooler params) -> $LOG" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%F %T')] DONE rc=$RC  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
  echo "[$(date '+%F %T')] DONE depth=$L rc=$RC  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$DRIVER"
done
echo "[$(date '+%F %T')] SWEEP COMPLETE" >> "$DRIVER"
