#!/bin/bash
# YouTube d=64 K=5: NN-pooler DEPTH sweep at constant width 32, with the pooler on the FAST lr.
# Sequential (one GPU). Depths 1, 2, 3, 5.  Override: DEPTHS="1 2" ./scripts/sweep_pooler_depth.sh
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
ROOT=$WD/logs/pooler_depth_sweep/d64_k5
DEPTHS=${DEPTHS:-"1 2 3 5"}
cd "$WD" || exit 1
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)

{
  echo "# DRIVER pooler_depth_sweep d64_k5  depths='$DEPTHS'"
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
    echo "# experiment=pooler_depth_sweep cell=d64_k5 tag=L$L"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
    echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0"
    echo "#       pooling = softmax(MLP([rec, pos, rad])), width 32, DEPTH $L  ($POOL pooler params)"
    echo "# lr groups: lr_embedding=1e-3 holds E (+pop_bias when on); lr_network=1e-2 holds"
    echo "#            geo_temp AND every pooler tensor. Verified on this commit."
    echo "#"
    echo "# PURPOSE: does pooler DEPTH help now that the pooler can actually train?"
    echo "#   CLAUDE.md records three YouTube capacity probes -- width 64, depth 2, the dev"
    echo "#   feature -- all failing, and concludes the YouTube pooler is NOT capacity-limited."
    echo "#   Every one of those ran with the pooler at the EMBEDDING lr (1e-3, single group)."
    echo "#   Since 987d061d the pooler rides lr_network=1e-2. Capacity that could not be"
    echo "#   trained is not capacity that was shown not to help -- so the conclusion is"
    echo "#   reopened, not contradicted."
    echo "#"
    echo "# YOUTUBE REFERENCES (d=64 K=5 seed 42, all at the OLD slow pooler lr 1e-3):"
    echo "#   depth 1, feats rec+pos+rad : 0.5551 test@ckpt, 0.5605 max, ep18 peak, stop ep27"
    echo "#   depth 2 (uncommitted edit) : 0.5538 test@ckpt, stop ep25   <- the probe being redone"
    echo "#   width 64 depth 1           : see logs/pooler_wide64/"
    echo "#   parameter-free rule        : 0.5677 (K=5), 0.5756 (K=10)   <- the bar to clear"
    echo "#   LB #1 GraphMixer           : 0.5887"
    echo "#   The learned pooler has NEVER beaten the parameter-free rule on YouTube."
    echo "#"
    echo "# READING THIS SWEEP: the L1 arm is the control and is bit-identical in construction"
    echo "#   to the committed default (verified), so L1-vs-0.5551 isolates the lr change alone"
    echo "#   and L2..L5-vs-L1 isolates depth at that lr. Both contrasts need L1 to be read."
    echo "#   Deeper arms draw different RNG, so a gap under ~0.01 between depths is not"
    echo "#   separable from init luck -- the same caveat as the feature ablation."
    echo "#"
    echo "# WATCH: (a) ESCAPE TIMING, not warmup height -- YouTube arms grind to ~0.36 then jump"
    echo "#   ~+0.15 over three epochs; the feature ablation showed warmup ordering INVERTS, so"
    echo "#   do not call an arm before its escape fires (~ep13-18);  (b) val/test drift -- the"
    echo "#   depth-1 rad arm lost 0.0054 between its true peak and its val-selected checkpoint;"
    echo "#   (c) whether a deeper arm escapes EARLIER, which is how extra capacity would show."
    echo "# Record BOTH test@val-checkpoint AND max test observed."
    echo "# ~100 s/epoch; arms run 25-30 epochs, so ~50 min each."
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
