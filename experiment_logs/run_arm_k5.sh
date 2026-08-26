#!/bin/bash
#   usage: run_arm_k5.sh [now|after-wikilink]
# without_cand_pop-5 arm: d=64, K_train=5, lr=1e-3, NO popularity channel.
# All 7 TGB-Seq datasets EXCEPT WikiLink (running separately as the K=10 cell).
# Layout mirrors experiment_logs/complete_runs/without_cand_pop/:
#   <arm>/<Dataset>/run-1/run.log   plus  <arm>/DRIVER_run-1.log
#
# Bipartite flags authoritative from tgb_seq/datasets/preprocess.py::bipartite_dict.
# Ascending by edge count (cheap-first): a partial run yields whole cells.
# RESUMABLE: a cell whose log already holds "=== Final results ===" is skipped.

MODE=${1:-after-wikilink}
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-wt-masterbr
ARM=$WD/experiment_logs/complete_runs/without_cand_pop-5
DRIVER=$ARM/DRIVER_run-1.log
D=64; K=5; TAG=run-1

ORDER=(GoogleLocal YouTube Flickr Patent ML-20M Taobao Yelp)
declare -A BIP=( [GoogleLocal]=1 [ML-20M]=1 [Taobao]=1 [Yelp]=1 \
                 [YouTube]=0 [Flickr]=0 [Patent]=0 )

mkdir -p "$ARM"
log() { echo "[$(date '+%F %T')] $*" >> "$DRIVER"; }
wikilink_running() { ps -eo args --no-headers | grep 'dataset WikiLink' | grep -qv grep; }

cd "$WD" || exit 1
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
log "=== without_cand_pop-5/$TAG START  d=$D K_train=$K lr=1e-3  mode=$MODE  branch=$BRANCH commit=$SHA  order=${ORDER[*]}"

if [ "$MODE" = "after-wikilink" ]; then
  log "waiting for the WikiLink K=10 cell to finish before starting"
  while wikilink_running; do sleep 300; done
  log "WikiLink finished; starting"
else
  log "running CONCURRENTLY with the WikiLink K=10 cell -- both share one GPU and are compute-contended; wall-clock in these logs is NOT comparable to solo runs (MRR is unaffected, seeds fixed)"
fi

for DS in "${ORDER[@]}"; do
  FLAG=""; [ "${BIP[$DS]}" = "1" ] && FLAG="--is-bipartite"
  OUT=$ARM/$DS/$TAG; mkdir -p "$OUT"; LOG=$OUT/run.log
  if [ -f "$LOG" ] && grep -q '=== Final results ===' "$LOG"; then
    log "SKIP  $DS (already complete)"; continue
  fi
  log "START $DS  d=$D K_train=$K lr=1e-3  $FLAG"
  {
    echo "# complete run: dataset=$DS arm=without_cand_pop-5 tag=$TAG d_emb=$D k_train=$K lr=1e-3"
    echo "# branch=$BRANCH commit=$SHA"
    echo "# started=$(date '+%F %T')"
    echo "# mode=$MODE (see DRIVER_run-1.log for whether this shared the GPU with WikiLink)"
    echo "# cmd: $PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq \\"
    echo "#        --dataset $DS --d-emb $D --k-train $K --lr 1e-3 \\"
    echo "#        --num-epochs 50 --early-stop-patience 3 \\"
    echo "#        --use-gpu --use-gpu-tempest $FLAG"
    echo
  } > "$LOG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train_link_property_prediction.py \
    --data-suite tgb-seq --dataset "$DS" --d-emb $D --k-train $K --lr 1e-3 \
    --num-epochs 50 --early-stop-patience 3 \
    --use-gpu --use-gpu-tempest $FLAG >> "$LOG" 2>&1
  RC=$?
  RES=$(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')
  MAXTEST=$(grep -o 'test [0-9.]*' "$LOG" | awk '{if($2>m)m=$2}END{printf "max_test_seen: %.4f", m}')
  log "DONE  $DS rc=$RC  $RES $MAXTEST"
done
log "=== without_cand_pop-5/$TAG COMPLETE ==="
