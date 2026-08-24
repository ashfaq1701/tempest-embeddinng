#!/bin/bash
# Full-run arm over all 8 TGB-Seq datasets at the chosen default d.
#   usage: run_arm.sh with_cand_pop|without_cand_pop [run_tag]
# Sequential (one GPU), cheap-first so a partial arm yields whole datasets.
# RESUMABLE: a cell whose log already holds "=== Final results ===" is skipped.

ARM=${1:?arm required: with_cand_pop | without_cand_pop}
TAG=${2:-run_1}
D=64

PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-wt-mainline
OUT=$WD/experiment_logs/complete_runs/$ARM
DRIVER=$OUT/DRIVER_$TAG.log

case "$ARM" in
  with_cand_pop)    POP="--use-cand-pop" ;;
  without_cand_pop) POP="" ;;
  *) echo "bad arm: $ARM" >&2; exit 1 ;;
esac

# Authoritative: tgb_seq/datasets/preprocess.py::bipartite_dict
declare -A BIP=( [ML-20M]=1 [Taobao]=1 [Yelp]=1 [GoogleLocal]=1 \
                 [Flickr]=0 [YouTube]=0 [Patent]=0 [WikiLink]=0 )
ORDER=(GoogleLocal YouTube ML-20M Patent Flickr Taobao Yelp WikiLink)

log() { echo "[$(date '+%F %T')] $*" >> "$DRIVER"; }
running() { ps -eo args --no-headers | grep 'train_link_property_prediction.py' | grep -qv grep; }

cd "$WD" || exit 1
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
log "=== $ARM/$TAG START  branch=$BRANCH commit=$SHA  d=$D  pop='$POP'  order=${ORDER[*]}"
while running; do log "waiting for an in-flight run"; sleep 120; done

for DS in "${ORDER[@]}"; do
  FLAG=""; [ "${BIP[$DS]}" = "1" ] && FLAG="--is-bipartite"
  LOG=$OUT/$DS/$TAG.log
  if [ -f "$LOG" ] && grep -q '=== Final results ===' "$LOG"; then
    log "SKIP  $DS (already complete)"; continue
  fi
  log "START $DS  d=$D $POP $FLAG"
  {
    echo "# complete run: dataset=$DS arm=$ARM tag=$TAG d_emb=$D"
    echo "# branch=$BRANCH commit=$SHA"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq \\"
    echo "#        --dataset $DS --d-emb $D --num-epochs 50 --early-stop-patience 3 \\"
    echo "#        --use-gpu --use-gpu-tempest $POP $FLAG"
    echo
  } > "$LOG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train_link_property_prediction.py \
    --data-suite tgb-seq --dataset "$DS" --d-emb $D \
    --num-epochs 50 --early-stop-patience 3 \
    --use-gpu --use-gpu-tempest $POP $FLAG >> "$LOG" 2>&1
  RC=$?
  log "DONE  $DS rc=$RC  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')"
done
log "=== $ARM/$TAG COMPLETE ==="
