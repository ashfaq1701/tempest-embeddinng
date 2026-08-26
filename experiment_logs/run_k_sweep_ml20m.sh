#!/bin/bash
# K_train sweep on ML-20M, d=64, NO pop bias. Third dataset after YouTube and
# GoogleLocal, same K grid so the three curves are directly comparable.
#
# QUEUED: waits for the run_k_sweep_all.sh DRIVER to exit (not merely for the
# GPU to look idle) -- otherwise it would race the parent sweep in the gap
# between its cells. Strictly one cell on the GPU at any time.
#
# ML-20M IS bipartite (tgb_seq/datasets/preprocess.py::bipartite_dict).
#
# Why this dataset is the informative third point: its negative-candidate pool
# is 9,646 unique destinations, 28-42x SMALLER than YouTube (402,422) or
# GoogleLocal (267,336), while its train split is 14.0M edges, 5-8x LARGER.
# YouTube's K curve rises monotonically to K=100; GoogleLocal's peaks at K=3-5
# then falls. If pool size drives the sign, ML-20M should look unlike both.
# Note K=100 here samples ~1% of the entire pool per query.
#
# Ascending K (cheap-first). RESUMABLE: a cell whose log already holds
# "=== Final results ===" is skipped.

PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-wt-masterbr
OUT=$WD/experiment_logs/k-sweep-ml20m
DRIVER=$OUT/DRIVER.log

DS=ML-20M
D=64
KS=(1 3 5 10 20 50 100)

mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" >> "$DRIVER"; }
parent_running() { ps -eo args --no-headers | grep 'run_k_sweep_all.sh' | grep -qv grep; }
train_running()  { ps -eo args --no-headers | grep 'train_link_property_prediction.py' | grep -qv grep; }

cd "$WD" || exit 1
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)

log "=== ML-20M K-sweep QUEUED behind run_k_sweep_all.sh  d=$D  K=${KS[*]}  branch=$BRANCH commit=$SHA"
while parent_running; do sleep 120; done
log "parent sweep finished"
while train_running; do sleep 60; done
log "GPU free; starting"

for K in "${KS[@]}"; do
  LOG=$OUT/k-$K.log
  if [ -f "$LOG" ] && grep -q '=== Final results ===' "$LOG"; then
    log "SKIP  K=$K (already complete)"; continue
  fi
  log "START K=$K"
  {
    echo "# K-sweep cell: dataset=$DS d_emb=$D k_train=$K patience=3 NO pop-bias"
    echo "# branch=$BRANCH commit=$SHA"
    echo "# started=$(date '+%F %T')"
    echo "# queued behind the YouTube+GoogleLocal K-sweep; nothing else on the GPU"
    echo "# cmd: $PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq \\"
    echo "#        --dataset $DS --d-emb $D --k-train $K --num-epochs 50 \\"
    echo "#        --early-stop-patience 3 --use-gpu --use-gpu-tempest --is-bipartite"
    echo
  } > "$LOG"
  PYTHONUNBUFFERED=1 $PY -u scripts/train_link_property_prediction.py \
    --data-suite tgb-seq --dataset "$DS" --d-emb $D --k-train "$K" \
    --num-epochs 50 --early-stop-patience 3 \
    --use-gpu --use-gpu-tempest --is-bipartite >> "$LOG" 2>&1
  RC=$?
  RES=$(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')
  MAXTEST=$(grep -o 'test [0-9.]*' "$LOG" | awk '{if($2>m)m=$2}END{printf "max_test_seen: %.4f", m}')
  log "DONE  K=$K rc=$RC  $RES $MAXTEST"
done
log "=== ML-20M K-sweep COMPLETE ==="
