#!/bin/bash
# K_train sweep: YouTube FIRST (all K), THEN GoogleLocal (all K).
# d=64, NO pop bias. STRICTLY SEQUENTIAL -- one cell at a time on one GPU.
# Supersedes the per-dataset drivers (*.superseded); do not run those as well.
#
# Bipartite flags authoritative from tgb_seq/datasets/preprocess.py::bipartite_dict:
#   GoogleLocal True  -> --is-bipartite
#   YouTube     False -> flag absent
#
# Ascending K (cheap-first) within each dataset: a partial night still yields
# the low-K end of a curve rather than one unfinished expensive cell.
#
# RESUMABLE: a cell whose log already holds "=== Final results ===" is skipped.
# A cell killed mid-run has no such block, so it restarts from scratch.

PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-wt-masterbr
ROOT=$WD/experiment_logs
DRIVER=$ROOT/DRIVER_k_sweep_all.log

D=64
KS=(1 3 5 10 20 50 100)
ORDER=(YouTube GoogleLocal)
declare -A BIP=( [GoogleLocal]=1 [YouTube]=0 )
declare -A OUTDIR=( [GoogleLocal]=k-sweep-googlelocal [YouTube]=k-sweep-youtube )

log() { echo "[$(date '+%F %T')] $*" >> "$DRIVER"; }
running() { ps -eo args --no-headers | grep 'train_link_property_prediction.py' | grep -qv grep; }

cd "$WD" || exit 1
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)

log "=== K-sweep ALL START  order=${ORDER[*]}  d=$D  K=${KS[*]}  branch=$BRANCH commit=$SHA  (no pop bias, sequential)"
while running; do log "waiting for an in-flight run to finish"; sleep 60; done

for DS in "${ORDER[@]}"; do
  OUT=$ROOT/${OUTDIR[$DS]}
  mkdir -p "$OUT"
  FLAG=""; [ "${BIP[$DS]}" = "1" ] && FLAG="--is-bipartite"
  log "---- dataset $DS START (bipartite=${BIP[$DS]}) ----"
  for K in "${KS[@]}"; do
    LOG=$OUT/k-$K.log
    if [ -f "$LOG" ] && grep -q '=== Final results ===' "$LOG"; then
      log "SKIP  $DS K=$K (already complete)"; continue
    fi
    log "START $DS K=$K"
    {
      echo "# K-sweep cell: dataset=$DS d_emb=$D k_train=$K patience=3 NO pop-bias"
      echo "# branch=$BRANCH commit=$SHA"
      echo "# started=$(date '+%F %T')"
      echo "# sequential sweep: GoogleLocal (all K) then YouTube (all K); nothing else on the GPU"
      echo "# cmd: $PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq \\"
      echo "#        --dataset $DS --d-emb $D --k-train $K --num-epochs 50 \\"
      echo "#        --early-stop-patience 3 --use-gpu --use-gpu-tempest $FLAG"
      echo
    } > "$LOG"
    PYTHONUNBUFFERED=1 $PY -u scripts/train_link_property_prediction.py \
      --data-suite tgb-seq --dataset "$DS" --d-emb $D --k-train "$K" \
      --num-epochs 50 --early-stop-patience 3 \
      --use-gpu --use-gpu-tempest $FLAG >> "$LOG" 2>&1
    RC=$?
    RES=$(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')
    MAXTEST=$(grep -o 'test [0-9.]*' "$LOG" | awk '{if($2>m)m=$2}END{printf "max_test_seen: %.4f", m}')
    log "DONE  $DS K=$K rc=$RC  $RES $MAXTEST"
  done
  log "---- dataset $DS curve COMPLETE ----"
done
log "=== K-sweep ALL COMPLETE ==="
