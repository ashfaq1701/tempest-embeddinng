#!/bin/bash
# Sweep one (d_emb, K_train) config across a list of TGB-Seq datasets, one at a time.
#
#   usage: scripts/run_sweep.sh <experiment> [tag] [d_emb] [k_train]
#   e.g.   scripts/run_sweep.sh no_pooler_temp run_1 64 5
#
# Per CLAUDE.md ("Where run logs live") output lands in the git-untracked logs/ tree:
#   logs/<experiment>/d<D>_k<K>/<tag>/<Dataset>.log   plus DRIVER.log alongside.
# Copy selected finished logs into experiment_logs/ afterwards; never write there directly.
# Override the root with OUT_ROOT=... and the dataset list with ORDER_LIST, e.g.
#   ORDER_LIST="GoogleLocal YouTube" scripts/run_sweep.sh no_pooler_temp run_1 64 5
# EXTRA_FLAGS is appended verbatim to every run, e.g. EXTRA_FLAGS=--use-pop-bias for the
# popularity arm (omitted = no pop bias, the flag is off by default).
# RESUMABLE: a dataset whose log already holds "=== Final results ===" is skipped.

EXP=${1:?experiment name required, e.g. no_pooler_temp}
TAG=${2:-run_1}
D=${3:-64}
K=${4:-5}
# Early-stop patience. Default 8, not 3: Patent's best val landed on epoch 1 under patience 3,
# so the counter never reset and the run died at epoch 4 during the warmup grind. A longer fuse
# can only find an equal-or-better val checkpoint -- it costs epochs, never quality.
PATIENCE=${PATIENCE:-8}

WD=/mnt/nfs2/inf/ms2420/tempest-embeddinng
source $WD/scripts/env.sh >/dev/null 2>&1   # module load Python/3.11.3 + CMake/3.26.3; sets $PY
OUT=${OUT_ROOT:-$WD/logs}/$EXP/d${D}_k${K}/$TAG
mkdir -p "$OUT"
DRIVER=$OUT/DRIVER.log

# Authoritative: tgb_seq/datasets/preprocess.py::bipartite_dict
declare -A BIP=( [ML-20M]=1 [Taobao]=1 [Yelp]=1 [GoogleLocal]=1 \
                 [Flickr]=0 [YouTube]=0 [Patent]=0 [WikiLink]=0 )
# Ascending by edge count; Yelp and WikiLink are run elsewhere.
ORDER=(${ORDER_LIST:-GoogleLocal YouTube Flickr Patent ML-20M Taobao})
EXTRA=${EXTRA_FLAGS:-}

log() { echo "[$(date '+%F %T')] $*" >> "$DRIVER"; }

cd "$WD" || exit 1
BRANCH=$(git rev-parse --abbrev-ref HEAD)
# A bare SHA that does not describe the code that ran is worse than no SHA, so name any
# modified tracked files in every log header (untracked files are ignored -- they are not code).
dirty() { git status --porcelain --untracked-files=no | awk '{print $2}' | tr '\n' ' '; }

log "=== $EXP/d${D}_k${K}/$TAG START  branch=$BRANCH commit=$(git rev-parse --short HEAD)  d=$D  K_train=$K  extra='$EXTRA'  order=${ORDER[*]}"
while pgrep -f 'train_link_property_prediction.py' >/dev/null; do
  log "waiting for an in-flight run"; sleep 120
done

for DS in "${ORDER[@]}"; do
  FLAG=""; [ "${BIP[$DS]}" = "1" ] && FLAG="--is-bipartite"
  LOG=$OUT/$DS.log
  if [ -f "$LOG" ] && grep -q '=== Final results ===' "$LOG"; then
    log "SKIP  $DS (already complete)"; continue
  fi
  # Re-read HEAD per dataset: the working tree can move under a long sweep, and the
  # header must name the code this cell actually ran, not the code the driver started on.
  SHA=$(git rev-parse --short HEAD); DIRTY=$(dirty)
  log "START $DS  d=$D K_train=$K lr=1e-3 patience=$PATIENCE commit=$SHA $EXTRA $FLAG"
  {
    echo "# sweep cell: dataset=$DS experiment=$EXP cell=d${D}_k${K} tag=$TAG"
    echo "#   d_emb=$D k_train=$K lr=1e-3 patience=$PATIENCE seed=42 extra='$EXTRA' NO pop-bias unless extra says so"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# UNCOMMITTED tracked changes on top of $SHA: $DIRTY"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq \\"
    echo "#        --dataset $DS --d-emb $D --k-train $K --lr 1e-3 \\"
    echo "#        --num-epochs 50 --early-stop-patience $PATIENCE \\"
    echo "#        --use-gpu --use-gpu-tempest $EXTRA $FLAG"
    echo
  } > "$LOG"
  # Unbuffered three ways so the log is live: PYTHONUNBUFFERED + python -u (Python-level)
  # and stdbuf (libc-level, since stdout is a file here and would otherwise block-buffer).
  PYTHONUNBUFFERED=1 stdbuf -oL -eL $PY -u scripts/train_link_property_prediction.py \
    --data-suite tgb-seq --dataset "$DS" --d-emb $D --k-train $K --lr 1e-3 \
    --num-epochs 50 --early-stop-patience $PATIENCE \
    --use-gpu --use-gpu-tempest $EXTRA $FLAG >> "$LOG" 2>&1
  RC=$?
  log "DONE  $DS rc=$RC  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')"
done
log "=== $EXP/d${D}_k${K}/$TAG COMPLETE ==="
