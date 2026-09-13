#!/bin/bash
# Three-geometry comparison at matched config, popularity channel ON.
#
# The geometries live on separate branches that differ ONLY in the head's geometry, so this
# driver checks each one out in turn and runs the same command. It is deliberately sequential:
# one GPU, and the working tree can only be on one branch at a time.
#
#   usage: run_geometry_popbias.sh <DATASET> [EXTRA_ARGS...]
#
# Bipartite flags are authoritative from tgb_seq/datasets/preprocess.py::bipartite_dict.
set -u
DATASET="${1:?usage: $0 <DATASET> [EXTRA_ARGS...]}"; shift
EXTRA="$*"
WD=/its/home/ms2420/tempest-embeddinng
PY=$WD/venv/bin/python
cd "$WD" || exit 1

case "$DATASET" in
  GoogleLocal|ML-20M|Taobao|Yelp) BIP="--is-bipartite" ;;
  *)                              BIP="" ;;
esac

# geometry label | branch
GEOMETRIES=(
  "sphere|feature/hypersphere"
  "euclidean|feature/euclidean"
)

ROOT=$WD/logs/geometry_popbias/d64_k5_popbias
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
START_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# A dirty tree cannot be checked out across branches without losing or carrying changes.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "ABORT: working tree has uncommitted tracked changes; commit or stash first." | tee -a "$DRIVER"
  exit 1
fi

{
  echo "# GEOMETRY COMPARISON  dataset=$DATASET  cell=d64_k5_popbias"
  echo "# d_emb=64 k_train=5 lr=1e-3 seed=42 patience 5 cap 100 --use-pop-bias"
  echo "# extra args: ${EXTRA:-(none)}"
  echo "# The three heads differ ONLY in geometry; every other moving part is identical."
  echo "#   ball      master              s = geo_temp * (-d_H(P_u,P_v)) + pop_bias   [reference]"
  echo "#   sphere    feature/hypersphere s = geo_temp * cos(P_u,P_v)    + pop_bias"
  echo "#   euclidean feature/euclidean   s = geo_temp * (-||P_u-P_v||)  + pop_bias"
  echo "# SEQUENTIAL: one GPU, and each arm needs its own branch checked out."
  echo "# started=$(date '+%F %T')  from branch $START_BRANCH"
  echo
} > "$DRIVER"

for entry in "${GEOMETRIES[@]}"; do
  IFS='|' read -r GEOM BRANCH <<< "$entry"

  if ! git checkout "$BRANCH" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] FAIL $GEOM: could not check out $BRANCH" >> "$DRIVER"
    continue
  fi
  SHA=$(git rev-parse --short HEAD)
  DIRTY=$(git status --porcelain --untracked-files=no | head -1)

  LOG=$ROOT/$GEOM/run_1/$DATASET.log
  mkdir -p "$(dirname "$LOG")"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $DATASET \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 100 --early-stop-patience 5 --use-pop-bias \
$BIP --use-gpu --use-gpu-tempest $EXTRA"
  {
    echo "# dataset=$DATASET ($([ -n "$BIP" ] && echo BIPARTITE || echo NON-bipartite))"
    echo "# geometry=$GEOM  branch=$BRANCH  commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
    echo "# d_emb=64 k_train=5 lr=1e-3 seed=42  patience 5  cap 100  --use-pop-bias"
    echo "# experiment=geometry_popbias cell=d64_k5_popbias tag=run_1"
    echo "# reference (ball head, master, scalar geo_temp + pop_bias): test MRR 0.6024"
    echo "# part of the comparison driven by scripts/run_geometry_popbias.sh; see ../../DRIVER.log"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"

  echo "[$(date '+%F %T')] START $GEOM ($BRANCH @ $SHA)" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%F %T')] DONE  $GEOM rc=$RC  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$DRIVER"
done

git checkout "$START_BRANCH" >/dev/null 2>&1
echo "[$(date '+%F %T')] COMPLETE (tree restored to $START_BRANCH)" >> "$DRIVER"
