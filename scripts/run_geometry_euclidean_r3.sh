#!/bin/bash
# EUCLIDEAN arm: all 8 TGB-Seq datasets at d64/K5/lr=1e-3, popularity channel ON.
# Runs on branch feature/euclidean, whose head is s = geo_temp * (-||P_u - P_v||) + pop_bias.
# Every other training file is byte-identical to master, so this is a geometry-only control.
#
# This machine produces replicate 3. Two other agents on other machines produce replicates 1
# and 2 into the same experiment_logs/other_geometries/euclidean/d64_k5_lr1e-3/<dataset>/ directories.
#
#   usage: run_archive_d64k5_popbias.sh [REPLICATE] [EXTRA_ARGS...]     (REPLICATE default 3)
#
# Runs write to logs/ ONLY. A finished run is COPIED into experiment_logs/ afterwards, never
# written there directly -- experiment_logs/ is the curated git-tracked archive (CLAUDE.md).
# A failed run (rc != 0) is left in logs/ and NOT copied, so the archive never gains a
# truncated or crashed log.
#
# Bipartite flags are authoritative from tgb_seq/datasets/preprocess.py::bipartite_dict.
set -u
REP="${1:-3}"; [ $# -gt 0 ] && shift
EXTRA="$*"
WD=/its/home/ms2420/tempest-embeddinng
PY=$WD/venv/bin/python
cd "$WD" || exit 1

# name|edges|bipartite-flag   ASCENDING by edge count, so results land soonest-first.
DATASETS=(
  "GoogleLocal|1.91M|--is-bipartite"
  "YouTube|3.29M|"
  "Flickr|7.22M|"
  "Patent|10.8M|"
  "ML-20M|14.5M|--is-bipartite"
  "Taobao|18.85M|--is-bipartite"
  "Yelp|19.8M|--is-bipartite"
  "WikiLink|34.2M|"
)

ROOT=$WD/logs/geometry_euclidean/d64_k5_lr1e-3
ARCHIVE=$WD/experiment_logs/other_geometries/euclidean/d64_k5_lr1e-3
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)

{
  echo "# ARCHIVE SWEEP  replicate=$REP  cell=d64_k5_lr1e-3  branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# geometry=euclidean  branch=feature/euclidean (head differs from master ONLY in model.py)"
  echo "# NOTE: driver scripts/run_geometry_euclidean_r3.sh is UNTRACKED; SHA covers the trained code only"
  echo "# d_emb=64 k_train=5 lr=1e-3 seed=42 patience 5  cap 100 --use-pop-bias"
  echo "# extra args: ${EXTRA:-(none)}"
  echo "# All 8 datasets, SEQUENTIAL (one GPU), ascending by edge count."
  echo "# Bipartite flags from tgb_seq/datasets/preprocess.py::bipartite_dict -- passing one"
  echo "# wrongly changes the negative-candidate pool, so these are not cosmetic:"
  echo "#   bipartite: GoogleLocal, ML-20M, Taobao, Yelp"
  echo "#   non-bip:   YouTube, Flickr, Patent, WikiLink"
  echo "# Runs write here; a finished run (rc=0) is COPIED to experiment_logs/ as ${REP}.log."
  echo "# started=$(date '+%F %T')"
  echo
} > "$DRIVER"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r NAME EDGES BIP <<< "$entry"
  LOWER=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
  LOG=$ROOT/$NAME/run_$REP/$NAME.log
  mkdir -p "$(dirname "$LOG")"

  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $NAME \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 100 --early-stop-patience 5 --use-pop-bias \
$BIP --use-gpu --use-gpu-tempest $EXTRA"
  {
    echo "# dataset=$NAME ($EDGES edges, $([ -n "$BIP" ] && echo BIPARTITE || echo NON-bipartite))"
    echo "# d_emb=64 k_train=5 lr=1e-3 seed=42  patience 5  cap 100  --use-pop-bias"
    echo "# experiment=geometry_euclidean cell=d64_k5_lr1e-3 tag=run_$REP  (archived as ${REP}.log)"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
    echo "# geometry=euclidean  branch=feature/euclidean (differs from master ONLY in model.py)"
    echo "# replicate $REP of 3; replicates 1 and 2 are produced by other agents on other machines."
    echo "# part of the sweep driven by scripts/run_geometry_euclidean_r3.sh; see ../../DRIVER.log"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"

  echo "[$(date '+%F %T')] START $NAME ($EDGES)" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  RC=$?
  SUMMARY=$(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')

  if [ $RC -eq 0 ] && grep -q 'best_test_mrr' "$LOG"; then
    mkdir -p "$ARCHIVE/$LOWER"
    cp "$LOG" "$ARCHIVE/$LOWER/${REP}.log"
    echo "[$(date '+%F %T')] DONE  $NAME rc=$RC  $SUMMARY -> archived $LOWER/${REP}.log" >> "$DRIVER"
  else
    echo "[$(date '+%F %T')] FAIL  $NAME rc=$RC  NOT archived (log kept in logs/)" >> "$DRIVER"
  fi
done
echo "[$(date '+%F %T')] SWEEP COMPLETE" >> "$DRIVER"
