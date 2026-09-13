#!/bin/bash
# LORENTZ geometry sweep, replicate 3 -- CONTINUATION: the 7 datasets other than Patent,
# ascending by edge count. Patent is archived by scripts/run_geometries_lorentz.sh.
# Head: s(u,v) = geo_temp * (-d_H) on the hyperboloid, NO popularity channel.
#
# Bipartite flags are authoritative from tgb_seq/datasets/preprocess.py::bipartite_dict --
# passing one wrongly changes the negative-candidate pool, so these are not cosmetic.
#
# Runs write to logs/ only; a finished run (rc=0 AND a best_test_mrr line) is COPIED to
# experiment_logs/. A failed run is left in logs/ and NOT copied.
set -u
REP="${1:-3}"; SEED="${2:-42}"
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1

DATASETS=(
  "GoogleLocal|1.91M|--is-bipartite"
  "YouTube|3.29M|"
  "Flickr|7.22M|"
  "ML-20M|14.5M|--is-bipartite"
  "Taobao|18.85M|--is-bipartite"
  "Yelp|19.8M|--is-bipartite"
  "WikiLink|34.2M|"
)

ROOT=$WD/logs/geometries_lorentz/run_$REP
ARCHIVE=$WD/experiment_logs/geometries/lorentz
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER_rest.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)

{
  echo "# LORENTZ SWEEP (continuation)  replicate=$REP  seed=$SEED  branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# NOTE: driver scripts/run_geometries_lorentz_rest.sh is UNTRACKED; SHA covers the trained code only"
  echo "# d_emb=64 k_train=5 num_walks_per_node=10 lr=1e-3 patience 5 cap 100; NO popularity channel"
  echo "# the 7 datasets other than Patent, ascending by edge count"
  echo "#   bipartite: GoogleLocal, ML-20M, Taobao, Yelp"
  echo "#   non-bip:   YouTube, Flickr, WikiLink"
  echo "# Tempest's walk RNG is NOT seeded on this branch, so runs are not bit-reproducible."
  echo "# started=$(date '+%F %T')"
  echo
} > "$DRIVER"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r NAME EDGES BIP <<< "$entry"
  LOWER=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
  LOG=$ROOT/$NAME/$NAME.log
  mkdir -p "$(dirname "$LOG")"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $NAME \
--d-emb 64 --k-train 5 --num-walks-per-node 10 --lr 1e-3 --seed $SEED --early-stop-patience 5 \
$BIP --use-gpu --use-gpu-tempest"
  {
    echo "# dataset=$NAME ($EDGES edges, $([ -n "$BIP" ] && echo BIPARTITE || echo NON-bipartite))"
    echo "# geometry=lorentz  replicate=$REP  seed=$SEED  (archived as ${REP}.log)"
    echo "# d_emb=64 k_train=5 num_walks_per_node=10 lr=1e-3 patience 5 cap 100; no popularity channel"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
    echo "# driven by scripts/run_geometries_lorentz_rest.sh (UNTRACKED); see ../DRIVER_rest.log"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  echo "[$(date '+%F %T')] START $NAME ($EDGES)" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  RC=$?
  SUMMARY=$(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')
  PEAK=$(grep -oP 'test \K[0-9.]+' "$LOG" | sort -rn | head -1)
  if [ $RC -eq 0 ] && grep -q 'best_test_mrr' "$LOG"; then
    mkdir -p "$ARCHIVE/$LOWER"
    cp "$LOG" "$ARCHIVE/$LOWER/${REP}.log"
    echo "[$(date '+%F %T')] DONE  $NAME rc=$RC  $SUMMARY peak=$PEAK -> archived $LOWER/${REP}.log" >> "$DRIVER"
  else
    echo "[$(date '+%F %T')] FAIL  $NAME rc=$RC  NOT archived (log kept in logs/)" >> "$DRIVER"
  fi
done
echo "[$(date '+%F %T')] SWEEP COMPLETE" >> "$DRIVER"
