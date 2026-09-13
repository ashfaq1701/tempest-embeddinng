#!/bin/bash
# LORENTZ sweep, replicate 3, seed 3. All 8 TGB-Seq datasets, sequential (one GPU),
# in the requested order: Patent, Flickr, Taobao, GoogleLocal, YouTube, ML-20M, Yelp, WikiLink.
# Intrinsic Lorentz manifold; NO popularity channel.
#
# First sweep on tempest-rw >= 1.0.6, so walks are seeded: a rerun at seed 3 should
# reproduce. Every earlier result predates this and had unseeded walks.
#
# Bipartite flags are authoritative from tgb_seq/datasets/preprocess.py::bipartite_dict --
# passing one wrongly changes the negative-candidate pool, so these are not cosmetic.
#
# Runs write to logs/ only; a finished run (rc=0 AND a best_test_mrr line) is COPIED to
# experiment_logs/. A failed or killed run is left in logs/ and NOT copied, so the archive
# never gains a truncated log. The script does not commit; that is done on request.
set -u
SEED="${1:-3}"; REP="${2:-3}"
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1

# Patent, Flickr, Taobao, GoogleLocal and YouTube are already archived at seed 3;
# these are the three that remain. ML-20M first -- it is the dataset the radius cap
# in projx actually binds on, and the only one that has never completed.
DATASETS=(
  "ML-20M|14.5M|--is-bipartite"
  "Yelp|19.8M|--is-bipartite"
  "WikiLink|34.2M|"
)

ROOT=$WD/logs/geometries_lorentz/run_${REP}_seed${SEED}
ARCHIVE=$WD/experiment_logs/geometries/lorentz
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
TRW=$($PY -c "import importlib.metadata as m;print(m.version('tempest-rw'))" 2>/dev/null)

{
  echo "# LORENTZ SWEEP  replicate=$REP  seed=$SEED  branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# NOTE: driver scripts/run_lorentz_sweep_seed3.sh is UNTRACKED; SHA covers the trained code only"
  echo "# tempest-rw=$TRW  (>=1.0.6 gives seeded, reproducible walks)"
  echo "# d_emb=64 k_train=5 num_walks_per_node=5 lr=1e-3 patience 5 cap 100; NO popularity channel"
  echo "# order: Patent, Flickr, Taobao, GoogleLocal, YouTube, ML-20M, Yelp, WikiLink"
  echo "#   bipartite: GoogleLocal, ML-20M, Taobao, Yelp"
  echo "#   non-bip:   Patent, Flickr, YouTube, WikiLink"
  echo "# started=$(date '+%F %T')"
  echo
} > "$DRIVER"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r NAME EDGES BIP <<< "$entry"
  LOWER=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
  LOG=$ROOT/$NAME/$NAME.log
  mkdir -p "$(dirname "$LOG")"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $NAME \
--d-emb 64 --k-train 5 --num-walks-per-node 5 --lr 1e-3 --seed $SEED --early-stop-patience 5 \
$BIP --use-gpu --use-gpu-tempest"
  {
    echo "# dataset=$NAME ($EDGES edges, $([ -n "$BIP" ] && echo BIPARTITE || echo NON-bipartite))"
    echo "# geometry=lorentz  replicate=$REP  seed=$SEED  (archived as ${REP}.log)"
    echo "# d_emb=64 k_train=5 num_walks_per_node=5 lr=1e-3 patience 5 cap 100; no popularity channel"
    echo "# branch=$BRANCH commit=$SHA  tempest-rw=$TRW"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
    echo "# driven by scripts/run_lorentz_sweep_seed3.sh (UNTRACKED); see ../DRIVER.log"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  echo "[$(date '+%F %T')] START $NAME ($EDGES)" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  RC=$?
  SUMMARY=$(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')
  PEAK=$(grep '^epoch ' "$LOG" | grep -oP 'test \K[0-9.]+' | sort -rn | head -1)
  # Collapse guard. A total collapse (every embedding at the origin) exits rc=0 and
  # reports a plausible best_test_mrr restored from a pre-collapse epoch, so rc alone
  # would bank a number from a dead run. Two signatures, either is fatal:
  #   r_max == 0.000  -- all points at the origin
  #   link == ln(1+K_train) = ln(6) = 1.7918 -- uniform over 1 positive + 5 negatives
  # At the origin dist() has a zero subgradient everywhere, so the state is absorbing.
  COLLAPSE=""
  grep '^epoch ' "$LOG" | grep -qE 'r_max=0\.000([^0-9]|$)' && COLLAPSE="r_max==0"
  grep '^epoch ' "$LOG" | grep -qE 'link=1\.791[0-9]' && COLLAPSE="${COLLAPSE:+$COLLAPSE,}link==ln(6)"
  if [ -n "$COLLAPSE" ]; then
    echo "[$(date '+%F %T')] COLLAPSE $NAME ($COLLAPSE) -- NOT archived, log kept in logs/" >> "$DRIVER"
  elif [ $RC -eq 0 ] && grep -q 'best_test_mrr' "$LOG"; then
    mkdir -p "$ARCHIVE/$LOWER"
    cp "$LOG" "$ARCHIVE/$LOWER/${REP}.log"
    echo "[$(date '+%F %T')] DONE  $NAME rc=$RC  $SUMMARY peak=$PEAK -> archived $LOWER/${REP}.log" >> "$DRIVER"
  else
    echo "[$(date '+%F %T')] FAIL  $NAME rc=$RC  NOT archived (log kept in logs/)" >> "$DRIVER"
  fi
done
echo "[$(date '+%F %T')] SWEEP COMPLETE" >> "$DRIVER"
