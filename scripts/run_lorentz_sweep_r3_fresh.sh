#!/bin/bash
# LORENTZ sweep, replicate 3, seed 5 -- projx outlier shave + transp fixes.
#
# The code under test. a8bc34c0 fixes LorentzManifold.transp, which formed <Y-X,V>_L as a
# difference of two nearly equal terms and lost radial momentum to float32 cancellation --
# systematically, so it compounded. That loop drove the ML-20M seed-5 geodesic step from
# 6.6888e-03 (flat for 13 epochs) to 149.8363 in one batch, overflowing cosh in expmap and
# producing NaN. Verified: same seed, only transp changed, all 16 epochs finite, step held
# at 6.6888e-03 throughout, r_max grew FURTHER (13.80 vs 11.44), MRR unchanged.
# The same commit drops clip_grad_norm_(E.weight, 1.0), which never fired (bit-identical to
# unclipped) and measured the Euclidean norm while the runaway was in the momentum buffer.
#
# No radius clamp, no grad clip, no retr cap. The (sum w)^2 midpoint floor remains.
#
# Superseded sweeps, all preserved under logs/geometries_lorentz/, none comparable to this:
#   run_3_seed3_stale_pre6c194d5a            mixed 1534d10f/ea0fc315, radius clamp, 6 datasets
#   run_3_seed3_partial_globalclip_6c194d5a  global clip, ML-20M only, 13ep, killed
#   run_3_seed3_partial_noclip_4435e40f_2ep  E-only clip, ML-20M only, 2ep, killed
# Every dataset below runs on one commit, so this column is like-for-like.
#
# Order: ascending edge count -- GoogleLocal, YouTube, Flickr, Patent, ML-20M, Taobao,
# Yelp, WikiLink. Cheapest first, so results land early and WikiLink (34.2M) is last.
#
# Bipartite flags are authoritative from tgb_seq/datasets/preprocess.py::bipartite_dict --
# read from venv/lib/python3.10/site-packages/ (the repo-relative path in CLAUDE.md does not
# exist). Verified on 2026-09-11: ML-20M, Taobao, Yelp, GoogleLocal True; rest False.
# Passing one wrongly changes the negative-candidate pool, so these are not cosmetic.
#
# Runs write to logs/ only; a finished run (rc=0 AND a best_test_mrr line) is COPIED to
# experiment_logs/, overwriting the stale 3.log there. A failed, collapsed or NaN'd run is
# left in logs/ and NOT copied, so the archive never gains a truncated or dead log.
# The script does not commit; that is done on request.
set -u
SEED="${1:-5}"; REP="${2:-3}"
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1

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

ROOT=$WD/logs/geometries_lorentz/run_${REP}_seed${SEED}
ARCHIVE=$WD/experiment_logs/geometries/lorentz
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
TRW=$($PY -c "import importlib.metadata as m;print(m.version('tempest-rw'))" 2>/dev/null)

{
  echo "# LORENTZ SWEEP (FRESH)  replicate=$REP  seed=$SEED  branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# NOTE: driver scripts/run_lorentz_sweep_r3_fresh.sh is UNTRACKED; SHA covers the trained code only"
  echo "# tempest-rw=$TRW  (>=1.0.6 gives seeded, reproducible walks)"
  echo "# code: (sum w)^2 midpoint floor, NaN-propagating _safe_sqrt, NO radius clamp,"
  echo "#       NO grad clip, NO weight decay, dense RiemannianAdam; projx shaves detached\n#       radius outliers (relative fence p50+2*(p99.9-p50), no absolute cap)"
  echo "# supersedes 3 earlier replicate-3 attempts; see the script header"
    echo "# d_emb=64 k_train=5 num_walks_per_node=5 lr=1e-3 patience 5 cap 100; NO popularity channel"
  echo "# order (ascending edges): GoogleLocal, YouTube, Flickr, Patent, ML-20M, Taobao, Yelp, WikiLink"
  echo "#   bipartite: GoogleLocal, ML-20M, Taobao, Yelp"
  echo "#   non-bip:   YouTube, Flickr, Patent, WikiLink"
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
    echo "# code: (sum w)^2 midpoint floor, NaN-propagating _safe_sqrt, NO radius clamp, NO grad clip, transp fixed"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
    echo "# driven by scripts/run_lorentz_sweep_r3_fresh.sh (UNTRACKED); see ../DRIVER.log"
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
  # would bank a number from a dead run. Three signatures, any is fatal:
  #   r_max == 0.000  -- all points at the origin
  #   link == ln(1+K_train) = ln(6) = 1.7918 -- uniform over 1 positive + 5 negatives
  #   link == nan     -- the new _safe_sqrt propagates NaN instead of swallowing it, so a
  #                      NaN now surfaces in the loss rather than silently poisoning weights
  # At the origin dist() has a zero subgradient everywhere, so the state is absorbing.
  COLLAPSE=""
  grep '^epoch ' "$LOG" | grep -qE 'r_max=0\.000([^0-9]|$)' && COLLAPSE="r_max==0"
  grep '^epoch ' "$LOG" | grep -qE 'link=1\.791[0-9]' && COLLAPSE="${COLLAPSE:+$COLLAPSE,}link==ln(6)"
  grep '^epoch ' "$LOG" | grep -qiE 'link=-?nan' && COLLAPSE="${COLLAPSE:+$COLLAPSE,}link==NaN"
  # An early stop at epoch <=2 is a patience artifact, not a result: Patent did exactly this
  # on the previous sweep (val spiked at ep1, collapsed, patience 5 ran out while it climbed
  # back) and archived best_test 0.1533 against a true 0.1931. Block it from the archive.
  SE=$(grep -oP 'stopped_at_epoch:\s+\K[0-9]+' "$LOG" | head -1)
  [ -n "$SE" ] && [ "$SE" -le 2 ] && COLLAPSE="${COLLAPSE:+$COLLAPSE,}stopped_at_epoch=$SE"
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
