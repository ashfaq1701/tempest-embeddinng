#!/bin/bash
# Full TGB-Seq sweep, ascending by edge count. Sequential: one dataset at a time.
#   usage: sweep_all_datasets.sh <EXPERIMENT> [EXTRA_ARGS...]
# Bipartite flags are authoritative from tgb_seq/datasets/preprocess.py::bipartite_dict.
EXPERIMENT="${1:?usage: $0 <EXPERIMENT> <DATASETS csv | ALL> [EXTRA_ARGS...]}"; shift
WANT="${1:?usage: $0 <EXPERIMENT> <DATASETS csv | ALL> [EXTRA_ARGS...]}"; shift
EXTRA="$*"
# LR is a parameter: `LR=3e-3 sweep_all_datasets.sh ...`. Default 1e-3 keeps old invocations
# byte-identical. The cell name records a non-default lr so the log path is self-describing.
LR="${LR:-1e-3}"
if [ "$LR" = "1e-3" ]; then DEFCELL=d64_k5; else DEFCELL="d64_k5_lr$(echo "$LR" | tr -d '.-')"; fi
CELL="${CELL:-$DEFCELL}"
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
ROOT=$WD/logs/$EXPERIMENT/$CELL
cd "$WD" || exit 1
mkdir -p "$ROOT"
DRIVER=$ROOT/DRIVER.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)

# name|edges|bipartite-flag   ascending by edge count
DATASETS=(
  "YouTube|3.29M|"
  "GoogleLocal|1.91M|--is-bipartite"
  "Flickr|7.22M|"
  "Patent|10.8M|"
  "ML-20M|14.5M|--is-bipartite"
  "Taobao|18.85M|--is-bipartite"
  "Yelp|19.8M|--is-bipartite"
  "WikiLink|34.2M|"
)

{
  echo "# SWEEP experiment=$EXPERIMENT cell=$CELL lr=$LR  branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# extra args: ${EXTRA:-(none)}"
  echo "# datasets requested: $WANT"
  echo "# run SEQUENTIALLY (one GPU), in the order listed below."
  echo "# Bipartite flags from tgb_seq/datasets/preprocess.py::bipartite_dict -- passing one wrongly"
  echo "# changes the negative-candidate pool, so these are not cosmetic:"
  echo "#   bipartite: GoogleLocal, ML-20M, Taobao, Yelp"
  echo "#   non-bip:   YouTube, Flickr, Patent, WikiLink"
  echo "# started=$(date '+%F %T')"
  echo
} > "$DRIVER"

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r NAME EDGES BIP <<< "$entry"
  # Run only the requested datasets, but keep this file's canonical bipartite flags.
  if [ "$WANT" != "ALL" ] && ! echo ",$WANT," | grep -q ",$NAME,"; then continue; fi
  LOG=$ROOT/$NAME/run_1/$NAME.log
  mkdir -p "$(dirname "$LOG")"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $NAME \
--d-emb 64 --k-train 5 --lr $LR --num-epochs 100 --early-stop-patience 5 \
$BIP --use-gpu --use-gpu-tempest $EXTRA"
  {
    echo "# dataset=$NAME ($EDGES edges, $([ -n "$BIP" ] && echo BIPARTITE || echo NON-bipartite))"
    echo "# d_emb=64 k_train=5 lr=$LR seed=42  patience 5  cap 100"
    echo "# experiment=$EXPERIMENT cell=$CELL tag=run_1"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
    echo "# part of the sweep driven by scripts/sweep_all_datasets.sh; see ../../DRIVER.log"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  echo "[$(date '+%F %T')] START $NAME ($EDGES)" >> "$DRIVER"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%F %T')] DONE  $NAME rc=$RC  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$DRIVER"
done
echo "[$(date '+%F %T')] SWEEP COMPLETE" >> "$DRIVER"
