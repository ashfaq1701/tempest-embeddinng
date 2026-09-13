#!/bin/bash
# Overnight queue. Steps, in order:
#   1. wait for the in-flight Patent (wpn 10) to be recorded, then stop the primary driver
#      (the RUNNING bash holds a pre-rewrite inode listing 6 more datasets and would
#       otherwise start GoogleLocal at wpn 10 without YouTube in its queue)
#   2. Patent at wpn 5                        -> logs only, NOT archived
#   3. DECIDE the winning wpn from max test of the two arms, then
#      Patent at --k-train 10 with that wpn   -> logs only, NOT archived
#   4. the other 7 datasets at wpn 10 via run_geometries_lorentz_rest.sh (archives each)
#
# Steps 2 and 3 are deliberately not archived: patent/3.log belongs to the wpn 10 arm
# from step 1, and archiving a comparison arm over it would destroy the baseline before
# the comparison is reported.
set -u
WD=/its/home/ms2420/tempest-embeddinng; PY=$WD/venv/bin/python
cd "$WD" || exit 1
SEED=42
DRIVER=$WD/logs/geometries_lorentz/run_3/DRIVER.log
WPN10_LOG=$WD/logs/geometries_lorentz/run_3/Patent/Patent.log
WPN5_LOG=$WD/logs/patent_wpn/d64_k5/wpn5/Patent.log
Q=$WD/logs/geometries_lorentz/run_3/QUEUE.log
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)

peak() { grep -oP 'test \K[0-9.]+' "$1" 2>/dev/null | sort -rn | head -1; }
ckpt() { grep -oP 'best_test_mrr[ =:]+\K[0-9.]+' "$1" 2>/dev/null | tail -1; }

run_patent() {   # $1=log  $2=wpn  $3=k_train  $4=tagline
  local LOG=$1 WPN=$2 KT=$3 TAG=$4
  mkdir -p "$(dirname "$LOG")"
  local DIRTY; DIRTY=$(git status --porcelain --untracked-files=no | head -1)
  local CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train $KT --num-walks-per-node $WPN --lr 1e-3 --seed $SEED --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
  {
    echo "# dataset=Patent (10.8M edges, NON-bipartite)"
    echo "# $TAG"
    echo "# geometry=lorentz  seed=$SEED"
    echo "# d_emb=64 k_train=$KT num_walks_per_node=$WPN lr=1e-3 patience 5 cap 100; no popularity channel"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes"
    echo "# driven by scripts/lorentz_queue.sh (UNTRACKED); NOT auto-archived (comparison arm)"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  echo "[$(date '+%F %T')] START $TAG" >> "$Q"
  PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] DONE  $TAG rc=$? peak=$(peak "$LOG") ckpt=$(ckpt "$LOG")" >> "$Q"
}

echo "[$(date '+%F %T')] queue armed: Patent(wpn10) -> Patent(wpn5) -> DECIDE -> Patent(k_train10,best wpn) -> 7 datasets(wpn10)" > "$Q"

# ---- step 1: wait out the in-flight run, then stop the primary driver ----
while true; do
  if grep -qE '^\[.*\] (DONE|FAIL)  Patent ' "$DRIVER" 2>/dev/null; then
    echo "[$(date '+%F %T')] $(grep -E '^\[.*\] (DONE|FAIL)  Patent ' "$DRIVER" | tail -1)" >> "$Q"; break
  fi
  pgrep -f "run_geometries_lorentz\.sh" > /dev/null || { echo "[$(date '+%F %T')] primary driver gone, no Patent record" >> "$Q"; break; }
  sleep 60
done
for p in $(pgrep -f "run_geometries_lorentz\.sh"); do kill "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f "train_link_property_prediction"); do kill "$p" 2>/dev/null; done
sleep 10
for p in $(pgrep -f "train_link_property_prediction"); do kill -9 "$p" 2>/dev/null; done
sleep 5
echo "[$(date '+%F %T')] primary driver stopped" >> "$Q"

# ---- step 2: wpn 5 ----
run_patent "$WPN5_LOG" 5 5 "ARM patent_wpn/wpn5: num_walks_per_node=5, k_train=5"

# ---- step 3: decide the winning wpn on max test, then k_train 10 ----
P10=$(peak "$WPN10_LOG"); P5=$(peak "$WPN5_LOG")
BEST_WPN=$($PY -c "
p10='''$P10'''.strip() or '0'; p5='''$P5'''.strip() or '0'
print(10 if float(p10) >= float(p5) else 5)")
{
  echo "[$(date '+%F %T')] DECIDE wpn: max test wpn10=$P10  wpn5=$P5  -> winner wpn=$BEST_WPN"
  echo "[$(date '+%F %T')]   (decided on max test, not test@val-checkpoint: val drift makes the"
  echo "[$(date '+%F %T')]    printed best_test_mrr not like-for-like across arms)"
} >> "$Q"
run_patent "$WD/logs/patent_ktrain/d64_ktrain10/wpn${BEST_WPN}/Patent.log" "$BEST_WPN" 10 \
  "ARM patent_ktrain/ktrain10: k_train=10, num_walks_per_node=$BEST_WPN (winning wpn)"

# ---- step 4: the other 7 datasets at wpn 10 ----
echo "[$(date '+%F %T')] START continuation: 7 datasets at wpn 10" >> "$Q"
"$WD/scripts/run_geometries_lorentz_rest.sh" 3 $SEED
echo "[$(date '+%F %T')] QUEUE COMPLETE" >> "$Q"
