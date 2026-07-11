#!/usr/bin/env bash
# run12.sh — overnight head bake-off: master + 3 velocity branches.
#   wiki:   each branch x {no-pf, pf}      = 8 runs
#   review: each branch x {no-pf} subsample = 4 runs
# Grouped by branch (checkout once per branch). Sequential — single GPU.
# Records val/test MRR + wall runtime per run to RESULTS.tsv as it goes.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=logs/manifold/run12
mkdir -p "$OUT"
RES="$OUT/RESULTS.tsv"
[ -f "$RES" ] || printf "branch\tdataset\tpf\tval\ttest\tepoch\truntime_s\tstatus\n" > "$RES"

PY=.venv/bin/python
COMMON="--d-emb 128 --batch-size 200 --eval-batch-size 20 --num-walks-per-node 10 \
--max-walk-len 20 --num-epochs 50 --early-stop-patience 5 --seed 42 \
--lr 5e-3 --lr-min 5e-6 \
--tempest-batch-window-multiplier -1.0 --use-gpu --use-gpu-tempest"

run_one () {  # $1=branch $2=dataset-label $3=pf(0/1) $4...=extra args
  local branch="$1" dlabel="$2" pf="$3"; shift 3
  local pfflag="" pftag="nopf"
  [ "$pf" = "1" ] && { pfflag="--use-pair-features"; pftag="pf"; }
  local log="$OUT/${branch//\//_}__${dlabel}__${pftag}.log"
  echo "=== [$(date +%H:%M:%S)] $branch / $dlabel / $pftag ===" | tee -a "$OUT/PROGRESS.log"
  local t0=$SECONDS
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY scripts/train.py $COMMON $pfflag "$@" > "$log" 2>&1
  local rc=$? dt=$((SECONDS - t0))
  local val test ep status
  val=$(grep -oP 'best_val_mrr:\s+\K[0-9.]+' "$log" | tail -1)
  test=$(grep -oP 'best_test_mrr:\s+\K[0-9.]+' "$log" | tail -1)
  ep=$(grep -oP 'stopped_at_epoch:\s+\K[0-9]+' "$log" | tail -1)
  if [ "$rc" = "0" ] && [ -n "$test" ]; then status=ok; else
    status="FAIL(rc=$rc)"; grep -qiE "out of memory|CUDA error" "$log" && status="OOM"; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$branch" "$dlabel" "$pftag" "${val:-NA}" "${test:-NA}" "${ep:-NA}" "$dt" "$status" >> "$RES"
  echo "    -> val=${val:-NA} test=${test:-NA} ep=${ep:-NA} ${dt}s [$status]" | tee -a "$OUT/PROGRESS.log"
}

WIKI="--dataset tgbl-wiki --k-train 300"
REVIEW="--dataset tgbl-review --k-train 100 --max-train-edges 110000 --max-eval-edges 25000"

for branch in master feature/velocity-head feature/velocity-perwalk-avg feature/velocity-mixture; do
  echo "########## checkout $branch ##########" | tee -a "$OUT/PROGRESS.log"
  git checkout -q "$branch" || { echo "checkout $branch FAILED" | tee -a "$OUT/PROGRESS.log"; continue; }
  run_one "$branch" wiki   0 $WIKI
  run_one "$branch" wiki   1 $WIKI
  run_one "$branch" review 0 $REVIEW
done

git checkout -q master
echo "########## ALL 12 DONE $(date) ##########" | tee -a "$OUT/PROGRESS.log"
column -t -s $'\t' "$RES" | tee -a "$OUT/PROGRESS.log"
