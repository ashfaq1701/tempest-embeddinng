#!/usr/bin/env bash
# sweep_dof.sh — candidate-tower DOF ablation, queued behind the running bbaa476 job.
# Runs from the exp/candidate-tangent-dof WORKTREE so the main tree stays free.
#   free       127 DOF, cold candidates move   (== 742fe55 as committed; expect ~0.2638)
#   free_seen  127 DOF, cold candidates frozen
#   partner      1 DOF along the geodesic to w_v, cold frozen (the rotation constraint)
# 'free' is a REPRODUCTION arm: if it does not land near 0.2638 the plumbing port changed something.
set -uo pipefail

WT=/tmp/claude-1022003/-its-home-ms2420/fdf943e6-1859-464b-b2be-70c15dea4eec/scratchpad/dof_exp
REPO=/its/home/ms2420/tempest-embeddinng
OUT=$REPO/logs/dof_ablation
RES="$OUT/RESULTS.tsv"
PY=$REPO/venv/bin/python
WAIT_PID="${1:-}"
mkdir -p "$OUT"

COMMON="--dataset tgbl-review --batch-size 1000 --eval-batch-size 1000 \
--max-walk-len 5 --num-walks-per-node 10 --lr-manifold 1e-3 --lr-model 1e-3 \
--early-stop-patience 5 --use-gpu --use-gpu-tempest"

[ -f "$RES" ] || printf "mode\thead_params\tbest_val\tbest_test\tbest_ep\tepochs_run\truntime_s\tstatus\tlog\n" > "$RES"

if [ -n "$WAIT_PID" ]; then
  echo "[$(date +%H:%M:%S)] waiting for bbaa476 (pid $WAIT_PID) to finish ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[$(date +%H:%M:%S)] bbaa476 done; starting DOF ablation"
fi

cd "$WT" || exit 1

run_mode () {
  local mode="$1"
  local log="$OUT/dof_${mode}_review_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%H:%M:%S)] START $mode"
  local t0=$SECONDS
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u scripts/train.py $COMMON --tangent-mode "$mode" > "$log" 2>&1
  local rc=$? dt=$((SECONDS - t0))
  local val test ep n params status
  val=$(grep -oP 'best_val_mrr:\s+\K[0-9.]+'  "$log" | tail -1)
  test=$(grep -oP 'best_test_mrr:\s+\K[0-9.]+' "$log" | tail -1)
  ep=$(grep -oP 'stopped_at_epoch:\s+\K[0-9]+' "$log" | tail -1)
  params=$(grep -oP 'head params:\s+\K[0-9,]+' "$log" | tail -1)
  n=$(grep -cE '^epoch ' "$log")
  if [ "$rc" = "0" ] && [ -n "$test" ]; then status=ok
  else status="FAIL(rc=$rc)"; grep -qiE "out of memory|CUDA error" "$log" && status="OOM"; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$mode" "${params:-NA}" "${val:-NA}" "${test:-NA}" "${ep:-NA}" "$n" "$dt" "$status" "$(basename "$log")" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE $mode -> val=${val:-NA} test=${test:-NA} ep=${ep:-NA} ${dt}s [$status]"
}

run_mode free
run_mode free_seen
run_mode partner

echo "[$(date +%H:%M:%S)] ===== DOF ABLATION COMPLETE ====="
column -t -s $'\t' "$RES"
