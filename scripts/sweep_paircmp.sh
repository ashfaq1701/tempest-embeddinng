#!/usr/bin/env bash
# sweep_paircmp.sh — overnight PairComparison capacity sweep on tgbl-review.
# Config 1 (H=4 r=16) is already running as $WAIT_PID; this script waits for it,
# then runs the remaining 6 configs SEQUENTIALLY (single GPU).
# Appends one row per run to RESULTS.tsv as it goes; per-run logs kept for curve analysis.
set -uo pipefail
cd /its/home/ms2420/tempest-embeddinng

OUT=logs/pair_comparison
RES="$OUT/RESULTS.tsv"
PY=venv/bin/python
WAIT_PID="${1:-}"

COMMON="--dataset tgbl-review --batch-size 1000 --eval-batch-size 1000 \
--max-walk-len 5 --num-walks-per-node 10 --lr-manifold 1e-3 --lr-model 1e-3 \
--use-gpu --use-gpu-tempest"

[ -f "$RES" ] || printf "config\tH\tr\thead_params\tbest_val\tbest_test\tbest_ep\tepochs_run\truntime_s\tstatus\tlog\n" > "$RES"

record () {  # $1=label $2=H $3=r $4=log $5=runtime $6=rc
  local label="$1" H="$2" r="$3" log="$4" dt="$5" rc="$6"
  local val test ep n params status
  val=$(grep -oP 'best_val_mrr:\s+\K[0-9.]+'  "$log" | tail -1)
  test=$(grep -oP 'best_test_mrr:\s+\K[0-9.]+' "$log" | tail -1)
  ep=$(grep -oP 'stopped_at_epoch:\s+\K[0-9]+' "$log" | tail -1)
  params=$(grep -oP 'head params:\s+\K[0-9,]+' "$log" | tail -1)
  n=$(grep -cE '^epoch ' "$log")
  if [ "$rc" = "0" ] && [ -n "$test" ]; then status=ok
  else status="FAIL(rc=$rc)"
       grep -qiE "out of memory|CUDA error" "$log" && status="OOM"; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$label" "$H" "$r" "${params:-NA}" "${val:-NA}" "${test:-NA}" "${ep:-NA}" \
    "$n" "$dt" "$status" "$(basename "$log")" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE $label -> val=${val:-NA} test=${test:-NA} ep=${ep:-NA} ${dt}s [$status]"
}

# ── Wait for the already-running config 1 (H=4 r=16), then record it ──
if [ -n "$WAIT_PID" ]; then
  echo "[$(date +%H:%M:%S)] waiting for running config H4_r16 (pid $WAIT_PID) ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  L1=$(ls -t "$OUT"/paircmp_H4_r16_*.log 2>/dev/null | head -1)
  [ -n "$L1" ] && record "H4_r16" 4 16 "$L1" "NA" 0
fi

# ── Remaining 6 configs, sequential ──
run_cfg () {  # $1=H $2=r
  local H="$1" r="$2" label="H${1}_r${2}"
  local log="$OUT/paircmp_${label}_review_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%H:%M:%S)] START $label"
  local t0=$SECONDS
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u scripts/train.py $COMMON --pair-heads "$H" --pair-rank "$r" > "$log" 2>&1
  local rc=$?
  record "$label" "$H" "$r" "$log" "$((SECONDS - t0))" "$rc"
}

run_cfg 8 32
run_cfg 8 16
run_cfg 2 64
run_cfg 4 64
run_cfg 8 64
run_cfg 1 256

echo "[$(date +%H:%M:%S)] ===== SWEEP COMPLETE ====="
column -t -s $'\t' "$RES"
