#!/usr/bin/env bash
# overnight_1v2proj.sh — 4 sequential runs on tgbl-review, code = 6d7e77f (log1p-hop PE) + score-weight probe.
#   1) SHARED  proj, lr-model 1e-3
#   2) SHARED  proj, lr-model 1e-2
#   --- live-swap model.py to the TWO-HEAD variant ---
#   3) TWOHEAD proj, lr-model 1e-3
#   4) TWOHEAD proj, lr-model 1e-2
# One-vs-two projection heads x lr, softmax-mixture dual-side walk head. Single GPU, sequential.
# model.py is swapped between the SHARED and TWOHEAD snapshots in logs/overnight_1v2proj/variants/.
set -uo pipefail
cd /its/home/ms2420/tempest-embeddinng

OUT=logs/overnight_1v2proj
VAR=$OUT/variants
RES=$OUT/RESULTS.tsv
PROG=$OUT/PROGRESS.log
PY=venv/bin/python
MODEL=tempest_walks/model.py

[ -f "$RES" ] || printf "run\tproj\tlr_model\thead_params\tbest_val\tbest_test\tbest_ep\tepochs_run\truntime_s\tfinal_w_PuEv\tfinal_w_EuPv\tfinal_w_PuPv\tstatus\tlog\n" > "$RES"

COMMON="--dataset tgbl-review --batch-size 1000 --eval-batch-size 1000 \
--num-walks-per-node-source-side 10 --max-walk-len-source-side 5 \
--num-walks-per-node-candidate-side 10 --max-walk-len-candidate-side 2 \
--lr-manifold 1e-3 --early-stop-patience 5 --use-gpu --use-gpu-tempest"

run_one () {  # $1=run-label $2=proj(SHARED|TWOHEAD) $3=lr_model
  local run="$1" proj="$2" lr="$3"
  cp "$VAR/model_${proj}.py" "$MODEL"        # live-swap the head variant onto disk BEFORE launch
  local log="$OUT/${run}_${proj}_lr${lr}_$(date +%Y%m%d_%H%M%S).log"
  echo "[$(date +%H:%M:%S)] START $run  proj=$proj  lr_model=$lr" | tee -a "$PROG"
  local t0=$SECONDS
  PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY -u scripts/train.py $COMMON --lr-model "$lr" > "$log" 2>&1
  local rc=$? dt=$((SECONDS - t0))
  local val test ep n params status wline w0 w1 w2
  val=$(grep -oP 'best_val_mrr:\s+\K[0-9.]+'  "$log" | tail -1)
  test=$(grep -oP 'best_test_mrr:\s+\K[0-9.]+' "$log" | tail -1)
  ep=$(grep -oP 'stopped_at_epoch:\s+\K[0-9]+' "$log" | tail -1)
  params=$(grep -oP 'head params:\s+\K[0-9,]+' "$log" | tail -1)
  n=$(grep -cE '^epoch ' "$log")
  wline=$(grep -oP 'w\[PuEv/EuPv/PuPv\]=\K[0-9./]+' "$log" | tail -1)
  w0=$(echo "$wline" | cut -d/ -f1); w1=$(echo "$wline" | cut -d/ -f2); w2=$(echo "$wline" | cut -d/ -f3)
  if [ "$rc" = "0" ] && [ -n "$test" ]; then status=ok
  else status="FAIL(rc=$rc)"; grep -qiE "out of memory|CUDA error" "$log" && status="OOM"; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run" "$proj" "$lr" "${params:-NA}" "${val:-NA}" "${test:-NA}" "${ep:-NA}" "$n" "$dt" \
    "${w0:-NA}" "${w1:-NA}" "${w2:-NA}" "$status" "$(basename "$log")" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE  $run -> val=${val:-NA} test=${test:-NA} ep=${ep:-NA} w=${wline:-NA} ${dt}s [$status]" | tee -a "$PROG"
}

echo "########## overnight 1-vs-2 projection sweep START $(date) ##########" | tee -a "$PROG"
run_one run1 SHARED  1e-3
run_one run2 SHARED  1e-2
run_one run3 TWOHEAD 1e-3
run_one run4 TWOHEAD 1e-2

# restore disk to the committed SHARED state (== 6d7e77f) so the tree is clean afterwards
cp "$VAR/model_SHARED.py" "$MODEL"

echo "########## ALL 4 DONE $(date) ##########" | tee -a "$PROG"
column -t -s $'\t' "$RES" | tee -a "$PROG"
