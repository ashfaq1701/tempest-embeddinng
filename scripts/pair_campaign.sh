#!/usr/bin/env bash
# Pair-feature overnight campaign driver. Runs a named list of experiments
# SEQUENTIALLY (single 8 GB GPU), full log per run, one summary line appended to
# logs/pair_features/RESULTS.tsv. Resilient: a failing run does not abort the rest.
#
# Usage: scripts/pair_campaign.sh <wave-name> "<name>|<flags>" "<name>|<flags>" ...
set -u
export PYTHONUNBUFFERED=1   # python block-buffers stdout when redirected; force live logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation on 8 GB
PY=.venv/bin/python
OUT=logs/pair_features
mkdir -p "$OUT"
RES="$OUT/RESULTS.tsv"
COMMON="--dataset tgbl-wiki --use-gpu --use-gpu-tempest --num-epochs 20 \
        --early-stop-patience 5 --eval-batch-size 50 --seed 42"

wave="$1"; shift
echo "### WAVE $wave  $(date '+%F %T')" >> "$RES"
for spec in "$@"; do
  name="${spec%%|*}"; flags="${spec#*|}"; [ "$flags" = "$name" ] && flags=""
  log="$OUT/${name}.log"
  echo ">>> [$(date '+%T')] $name   flags: $flags"
  stdbuf -oL -eL $PY scripts/train.py $COMMON $flags > "$log" 2>&1
  val=$(grep -oP 'best_val_mrr:\s+\K[0-9.]+' "$log" | tail -1)
  test=$(grep -oP 'best_test_mrr:\s+\K[0-9.]+' "$log" | tail -1)
  ep=$(grep -oP 'stopped_at_epoch:\s+\K[0-9]+' "$log" | tail -1)
  printf '%s\t%s\tval=%s\ttest=%s\tep=%s\t[%s]\n' \
         "$wave" "$name" "${val:-FAIL}" "${test:-FAIL}" "${ep:-?}" "$flags" >> "$RES"
  echo "<<< [$(date '+%T')] $name  val=${val:-FAIL} test=${test:-FAIL} ep=${ep:-?}"
done
echo "### WAVE $wave DONE $(date '+%F %T')" >> "$RES"
