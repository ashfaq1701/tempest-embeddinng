#!/bin/bash
# Run the alignment+uniformity follow-up cells missed by the §4.8.1 wrapper.
# (Now reduced: λ_link 0.3/1.0 were picked up by the wrapper after-all when
# bash re-read the loop body. Only `alignment + normbrake` remains to test —
# the diagnostic-targeted fix for the 50-epoch cliff observed on wiki anchor.)
set -e
TS=$(date +%H%M%S)
mkdir -p runs

# alignment + normbrake (no joint training).
log="runs/sec481_tgbl-wiki_A_nb_seed42_align_${TS}.log"
echo "--- alignment + normbrake (λ_nb=0.1, no joint) → ${log} ---"
.venv/bin/python -u -m scripts.train --tgb-name tgbl-wiki --use-gpu \
  --num-epochs 50 --early-stop-patience 5 --seed 42 \
  --primary-loss alignment --lambda-normbrake 0.1 \
  --normbrake-threshold 3.87 \
  --head-mode cross_table > "${log}" 2>&1
echo "  exit=$?"
tail -7 "${log}"

echo "=== alignment follow-up cell done ==="
