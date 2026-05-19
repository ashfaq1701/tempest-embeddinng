#!/bin/bash
# Run the 2 alignment+uniformity λ_link cells missed by the §4.8.1 wrapper.
set -e
TS=$(date +%H%M%S)
mkdir -p runs
for JL in 0.3 1.0; do
  log="runs/sec481_tgbl-wiki_A_jl${JL}_seed42_align_${TS}.log"
  echo "--- alignment λ_link=${JL} → ${log} ---"
  .venv/bin/python -u -m scripts.train --tgb-name tgbl-wiki --use-gpu \
    --num-epochs 50 --early-stop-patience 5 --seed 42 \
    --primary-loss alignment --lambda-normbrake 0 \
    --normbrake-threshold 3.87 \
    --lambda-link "${JL}" \
    --head-mode cross_table > "${log}" 2>&1
  echo "  exit=$?"
  tail -7 "${log}"
done
echo "=== alignment λ_link cells done ==="
