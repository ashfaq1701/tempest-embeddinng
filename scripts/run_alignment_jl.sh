#!/bin/bash
# Run the alignment+uniformity follow-up cells missed by the §4.8.1 wrapper:
#   - λ_link sweep (0.3, 1.0)
#   - normbrake on top of alignment (the diagnostic-targeted fix for the
#     50-epoch cliff observed on wiki anchor)
set -e
TS=$(date +%H%M%S)
mkdir -p runs

# Step 1 — λ_link sweep on alignment+uniformity (joint training on anchor)
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

# Step 2 — alignment + normbrake (no joint training).  Tests whether the
# diagnostic-derived norm-brake fixes the wiki anchor's 50-epoch cliff.
log="runs/sec481_tgbl-wiki_A_nb_seed42_align_${TS}.log"
echo "--- alignment + normbrake (λ_nb=0.1, no joint) → ${log} ---"
.venv/bin/python -u -m scripts.train --tgb-name tgbl-wiki --use-gpu \
  --num-epochs 50 --early-stop-patience 5 --seed 42 \
  --primary-loss alignment --lambda-normbrake 0.1 \
  --normbrake-threshold 3.87 \
  --head-mode cross_table > "${log}" 2>&1
echo "  exit=$?"
tail -7 "${log}"

echo "=== alignment follow-up cells done ==="
