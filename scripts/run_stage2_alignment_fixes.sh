#!/bin/bash
# Stage 2 — alignment+uniformity architectural fixes, long-training (50 ep,
# no early stop) on wiki. Tests which fix(es) prevent the 50-epoch cliff
# observed in the Phase 0.5 diagnostic (0.7070 -> 0.4269 over 50 epochs).
#
# Goal (per user direction 2026-05-20): smooth loss decrease + smooth MRR
# increase over the full 50-epoch horizon.
set -e
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p runs

summary="runs/stage2_alignment_${TS}_SUMMARY.log"
{
echo "=== Stage 2 alignment fixes long-train — ts=${TS} ==="
echo "  num_epochs=50  early-stop disabled  --log-debug"

# 6 cells:
#   1. A_long_baseline    — pure alignment+uniformity (reproduces the diag).
#   2. A_long_nb          — alignment + normbrake (threshold 3.87).
#   3. A_long_dr0.3       — alignment + link MLP dropout 0.3.
#   4. A_long_ed0.3       — alignment + embedding dropout 0.3.
#   5. A_long_d5          — alignment + n_layers=5 (deeper link MLP).
#   6. A_long_full        — alignment + nb + n_layers=5 + link_dr 0.3 + emb_dr 0.3 (kitchen sink).

for spec in \
  "A_long_baseline:0:3:0.0:0.0" \
  "A_long_nb:0.1:3:0.0:0.0" \
  "A_long_dr0.3:0:3:0.3:0.0" \
  "A_long_ed0.3:0:3:0.0:0.3" \
  "A_long_d5:0:5:0.0:0.0" \
  "A_long_full:0.1:5:0.3:0.3"; do
  IFS=':' read tag nb nl ldr edr <<< "$spec"
  log="runs/long_align_${tag}_${TS}.log"
  echo "--- ${tag} (nb=${nb} n_layers=${nl} link_dr=${ldr} emb_dr=${edr}) → ${log} ---"
  .venv/bin/python -u -m scripts.train --tgb-name tgbl-wiki --use-gpu \
    --num-epochs 50 --early-stop-patience 999 --seed 42 \
    --primary-loss alignment --lambda-normbrake "${nb}" \
    --normbrake-threshold 3.87 \
    --link-mlp-n-layers "${nl}" \
    --link-mlp-dropout "${ldr}" \
    --embedding-dropout "${edr}" \
    --head-mode cross_table \
    --log-debug > "${log}" 2>&1
  rc=$?
  echo "  exit=${rc}  log=${log}"
  tail -12 "${log}"
done

echo "=== Stage 2 DONE ==="
} 2>&1 | tee "${summary}"
