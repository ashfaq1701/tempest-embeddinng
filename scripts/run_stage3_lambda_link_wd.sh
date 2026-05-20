#!/bin/bash
# Stage 3 — λ_link sweep on alignment+normbrake + link MLP weight_decay
# sweep, 50 ep no early-stop on wiki seed 42.
#
# Hypothesis (user, 2026-05-20): on alignment+uniformity the residual -0.11
# Stage-2 cliff comes partly from E_context grad collapse (0.005 → 0.0001
# by ep 7) — Adam momentum keeps moving E_context past saturation. Joint
# training (λ_link > 0) gives E_context a SECOND gradient source from link
# BCE that doesn't saturate with alignment loss. InfoNCE+λ_link failed
# because InfoNCE's gradient geometry fights BCE's; alignment's geometry
# composes more cleanly (both pull target·context up for related pairs).
#
# In parallel: link MLP weight_decay tests the OTHER cliff driver
# (link_w_norm 0.28 → 1.83 even with normbrake). These are complementary,
# not redundant.
#
# All cells: alignment + normbrake (λ=0.1, threshold=3.87), 50 ep,
# patience=999, --log-debug, seed 42.
set -e
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p runs

summary="runs/stage3_${TS}_SUMMARY.log"
{
echo "=== Stage 3 λ_link + weight_decay_link sweep — ts=${TS} ==="
echo "  base: alignment+uniformity + normbrake λ=0.1, threshold=3.87"
echo "  num_epochs=50  early-stop disabled  --log-debug"

# 6 cells:
#   λ_link sweep (4):
#     1. L0       — λ_link=0 (control, must match A_long_nb)
#     2. L0.1     — λ_link=0.1 (gentle joint)
#     3. L0.3     — λ_link=0.3 (stronger)
#     4. L1.0     — λ_link=1.0 (full joint)
#   weight_decay_link sweep (2):
#     5. WD1e-4   — link MLP wd 1e-4
#     6. WD1e-3   — link MLP wd 1e-3
#
# spec format: tag:lambda_link:weight_decay_link
for spec in \
  "L0:0.0:0.0" \
  "L0.1:0.1:0.0" \
  "L0.3:0.3:0.0" \
  "L1.0:1.0:0.0" \
  "WD1e-4:0.0:1e-4" \
  "WD1e-3:0.0:1e-3"; do
  IFS=':' read tag ll wdl <<< "$spec"
  log="runs/stage3_${tag}_${TS}.log"
  echo "--- ${tag} (λ_link=${ll}  wd_link=${wdl}) → ${log} ---"
  .venv/bin/python -u -m scripts.train --tgb-name tgbl-wiki --use-gpu \
    --num-epochs 50 --early-stop-patience 999 --seed 42 \
    --primary-loss alignment \
    --lambda-normbrake 0.1 --normbrake-threshold 3.87 \
    --lambda-link "${ll}" \
    --weight-decay-link "${wdl}" \
    --head-mode cross_table \
    --log-debug > "${log}" 2>&1
  rc=$?
  echo "  exit=${rc}  log=${log}"
  tail -12 "${log}"
done

echo "=== Stage 3 DONE ==="
} 2>&1 | tee "${summary}"
