#!/bin/bash
# Stage 5 — uniformity hyperparameter sweep, 50 ep no early-stop on wiki
# seed 42.
#
# Tests two hypotheses about uniformity loss (v2.4 §11):
#   H1: eta_uniform too high → uniformity's neg pull dominates alignment
#   H2: effective uniformity neg count too high → too-low gradient
#       variance, Adam momentum keeps moving E_context past saturation
#
# Base config: alignment + normbrake (λ=0.1, threshold=3.87) +
# weight_decay_link=1e-4 + Stage 4 winner. Stage 4 winner is parameterized
# via env vars STAGE4_HIST_NEG_RATIO and STAGE4_LAMBDA_LINK; defaults
# match current best-known config.
#
# Usage:
#   STAGE4_HIST_NEG_RATIO=0.5 STAGE4_LAMBDA_LINK=0.0 \
#       bash scripts/run_stage5_uniformity_sweep.sh
#
# IMPORTANT: do not launch until Stage 4 lands. Update env vars to the
# Stage 4 winning values before launching.
set -e
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p runs

HNR="${STAGE4_HIST_NEG_RATIO:-0.5}"
LL="${STAGE4_LAMBDA_LINK:-0.0}"

summary="runs/stage5_${TS}_SUMMARY.log"
{
echo "=== Stage 5 uniformity sweep — ts=${TS} ==="
echo "  base: alignment + normbrake λ=0.1 thr=3.87 + WD_link=1e-4"
echo "        + hist_neg_ratio=${HNR} (Stage 4 winner)"
echo "        + lambda_link=${LL} (Stage 4 winner)"
echo "  num_epochs=50  early-stop disabled  --log-debug  seed=42"

# 6 cells:
#   eta_uniform sweep (H1):
#     U_base   — eta=1.0, cap=20000 (control = current locked baseline)
#     U_lo     — eta=0.3, cap=20000
#     U_lower  — eta=0.1, cap=20000
#   uniformity_cap sweep (H2):
#     N_half   — eta=1.0, cap=200  (~half typical batch unique nodes)
#     N_quarter — eta=1.0, cap=100 (~quarter typical)
#     Combined:
#     Both_lo  — eta=0.3, cap=200
#
# spec format: tag:eta_uniform:uniformity_cap
for spec in \
  "U_base:1.0:20000" \
  "U_lo:0.3:20000" \
  "U_lower:0.1:20000" \
  "N_half:1.0:200" \
  "N_quarter:1.0:100" \
  "Both_lo:0.3:200"; do
  IFS=':' read tag eta cap <<< "$spec"
  log="runs/stage5_${tag}_${TS}.log"
  echo "--- ${tag} (eta_uniform=${eta}  uniformity_cap=${cap}) → ${log} ---"
  .venv/bin/python -u -m scripts.train --tgb-name tgbl-wiki --use-gpu \
    --num-epochs 50 --early-stop-patience 999 --seed 42 \
    --primary-loss alignment \
    --lambda-normbrake 0.1 --normbrake-threshold 3.87 \
    --weight-decay-link 1e-4 \
    --lambda-link "${LL}" \
    --hist-neg-ratio "${HNR}" \
    --eta-uniform "${eta}" \
    --uniformity-cap "${cap}" \
    --head-mode cross_table \
    --log-debug > "${log}" 2>&1
  rc=$?
  echo "  exit=${rc}  log=${log}"
  tail -12 "${log}"
done

echo "=== Stage 5 DONE ==="
} 2>&1 | tee "${summary}"
