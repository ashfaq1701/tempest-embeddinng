#!/bin/bash
# Stage 4 — hist_neg_ratio × joint-training sweep, 50 ep no early-stop on
# wiki seed 42.
#
# Hypothesis (user 2026-05-20, post-L0.1 collapse): joint training (λ_link>0)
# fails because historical negatives in BCE fight alignment supervision.
# Alignment pulls target(u) toward context(v_hist); BCE on hist negative
# (u, v_hist) pushes target(u) away from context(v_hist). Destructive
# interference at the same embedding pairs.
#
# Critical test: J_hnr0 (joint λ=0.1, hist_neg_ratio=0, pure random negs).
# If hypothesis right, this should NOT collapse — random negs are unlikely
# to be in u's walk history.
#
# Also samples the decoupled column to check the existing hist_neg_ratio=0.5
# default isn't accidentally suboptimal.
#
# All cells: alignment + normbrake (λ=0.1, threshold=3.87) +
# weight_decay_link=1e-4 (Stage 3 BREAKTHROUGH — closes residual cliff;
# val drop -0.014 vs nb-only -0.11). This is the locked production base.
# 50 ep, patience=999, --log-debug, seed 42.
set -e
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p runs

summary="runs/stage4_${TS}_SUMMARY.log"
{
echo "=== Stage 4 hist_neg_ratio × λ_link sweep — ts=${TS} ==="
echo "  base: alignment + normbrake λ=0.1 threshold=3.87 + WD_link=1e-4"
echo "  num_epochs=50  early-stop disabled  --log-debug  seed=42"

# 8 cells: 2×4 grid (λ_link ∈ {0, 0.1}) × (hist_neg_ratio ∈ {0, 0.25, 0.5, 0.75})
#
# spec format: tag:lambda_link:hist_neg_ratio
for spec in \
  "D_hnr0:0.0:0.0" \
  "D_hnr0.25:0.0:0.25" \
  "D_hnr0.5:0.0:0.5" \
  "D_hnr0.75:0.0:0.75" \
  "J_hnr0:0.1:0.0" \
  "J_hnr0.25:0.1:0.25" \
  "J_hnr0.5:0.1:0.5" \
  "J_hnr0.75:0.1:0.75"; do
  IFS=':' read tag ll hnr <<< "$spec"
  log="runs/stage4_${tag}_${TS}.log"
  echo "--- ${tag} (λ_link=${ll}  hist_neg_ratio=${hnr}) → ${log} ---"
  .venv/bin/python -u -m scripts.train --tgb-name tgbl-wiki --use-gpu \
    --num-epochs 50 --early-stop-patience 999 --seed 42 \
    --primary-loss alignment \
    --lambda-normbrake 0.1 --normbrake-threshold 3.87 \
    --weight-decay-link 1e-4 \
    --lambda-link "${ll}" \
    --hist-neg-ratio "${hnr}" \
    --head-mode cross_table \
    --log-debug > "${log}" 2>&1
  rc=$?
  echo "  exit=${rc}  log=${log}"
  tail -12 "${log}"
done

echo "=== Stage 4 DONE ==="
} 2>&1 | tee "${summary}"
