#!/bin/bash
# Auto-calibrating review sweep (TIGHTENED for ~8-hr budget):
#   1. 2-ep alignment anchor on review (sampled eval) to measure col-norm.
#   2. threshold = 1.5 × measured col-norm.
#   3. Launch 6-cell sweep:
#        alignment × {no nb, +nb}
#        triplet × {no nb, +nb}
#        sgns × {no nb, +nb}
#      InfoNCE dropped — definitively rejected on wiki (joint training
#      monotonically worsened; cliff drop 0.025 -> 0.151 as λ_link rose
#      0 -> 1; loss family is fundamental wrong for this task).
#   4. Each cell: --num-epochs 6, --early-stop-patience 2,
#      --monitor-sample-pct 0.05, final full eval at end.
#
# Usage:
#   bash scripts/run_review_sweep_auto.sh [seed=42] [ts-tag] [sample_pct=0.05]
set -e
SEED="${1:-42}"
TS="${2:-$(date +%Y%m%d_%H%M%S)}"
SAMPLE="${3:-0.05}"
mkdir -p runs

# Step 1 — anchor calibration (sampled eval for speed)
cal_log="runs/review_calib_seed${SEED}_${TS}.log"
echo "=== Step 1: review anchor calibration (2 ep, alignment, sampled=${SAMPLE}) → ${cal_log} ==="
.venv/bin/python -u -m scripts.train --tgb-name tgbl-review --use-gpu \
  --num-epochs 2 --early-stop-patience 5 --seed "${SEED}" \
  --primary-loss alignment \
  --monitor-sample-pct "${SAMPLE}" \
  --skip-final-full-eval > "${cal_log}" 2>&1
echo "exit=$?  log=${cal_log}"
tail -25 "${cal_log}"

# Step 2 — extract calibrated threshold from per_epoch_col_norm
COL_NORM=$(grep -E "per_epoch_col_norm" "${cal_log}" | tail -1 | sed 's/.*: //' | awk -F',' '{print $NF}' | tr -d ' ')
if [ -z "${COL_NORM}" ]; then
  echo "ERROR: could not read per_epoch_col_norm from ${cal_log}"
  exit 1
fi
NB_THR=$(.venv/bin/python -c "print(round(1.5 * float('${COL_NORM}'), 4))")
echo "=== Calibrated: col_norm=${COL_NORM}  normbrake_threshold=${NB_THR} ==="

# Step 3 — 6-cell sweep (alignment × 2 + triplet × 2 + sgns × 2)
summary="runs/sweep_tgbl-review_seed${SEED}_${TS}_SUMMARY.log"
{
  echo "=== Review loss sweep — seed=${SEED} ts=${TS} ==="
  echo "  num_epochs=6  patience=2  sample=${SAMPLE}  normbrake_threshold=${NB_THR}"
  for spec in \
    "A:alignment:0:cellA_alignment" \
    "A_nb:alignment:0.1:cellA_alignment_nb" \
    "T:triplet:0:cellT_triplet" \
    "T_nb:triplet:0.1:cellT_triplet_nb" \
    "S:sgns:0:cellS_sgns" \
    "S_nb:sgns:0.1:cellS_sgns_nb"; do
    IFS=':' read cellid loss nb tag <<< "$spec"
    log="runs/sweep_tgbl-review_${tag}_seed${SEED}_${TS}.log"
    echo "--- Cell ${cellid}: ${tag} (loss=${loss} nb=${nb}) → ${log} ---"
    .venv/bin/python -u -m scripts.train --tgb-name tgbl-review --use-gpu \
      --num-epochs 6 --early-stop-patience 2 --seed "${SEED}" \
      --primary-loss "${loss}" --lambda-normbrake "${nb}" \
      --normbrake-threshold "${NB_THR}" \
      --head-mode cross_table \
      --monitor-sample-pct "${SAMPLE}" \
      --skip-final-full-eval > "${log}" 2>&1
    rc=$?
    echo "Cell ${cellid} exit=${rc}  log=${log}"
    tail -10 "${log}"
  done
  echo "=== REVIEW SWEEP DONE: seed=${SEED} ==="
} 2>&1 | tee "${summary}"
