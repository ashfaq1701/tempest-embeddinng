#!/bin/bash
# Generic loss-family sweep wrapper.
#
# Usage:
#   bash scripts/run_loss_sweep.sh <tgb-name> <seed> <normbrake-threshold> <ts-tag> [num_epochs] [extra-args...]
#
# Runs all 4 primary losses × {with normbrake, without normbrake} = 8 cells.
# Plus 1 cell for `alignment + normbrake` (to compare the anchor against the
# normbrake-augmented baseline — relevant only when calibration target is the
# anchor itself).
set -e
TGB="${1}"
SEED="${2}"
NB_THR="${3}"
TS="${4}"
NUM_EPOCHS="${5:-50}"
shift 5 || true
EXTRA="$@"

mkdir -p runs
summary="runs/sweep_${TGB}_seed${SEED}_${TS}_SUMMARY.log"
{
  echo "=== Loss sweep — dataset=${TGB} seed=${SEED} ts=${TS} ==="
  echo "  num_epochs=${NUM_EPOCHS}  normbrake_threshold=${NB_THR}  extra=${EXTRA}"
  for spec in \
    "A:alignment:0:cellA_alignment" \
    "A_nb:alignment:0.1:cellA_alignment_nb" \
    "1:infonce:0:cell1_infonce" \
    "2:triplet:0:cell2_triplet" \
    "3:sgns:0:cell3_sgns" \
    "4:infonce:0.1:cell4_infonce_nb" \
    "5:triplet:0.1:cell5_triplet_nb" \
    "6:sgns:0.1:cell6_sgns_nb"; do
    IFS=':' read cellid loss nb tag <<< "$spec"
    log="runs/sweep_${TGB}_${tag}_seed${SEED}_${TS}.log"
    echo "--- Cell ${cellid}: ${tag} (loss=${loss} nb=${nb}) → ${log} ---"
    .venv/bin/python -u -m scripts.train --tgb-name "${TGB}" --use-gpu \
      --num-epochs "${NUM_EPOCHS}" --early-stop-patience 5 --seed "${SEED}" \
      --primary-loss "${loss}" --lambda-normbrake "${nb}" \
      --normbrake-threshold "${NB_THR}" \
      --head-mode cross_table ${EXTRA} > "${log}" 2>&1
    rc=$?
    echo "Cell ${cellid} exit=${rc}  log=${log}"
    tail -7 "${log}"
  done
  echo "=== SWEEP DONE: ${TGB} seed=${SEED} ==="
} 2>&1 | tee "${summary}"
