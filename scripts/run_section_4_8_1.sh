#!/bin/bash
# §4.8.1 — Joint training (λ_link) sweep on cliff-prone configurations.
#
# Usage:
#   bash scripts/run_section_4_8_1.sh <tgb-name> <seed> <ts-tag> <normbrake_threshold>
#
# Tests whether BCE-into-embeddings stabilises the InfoNCE rapid-breakdown
# and SGNS+nb seed-sensitivity observed in the §4.7 wiki sweep.
set -e
TGB="${1:-tgbl-wiki}"
SEED="${2:-42}"
TS="${3:-$(date +%Y%m%d_%H%M%S)}"
NB_THR="${4:-3.87}"

mkdir -p runs
summary="runs/sec481_${TGB}_seed${SEED}_${TS}_SUMMARY.log"
{
  echo "=== §4.8.1 λ_link sweep — dataset=${TGB} seed=${SEED} ts=${TS} ==="
  # On the cliff-prone configs: InfoNCE alone, InfoNCE+nb, SGNS+nb
  # Plus Triplet sweep for completeness (does joint training help even the stable winner?)
  for spec in \
    "I_jl0.0:infonce:0:0.0" \
    "I_jl0.1:infonce:0:0.1" \
    "I_jl0.3:infonce:0:0.3" \
    "I_jl1.0:infonce:0:1.0" \
    "T_jl0.3:triplet:0:0.3" \
    "T_jl1.0:triplet:0:1.0" \
    "S_jl0.0:sgns:0.1:0.0" \
    "S_jl0.3:sgns:0.1:0.3" \
    "S_jl1.0:sgns:0.1:1.0" \
    "A_jl0.3:alignment:0:0.3" \
    "A_jl1.0:alignment:0:1.0"; do
    IFS=':' read tag loss nb jl <<< "$spec"
    log="runs/sec481_${TGB}_${tag}_seed${SEED}_${TS}.log"
    echo "--- ${tag} (loss=${loss} nb=${nb} λ_link=${jl}) → ${log} ---"
    .venv/bin/python -u -m scripts.train --tgb-name "${TGB}" --use-gpu \
      --num-epochs 50 --early-stop-patience 5 --seed "${SEED}" \
      --primary-loss "${loss}" --lambda-normbrake "${nb}" \
      --normbrake-threshold "${NB_THR}" \
      --lambda-link "${jl}" \
      --head-mode cross_table > "${log}" 2>&1
    rc=$?
    echo "${tag} exit=${rc}"
    tail -7 "${log}"
  done
  echo "=== §4.8.1 SWEEP DONE: ${TGB} seed=${SEED} ==="
} 2>&1 | tee "${summary}"
