#!/bin/bash
# Multi-seed validation of a locked configuration.
#
# Usage:
#   bash scripts/run_multiseed.sh <tgb-name> <ts-tag> <primary> <nb> <jl> \
#       <nl> <link_dr> <emb_dr> <normbrake_threshold>
#
# Runs the same config on seeds {42, 7, 13} sequentially.
set -e
TGB="${1:-tgbl-wiki}"
TS="${2:-$(date +%Y%m%d_%H%M%S)}"
PRIMARY="${3:-triplet}"
NB="${4:-0}"
JL="${5:-0}"
NL="${6:-3}"
LDR="${7:-0.0}"
EDR="${8:-0.0}"
NB_THR="${9:-3.87}"

mkdir -p runs
summary="runs/multiseed_${TGB}_${PRIMARY}_${TS}_SUMMARY.log"
{
  echo "=== Multi-seed — ${TGB} ts=${TS} ==="
  echo "  primary=${PRIMARY} nb=${NB} λ_link=${JL} n_layers=${NL} link_dr=${LDR} emb_dr=${EDR}"
  for SEED in 42 7 13; do
    log="runs/multiseed_${TGB}_${PRIMARY}_seed${SEED}_${TS}.log"
    echo "--- seed ${SEED} → ${log} ---"
    .venv/bin/python -u -m scripts.train --tgb-name "${TGB}" --use-gpu \
      --num-epochs 50 --early-stop-patience 5 --seed "${SEED}" \
      --primary-loss "${PRIMARY}" --lambda-normbrake "${NB}" \
      --normbrake-threshold "${NB_THR}" \
      --lambda-link "${JL}" \
      --link-mlp-n-layers "${NL}" \
      --link-mlp-dropout "${LDR}" \
      --embedding-dropout "${EDR}" \
      --head-mode cross_table > "${log}" 2>&1
    echo "  seed ${SEED} exit=$?"
    tail -7 "${log}"
  done
  echo "=== MULTI-SEED DONE ==="
} 2>&1 | tee "${summary}"
