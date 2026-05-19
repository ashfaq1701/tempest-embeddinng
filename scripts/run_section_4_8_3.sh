#!/bin/bash
# §4.8.3 — Long-training plateau validation on the locked architecture.
#
# Usage:
#   bash scripts/run_section_4_8_3.sh <tgb-name> <seed> <ts-tag> \
#       <primary> <nb> <jl> <nl> <link_dr> <emb_dr> <normbrake_threshold>
#
# Runs the final-locked configuration for 100 epochs with NO early stopping
# and --log-debug to track per-epoch val/test/col-norm/grad-norm.
set -e
TGB="${1:-tgbl-wiki}"
SEED="${2:-42}"
TS="${3:-$(date +%Y%m%d_%H%M%S)}"
PRIMARY="${4:-triplet}"
NB="${5:-0}"
JL="${6:-0}"
NL="${7:-3}"
LDR="${8:-0.0}"
EDR="${9:-0.0}"
NB_THR="${10:-3.87}"

mkdir -p runs
log="runs/sec483_long_${TGB}_${PRIMARY}_seed${SEED}_${TS}.log"
echo "=== §4.8.3 long-training — ${TGB} seed=${SEED} ts=${TS} ==="
echo "  primary=${PRIMARY} nb=${NB} λ_link=${JL} n_layers=${NL} link_dr=${LDR} emb_dr=${EDR}"
echo "  log=${log}"
.venv/bin/python -u -m scripts.train --tgb-name "${TGB}" --use-gpu \
  --num-epochs 100 --early-stop-patience 999 --seed "${SEED}" \
  --primary-loss "${PRIMARY}" --lambda-normbrake "${NB}" \
  --normbrake-threshold "${NB_THR}" \
  --lambda-link "${JL}" \
  --link-mlp-n-layers "${NL}" \
  --link-mlp-dropout "${LDR}" \
  --embedding-dropout "${EDR}" \
  --head-mode cross_table \
  --log-debug > "${log}" 2>&1
echo "exit=$?  log=${log}"
tail -15 "${log}"
