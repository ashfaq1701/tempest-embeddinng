#!/bin/bash
# §4.8.2 — Architectural fix sweep on the §4.8.1 winner.
#
# Usage:
#   bash scripts/run_section_4_8_2.sh <tgb-name> <seed> <ts-tag> \
#       <winning-primary> <winning-nb> <winning-jl> <normbrake_threshold>
#
# Tests deeper link MLP, link dropout, and embedding dropout on top of the
# §4.8.1 best (primary + λ_link) configuration.
set -e
TGB="${1:-tgbl-wiki}"
SEED="${2:-42}"
TS="${3:-$(date +%Y%m%d_%H%M%S)}"
PRIMARY="${4:-triplet}"
NB="${5:-0}"
JL="${6:-0}"
NB_THR="${7:-3.87}"

mkdir -p runs
summary="runs/sec482_${TGB}_${PRIMARY}_seed${SEED}_${TS}_SUMMARY.log"
{
  echo "=== §4.8.2 architectural sweep — dataset=${TGB} seed=${SEED} ts=${TS} ==="
  echo "Base config: primary=${PRIMARY} normbrake=${NB} lambda_link=${JL}"
  for spec in \
    "A_d5:5:0.0:0.0" \
    "A_dr0.1:3:0.1:0.0" \
    "A_dr0.3:3:0.3:0.0" \
    "A_ed0.1:3:0.0:0.1" \
    "A_ed0.3:3:0.0:0.3" \
    "A_d5dr0.1:5:0.1:0.0" \
    "A_d5ed0.1:5:0.0:0.1"; do
    IFS=':' read tag nl dr ed <<< "$spec"
    log="runs/sec482_${TGB}_${PRIMARY}_${tag}_seed${SEED}_${TS}.log"
    echo "--- ${tag} (n_layers=${nl} link_dropout=${dr} emb_dropout=${ed}) → ${log} ---"
    .venv/bin/python -u -m scripts.train --tgb-name "${TGB}" --use-gpu \
      --num-epochs 50 --early-stop-patience 5 --seed "${SEED}" \
      --primary-loss "${PRIMARY}" --lambda-normbrake "${NB}" \
      --normbrake-threshold "${NB_THR}" \
      --lambda-link "${JL}" \
      --link-mlp-n-layers "${nl}" \
      --link-mlp-dropout "${dr}" \
      --embedding-dropout "${ed}" \
      --head-mode cross_table > "${log}" 2>&1
    rc=$?
    echo "${tag} exit=${rc}"
    tail -7 "${log}"
  done
  echo "=== §4.8.2 SWEEP DONE: ${TGB} seed=${SEED} ${PRIMARY} ==="
} 2>&1 | tee "${summary}"
