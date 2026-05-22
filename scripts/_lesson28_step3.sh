#!/usr/bin/env bash
# Lesson 28 Step 3 verification — three 50-ep wiki cells under both bug fixes.
#
#   W_off_fixed    : encoder OFF, tables train, K=5 (locked default)
#   W_gru_k1_fixed : encoder ON,  tables train, K=1
#   Sanity_fixed   : encoder ON,  tables FROZEN, K=1
#
# Each cell: seed 42, --early-stop-patience 999 (run all 50 epochs),
# --log-debug for per-epoch col_norm + link_w_norm trajectory.
# Logs go to runs/lesson28_step3_<cell>_<stamp>.log.
set -euo pipefail
PY=/home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

COMMON=(
  --tgb-name tgbl-wiki --use-gpu
  --num-epochs 50 --early-stop-patience 999
  --seed 42 --log-debug
)

run_cell() {
  local NAME=$1; shift
  local LOG="runs/lesson28_step3_${NAME}_${STAMP}.log"
  echo "=== ${NAME} → ${LOG} ==="
  echo "    args: $*"
  PYTHONUNBUFFERED=1 "${PY}" -u -m scripts.train "${COMMON[@]}" "$@" > "${LOG}" 2>&1
  echo "=== ${NAME} done ==="
}

# W_off: encoder off; K stays at 5 (the locked default for the
# baseline). e_t_u becomes target(u) — static identity lookup.
run_cell W_off  --no-use-walk-encoder

# W_gru_k1: encoder on (default), tables train (default), K=1.
run_cell W_gru_k1  --num-walks-per-node 1

# Sanity: encoder on, K=1, tables frozen.
run_cell Sanity  --num-walks-per-node 1 --freeze-tables

echo "=== Lesson 28 Step 3 cells complete ==="
