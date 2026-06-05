#!/usr/bin/env bash
# Phase 1 — channel ablations on the Phase-0 direction winner.
# 4 variants, 15 epochs each, sequential. ~4h per variant → ~16h total.
#
# Set WINNER below to the Phase-0 winning direction
# ('forward', 'backward', or 'both') before launching.
#
# Each ablation removes ONE component of V0; comparison to the V0 run
# (rerun here as the baseline) isolates that component's contribution.
set -euo pipefail
cd "$(dirname "$0")/.."

# ────────────────────────────────────────────────────────────────────
# Set this AFTER Phase 0 finishes. Pick the direction with the highest
# best_test_mrr (or best_val if tied).
WINNER=both     # TODO: replace with Phase-0 winner before launching
# ────────────────────────────────────────────────────────────────────

PY=./.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(
    --dataset tgbl-wiki
    --use-gpu --use-gpu-tempest
    --seed 42
    --num-epochs 15
    --early-stop-patience 0
    --batch-size 500
    --eval-batch-size 50
    --d-emb 128
    --link-pred-num-walks-per-node 10
    --link-pred-max-walk-len 20
    --link-head-chunk-c 8
    --link-head-direction "$WINNER"
    --export-best-embedding-table
)

run() {
    TAG=$1; shift
    LOG="logs/iterations/v2_${TAG}_15ep_seed42_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p logs/iterations
    echo "============================================================"
    echo "[$TAG]  → $LOG"
    echo "============================================================"
    "$PY" -u scripts/train.py "${COMMON[@]}" "$@" > "$LOG" 2>&1
    echo "  [$TAG] done"
    grep -E "best_val_mrr|best_test_mrr|saved to" "$LOG" | tail -3
}

# V1 — drop the time channel
run V1_no_time \
    --link-head-no-time-channel

# V2 — replace per-dim primitives with scalar cosine
run V2_cos_only \
    --link-head-sim-primitives cosine_only

# V3 — walks tower OFF, direct (E[u], E[v]) only
run V3_direct_only \
    --link-head-direct-only

# V4 — drop the K (hop) channel
run V4_no_K \
    --link-head-no-K-channel

echo "Phase 1 complete."
