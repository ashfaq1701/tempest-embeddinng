#!/usr/bin/env bash
# Phase 0 — direction sweep on V0 (full head), 15 epochs each.
# Sequential: V0_fwd → V0_bwd → V0_both. ~4 h each → ~12 h total.
#
# Each run uses the iter-6 embedding-loss recipe (always-on forward
# alignment + inverse-degree seed weighting). d_emb=128 + chunk_C=8
# is the memory budget that fits an 8 GB GPU with the walk-mediated
# head's per-position broadcast.
set -euo pipefail
cd "$(dirname "$0")/.."

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

run V0_fwd  --link-head-direction forward
run V0_bwd  --link-head-direction backward
run V0_both --link-head-direction both

echo "Phase 0 complete."
