#!/usr/bin/env bash
# Embedding-direction ablation. 2 runs:
#   E0_both  — default (5 fwd + 5 bwd alignment)
#   E0_bwd   — backward only (10 bwd alignment)
# Fair compute: both runs spend the same total K=10 walks/seed.
# 15-epoch cap with --early-stop-patience 5 (stop after 5 consecutive
# non-improving epochs).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON=(
    --dataset tgbl-wiki
    --use-gpu --use-gpu-tempest
    --seed 42
    --num-epochs 15
    --early-stop-patience 5
    --batch-size 500
    --eval-batch-size 50
    --d-emb 128
    --embedding-num-walks-per-node 10
    --embedding-max-walk-len 20
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

run E0_both --embedding-direction both
run E0_bwd  --embedding-direction backward

echo "Embedding-direction sweep complete."
