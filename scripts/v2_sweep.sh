#!/usr/bin/env bash
# Sweep schedule for the v2 link-prediction head.
# Six runs total:
#   Phase 0 — direction sweep on V0 (full head): forward / backward / both.
#             Winner becomes the "final V0" carried into Phase 1.
#   Phase 1 — ablations on final V0: -time, -primitives→cos, -walks→direct-only.
#
# Each run uses the iter-6 embedding-loss recipe (forward-walks alignment +
# inverse-degree seed weighting always on; d_emb=256). The head's compute
# scales with eval batch size × candidates, so for wiki keep eval batch
# small (default 50 → may need 25 or even 16).
#
# Per-run cost: ~110 min (d_emb=256, 50 epochs). Total: ~11 hours.
#
# Output:
#   logs/iterations/v2_<TAG>_seed42_<timestamp>.log
#   logs/embeddings/tgbl-wiki_seed42_demb256_v2_<TAG>_ep<best_val>.npy
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
COMMON=(
    --dataset tgbl-wiki
    --use-gpu --use-gpu-tempest
    --seed 42
    --num-epochs 50
    --early-stop-patience 0
    --batch-size 500
    --eval-batch-size 25
    --d-emb 256
    --link-pred-num-walks-per-node 10
    --link-pred-max-walk-len 20
    --export-best-embedding-table
)

run() {
    TAG=$1; shift
    LOG="logs/iterations/v2_${TAG}_seed42_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p logs/iterations
    echo "[$TAG]  → $LOG"
    nohup "$PY" -u scripts/train.py "${COMMON[@]}" "$@" > "$LOG" 2>&1 &
    PID=$!
    echo "  PID $PID"
    wait $PID
}

# ──────────────────────────────────────────────────────────────────────
# Phase 0 — direction sweep on V0 (full head)
# ──────────────────────────────────────────────────────────────────────

# V0-fwd: 10 walks forward only.
run V0_fwd \
    --link-head-direction forward

# V0-bwd: 10 walks backward only.
run V0_bwd \
    --link-head-direction backward

# V0-both: 5 walks forward + 5 walks backward (head splits internally).
run V0_both \
    --link-head-direction both

# Inspect best of the three; set WINNER=<fwd|bwd|both> below.
# WINNER=$(...)

# ──────────────────────────────────────────────────────────────────────
# Phase 1 — ablations on final V0
# Set WINNER and re-run only Phase 1 runs after Phase 0 lands.
# ──────────────────────────────────────────────────────────────────────

WINNER=both  # placeholder; update after Phase 0 results

# V1: V0 minus the per-position time channel.
run V1_no_time \
    --link-head-direction "$WINNER" \
    --link-head-no-time-channel

# V2: V0 with scalar cosine instead of Hadamard+absdiff primitives.
run V2_cos_only \
    --link-head-direction "$WINNER" \
    --link-head-sim-primitives cosine_only

# V3: walks tower OFF — direct (E[u], E[v]) bypass only.
run V3_direct_only \
    --link-head-direct-only
