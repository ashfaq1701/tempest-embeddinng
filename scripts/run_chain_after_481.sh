#!/bin/bash
# Chain runner — waits for §4.8.1 wrapper to finish, then runs:
#   Step 1: alignment+uniformity λ_link cells (A_jl0.3, A_jl1.0)
#   Step 2: review sweep with sampled-eval (3×2 + alignment×2 = 8 cells)
# Each step is GPU-serialised behind the previous.
set -e
ts_global=$(date +%Y%m%d_%H%M%S)
echo "=== chain runner — start ts=${ts_global} ==="

# Wait until the §4.8.1 wrapper no longer has python child training.
echo "[wait] for §4.8.1 wrapper to release GPU..."
while pgrep -f "scripts.train.*tgbl-wiki.*lambda-link" > /dev/null 2>&1; do
  sleep 60
done
echo "[wait] §4.8.1 GPU freed at $(date '+%H:%M:%S')"

# Step 1: alignment λ_link cells (~25 min)
echo "[chain] Step 1: alignment+uniformity λ_link cells"
bash scripts/run_alignment_jl.sh 2>&1 | tee "runs/chain_step1_alignment_jl_${ts_global}.log"

# Step 2: review sweep (~3-5 hr with sampled eval at 5%)
echo "[chain] Step 2: review sweep (auto-calibrated normbrake)"
bash scripts/run_review_sweep_auto.sh 42 "${ts_global}" 0.05 2>&1 | tee "runs/chain_step2_review_sweep_${ts_global}.log"

echo "=== chain runner — done at $(date '+%H:%M:%S') ==="
