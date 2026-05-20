#!/bin/bash
# Autonomous chain — runs review sweep + architectural fix sweeps focused on
# the user's goal: smooth loss decrease + smooth MRR increase across 50
# epochs (no rapid-breakdown after early epochs).
#
# Total budget: ~13 hours (within 14-hr cap).
#
# Stages (sequential — GPU is single):
#   1. Review sweep: 6 cells (alignment×2 + triplet×2 + sgns×2), 6 ep, p=2,
#      5% sample. Wall ~8 hr.
#   2. Alignment + architectural fixes, LONG TRAINING (50 ep, no early stop)
#      on wiki — characterizes whether dropout / deeper MLP / normbrake fix
#      the 50-epoch alignment cliff. 5 cells × ~25-30 min = ~2.5 hr.
#   3. Triplet + architectural fixes, normal early-stop on wiki — tests
#      whether deeper/dropout lift the ceiling. 4 cells × ~10 min = ~40 min.
#   4. Generate final summary + smoothness rankings.
set -e
TS=$(date +%Y%m%d_%H%M%S)
mkdir -p runs
MASTER="runs/autonomous_chain_${TS}.log"

wait_gpu_free() {
  echo "[$(date '+%H:%M:%S')] waiting for GPU..."
  local i=0
  while pgrep -f "scripts.train.*--use-gpu" > /dev/null 2>&1; do
    sleep 60
    i=$((i+1))
    if [ $i -ge 30 ]; then
      echo "[$(date '+%H:%M:%S')] GPU still busy after 30 min — proceeding"
      return
    fi
  done
  echo "[$(date '+%H:%M:%S')] GPU free"
}

{
echo "=== autonomous chain start ts=${TS} ==="

# ---------------------------------------------------------------------- #
# STAGE 1: review sweep (6 cells, ~8 hr)
# ---------------------------------------------------------------------- #
wait_gpu_free
echo "[$(date '+%H:%M:%S')] STAGE 1: review sweep"
bash scripts/run_review_sweep_auto.sh 42 "${TS}_review" 0.05
echo "[$(date '+%H:%M:%S')] STAGE 1: done"

# ---------------------------------------------------------------------- #
# STAGE 2: alignment + architectural fixes on wiki, LONG-TRAINING 50 ep
# (no early stop) to expose the cliff and test whether each fix smooths it.
# ---------------------------------------------------------------------- #
wait_gpu_free
echo "[$(date '+%H:%M:%S')] STAGE 2: alignment long-training fixes (50 ep, no early stop)"
for spec in \
  "A_long_nb:0.1:3:0.0:0.0" \
  "A_long_dr0.3:0:3:0.3:0.0" \
  "A_long_ed0.3:0:3:0.0:0.3" \
  "A_long_d5:0:5:0.0:0.0" \
  "A_long_full:0.1:5:0.3:0.3"; do
  IFS=':' read tag nb nl ldr edr <<< "$spec"
  log="runs/long_align_${tag}_${TS}.log"
  echo "[$(date '+%H:%M:%S')] -- ${tag} (nb=${nb} nl=${nl} link_dr=${ldr} emb_dr=${edr}) → ${log}"
  .venv/bin/python -u -m scripts.train --tgb-name tgbl-wiki --use-gpu \
    --num-epochs 50 --early-stop-patience 999 --seed 42 \
    --primary-loss alignment --lambda-normbrake "${nb}" \
    --normbrake-threshold 3.87 \
    --link-mlp-n-layers "${nl}" \
    --link-mlp-dropout "${ldr}" \
    --embedding-dropout "${edr}" \
    --head-mode cross_table \
    --log-debug > "${log}" 2>&1
  echo "  exit=$?  log=${log}"
  tail -10 "${log}"
done

# ---------------------------------------------------------------------- #
# STAGE 3: Triplet + architectural fixes on wiki, normal early-stop.
# Triplet already plateaus cleanly; goal here is to LIFT the ceiling
# (0.7112 → ?) via more capacity in the link MLP.
# ---------------------------------------------------------------------- #
wait_gpu_free
echo "[$(date '+%H:%M:%S')] STAGE 3: Triplet architectural fixes on wiki"
for spec in \
  "T_dr0.1:0:3:0.1:0.0" \
  "T_dr0.3:0:3:0.3:0.0" \
  "T_ed0.1:0:3:0.0:0.1" \
  "T_d5:0:5:0.0:0.0"; do
  IFS=':' read tag nb nl ldr edr <<< "$spec"
  log="runs/sec482_tgbl-wiki_triplet_${tag}_${TS}.log"
  echo "[$(date '+%H:%M:%S')] -- ${tag} (nb=${nb} nl=${nl} link_dr=${ldr} emb_dr=${edr}) → ${log}"
  .venv/bin/python -u -m scripts.train --tgb-name tgbl-wiki --use-gpu \
    --num-epochs 50 --early-stop-patience 5 --seed 42 \
    --primary-loss triplet --lambda-normbrake "${nb}" \
    --normbrake-threshold 3.87 \
    --link-mlp-n-layers "${nl}" \
    --link-mlp-dropout "${ldr}" \
    --embedding-dropout "${edr}" \
    --head-mode cross_table > "${log}" 2>&1
  echo "  exit=$?  log=${log}"
  tail -10 "${log}"
done

# ---------------------------------------------------------------------- #
# STAGE 4: final summary
# ---------------------------------------------------------------------- #
echo "[$(date '+%H:%M:%S')] STAGE 4: final summary"
final="runs/FINAL_SUMMARY_${TS}.txt"
{
  echo "=========================================="
  echo "== Autonomous chain final  ts=${TS}"
  echo "=========================================="
  echo
  echo "--- WIKI §4.7 (loss family + normbrake, no λ_link) — 6 baseline cells ---"
  .venv/bin/python scripts/summarize_runs.py 'runs/lossmining_v2_cell*.log' 2>&1 | head -25 || true
  echo
  echo "--- WIKI §4.7 multi-seed Triplet vs SGNS+nb across {42, 7, 13} ---"
  .venv/bin/python scripts/summarize_runs.py 'runs/lossmining_v2_cell{2,6}_*seed*.log' 2>&1 | head -15 || true
  echo
  echo "--- WIKI §4.8.1 (λ_link sweep on all loss families) ---"
  .venv/bin/python scripts/summarize_runs.py 'runs/sec481_tgbl-wiki_*.log' 2>&1 | head -30 || true
  echo
  echo "--- REVIEW §4.7 ---"
  .venv/bin/python scripts/summarize_runs.py "runs/sweep_tgbl-review_*${TS}_review*.log" 2>&1 | head -25 || true
  echo
  echo "--- WIKI alignment + architectural fixes (50 ep, no early stop) ---"
  .venv/bin/python scripts/summarize_runs.py "runs/long_align_*${TS}*.log" 2>&1 | head -20 || true
  echo
  echo "--- WIKI Triplet + architectural fixes ---"
  .venv/bin/python scripts/summarize_runs.py "runs/sec482_tgbl-wiki_triplet_*${TS}*.log" 2>&1 | head -20 || true
} > "${final}"
cat "${final}"

echo "=== autonomous chain done at $(date '+%H:%M:%S') ==="
} 2>&1 | tee "${MASTER}"
