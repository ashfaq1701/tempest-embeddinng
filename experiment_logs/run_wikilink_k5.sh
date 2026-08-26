#!/bin/bash
# WikiLink cell of the without_cand_pop-5 arm: d=64, K_train=5, lr=1e-3, no pop channel.
# Run on THIS machine; a second A40 runs the other 7 datasets of the same arm
# (see run_arm_k5.sh, whose ORDER deliberately excludes WikiLink).
# WikiLink is NOT bipartite (tgb_seq/datasets/preprocess.py::bipartite_dict) -> no flag.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-wt-masterbr
OUT=$WD/experiment_logs/complete_runs/without_cand_pop-5/WikiLink/run-1
LOG=$OUT/run.log
DRIVER=$WD/experiment_logs/complete_runs/without_cand_pop-5/DRIVER_run-1_wikilink.log
mkdir -p "$OUT"
cd "$WD" || exit 1
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "[$(date '+%F %T')] START WikiLink  d=64 K_train=5 lr=1e-3   (this machine; branch=$BRANCH commit=$SHA)" >> "$DRIVER"
{
  echo "# complete run: dataset=WikiLink arm=without_cand_pop-5 tag=run-1 d_emb=64 k_train=5 lr=1e-3"
  echo "# branch=$BRANCH commit=$SHA"
  echo "# started=$(date '+%F %T')"
  echo "# run on a second A40 in parallel with the other 7 datasets of this arm; sole job on this GPU"
  echo "# cmd: $PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq \\"
  echo "#        --dataset WikiLink --d-emb 64 --k-train 5 --lr 1e-3 \\"
  echo "#        --num-epochs 50 --early-stop-patience 3 \\"
  echo "#        --use-gpu --use-gpu-tempest  "
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $PY -u scripts/train_link_property_prediction.py \
  --data-suite tgb-seq --dataset WikiLink --d-emb 64 --k-train 5 --lr 1e-3 \
  --num-epochs 50 --early-stop-patience 3 \
  --use-gpu --use-gpu-tempest >> "$LOG" 2>&1
RC=$?
echo "[$(date '+%F %T')] DONE  WikiLink rc=$RC  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$DRIVER"
