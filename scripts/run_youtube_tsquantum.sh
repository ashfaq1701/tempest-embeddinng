#!/bin/bash
# YouTube d=64 K=5, encoded NN pooler, single lr, with the TIMESTAMP-GRID FLOOR on the
# TimeEncoding ladder (7c7c112a). Clean A/B: identical to logs/encoded_single_lr/ in every
# respect except lam_min.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/encoded_tsquantum/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE param group)  NO pop bias  seed=42"
  echo "# experiment=encoded_tsquantum cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([TimeEncoding(age) | pos_embedding | rad])), 762 head params"
  echo "#       encoder dims: time_dim=16 pos_dim=4 hidden_dim=32 (defaults)"
  echo "#"
  echo "# THE ONE VARIABLE: TimeEncoding's shortest wavelength."
  echo "#   before (b86d2080): lam_min = 1e-4 * T_train = 1,702 s = 0.020 d"
  echo "#   after  (7c7c112a): lam_min = 2.5 * ts_quantum = 216,000 s = 2.50 d"
  echo "#   YouTube's train split has 175 distinct timestamps, 24 h apart -- daily snapshots, so"
  echo "#   no two ages differ by less than a day. The old bottom rung was 50x finer than"
  echo "#   anything the data can express: 4 of 8 frequencies sat below Nyquist and emitted a"
  echo "#   deterministic hash of the day index (lag-1 autocorr +0.13, -0.73, -0.57) while"
  echo "#   looking healthy by variance (std 0.69, 0.52, 0.73). Now 0 of 8 are below Nyquist."
  echo "#   EVERYTHING ELSE IS IDENTICAL to the baseline: same seed, d, K, lr, patience, dims."
  echo "#"
  echo "# BASELINE -- logs/encoded_single_lr/d64_k5/run_1, commit b86d2080, aliased ladder:"
  echo "#   escape at ep20 (0.4624 -> 0.5077 -> 0.5250), stop ep42"
  echo "#   test@val-ckpt 0.5487   max test 0.5487   best val 0.6483"
  echo "#"
  echo "# OTHER YOUTUBE REFERENCES (d=64 K=5 seed 42):"
  echo "#   raw-scalar pooler, 1 lr group : 0.5551 ckpt, 0.5605 max, escape ep14, stop ep27"
  echo "#   raw-scalar pooler, 2 lr groups: 0.5564 ckpt, 0.5585 max, escape ep7,  stop ep12"
  echo "#   BEST EVER learned pooler here : 0.5645 (depth 3, mnia, 2 groups; arm was cut short)"
  echo "#   parameter-free pooling rule   : 0.5677 (K=5), 0.5756 (K=10)   <- the bar to clear"
  echo "#   LB #1 GraphMixer              : 0.5887"
  echo "#   The learned pooler has NEVER beaten the parameter-free rule on YouTube."
  echo "#"
  echo "# WHAT WOULD COUNT: the baseline lost 8 of its 15 periodic channels to noise, so the MLP"
  echo "#   had to learn to ignore half its input. If that cost anything, this run should escape"
  echo "#   EARLIER than ep20 and/or plateau higher than 0.5487. A null result is informative"
  echo "#   too -- it would say the pooler was never input-limited on this dataset, which is"
  echo "#   what three earlier capacity probes (width 64, depth 2, the dev feature) also found."
  echo "# WATCH: (a) the printed ts_quantum (expect 86400.0) -- confirms the floor engaged;"
  echo "#   (b) escape epoch vs ep20; (c) plateau vs 0.5487, then 0.5605, then 0.5677;"
  echo "#   (d) val/test drift -- the baseline's val kept creeping to ep42 while test sat flat."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~100 s/epoch; the baseline ran 47 epochs in ~80 min."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
