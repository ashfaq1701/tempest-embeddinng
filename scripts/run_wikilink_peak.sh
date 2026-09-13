#!/bin/bash
# WikiLink d=64 K=5 on the SETTLED head, run to CONVERGENCE to find the pooler's peak.
# No local edits -- HEAD fully describes this run.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/wikilink_peak/d64_k5/run_1/WikiLink.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset WikiLink \
--d-emb 64 --k-train 5 --lr-embedding 1e-3 --lr-network 1e-3 --num-epochs 50 --early-stop-patience 3 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=WikiLink (NON-bipartite) d_emb=64 k_train=5 lr=1e-3  NO pop bias  seed=42"
  echo "# experiment=wikilink_peak cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA alone does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0, ONE optimizer group"
  echo "#       pooling = softmax(MLP([rec, pos, rad])), 3 -> 32 -> 1"
  echo "#       162 head params (161 pooler + 1 geo_temp)"
  echo "#"
  echo "# PURPOSE: find the PEAK. The only prior NN-pooler WikiLink run was KILLED at ep4 while"
  echo "#   still rising, so 'the pooler helps WikiLink' has never been settled at convergence."
  echo "# That killed run (commit 5b953ac1, hidden=8*N_FEAT=24, 121 pooler params) gave:"
  echo "#   ep1 0.4455  ep2 0.5551  ep3 0.5883  ep4 0.6050   val rose monotonically throughout"
  echo "#   per-epoch gains halved cleanly (+0.110, +0.033, +0.017) -> crude asymptote ~0.62"
  echo "# NOTE this run is NOT a strict continuation: hidden is 32 here vs 24 there (161 vs 121"
  echo "#   pooler params). Same features, same everything else."
  echo "#"
  echo "# WikiLink references (d=64, K=5, lr=1e-3, seed 42, no pop), FIXED parameter-free pooler:"
  echo "#   1-param pooling temp + linear geo_temp : 0.5828 @ ep7  (converged)"
  echo "#   NO pooling temp      + linear geo_temp : 0.5430 @ ep7  (converged, stop ep10)  <- BASELINE"
  echo "#   NO pooling temp      + EXP geo_temp    : 0.6108 @ ep8  (val-peaked, best 1-param)"
  echo "#   LB #1 TGN = 0.6294.  Our record 0.7904 @ ep44 (a much larger head)."
  echo "#"
  echo "# WHY THIS RUN MATTERS: on YouTube the SAME pooler LOSES to the parameter-free rule"
  echo "#   (best 0.5605 vs 0.5677 K=5 / 0.5756 K=10), and three capacity probes there -- width 64,"
  echo "#   depth 2, and the dev feature -- all failed to help, so the YouTube pooler is not"
  echo "#   capacity-limited. WikiLink appears to disagree. The curve shapes differ too: YouTube"
  echo "#   grinds ~12 epochs then JUMPS +0.15 (an 'escape'); WikiLink decelerates smoothly from"
  echo "#   ep1 with no escape at all. If WikiLink converges above 0.5430 the disagreement is real"
  echo "#   and dataset-conditional; if it lands at or below, the ep1-4 lead was a warmup artifact"
  echo "#   -- exactly the trap that made dev look good on YouTube before it finished third."
  echo "#"
  echo "# WATCH: (a) the peak and its epoch; (b) whether a val/test divergence opens -- on YouTube"
  echo "#   every arm lost 0.001-0.005 between its true peak and its val-selected checkpoint;"
  echo "#   (c) epoch count -- the fixed-pooler baselines here converge by ep7-10."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~31 min/epoch: expect 5-8 h to convergence."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
