#!/bin/bash
# YouTube d=64 K=5 with a DEPTH-2 NN pooler: 3 -> 32 -> 32 -> 1 (1217 pooler params).
# The second hidden layer is a LOCAL, UNCOMMITTED edit to link_property_prediction/model.py --
# the committed code at HEAD is depth 1. See the log header for the exact diff.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/pooler_depth2/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -5)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr-embedding 1e-3 --lr-network 1e-3 --num-epochs 50 --early-stop-patience 3 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5 lr=1e-3  NO pop bias  seed=42"
  echo "# experiment=pooler_depth2 cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  echo "#"
  echo "# *** THE SHA DOES NOT DESCRIBE THE CODE THAT RAN ***"
  echo "# The working tree carries an UNCOMMITTED change; git status --porcelain says:"
  echo "$DIRTY" | sed 's/^/#   /'
  echo "# The change, in BagWeights.__init__, is exactly this -- one extra (hidden, hidden)"
  echo "# block plus a GELU, turning the pooler MLP from depth 1 into depth 2:"
  echo "#     self.net = nn.Sequential(nn.Linear(self.N_FEAT, self.hidden), nn.GELU(),"
  echo "#  +                           nn.Linear(self.hidden, self.hidden), nn.GELU(),"
  echo "#                              nn.Linear(self.hidden, 1))"
  echo "# Nothing else differs from $SHA. To reproduce: check out $SHA and apply that one edit."
  echo "# It is deliberately uncommitted -- the depth knob was reverted in 19eb15f3 because it"
  echo "#   is a test in progress, not a setting the model should carry."
  echo "#"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0, ONE optimizer group"
  echo "#       pooling = softmax(MLP([rec, pos, rad])), 3 -> 32 -> 32 -> 1"
  echo "#       1218 head params (1217 pooler + 1 geo_temp)  vs 162 at depth 1"
  echo "# THE BASELINE -- same dataset/config, depth-1 pooler, commit e6b6079c, arm f_rec_pos_rad:"
  echo "#   test@val-ckpt 0.5551 (ep27)   MAX test 0.5605 (ep18)   escape ep13   ran 30 ep"
  echo "#   Its escape steps were +0.037, +0.069, +0.040 at ep13-15."
  echo "# Depth 2 is 7.6x the pooler params (161 -> 1217). It also draws DIFFERENT RNG than the"
  echo "#   depth-1 arm, so per the feature-ablation caveat a gap under ~0.01 is not separable"
  echo "#   from init luck. Only a clear move off 0.5605 counts."
  echo "# Other YouTube references (d=64, K=5 unless noted, seed 42, no pop):"
  echo "#   fixed parameter-free pooler + ptemp, K=5  : 0.5677 (still rising at the 50-ep cap)"
  echo "#   fixed parameter-free pooler + ptemp, K=10 : 0.5756 (36 ep, converged)"
  echo "#   LB #1 YouTube = GraphMixer 0.5887."
  echo "# WATCH: (a) does the escape fire earlier/later than ep13, and from what plateau;"
  echo "#        (b) val/test drift -- the depth-1 arm lost 0.0054 between its ep18 peak and its"
  echo "#            ep27 checkpoint to +0.0006 val flickers resetting patience."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'best_val_mrr|best_test_mrr|stopped_at_epoch' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
