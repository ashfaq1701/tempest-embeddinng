#!/bin/bash
# YouTube d=64 K=5, encoded pooler, single lr, floored ladder, and NO GELU in the pooler.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/encoded_nogelu/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE param group)  NO pop bias  seed=42"
  echo "# experiment=encoded_nogelu cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H(P_u, P_v)), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([TimeEncoding(age) | pos_embedding | rad])), 762 head params"
  echo "#       encoder dims: time_dim=16 pos_dim=4 hidden_dim=32 (defaults)"
  echo "#"
  echo "# THE ONE VARIABLE: the GELU in the pooler MLP is REMOVED."
  echo "#   The two Linears then compose into a single affine map, and hidden (32) >= n_feat (21)"
  echo "#   so there is no rank bottleneck: the pooler is exactly softmax(w . feat + b) --"
  echo "#   737 stored parameters carrying 22 degrees of freedom. Verified equal to an explicit"
  echo "#   affine map to 1e-5. Everything else matches logs/encoded_tsquantum/ exactly:"
  echo "#   same commit lineage (6ffc070c is 7c7c112a + this one edit), seed, d, K, lr, patience,"
  echo "#   time_dim/pos_dim/hidden_dim, and the ts_quantum ladder floor."
  echo "#"
  echo "# WHAT IS AND IS NOT LOST: expressive power in AGE survives -- the time features are a"
  echo "#   fixed Fourier basis, so a linear readout of them is a Fourier series in age, able to"
  echo "#   represent a rich recency curve. What dies is INTERACTION: the pooler can no longer"
  echo "#   make its use of recency depend on the token position or its hyperbolic radius."
  echo "#   So this run asks: does the YouTube pooler need cross-feature interaction at all?"
  echo "#"
  echo "# BASELINES (identical config except where noted):"
  echo "#   floored ladder + GELU  (7c7c112a): ckpt 0.5456  max 0.5457  escape ep19-20  ran 50 ep"
  echo "#   aliased ladder + GELU  (b86d2080): ckpt 0.5487  max 0.5487  escape ep20     stop ep42"
  echo "#"
  echo "# OTHER YOUTUBE REFERENCES (d=64 K=5 seed 42):"
  echo "#   raw-scalar pooler, 1 lr group : 0.5551 ckpt, 0.5605 max, escape ep14, stop ep27"
  echo "#   raw-scalar pooler, 2 lr groups: 0.5564 ckpt, 0.5585 max, escape ep7,  stop ep12"
  echo "#   BEST EVER learned pooler here : 0.5645 (depth 3, mnia, 2 groups; arm was cut short)"
  echo "#   parameter-free pooling rule   : 0.5677 (K=5), 0.5756 (K=10)   <- the bar to clear"
  echo "#   LB #1 GraphMixer              : 0.5887"
  echo "#   The learned pooler has NEVER beaten the parameter-free rule on YouTube."
  echo "#"
  echo "# WATCH: (a) escape epoch vs ep19-20; (b) plateau vs 0.5457 (the matched GELU arm);"
  echo "#   (c) whether the curve is SMOOTHER -- a linear readout should be easier to optimise;"
  echo "#   (d) val/test drift, which cost the GELU arm 0.0001 and the rad arm 0.0054 historically."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~100 s/epoch; the baseline ran 47 epochs in ~80 min."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
