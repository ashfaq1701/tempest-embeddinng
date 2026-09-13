#!/bin/bash
# YouTube d=64 K=5: pooler on [log1p(age) NO z-score | one-hot position | rad], single lr, linear temp.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/logage_nozscore/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE group)  NO pop bias  seed=42"
  echo "# experiment=logage_nozscore cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([log1p(age) NO z-score | onehot(pos) | rad])), width 32, depth 1"
  echo "#       n_feat = 1 + max_walk_len(5) + 1 = 7   ->   290 head params (was 762)"
  echo "#"
  echo "# THE CHANGE vs the encoded control: the pooler carries NO dataset-derived time constant."
  echo "#   TimeEncoding, its ts_quantum ladder floor, the learned position embedding and t_train"
  echo "#   are all gone. Ages are log1p-transformed then standardised to mean 0 / std 1 against"
  echo "#   statistics pooled over both bags of the batch; position is one-hot over hop 1..5 with"
  echo "#   the padding column dropped."
  echo "#"
  echo "# THIS IS AN ABLATION: the z-score is REMOVED, log1p alone remains. Everything else --"
  echo "#   seed, d, K, lr, patience, one-hot position, rad, optimiser, score -- matches"
  echo "#   logs/logage_zscore/ exactly. 96f016f0 is 94a0d155 minus the standardisation."
  echo "#"
  echo "# WHAT IT COSTS, measured on YouTube-scale ages: the feature becomes mean 12.9, std 2.9,"
  echo "#   range [0, 14.5]. Its siblings are one-hot (0/1) and rad (order 1), so the age channel"
  echo "#   now enters at ~3x their spread and ~13x their mean offset."
  echo "#"
  echo "# THE PRIOR, and it is a real prediction: log1p and z-score are NOT symmetric. log1p"
  echo "#   changes the SHAPE of the age distribution (skew +1.83 -> -1.07 on real tokens) and"
  echo "#   nothing else in the pipeline can. The z-score is AFFINE, so it cannot change shape at"
  echo "#   all -- it only sets scale and offset, and the first Linear has a weight and a bias"
  echo "#   that can in principle absorb both. So the expectation is that this ablation costs"
  echo "#   LITTLE and the log1p carries the result. A large drop would mean the optimiser does"
  echo "#   not in fact absorb the scale at this lr and init, which is worth knowing on its own."
  echo "#"
  echo "# BASELINES, all YouTube d=64 K=5 seed 42, identical except where named:"
  echo "#   encoded pooler, single lr, floored ladder : 0.5457  stop 50 (cap)  762 par  <- CONTROL"
  echo "#   log1p + Z-SCORE (the full version)        : 0.5793 max @ep25 / 0.5757 ckpt @ep35, 290 par  <- THE ARM BEING ABLATED"
  echo "#   min-max age + onehot pos (superseded)     : 0.3653 @ep9 KILLED pre-escape  290 par"
  echo "#   encoded pooler, single lr, aliased ladder : 0.5487  stop ep42      762 par"
  echo "#   encoded pooler, LOG temp                  : 0.5609  stop ep25      762 par  <- our best"
  echo "#   encoded pooler, lr-temp split             : 0.5537  stop ep15      762 par"
  echo "#   two-term score w.[-d, r_u*r_v]            : 0.5324 @ep31 (killed, still rising)"
  echo "#   RAW-SCALAR pooler [rec,pos,rad], 1 group  : 0.5551 ckpt / 0.5605 max, 162 par"
  echo "#   fixed-rule pooling, 2-param head          : 0.5752  stop ep17    <- best in repo"
  echo "#   LB #1 GraphMixer                          : 0.5887"
  echo "#"
  echo "# WATCH: (a) ESCAPE EPOCH -- the z-score arm escaped at ep13, six epochs before the"
  echo "#   control, and did so WITHOUT the geo_temp surge every earlier arm showed. If the"
  echo "#   ablation still escapes at ~ep13 the escape belongs to log1p, not the scaling;"
  echo "#   (b) geo_temp -- LINEAR and single-group here, so expect the slow crawl toward ~44 by"
  echo "#   ep40 rather than the ~79 the faster arms reached;  (c) whether 290 params beat 762;"
  echo "#   (d) val/test drift, which has been 0.0000 on recent arms."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~100 s/epoch; recent YouTube runs went 15-50 epochs, so budget 30-90 min."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
