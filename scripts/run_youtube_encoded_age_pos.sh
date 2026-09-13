#!/bin/bash
# YouTube d=64 K=5: pooler on [TimeEncoder(age) | hop_emb(hop) | rad], single lr, linear temp.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/encoded_age_pos/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE group)  NO pop bias  seed=42"
  echo "# experiment=encoded_age_pos cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([TimeEncoder(age) | hop_emb(hop) | rad])), width 32, depth 1"
  echo "#       d_time=16 (1 monotone linear + 15 cos/sin over log-age), d_pos=4, +1 rad"
  echo "#       n_feat = 21  ->  762 head params"
  echo "#"
  echo "# THE CHANGE: an encoded pooler again, but the ladder is over LOG-age and its normaliser"
  echo "#   is a CONSTANT. u = log1p(age)/24, clamped at 1. LMAX=24 is bounded by arithmetic, not"
  echo "#   measurement: 1 s is 0.69, 1 year 17.3, three centuries 23.0. Frequencies are a fixed"
  echo "#   geometric ladder over wavelengths 0.05..2.0 in u, NOT learnable -- d/dw of cos(xw)"
  echo "#   scales with x and destabilises on large timestamps (Cong et al., ICLR 2023)."
  echo "#   Hop is a learned embedding, max_walk_len+1 rows with padding_idx=0, init std 0.02."
  echo "#"
  echo "# WHY A CONSTANT NORMALISER: it makes the encoder a PURE FUNCTION OF AGE. The same age"
  echo "#   encodes identically in every batch, on every dataset, and at every point in a stream."
  echo "#   Reference points: 1 s u=0.029, 1 h 0.341, 1 d 0.474, 1 mo 0.615, 1 yr 0.720, 25 yr"
  echo "#   0.854. YouTube's full span lands at u=0.699, so this dataset uses the lower 70% of"
  echo "#   the range rather than stretching to fill [0,1] -- that is the price of the mapping"
  echo "#   being absolute, and the thing to watch if the result disappoints."
  echo "#"
  echo "# EVERY PRIOR FIXED-LADDER ATTEMPT FAILED, and for a reason this design targets. The old"
  echo "#   TimeEncoding laddered over RAW seconds normalised by T_train, which put 4 of 8"
  echo "#   frequencies below Nyquist on YouTube (daily grid, 175 distinct timestamps) -- they"
  echo "#   emitted a deterministic hash of the day index. Log-age gives uniform resolution per"
  echo "#   decade instead, which is what the measured age distribution wants (skew +1.83 raw)."
  echo "#"
  echo "# BASELINES, all YouTube d=64 K=5 seed 42, identical except where named:"
  echo "#   log1p(age) + raw pos, NO standardisation (master) : 0.5625  stop ep21   162 par"
  echo "#   log1p + z-score, one-hot pos                      : 0.5793 max / 0.5757 ckpt  290 par"
  echo "#   log1p + z-score, z-scored scalar pos              : 0.5787 max @ep17 (not converged)"
  echo "#   log1p only, one-hot pos                           : 0.5617  stop ep18   290 par"
  echo "#   OLD raw-seconds TimeEncoding, floored ladder      : 0.5457  stop 50 (cap)  762 par"
  echo "#   OLD raw-seconds TimeEncoding, aliased ladder      : 0.5487  stop ep42      762 par"
  echo "#   raw-scalar pooler [rec,pos,rad], mnia scale       : 0.5551 ckpt / 0.5605 max  162 par"
  echo "#   fixed-rule pooling, 2-param head                  : 0.5752  stop ep17"
  echo "#   LB #1 GraphMixer                                  : 0.5887"
  echo "#"
  echo "# WHAT WOULD COUNT: 762 params on a log-age ladder against 162 on a bare log1p scalar."
  echo "#   Beating 0.5793 makes this the best learned pooler here; landing near 0.5457 would say"
  echo "#   the encoder width is the problem regardless of what the ladder is laid over."
  echo "#"
  echo "# WATCH: (a) escape epoch -- the log1p arms escaped ep13 (standardised) and ep17 (not);"
  echo "#   the old raw-seconds encoders escaped ep19-20;  (b) geo_temp, which crawls to ~44 on"
  echo "#   single-group linear arms;  (c) val/test drift, 0.0000 on most recent arms."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~100 s/epoch; recent YouTube runs went 18-50 epochs, so budget 30-90 min."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
