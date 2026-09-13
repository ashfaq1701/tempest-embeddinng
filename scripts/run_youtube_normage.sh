#!/bin/bash
# YouTube d=64 K=5: pooler on [normalised age | one-hot position | rad], single lr, linear temp.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/normage/d64_k5/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr 1e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=1e-3 (SINGLE group)  NO pop bias  seed=42"
  echo "# experiment=normage cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([age_norm | onehot(pos) | rad])), width 32, depth 1"
  echo "#       n_feat = 1 + max_walk_len(5) + 1 = 7   ->   290 head params (was 762)"
  echo "#"
  echo "# THE CHANGE: the pooler now carries NO dataset-derived time constant."
  echo "#   TimeEncoding and its ts_quantum ladder floor are gone; so is the learned position"
  echo "#   embedding and t_train. Ages are min-max normalised to [0,1] per QUERY PAIR against a"
  echo "#   min/max pooled across the source and candidate bags, and position is one-hot over"
  echo "#   hop 1..5 with the padding column dropped."
  echo "#"
  echo "# WHY: every fixed time scale tried so far mis-scales somewhere on the suite. Measured"
  echo "#   over real walk tokens (K=5 L=5, 16,384 edges x 3 seeds), the median of age/mnia runs"
  echo "#   0.10 GoogleLocal / 0.28 Yelp / 0.67 Patent / 1.03 YouTube / 1.04 ML-20M / 1.08"
  echo "#   WikiLink / 4.74 Flickr -- a 47x spread. And the fixed frequency ladder had 4 of 8"
  echo "#   frequencies BELOW NYQUIST on YouTube and Flickr, 3/8 on Patent, 2/8 on WikiLink."
  echo "#   A per-pair min-max sidesteps both failure modes at the cost of making the same"
  echo "#   absolute age mean different things in different pairs."
  echo "#"
  echo "# THAT COST IS THE RISK, and it is the thing to watch. Normalisation is per query pair,"
  echo "#   so a token 3 days old maps to 0.2 in one comparison and 0.9 in another depending on"
  echo "#   the oldest token each bag happens to contain. The pooler can no longer learn an"
  echo "#   absolute recency curve -- only a relative one. Whether that is a feature (scale"
  echo "#   invariance) or a bug (destroyed signal) is exactly what this run measures."
  echo "#   Note the seed slot is age 0 and non-padded, so the min is structurally 0 and this"
  echo "#   reduces to dividing by the oldest token in the pair."
  echo "#"
  echo "# BASELINES, all YouTube d=64 K=5 seed 42, identical except where named:"
  echo "#   encoded pooler, single lr, floored ladder : 0.5457  stop 50 (cap)  762 par  <- CONTROL"
  echo "#   encoded pooler, single lr, aliased ladder : 0.5487  stop ep42      762 par"
  echo "#   encoded pooler, LOG temp                 : 0.5609  stop ep25      762 par  <- our best"
  echo "#   encoded pooler, lr-temp split            : 0.5537  stop ep15      762 par"
  echo "#   two-term score w.[-d, r_u*r_v]           : 0.5324 @ep31 (killed, still rising)"
  echo "#   RAW-SCALAR pooler [rec,pos,rad], 1 group : 0.5551 ckpt / 0.5605 max, 162 par"
  echo "#   fixed-rule pooling, 2-param head         : 0.5752  stop ep17    <- best in repo"
  echo "#   LB #1 GraphMixer                         : 0.5887"
  echo "#"
  echo "# WATCH: (a) escape epoch -- control ep19-20, log temp ep11-12, 2-param reference none;"
  echo "#   (b) geo_temp -- it is LINEAR and single-group here, so expect the slow crawl to ~44"
  echo "#   by ep40 rather than the ~79 the faster arms reached;  (c) whether 290 params beat"
  echo "#   762 -- if so the encoders were never earning their width;  (d) val/test drift."
  echo "# Record BOTH test@val-checkpoint AND max test observed."
  echo "# ~100 s/epoch; recent YouTube runs went 15-50 epochs, so budget 30-90 min."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
