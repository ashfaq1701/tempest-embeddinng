#!/bin/bash
# YouTube d=64 K=5: normalised Gromov product, NO temperature. Parameterised by lr.
#   s(u,v) = -d_H(P_u,P_v) / (rad(P_u) + rad(P_v))
# usage: run_youtube_gromov_notemp_lr.sh <LR_TAG> <LR>     e.g.  lr1e-2 1e-2
LR_TAG="$1"; LR="$2"
[ -z "$LR_TAG" ] || [ -z "$LR" ] && { echo "usage: $0 <LR_TAG> <LR>"; exit 2; }
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/gromov_notemp_lr/$LR_TAG/run_1/YouTube.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --lr $LR --num-epochs 100 --early-stop-patience 3 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=YouTube (NON-bipartite) d_emb=64 k_train=5  lr=$LR (single group)  seed=42"
  echo "# experiment=gromov_notemp_lr cell=$LR_TAG tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = -d_H(P_u,P_v) / (rad(P_u) + rad(P_v))   <- NO temperature, no learned scalar"
  echo "#       161 head params: the pooler MLP alone. init_irange 1e-3."
  echo "#       pooling = softmax(MLP([log1p(age) | raw pos | rad])), width 32, unchanged"
  echo "#"
  echo "# PAIRED RUN. This cell is one of TWO launched together from a cold start, sharing the GPU,"
  echo "#   identical in every respect except lr: cells lr1e-3 and lr1e-2 under this experiment."
  echo "#   They were started in parallel deliberately so the comparison is coupled -- same commit,"
  echo "#   same machine state, same wall-clock. Compare ONLY against the sibling cell; earlier"
  echo "#   solo runs at lr 1e-3 ran without GPU contention and are not wall-clock comparable"
  echo "#   (epochs here will be slower for both cells; MRR is unaffected)."
  echo "#"
  echo "# WHY lr MATTERS ON THIS HEAD SPECIFICALLY. There is no temperature, so the model cannot"
  echo "#   sharpen the softmax at all -- the similarity is bounded in [0,1] with effective spread"
  echo "#   ~0.25 (measured [0.5555,0.8136] at init), capping CE ~0.16-0.47 below chance"
  echo "#   (log 6 = 1.7918). Every bit of progress has to come from MOVING THE GEOMETRY, so the"
  echo "#   embedding learning rate is the only lever that sets how fast the head can improve."
  echo "#   The tempered sibling reached its scale via geo_temp instead, which is why lr mattered"
  echo "#   less there."
  echo "#"
  echo "# lr 1e-2 IS A REAL DIVERGENCE RISK, and that is informative rather than a failure:"
  echo "#   E is a geoopt.ManifoldParameter stepped by RiemannianAdam. Field notes record that even"
  echo "#   3e-3 on a ManifoldParameter was untested and a blow-up was considered a live outcome."
  echo "#   1e-2 is 10x the standard rate. WATCH for NaN, |E|max pinning at 1.0, or link exploding."
  echo "#"
  echo "# REFERENCES (YouTube d=64 K=5 seed 42, no pop bias, SAME pooler in every arm, lr 1e-3):"
  echo "#   geo_temp * (-d)                : 0.5625 @ep21  <- BASELINE, unbeaten"
  echo "#   -d / (rad_u * rad_v)           : 0.4399 @ep8"
  echo "#   -rad_u * rad_v * d             : 0.4264 @ep19"
  echo "#   geo_temp * (-d/(rad_u+rad_v))  : 0.3928 @ep12  <- tempered version of THIS head"
  echo "#   -d / rad_v                     : 0.2529 @ep6   <- the other un-tempered head"
  echo "#   this head at lr 1e-3, ep1      : link 1.6091  val 0.2664  test 0.2250"
  echo "#   LB #1 GraphMixer               : 0.5887"
  echo "#"
  echo "# WATCH: (a) link -- if it parks near 1.5-1.6 while val stalls, the bounded-similarity cap"
  echo "#   is biting and lr will not rescue it;  (b) whether val peaks then DECLINES -- that is"
  echo "#   memorisation surviving the temperature removal, which refutes the reason this head"
  echo "#   exists;  (c) |E|mean -- the baseline escapes at 0.246, the tempered Gromov arm ended at"
  echo "#   0.086. If a higher lr drives |E| toward 0.246 AND the escape fires, that is direct"
  echo "#   evidence for the CAPACITY hypothesis (hyperbolic volume grows exponentially with"
  echo "#   radius; scaling a configuration outward REORDERS it past |x| ~ 0.4, measured);"
  echo "#   (d) record BOTH max test and test@val-checkpoint;"
  echo "#   (e) patience is 3, tighter than the default 5 -- a 3-epoch flat spell ends the run."
  echo "# ~100 s/epoch solo; EXPECT SLOWER with two runs sharing the GPU."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
