#!/usr/bin/env bash
# CURRENT MASTER: cos_o pooler (post-triangle_cos) on the [Q,K,L] walk_tokens builder. n_feat 4.
#   usage: run_cosomaster_arm.sh <CODE> <EXP> <DS> <yes|no bip> <TAG> <PAT> <SEED> <EPOCHS> <WPN> <MWL> <NL>
set -u
CODE="${1:?}"; EXPERIMENT="${2:?}"; DS="${3:?}"; BIP="${4:?}"; TAG="${5:?}"; PAT="${6:?}"
SEED="${7:?}"; EPOCHS="${8:?}"; WPN="${9:?}"; MWL="${10:?}"; NL="${11:?}"; NHEADS="${12:-4}"
MASTER=/mnt/nfs2/inf/ms2420/tempest-embeddinng
PY=$MASTER/venv/bin/python
LR=1e-3; HID=32
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"

M=link_property_prediction/model.py; W=link_property_prediction/walk_tokens.py
grep -q 'torch.stack(\[age, pos, a, cos_o\], dim=-1)' $M || { echo "ABORT: features are not [age, pos, d_mid, cos_o]" >&2; exit 3; }
grep -q 'cos_o = F.cosine_similarity(xt, mid.unsqueeze(-2), dim=-1, eps=_COS_EPS) \* u' $M || { echo "ABORT: cos_o is not the coordinate cosine" >&2; exit 3; }
grep -q 'triangle_cos' $M                 && { echo "ABORT: triangle_cos is back -- the intrinsic chart needs no law of cosines" >&2; exit 3; }
grep -q '_COS_EPS = 1e-12' $M             || { echo "ABORT: _COS_EPS missing" >&2; exit 3; }
grep -q 'self.n_feat = 4' $M              || { echo "ABORT: n_feat is not 4" >&2; exit 3; }
grep -q 'feats = standardise(torch.stack' $M || { echo "ABORT: features not standardised via the module-level standardise()" >&2; exit 3; }
grep -q 'def standardise' $M              || { echo "ABORT: module-level standardise() missing" >&2; exit 3; }
grep -q '_standardise' $M                 && { echo "ABORT: the old BagWeights._standardise staticmethod is still present" >&2; exit 3; }
grep -q 'self.geom.weighted_midpoint(' $M || { echo "ABORT: pooler does not call weighted_midpoint (renamed from midpoint)" >&2; exit 3; }
grep -q 'self.geom.midpoint(' $M          && { echo "ABORT: the old geom.midpoint name is still in use" >&2; exit 3; }
grep -q 'q_off\|n_queries\|logmap0\|log_alpha\|d0_mid' $M && { echo "ABORT: another arm's machinery present" >&2; exit 3; }
grep -q 'nn.MultiheadAttention\|nn.Conv1d\|nn.GRU\|nn.LayerNorm' $M && { echo "ABORT: attention/conv/GRU/LayerNorm present" >&2; exit 3; }
grep -q '\[Q, K, L\] fields (walk axis explicit)' $W || { echo "ABORT: walk_tokens is not the [Q,K,L] builder (master moved to it in 4b1a919)" >&2; exit 3; }
grep -q 'Flattened \[Q, T\] fields' $W   && { echo "ABORT: walk_tokens is the OLD flat [Q,T] builder" >&2; exit 3; }
grep -q 'tokens.nodes.flatten(1)' $M      || { echo "ABORT: the pooler does not flatten [Q,K,L] -> [Q,T] at the consumer" >&2; exit 3; }
grep -q 'tokens.ages.flatten(1)' $M       || { echo "ABORT: ages not flattened at the consumer" >&2; exit 3; }
grep -q 'tokens.positions.flatten(1)' $M  || { echo "ABORT: positions not flattened at the consumer" >&2; exit 3; }
grep -q 'reshape(q, -1)' $M               && { echo "ABORT: reshape(q,-1) is ambiguous for an empty batch; master uses flatten(1)" >&2; exit 3; }
grep -q 'w_self\|w_prev\|h_self\|h_prev\|log_alpha\|_walk_memory\|nn.Conv1d' $M && { echo "ABORT: a sequence-pooler arm is present -- this driver runs plain master" >&2; exit 3; }
grep -q 'bag_mean\|r_mid =\|_DENOM_FLOOR' $M && { echo "ABORT: bag-mean normaliser present" >&2; exit 3; }
grep -q 'n_layers_pooler' scripts/train_link_property_prediction.py || { echo "ABORT: --n-layers-pooler missing" >&2; exit 3; }

CELL=d64_k5_wpn${WPN}_mwl${MWL}_lr${LR}_pat${PAT}_h${HID}_nl${NL}
OUT=$MASTER/logs/$EXPERIMENT/$CELL/$TAG
mkdir -p "$OUT"; LOG=$OUT/$DS.log
BRANCH=$(git rev-parse --abbrev-ref HEAD); COMMIT=$(git rev-parse --short HEAD)
DIRTY=""; [ -n "$(git status --porcelain -- link_property_prediction scripts)" ] && \
  DIRTY="  *** WORKING TREE DIRTY -- SHA does NOT describe the code that ran ***"
BIPFLAG=""; [ "$BIP" = "yes" ] && BIPFLAG="--is-bipartite"

CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $DS \
--data-root $MASTER/datasets --d-emb 64 --k-train 5 --num-walks-per-node $WPN --max-walk-len $MWL \
--lr $LR --num-epochs $EPOCHS --early-stop-patience $PAT --hidden-dim $HID --n-layers-pooler $NL \
--seed $SEED --use-gpu --use-gpu-tempest $BIPFLAG"

{
  echo "# dataset=$DS bipartite=$BIP (flag: '${BIPFLAG:-none}')"
  echo "# experiment=$EXPERIMENT cell=$CELL tag=$TAG   *** NOT ARCHIVED ***"
  echo "# ARM = $BRANCH $COMMIT: CURRENT MASTER. cos_o pooler, post-triangle_cos, on the"
  echo "#   [Q,K,L] walk_tokens builder. feats = standardise([log1p(age), pos, d_mid, cos_o])."
  echo "#   n_feat 4. Pooler 1,249 at nl2, 193 at nl1. THIS RUN nl$NL."
  echo "# WHY THIS RUN EXISTS: four of the five cos_o references in use were produced at commit"
  echo "#   2d7b739, BEFORE the triangle_cos numerics fix. On YouTube that fix was worth +0.0398"
  echo "#   (0.5469 -> 0.5867), so Flickr 0.6368, ML-20M 0.2518, Yelp 0.6610 and WikiLink 0.6554"
  echo "#   are all plausibly DEPRESSED baselines, and every new arm has been judged against"
  echo "#   them. This produces a uniform post-fix baseline set from ONE commit."
  echo "# YouTube is included deliberately even though 388f5ef already gives a post-fix 0.5867:"
  echo "#   master has since moved to the [Q,K,L] builder (4b1a919), which was proven bit-identical"
  echo "#   in a unit test (all six fields, pooled points, full scores and all eight gradients"
  echo "#   equal, max|diff| 0.000e+00). A YouTube run that lands on 0.5867 confirms that"
  echo "#   end-to-end through training; one that does not means the refactor is NOT inert."
  echo "# nl PER DATASET: nl1 for WikiLink, nl2 for the rest. That matches the existing"
  echo "#   references and is the only reason WikiLink can be compared with the nl1-family arms"
  echo "#   (ema-nl1 194 params, ph-nl1 225) at all."
  echo "# COMPARISON SET, seed 5, all pre-fix except YouTube:"
  echo "#   cos_o  YouTube 0.5867  Flickr 0.6368  ML-20M 0.2518  Yelp 0.6610  WikiLink 0.6554"
  echo "#   ema-nl1 (194 params)   YouTube 0.5668  Flickr 0.6312  ML-20M 0.2513"
  echo "#   ph-nl1  (225 params)   YouTube 0.5584*  Flickr 0.6302  ML-20M 0.2500   * ep100 cap, a floor"
  echo "#   pre-merge master nl2   YouTube 0.6149  Flickr 0.6366  ML-20M 0.2469 -- still the best YouTube"
  echo "# ACCEPTANCE TEST IS THE TRAINING LOSS, NOT MRR. Ten arms obey one law: driving train"
  echo "#   link loss below master\x27s widens the val->test gap (monotone 5/5 on YouTube). The"
  echo "#   WikiLink winners are the one counterexample. HEALTHY = train loss comparable to"
  echo "#   master with better val AND test. See ATTENTION_ARMS_MESSAGE.md sec 2."
  echo "# BASELINES seed 5, same config: master nl2 YouTube 0.6149 Flickr 0.6366 ML-20M 0.2469"
  echo "#   Yelp 0.6481 (ep14, killed, FLOOR); master nl1 WikiLink 0.6513 (ep14, ran to 24)."
  echo "#   Record max test AND best_test_mrr -- they diverge."
  echo "# d_emb=64 k_train=5 wpn=$WPN mwl=$MWL lr=$LR hidden=$HID n_layers_pooler=$NL epochs=$EPOCHS patience=$PAT seed=$SEED"
  echo "# code=$CODE (detached git worktree) commit=$COMMIT$DIRTY"
  echo "#   model.py md5=$(md5sum $M | cut -d' ' -f1)  walk_tokens.py md5=$(md5sum $W | cut -d' ' -f1)"
  echo "# host=$(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-none}  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo ""
} > "$LOG"
echo "[$(date '+%F %T')] START $DS wpn=$WPN tag=$TAG commit=$COMMIT host=$(hostname) job=${SLURM_JOB_ID:-none}" >> "$OUT/DRIVER.log"
$CMD >> "$LOG" 2>&1
rc=$?
echo "=== rc=$rc finished=$(date '+%F %T') ===" >> "$LOG"
MAXT=$(grep -oE 'test [0-9.]+ \(new best\)' "$LOG" | grep -oE '[0-9.]+' | sort -g | tail -1)
echo "[$(date '+%F %T')] DONE  $DS rc=$rc $(grep -oE 'best_test_mrr: *[0-9.]+' "$LOG" | tail -1)  max_test=${MAXT:-none}  eps=$(grep -c '^epoch' "$LOG")" >> "$OUT/DRIVER.log"
