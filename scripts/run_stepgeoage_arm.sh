#!/usr/bin/env bash
# cos_o pooler + THREE step columns: the time gap, the step length, and the step direction.
#   usage: run_stepgeoage_arm.sh <CODE> <EXP> <DS> <yes|no bip> <TAG> <PAT> <SEED> <EPOCHS> <WPN> <MWL> <NL>
set -u
CODE="${1:?}"; EXPERIMENT="${2:?}"; DS="${3:?}"; BIP="${4:?}"; TAG="${5:?}"; PAT="${6:?}"
SEED="${7:?}"; EPOCHS="${8:?}"; WPN="${9:?}"; MWL="${10:?}"; NL="${11:?}"; NHEADS="${12:-4}"
MASTER=/mnt/nfs2/inf/ms2420/tempest-embeddinng
PY=$MASTER/venv/bin/python
LR=1e-3; HID=32
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"

M=link_property_prediction/model.py; W=link_property_prediction/walk_tokens.py
# --- [Q, K, L] tokens: this arm needs the walk axis to have a "previous hop" at all ---
grep -q '\[Q, K, L\] fields (walk axis explicit)' $W || { echo "ABORT: walk_tokens is not the [Q,K,L] builder" >&2; exit 3; }
grep -q 'Flattened \[Q, T\] fields' $W    && { echo "ABORT: walk_tokens is the OLD flat [Q,T] builder" >&2; exit 3; }
grep -q 'shuffle_walk_order=False' link_property_prediction/walks.py || { echo "ABORT: shuffle_walk_order is not False" >&2; exit 3; }
grep -q 'Seed sits at row position ``lens-1``' link_property_prediction/walks.py || { echo "ABORT: walk layout contract changed -- re-derive which index is the previous hop" >&2; exit 3; }
# --- DIRECTION: the previous hop is l-1 (older). l+1 would read the token own future ---
grep -q 'xt_prev = F.pad(xt\[:, :, :-1\], (0, 0, 1, 0))' $M || { echo "ABORT: xt_prev is not the l-1 shift. Index 0 is the OLDEST hop (measured: ages 50,40,30,20,10,0 along increasing index)" >&2; exit 3; }
grep -q 'has_prev = F.pad(u\[:, :, :-1\], (1, 0)) \* u' $M || { echo "ABORT: has_prev mask missing -- the oldest hop must be zeroed" >&2; exit 3; }
grep -q 'b_prev = F.pad(b\[:, :, :-1\], (1, 0))' $M || { echo "ABORT: r_prev missing" >&2; exit 3; }
# --- the two step columns, both through triangle_cos ---
grep -q 'd_step = self.geom.dist(xt, xt_prev) \* has_prev' $M || { echo "ABORT: d_step missing" >&2; exit 3; }
grep -q 'cos_step = triangle_cos(b, b_prev, d_step) \* has_prev' $M || { echo "ABORT: cos_step not from triangle_cos(r_tok, r_prev, d_step)" >&2; exit 3; }
grep -q 'cos_at_origin' $M                && { echo "ABORT: cos_at_origin is back -- this arm reuses triangle_cos" >&2; exit 3; }
# --- ZERO-INIT: the arm must START as master, columns enter by gradient only ---
grep -q 'self.n_feat_base = 4' $M         || { echo "ABORT: n_feat_base missing -- layer 0 must be drawn at master width" >&2; exit 3; }
grep -q 'layers = \[nn.Linear(self.n_feat_base, self.hidden), nn.GELU()\]' $M || { echo "ABORT: layer 0 is not drawn at n_feat_base, so RNG will not match master" >&2; exit 3; }
grep -q 'wide.weight.zero_()' $M          || { echo "ABORT: new columns not zero-initialised" >&2; exit 3; }
grep -q 'wide.weight\[:, :self.n_feat_base\] = narrow.weight' $M || { echo "ABORT: master columns not copied into the widened layer" >&2; exit 3; }
# --- master geometry, unchanged ---
grep -q 'cos_o = triangle_cos(b, c, a) \* u' $M || { echo "ABORT: cos_o not from triangle_cos(r_tok, r_mid, d_mid)" >&2; exit 3; }
grep -q 'def triangle_cos' $M             || { echo "ABORT: triangle_cos helper missing" >&2; exit 3; }
grep -q 'use_e = (p < small) & (q < small)' $M || { echo "ABORT: no Euclidean small-side branch" >&2; exit 3; }
grep -q 'return out.clamp(-1.0, 1.0)' $M  || { echo "ABORT: triangle_cos does not clamp to [-1,1]" >&2; exit 3; }
grep -q 'dims = tuple(range(feat.dim() - 1))' $M || { echo "ABORT: standardiser is not rank-agnostic" >&2; exit 3; }
grep -q 'logits.masked_fill(~valid_flat, float("-inf")), dim=-1' $M || { echo "ABORT: not a single softmax over all K*L tokens" >&2; exit 3; }
grep -q 'n_layers_pooler' scripts/train_link_property_prediction.py || { echo "ABORT: --n-layers-pooler missing" >&2; exit 3; }
grep -q 'w_self\|w_prev\|h_self\|h_prev\|log_alpha\|_walk_memory\|nn.Conv1d\|nn.MultiheadAttention\|nn.GRU\|nn.LayerNorm' $M && { echo "ABORT: machinery from another arm is present" >&2; exit 3; }
grep -q 'bag_mean\|_DENOM_FLOOR\|theta_o\|d0_mid\|d0_tok' $M && { echo "ABORT: machinery from another arm is present" >&2; exit 3; }
grep -q 'self.n_feat = 7' $M                || { echo "ABORT: n_feat is not 7" >&2; exit 3; }
grep -q 'torch.stack(\[age, pos, a, cos_o, age_step, d_step, cos_step\], dim=-1)' $M || { echo "ABORT: features are not [age,pos,d_mid,cos_o,age_step,d_step,cos_step]" >&2; exit 3; }
grep -q 'age_prev = F.pad(raw_age\[:, :, :-1\], (1, 0))' $M || { echo "ABORT: age_prev missing" >&2; exit 3; }
grep -q 'age_step = torch.log1p((age_prev - raw_age).clamp_min(0)) \* has_prev' $M || { echo "ABORT: age_step is not log1p(age_prev - age_l). The reverse is NEGATIVE in this time-ordered layout and log1p of it is NaN -- measured" >&2; exit 3; }
grep -q 'raw_age = tokens.ages.clamp_min(0).to(xt.dtype)' $M || { echo "ABORT: the gap must come from RAW ages, not from the log1p(age) column" >&2; exit 3; }
CELL=d64_k5_wpn${WPN}_mwl${MWL}_lr${LR}_pat${PAT}_h${HID}_nl${NL}_step3
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
  echo "# ARM = $BRANCH $COMMIT: cos_o pooler + THREE step columns: the time gap, the step length, and the step direction."
  echo "#   age_step = log1p(age_{l-1} - age_l)            TIME gap to the previous hop"
  echo "#   d_step   = d(x_l, x_{l-1})                     geodesic length of the step"
  echo "#   cos_step = triangle_cos(r_tok, r_prev, d_step) angle at O between consecutive hops"
  echo "# 1,345 params = master 1,249 + 3 cols x 32."
  echo "# SIGN: the spec asked for log1p(age_l - age_{l-1}), which is NEGATIVE here (ages decrease"
  echo "#   with index: measured 50,40,30,20,10,0) and gives log1p -> NaN. Verified nan on a real"
  echo "#   walk. This arm uses age_{l-1} - age_l, from RAW ages, clamped at 0."
  echo "# ZERO-INIT, EXACT: layer 0 is drawn at master width 4 (so it consumes master RNG)"
  echo "#   then widened with ZERO columns. Verified at seed 5: E.weight equal, first four"
  echo "#   columns equal, new columns all zero, and the forward pass on real walks is"
  echo "#   BITWISE equal to master, max|diff| 0.000e+00. The arm STARTS as master and the"
  echo "#   new columns enter only by gradient."
  echo "# DIRECTION, MEASURED: index 0 is the OLDEST hop, lens-1 the seed, so the previous"
  echo "#   hop is l-1 and every new column is ZERO at the oldest hop. l+1 would read the"
  echo "#   token own FUTURE -- that bug cost five runs earlier; the guards pin it."
  echo "# READ THE TRAINING LOSS AND logit_sd BEFORE MRR. A column that undercuts master"
  echo "#   training loss and widens the logit spread is the familiar pattern and has not"
  echo "#   transferred in this project."
  echo "# BASELINE, post-fix master, seed 5: YouTube 0.5867  Flickr 0.6377 (same commit)."
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
