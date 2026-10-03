#!/usr/bin/env bash
# cos_o pooler + ONE WALK-LOCAL CONV on [Q, K, L] tokens. stem -> Conv1d(k) -> head.
#   usage: run_wconv1_arm.sh <CODE> <EXP> <DS> <yes/no bip> <TAG> <PAT> <SEED> <EPOCHS> <WPN> <MWL> <KS>
set -u
CODE="${1:?}"; EXPERIMENT="${2:?}"; DS="${3:?}"; BIP="${4:?}"; TAG="${5:?}"; PAT="${6:?}"
SEED="${7:?}"; EPOCHS="${8:?}"; WPN="${9:?}"; MWL="${10:?}"; KS="${11:?}"
MASTER=/mnt/nfs2/inf/ms2420/tempest-embeddinng
PY=$MASTER/venv/bin/python
LR=1e-3; HID=32
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"

M=link_property_prediction/model.py; W=link_property_prediction/walk_tokens.py; TR=link_property_prediction/trainer.py
S=scripts/train_link_property_prediction.py
# --- walk_tokens must be the [Q, K, L] version, NOT the flattened one ---
grep -q '\[Q, K, L\] fields (walk axis explicit)' $W || { echo "ABORT: walk_tokens is not the [Q,K,L] version" >&2; exit 3; }
grep -q 'Flattened \[Q, T\] fields' $W && { echo "ABORT: walk_tokens is still the flat [Q,T] version" >&2; exit 3; }
grep -q 'reshape(q, k, length)' $W        || { echo "ABORT: walk_tokens does not reshape to [Q,K,L]" >&2; exit 3; }
grep -q 'shuffle_walk_order=False' link_property_prediction/walks.py || { echo "ABORT: shuffle_walk_order is not False -- the [Q,K,L] reshape would scramble walks" >&2; exit 3; }
# --- the conv itself ---
grep -q 'nn.Conv1d(self.hidden, self.hidden, kernel_size, padding=kernel_size // 2)' $M || { echo "ABORT: conv is not Conv1d(H,H,k,same)" >&2; exit 3; }
grep -q 'h.reshape(q \* k, l, self.hidden).transpose(1, 2)' $M || { echo "ABORT: conv does not run per-walk over the L axis" >&2; exit 3; }
grep -q 'self.stem = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU())' $M || { echo "ABORT: per-token stem missing -- the arm must have a purely per-token stage first" >&2; exit 3; }
grep -q 'self.head = nn.Linear(self.hidden, 1)' $M || { echo "ABORT: head missing" >&2; exit 3; }
grep -q 'raise ValueError(f"kernel_size must be odd' $M || { echo "ABORT: odd-kernel check missing" >&2; exit 3; }
grep -q 'self.stem(feats) \* m.unsqueeze(-1)' $M || { echo "ABORT: padding not zeroed before the conv" >&2; exit 3; }
# --- the cos_o geometry, unchanged from master ---
grep -q 'torch.stack(\[age, pos, a, cos_o\], dim=-1)' $M || { echo "ABORT: features are not [age, pos, d_mid, cos_o]" >&2; exit 3; }
grep -q 'cos_o = triangle_cos(b, c, a) \* m' $M || { echo "ABORT: cos_o not from triangle_cos(r_tok, r_mid, d_mid)" >&2; exit 3; }
grep -q 'def triangle_cos' $M              || { echo "ABORT: triangle_cos helper missing" >&2; exit 3; }
grep -q 'use_e = (p < small) & (q < small)' $M || { echo "ABORT: no Euclidean small-side branch" >&2; exit 3; }
grep -q 'return out.clamp(-1.0, 1.0)' $M   || { echo "ABORT: triangle_cos does not clamp to [-1,1]" >&2; exit 3; }
grep -q 'self.n_feat = 4' $M               || { echo "ABORT: n_feat is not 4" >&2; exit 3; }
grep -q 'self._standardise(torch.stack' $M || { echo "ABORT: features not standardised" >&2; exit 3; }
grep -q 'dims = tuple(range(feat.dim() - 1))' $M || { echo "ABORT: standardiser is not rank-agnostic" >&2; exit 3; }
# --- one softmax over the whole K*L bag ---
grep -q 'logits.masked_fill(~valid_flat, float("-inf")), dim=-1' $M || { echo "ABORT: not a single softmax over all K*L tokens" >&2; exit 3; }
# --- plumbing: kernel-size in, n-layers OUT ---
grep -q 'kernel-size-pooler' $S            || { echo "ABORT: --kernel-size-pooler not plumbed" >&2; exit 3; }
grep -q 'kernel_size_pooler=args.kernel_size_pooler' $S || { echo "ABORT: --kernel-size-pooler not passed to TrainerConfig" >&2; exit 3; }
grep -q 'kernel_size_pooler: int = 3' $TR  || { echo "ABORT: trainer config lacks kernel_size_pooler" >&2; exit 3; }
grep -q 'kernel_size=int(config.kernel_size_pooler)' $TR || { echo "ABORT: kernel_size not passed to LinkPredHead" >&2; exit 3; }
grep -rq 'n_layers' --include=*.py link_property_prediction $S && { echo "ABORT: n_layers plumbing still present in the package or train script -- it was meant to be removed (other arms' shell drivers may mention it; that is fine)" >&2; exit 3; }
# --- nothing from another arm ---
grep -q 'nn.MultiheadAttention\|nn.GRU\|nn.LayerNorm' $M && { echo "ABORT: attention/GRU/LayerNorm present" >&2; exit 3; }
grep -q 'q_off\|n_queries\|logmap0\|log_alpha\|d0_mid\|d0_tok\|theta_o\|angle_at_origin' $M && { echo "ABORT: machinery from another arm is present" >&2; exit 3; }
grep -q 'bag_mean\|_DENOM_FLOOR' $M        && { echo "ABORT: bag-mean normaliser present" >&2; exit 3; }
grep -q 'lr-pooler\|pooler-wd' $S          && { echo "ABORT: a pooler lr/wd knob is plumbed -- this arm must be plain master config" >&2; exit 3; }
[ $((KS % 2)) -eq 1 ] || { echo "ABORT: KS=$KS is even; kernel must be odd" >&2; exit 3; }

CELL=d64_k5_wpn${WPN}_mwl${MWL}_lr${LR}_pat${PAT}_h${HID}_ks${KS}
OUT=$MASTER/logs/$EXPERIMENT/$CELL/$TAG
mkdir -p "$OUT"; LOG=$OUT/$DS.log
BRANCH=$(git rev-parse --abbrev-ref HEAD); COMMIT=$(git rev-parse --short HEAD)
DIRTY=""; [ -n "$(git status --porcelain -- link_property_prediction scripts)" ] && \
  DIRTY="  *** WORKING TREE DIRTY -- SHA does NOT describe the code that ran ***"
BIPFLAG=""; [ "$BIP" = "yes" ] && BIPFLAG="--is-bipartite"

CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $DS \
--data-root $MASTER/datasets --d-emb 64 --k-train 5 --num-walks-per-node $WPN --max-walk-len $MWL \
--lr $LR --num-epochs $EPOCHS --early-stop-patience $PAT --hidden-dim $HID --kernel-size-pooler $KS \
--seed $SEED --use-gpu --use-gpu-tempest $BIPFLAG"

{
  echo "# dataset=$DS bipartite=$BIP (flag: '${BIPFLAG:-none}')"
  echo "# experiment=$EXPERIMENT cell=$CELL tag=$TAG   *** NOT ARCHIVED ***"
  echo "# ARM = $BRANCH $COMMIT: cos_o pooler + ONE WALK-LOCAL CONV, on [Q, K, L] tokens."
  echo "#   walk_tokens keeps the walk axis EXPLICIT: fields are [Q, K, L], not a flat [Q, T]."
  echo "#   Tempest's row layout is unchanged, so this is a RESHAPE of [Q*K, L], not different"
  echo "#   sampling. shuffle_walk_order=False (walks.py:57) is what makes that reshape valid."
  echo "#   per token: standardise([log1p(age), pos, d_mid, cos_o]) -> Linear(4,$HID)+GELU (stem)"
  echo "#   then Conv1d($HID,$HID,k=$KS,padding=k//2) ALONG L, PER WALK -> GELU -> Linear($HID,1)"
  echo "#   one softmax over all K*L tokens -> weighted Lorentz midpoint."
  echo "#   WALKS NEVER SEE EACH OTHER except through that final softmax."
  echo "# VERIFIED on synthetic [Q,K,L] tokens incl. a cold query:"
  echo "#   WALK-LOCALITY at the logit level -- perturbing query0/walk1 moves that walk's logits"
  echo "#     by 0.552 and EVERY other walk of that query, and all of query 1, by exactly 0.000."
  echo "#   RECEPTIVE FIELD is exactly k -- perturbing token 3 changes logit indices [3] at k=1,"
  echo "#     [2,3,4] at k=3, [1,2,3,4,5] at k=5."
  echo "#   Gradients reach E (4.53), conv.weight (8.1e-03), stem and head. Even k raises."
  echo "#   head.bias is a DEAD parameter (softmax is shift-invariant, grad 1.7e-10)."
  echo "# POOLER PARAMS: k=1 -> 1,249   k=3 -> 3,297   k=5 -> 5,345.  THIS RUN k=$KS."
  echo "#   k=1 is the no-context CONTROL: the conv collapses to a per-token Linear and the"
  echo "#   pooler is a 2-stage per-token MLP with 1,249 params -- the SAME COUNT as merged"
  echo "#   master's cos_o pooler. Run k=1 before reading k=3 as a context effect, or depth"
  echo "#   and context are confounded, which is exactly what sank the earlier conv arms."
  echo "# --n-layers-pooler IS GONE on this branch, replaced by --kernel-size-pooler. So there"
  echo "#   is no nl1/nl2 distinction any more: all five datasets run one shape. The WikiLink"
  echo "#   baselines below were nl1, so that comparison mixes in a pooler-shape change."
  echo "# PRIOR ART, and the reason to be sceptical: CLAUDE.md records a hierarchical-conv arm"
  echo "#   (cb3405e) and a conv+pointwise-stem arm (6005d39) that both landed INSIDE NOISE of"
  echo "#   master on 3-5 datasets, at 1.3-1.6x the cost per epoch. The stem arm's lesson was"
  echo "#   that a purely per-token stage must come FIRST -- this arm has that by construction."
  echo "#   A hierarchy over [Q,K,L] cannot enlarge the hypothesis class: any non-negative"
  echo "#   W[k,l] factors as w_walk[k]*w_tok[k,l]. A CONV is not that factorisation, but the"
  echo "#   null result is still the prior. Two instabilities to watch: WikiLink diverged on the"
  echo "#   stem arm (train loss RISING from ep9, r_mean +0.20/epoch) and Yelp ran r_max to 8.14."
  echo "# ACCEPTANCE TEST IS THE TRAINING LOSS, NOT MRR. Driving train link loss below master's"
  echo "#   widens the val->test gap, monotone 5/5 on YouTube. HEALTHY = loss comparable to"
  echo "#   master with better val AND test. See ATTENTION_ARMS_MESSAGE.md sec 2."
  echo "# BASELINES seed 5, same config. MASTER NOW MEANS cos_o (merged 1ed16df):"
  echo "#   cos_o    YouTube 0.5867  Flickr 0.6368  ML-20M 0.2518  Yelp 0.6610  WikiLink 0.6554"
  echo "#   pre-merge master  YouTube 0.6149  Flickr 0.6366  ML-20M 0.2469  Yelp 0.6481*  WikiLink 0.6513"
  echo "#   * hand-killed floor at ep14. YouTube: pre-merge master still beats cos_o by 0.0282"
  echo "#   and holds that dataset against every arm tried, so YouTube is the one to watch."
  echo "#   cos_o YouTube over 3 seeds: 0.5860 / 0.5867 / 0.5914, mean 0.5880 sd 0.0029."
  echo "#   Record max test AND best_test_mrr -- they diverge, and drift has decided ties."
  echo "# d_emb=64 k_train=5 wpn=$WPN mwl=$MWL lr=$LR hidden=$HID kernel_size=$KS epochs=$EPOCHS patience=$PAT seed=$SEED"
  echo "# code=$CODE (detached git worktree) commit=$COMMIT$DIRTY"
  echo "#   model.py md5=$(md5sum $M | cut -d' ' -f1)  walk_tokens.py md5=$(md5sum $W | cut -d' ' -f1)"
  echo "# host=$(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-none}  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo ""
} > "$LOG"
echo "[$(date '+%F %T')] START $DS ks=$KS tag=$TAG commit=$COMMIT host=$(hostname) job=${SLURM_JOB_ID:-none}" >> "$OUT/DRIVER.log"
$CMD >> "$LOG" 2>&1
rc=$?
echo "=== rc=$rc finished=$(date '+%F %T') ===" >> "$LOG"
MAXT=$(grep -oE 'test [0-9.]+ \(new best\)' "$LOG" | grep -oE '[0-9.]+' | sort -g | tail -1)
echo "[$(date '+%F %T')] DONE  $DS rc=$rc $(grep -oE 'best_test_mrr: *[0-9.]+' "$LOG" | tail -1)  max_test=${MAXT:-none}  eps=$(grep -c '^epoch' "$LOG")" >> "$OUT/DRIVER.log"
