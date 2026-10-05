#!/usr/bin/env bash
# MASTER POOLER + TWO-CHANNEL SCORER: w[0]*std(-geo) + w[1]*std(cos_uv). geo_temp removed.
#   usage: run_scorer2ch_arm.sh <CODE> <EXP> <DS> <yes|no bip> <TAG> <PAT> <SEED> <EPOCHS> <WPN> <MWL> <NL>
set -u
CODE="${1:?}"; EXPERIMENT="${2:?}"; DS="${3:?}"; BIP="${4:?}"; TAG="${5:?}"; PAT="${6:?}"
SEED="${7:?}"; EPOCHS="${8:?}"; WPN="${9:?}"; MWL="${10:?}"; NL="${11:?}"; NHEADS="${12:-4}"
MASTER=/mnt/nfs2/inf/ms2420/tempest-embeddinng
PY=$MASTER/venv/bin/python
LR=1e-3; HID=32
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"

M=link_property_prediction/model.py; W=link_property_prediction/walk_tokens.py
# --- pooler must be master's, untouched ---
grep -q 'torch.stack(\[age, pos, a, cos_o\], dim=-1)' $M || { echo "ABORT: pooler features are not [age, pos, d_mid, cos_o]" >&2; exit 3; }
grep -q 'cos_o = triangle_cos(b, c, a) \* u' $M || { echo "ABORT: cos_o not from triangle_cos" >&2; exit 3; }
grep -q 'self.n_feat = 4' $M              || { echo "ABORT: pooler n_feat is not 4" >&2; exit 3; }
grep -q 'def triangle_cos' $M             || { echo "ABORT: triangle_cos helper missing" >&2; exit 3; }
grep -q 'use_e = (p < small) & (q < small)' $M || { echo "ABORT: no Euclidean small-side branch" >&2; exit 3; }
grep -q 'return out.clamp(-1.0, 1.0)' $M  || { echo "ABORT: triangle_cos does not clamp" >&2; exit 3; }
# --- the TWO-CHANNEL scorer ---
grep -q 'def standardise' $M              || { echo "ABORT: module-level standardise missing" >&2; exit 3; }
grep -q 'dims = tuple(range(feat.dim() - 1))' $M || { echo "ABORT: standardise is not rank-agnostic" >&2; exit 3; }
grep -q 'self.w = nn.Parameter(torch.tensor(\[1.0, 1.0\]))' $M || { echo "ABORT: channel weights w are not init [1,1]" >&2; exit 3; }
grep -q 'cos_uv = triangle_cos(r_u, r_v, geo)' $M || { echo "ABORT: cos_uv not from triangle_cos(r_u, r_v, geo)" >&2; exit 3; }
grep -q 'chan = standardise(torch.stack(\[-geo, cos_uv\], dim=-1))' $M || { echo "ABORT: channels not standardised batch-wide and unmasked" >&2; exit 3; }
grep -q 'return (chan \* self.w).sum(-1)' $M || { echo "ABORT: score is not the learned linear combination" >&2; exit 3; }
grep -q 'self.geo_temp' $M                && { echo "ABORT: a self.geo_temp PARAMETER is still present -- it is redundant under standardisation and must be gone (the docstring may mention it; code must not define it)" >&2; exit 3; }
grep -q 'w_cos=' link_property_prediction/trainer.py || { echo "ABORT: trainer does not log w_cos -- that is the collapse diagnostic" >&2; exit 3; }
grep -q 'w_cos_frac' link_property_prediction/trainer.py || { echo "ABORT: trainer does not log w_cos_frac" >&2; exit 3; }
# --- [Q,K,L] builder, flattened at the consumer ---
grep -q '\[Q, K, L\] fields (walk axis explicit)' $W || { echo "ABORT: walk_tokens is not the [Q,K,L] builder" >&2; exit 3; }
grep -q 'tokens.nodes.flatten(1)' $M      || { echo "ABORT: pooler does not flatten at the consumer" >&2; exit 3; }
grep -q 'n_layers_pooler' scripts/train_link_property_prediction.py || { echo "ABORT: --n-layers-pooler missing" >&2; exit 3; }
grep -q 'w_self\|w_prev\|log_alpha\|_walk_memory\|d_step\|nn.Conv1d\|nn.MultiheadAttention' $M && { echo "ABORT: machinery from another arm is present" >&2; exit 3; }
CELL=d64_k5_wpn${WPN}_mwl${MWL}_lr${LR}_pat${PAT}_h${HID}_nl${NL}_2ch
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
  echo "# ARM = $BRANCH $COMMIT: master pooler + TWO-CHANNEL SCORER, geo_temp removed."
  echo "#   score = w[0]*std(-geo) + w[1]*std(cos_uv),  w init [1,1], standardised batch-wide."
  echo "#   Pooler is bitwise identical to master (E, all six net tensors, pooled points)."
  echo "# WHY: the two channels are scale-mismatched ~70x at init and ~0.03x late -- |d(u,v)|"
  echo "#   grows 0.0014 -> 5.5 as r_mean goes 1e-3 -> 3.1 while |cos| sits at ~0.1 throughout,"
  echo "#   so the ratio INVERTS by ~2000x and no fixed rescaling works. Standardising pins both"
  echo "#   to sd 1.000 at every radius (measured r = 0.001 .. 6.0). geo_temp is then redundant:"
  echo "#   standardising geo over the batch IS a batch-adaptive temperature."
  echo "# WATCH r_mean AND w_cos FROM EPOCH 1, NOT MRR. CLAUDE.md records the earlier [-d_H, cos]"
  echo "#   scorer peaking at ep2 then collapsing: cos is exactly scale-invariant, so it gives a"
  echo "#   cheap angular win and the embedding never spreads (r_mean 0.049 vs a healthy 0.225 at"
  echo "#   ep1, r_max 1.038 vs 3.116). Standardising -geo DISCARDS the radius signal too --"
  echo "#   d_H within-query spread grows 1420x from init to r=3.1 while cos stays at 1.00x -- so"
  echo "#   after standardisation a collapsed and a healthy embedding look IDENTICAL to the"
  echo "#   scorer. w_cos growing while r_mean stays low IS the collapse. The epoch line now"
  echo "#   carries w_geo, w_cos and w_cos_frac next to r_mean for exactly this."
  echo "# w init [1,1] means the scorer is NOT master at init -- cos contributes equally from"
  echo "#   step one. Table A measured arms starting away from master costing 0.016-0.022 against"
  echo "#   ~0.003 for zero-init arms, so w = [1, 0] is the follow-up if this collapses."
  echo "# KNOWN COST: batch statistics differ train vs eval -- divisor sd per channel is"
  echo "#   [0.00132, 0.34546] at C=6 (train, K_train 5) and [0.00129, 0.34098] at C=21, ~2%."
  echo "# cos_uv at init: mean +0.506 sd 0.351 range 0.008..1.000, 0% clamped."
  echo "# BASELINE, post-fix master, seed 5: YouTube 0.5867  Flickr 0.6377  (same commit lineage)."
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
