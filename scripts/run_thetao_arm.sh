#!/usr/bin/env bash
# MASTER + theta_o: the angle at the ORIGIN between the bag centre and the token, from COORDINATES. n_feat 4.
#   usage: run_thetao_arm.sh <CODE> <EXP> <DS> <yes|no bip> <TAG> <PAT> <SEED> <EPOCHS> <WPN> <MWL> <NL>
set -u
CODE="${1:?}"; EXPERIMENT="${2:?}"; DS="${3:?}"; BIP="${4:?}"; TAG="${5:?}"; PAT="${6:?}"
SEED="${7:?}"; EPOCHS="${8:?}"; WPN="${9:?}"; MWL="${10:?}"; NL="${11:?}"; NHEADS="${12:-4}"
MASTER=/mnt/nfs2/inf/ms2420/tempest-embeddinng
PY=$MASTER/venv/bin/python
LR=1e-3; HID=32
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"

M=link_property_prediction/model.py; W=link_property_prediction/walk_tokens.py
grep -q 'torch.stack(\[age, pos, d_mid, theta_o\], dim=-1)' $M || { echo "ABORT: features are not [age, pos, d_mid, theta_o]" >&2; exit 3; }
grep -q 'theta_o = angle_at_origin(xt, mid.unsqueeze(-2)) \* u' $M || { echo "ABORT: theta_o not from angle_at_origin(token, mid)" >&2; exit 3; }
grep -q 'def angle_at_origin' $M          || { echo "ABORT: angle_at_origin helper missing" >&2; exit 3; }
grep -q 'theta = torch.atan2(sin, cos)' $M || { echo "ABORT: angle not formed with atan2 -- arccos loses precision at small angles" >&2; exit 3; }
grep -q 'sin = (ux - cos.unsqueeze(-1) \* uy).norm(dim=-1)' $M || { echo "ABORT: perpendicular component not formed as written" >&2; exit 3; }
grep -q 'ux = x / nx.clamp_min(floor)' $M || { echo "ABORT: x direction not normalised with a floor" >&2; exit 3; }
grep -q 'uy = y / ny.clamp_min(floor)' $M || { echo "ABORT: y direction not normalised with a floor" >&2; exit 3; }
grep -q 'ok = (nx.squeeze(-1) > floor) & (ny.squeeze(-1) > floor)' $M || { echo "ABORT: origin-degenerate case not detected" >&2; exit 3; }
grep -q 'torch.where(ok, theta, torch.zeros_like(theta))' $M || { echo "ABORT: degenerate case not zeroed" >&2; exit 3; }
grep -q '_DIR_FLOOR = 1e-12' $M           || { echo "ABORT: _DIR_FLOOR missing or not 1e-12" >&2; exit 3; }
grep -q 'd_mid = self.geom.dist(xt, mid.unsqueeze(-2))' $M || { echo "ABORT: d_mid missing" >&2; exit 3; }
grep -q 'self.n_feat = 4' $M              || { echo "ABORT: n_feat is not 4" >&2; exit 3; }
grep -q 'self._standardise(torch.stack' $M || { echo "ABORT: features not standardised" >&2; exit 3; }
grep -q 'triangle_cos\|cos_o' $M          && { echo "ABORT: cos_o machinery present -- this arm replaces it with theta_o" >&2; exit 3; }
grep -q 'torch.cosh\|torch.sinh' $M       && { echo "ABORT: a cosh/sinh CALL is present -- theta_o comes from COORDINATES, not a law of cosines (the docstring may mention sinh; code must not call it)" >&2; exit 3; }
grep -q 'q_off\|n_queries\|logmap0\|log_alpha\|d0_mid\|d0_tok' $M && { echo "ABORT: machinery from another arm is present" >&2; exit 3; }
grep -q 'nn.MultiheadAttention\|nn.Conv1d\|nn.GRU\|nn.LayerNorm' $M && { echo "ABORT: attention/conv/GRU/LayerNorm present" >&2; exit 3; }
grep -q 'Flattened \[Q, T\] fields' $W   || { echo "ABORT: walk_tokens is not the flat [Q,T] version" >&2; exit 3; }
grep -q 'bag_mean\|r_mid =\|_DENOM_FLOOR' $M && { echo "ABORT: bag-mean normaliser present" >&2; exit 3; }
grep -q 'lr-pooler\|pooler-wd' scripts/train_link_property_prediction.py && { echo "ABORT: a pooler lr/wd knob is plumbed -- this arm must be plain master config" >&2; exit 3; }
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
  echo "# ARM = $BRANCH $COMMIT: MASTER + theta_o, the angle at the ORIGIN between the bag centre and the token."
  echo "#   feats = standardise([log1p(age), pos, d_mid, theta_o]).  n_feat 4. Pooler 1,217 -> 1,249 (1.03x)."
  echo "#   From COORDINATES, not distances. In the intrinsic Lorentz chart a point at radius r on"
  echo "#   ray direction n has x = sinh(r) * n, so the angle at O between two points is the"
  echo "#   EUCLIDEAN angle between their coordinate vectors. Formed as atan2(sin, cos) with"
  echo "#   sin = |perpendicular part of ux w.r.t. uy|, which is accurate at every angle including"
  echo "#   the small sectors where arccos loses precision. No cosh/sinh, no distances."
  echo "#   Range [0, pi]: 0 same ray as the centre, pi/2 orthogonal, pi opposite sides of the root."
  echo "#   A point AT the origin has no direction and returns 0."
  echo "# CHART CLAIM VERIFIED: dist0 = 2 sqrt(k) asinh(sqrt(w/2)) reduces to asinh(||x||) at k=1,"
  echo "#   so ||x|| = sinh(r) and the coordinate direction IS the ray direction. The ANGLE is also"
  echo "#   k-independent, since curvature only rescales radius -- so this column is safe at any k."
  echo "# *** READ THIS BEFORE INTERPRETING THE RESULT ***"
  echo "#   theta_o IS EXACTLY arccos(cos_o). Measured against the law-of-cosines route in float64:"
  echo "#   agreement to 1.6e-15 at r 1.61, 3.3e-15 at 0.25, 4.6e-13 at 0.02, 1.9e-08 at 4e-04"
  echo "#   (that last one is cos_o being the looser of the two, not theta_o). So this arm carries"
  echo "#   IDENTICAL INFORMATION to the cos_o column -- it is a monotone reparameterisation, not a"
  echo "#   new feature. The MLP is nonlinear so the arms are not literally equivalent, but any gain"
  echo "#   here is CONDITIONING or NUMERICS, not new geometry. Do not report it as a new signal."
  echo "#   The right reference is therefore cos_o, not just master."
  echo "# NUMERICS ARE BETTER, BY THREE ORDERS AT THE WORST SCALE. float32 err vs float64 truth:"
  echo "#   theta_o is 1.2e-07 FLAT at r = 4e-04, 3.9e-03, 0.02 and 1.61. arccos(cos_o) is 7.7e-07,"
  echo "#   2.3e-06, 2.5e-04 and 7.2e-07 at those same radii. atan2 is accurate at small angles"
  echo "#   where arccos is not, and no cosh DIFFERENCE is formed, so the cancellation that"
  echo "#   triangle_cos exists to dodge cannot arise here at all."
  echo "# THE CONDITIONING ARGUMENT IS NOT VERIFIED -- it is the open question this arm answers."
  echo "#   The hope: arccos expands the region near cos=+1 where tokens pile up, and cos_o was"
  echo "#   measured to clamp 0.72% of tokens at +1 at r_mean 1.61, where clamping zeroes the"
  echo "#   gradient; theta_o has no clamp. BUT that 0.72% could NOT be reproduced: with"
  echo "#   independent random directions at d=64 the pairs are near-orthogonal (theta_o mean"
  echo "#   1.5705, cos_o mean 0.0003) and nothing clamps at all, so a synthetic draw is not a"
  echo "#   usable proxy for trained embeddings. Treat the clamping story as UNTESTED."
  echo "# COS_O REFERENCES, seed 5, same config -- THE ARM TO BEAT:"
  echo "#   YouTube 0.5867  Flickr 0.6368  ML-20M 0.2518  Yelp 0.6610  WikiLink 0.6554"
  echo "#   NOTE: four of those five (all but YouTube) were run at 2d7b739, BEFORE the triangle_cos"
  echo "#   numerics fix that was worth +0.0398 on YouTube. So they are plausibly conservative, and"
  echo "#   a theta_o win over them on those four datasets is partly a numerics win, not a shape win."
  echo "# RELATED NULL RESULT, so do not re-run it: --lr-pooler 1e-4 on cos_o YouTube held logit_sd"
  echo "#   at 5.6-6.2 against cos_o 9.5-13.6 for 60 epochs -- the intended 2x unsharpening -- and"
  echo "#   scored 0.5786 max test, BELOW cos_o 0.5834 and master 0.6149. Over-sharpening is a"
  echo "#   CORRELATE of the cos_o deficit, not its cause. See GEOMETRIC_FEATURES_REPORT.md sec 3a."
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
