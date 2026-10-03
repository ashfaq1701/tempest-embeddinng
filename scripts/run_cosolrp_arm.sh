#!/usr/bin/env bash
# MASTER + cos_o, DIAGNOSTIC build, WITH --lr-pooler: a separate lr for the pooler MLP only.
#   DIAGNOSTIC build: grad norms, realised update size, pooling sharpness.
#   usage: run_cosolrp_arm.sh <CODE> <EXP> <DS> <yes/no bip> <TAG> <PAT> <SEED> <EPOCHS> <WPN> <MWL> <NL> <LRP>
set -u
CODE="${1:?}"; EXPERIMENT="${2:?}"; DS="${3:?}"; BIP="${4:?}"; TAG="${5:?}"; PAT="${6:?}"
SEED="${7:?}"; EPOCHS="${8:?}"; WPN="${9:?}"; MWL="${10:?}"; NL="${11:?}"; LRP="${12:?}"
MASTER=/mnt/nfs2/inf/ms2420/tempest-embeddinng
PY=$MASTER/venv/bin/python
LR=1e-3; HID=32
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"

M=link_property_prediction/model.py; W=link_property_prediction/walk_tokens.py
grep -q 'torch.stack(\[age, pos, a, cos_o\], dim=-1)' $M || { echo "ABORT: features are not [age, pos, d_mid, cos_o]" >&2; exit 3; }
grep -q 'cos_o = triangle_cos(b, c, a) \* u' $M || { echo "ABORT: cos_o not from triangle_cos(r_tok, r_mid, d_mid)" >&2; exit 3; }
grep -q 'def triangle_cos' $M             || { echo "ABORT: triangle_cos helper missing" >&2; exit 3; }
grep -q 'self._diag = pooling_probe(' $M  || { echo "ABORT: pooling_probe not wired into forward" >&2; exit 3; }
grep -q 'from .probe import pooling_probe' $M || { echo "ABORT: probe module not imported" >&2; exit 3; }
[ -f link_property_prediction/probe.py ]    || { echo "ABORT: probe.py missing" >&2; exit 3; }
grep -q 'def pooling_probe' link_property_prediction/probe.py || { echo "ABORT: probe.py lacks pooling_probe" >&2; exit 3; }
grep -q 'def tail_probe' link_property_prediction/probe.py || { echo "ABORT: probe.py lacks tail_probe" >&2; exit 3; }
grep -q 'def _tau_sweep' link_property_prediction/trainer.py || { echo "ABORT: tau sweep missing" >&2; exit 3; }
grep -q 'logit_tau' $M                    || { echo "ABORT: logit_tau knob missing from the pooler" >&2; exit 3; }
grep -q '_deg_tail' link_property_prediction/trainer.py || { echo "ABORT: degree-decile setup missing" >&2; exit 3; }
grep -q 'grad_tail_frac' link_property_prediction/trainer.py || { echo "ABORT: grad_tail_frac missing" >&2; exit 3; }
grep -q 'with torch.no_grad():' $M       || { echo "ABORT: diagnostics must be under no_grad" >&2; exit 3; }
grep -q 'relN.: dN / max(nN' link_property_prediction/trainer.py || { echo "ABORT: trainer lacks the realised-update probe" >&2; exit 3; }
grep -q 'logit_sd=' link_property_prediction/trainer.py || { echo "ABORT: trainer lacks the pooling-sharpness probe" >&2; exit 3; }
grep -q 'return out.clamp(-1.0, 1.0)' $M  || { echo "ABORT: triangle_cos does not clamp to [-1,1]" >&2; exit 3; }
grep -q 'use_e = (p < small) & (q < small)' $M || { echo "ABORT: no Euclidean small-side branch" >&2; exit 3; }
grep -q 'torch.where((p <= 0) | (q <= 0)' $M || { echo "ABORT: degenerate vertex not zeroed" >&2; exit 3; }
grep -q 'b = self.geom.dist0(xt)' $M      || { echo "ABORT: r_tok missing" >&2; exit 3; }
grep -q 'c = self.geom.dist0(mid).unsqueeze(-1)' $M || { echo "ABORT: r_mid missing" >&2; exit 3; }
grep -q 'self.n_feat = 4' $M              || { echo "ABORT: n_feat is not 4" >&2; exit 3; }
grep -q 'self._standardise(torch.stack' $M || { echo "ABORT: features not standardised" >&2; exit 3; }
grep -q 'q_off\|n_queries\|logmap0\|log_alpha\|d0_mid' $M && { echo "ABORT: another arm's machinery present" >&2; exit 3; }
grep -q 'nn.MultiheadAttention\|nn.Conv1d\|nn.GRU\|nn.LayerNorm' $M && { echo "ABORT: attention/conv/GRU/LayerNorm present" >&2; exit 3; }
grep -q 'Flattened \[Q, T\] fields' $W   || { echo "ABORT: walk_tokens is not the flat [Q,T] version" >&2; exit 3; }
grep -q 'bag_mean\|r_mid =\|_DENOM_FLOOR' $M && { echo "ABORT: bag-mean normaliser present" >&2; exit 3; }
grep -q 'n_layers_pooler' scripts/train_link_property_prediction.py || { echo "ABORT: --n-layers-pooler missing" >&2; exit 3; }
grep -q 'lr-pooler' scripts/train_link_property_prediction.py || { echo "ABORT: --lr-pooler not plumbed" >&2; exit 3; }
grep -q 'lr_pooler=args.lr_pooler' scripts/train_link_property_prediction.py || { echo "ABORT: --lr-pooler not passed into TrainerConfig" >&2; exit 3; }
grep -q 'lr_pooler: Optional\[float\] = None' link_property_prediction/trainer.py || { echo "ABORT: trainer config lacks lr_pooler (or its default is not None)" >&2; exit 3; }
grep -q 'name.startswith("bag_weights.net.")' link_property_prediction/trainer.py || { echo "ABORT: pooler lr not scoped to bag_weights.net.*" >&2; exit 3; }
grep -q 'param group split dropped tensors' link_property_prediction/trainer.py || { echo "ABORT: no coverage assert on the param split" >&2; exit 3; }
grep -q '{"params": nn_params, "lr": float(config.lr_pooler)}' link_property_prediction/trainer.py || { echo "ABORT: pooler group does not carry its own lr" >&2; exit 3; }
grep -q 'weight_decay' link_property_prediction/trainer.py && { echo "ABORT: weight decay present -- this arm varies lr ONLY" >&2; exit 3; }
grep -q 'lrP=' link_property_prediction/trainer.py || { echo "ABORT: epoch line does not report the pooler lr" >&2; exit 3; }
[ "$LRP" != "$LR" ] || { echo "ABORT: LRP == LR ($LRP) -- that is the single-group baseline, not this arm" >&2; exit 3; }

CELL=d64_k5_wpn${WPN}_mwl${MWL}_lr${LR}_pat${PAT}_h${HID}_nl${NL}_lrP${LRP}
OUT=$MASTER/logs/$EXPERIMENT/$CELL/$TAG
mkdir -p "$OUT"; LOG=$OUT/$DS.log
BRANCH=$(git rev-parse --abbrev-ref HEAD); COMMIT=$(git rev-parse --short HEAD)
DIRTY=""; [ -n "$(git status --porcelain -- link_property_prediction scripts)" ] && \
  DIRTY="  *** WORKING TREE DIRTY -- SHA does NOT describe the code that ran ***"
BIPFLAG=""; [ "$BIP" = "yes" ] && BIPFLAG="--is-bipartite"

CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $DS \
--data-root $MASTER/datasets --d-emb 64 --k-train 5 --num-walks-per-node $WPN --max-walk-len $MWL \
--lr $LR --lr-pooler $LRP --num-epochs $EPOCHS --early-stop-patience $PAT --hidden-dim $HID --n-layers-pooler $NL \
--seed $SEED --use-gpu --use-gpu-tempest $BIPFLAG"

{
  echo "# dataset=$DS bipartite=$BIP (flag: '${BIPFLAG:-none}')"
  echo "# experiment=$EXPERIMENT cell=$CELL tag=$TAG   *** NOT ARCHIVED ***"
  echo "# ARM = $BRANCH $COMMIT: MASTER + cos_o, the angle at the ORIGIN between the bag centre and the token."
  echo "#   feats = standardise([log1p(age), pos, d_mid, cos_o]).  n_feat 4. Pooler 1,217 -> 1,249 (1.03x)."
  echo "#   O, M, X with a = d(M,X) = d_mid,  b = d(O,X) = r_tok,  c = d(O,M) = r_mid."
  echo "#   cos_o = (cosh b cosh c - cosh a) / (sinh b sinh c)   (hyperbolic law of cosines at ORIGIN O)"
  echo "#   +1 same ray from the origin (same branch of the hierarchy), 0 unrelated, -1 opposite sides"
  echo "#   DISTANCES ONLY: no chart, no coordinate frame, so invariant under every isometry"
  echo "#   fixing the origin -- unlike a tangent-space offset, which reads the arbitrary frame."
  echo "# MEASURED CORRECT, float64, 4000 pts/scale vs the true angle: max err 4.6e-11 at r=1e-3,"
  echo "#   ~1e-15 at r 0.1-5. Ground truth from tangent vectors in the metric at the vertex."
  echo "# FLOAT32 CANCELLATION AT INIT RADII IS NOW FIXED by triangle_cos, which switches to the"
  echo "#   EUCLIDEAN law of cosines when BOTH adjacent sides are < 1e-2. Measured float32 error"
  echo "#   against the true angle: at r_mean 4e-04 the old inline form was 7.7e-01 (the column"
  echo "#   was NOISE, not signal) and triangle_cos is 3.3e-07; at 3.9e-03, 2.2e-02 -> 5.7e-06."
  echo "#   At r >= 0.02 the two forms are bit-identical (Euclidean branch unused). So epoch 1 is"
  echo "#   now meaningful, where in the earlier cos_o runs it was not."
  echo "# DISTRIBUTION at a trained-WikiLink-like scale (r_mean 1.61): mean +0.266  sd 0.154  range -0.150..+1.000  0.72% clamped at +1  corr(d_mid) -0.531"
  echo "# CONTEXT: the RADIAL columns just won WikiLink for the first time in this project --"
  echo "#   three-geom 0.6739 and d0_tok-only 0.6659 vs master 0.6513, both peaking ep15 with ZERO"
  echo "#   val/test drift -- while LOSING YouTube (-0.031 to -0.040) and Yelp. This arm is the"
  echo "#   ANGULAR complement, and YouTube/Flickr are exactly where the radial columns failed."
  echo "# DIAGNOSTIC BUILD, and this arm DELIBERATELY changes the trajectory, so epoch 1 is NOT"
  echo "#   expected to match the 1.6593 / 0.1503 / 0.1298 reference -- a lower pooler lr moves the"
  echo "#   first step. What IS still expected: init is identical (seed_all runs before model"
  echo "#   construction and the feature count is unchanged at 4), so epoch-1 DIVERGENCE must be"
  echo "#   SMALL and in the pooler only. A large epoch-1 gap means the param split is wrong."
  echo "#   The probe reads themselves remain under no_grad / on detached copies."
  echo "# EXTRA PER-EPOCH FIELDS (averaged over the epoch\x27s batches):"
  echo "#   |gE| |gNN| |gT|  grad norms of E / pooler MLP / geo_temp, after backward before the step"
  echo "#   relE relNN       REALISED ||dp||/||p||. Adam\x27s nominal step is ~lr=1e-3, so relNN >> 1e-3"
  echo "#                    with a rising loss is the overshoot signature -- measured, not inferred"
  echo "#   logit_sd         within-bag spread of the pre-softmax logits (features are unit-variance,"
  echo "#                    so this is comparable across arms)"
  echo "#   w_ent            bag softmax entropy / log(n_valid): 1.0 = UNIFORM pooling, 0.0 = all"
  echo "#                    mass on one token. NOTE: CLAUDE.md claims the parameter-free uniform"
  echo "#                    rule BEATS the learned pooler on YouTube. The tau sweep MEASURED"
  echo "#                    uniform at test 0.3415 (cos_o) and 0.3449 (master), i.e. 0.27 BELOW"
  echo "#                    the learned pooler. That claim does NOT hold in this configuration,"
  echo "#                    so uniform is a FLOOR to stay away from, not a target."
  echo "#   w_max            mean largest bag weight"
  echo "#   fsd0..3          per-feature sd of the standardised block [age, pos, d_mid, cos_o];"
  echo "#                    all should be ~1.0, a drift means the masked standardiser is misbehaving"
  echo "# WHY THIS ARM: the softmax-collapse branch of that diagnosis CAME BACK POSITIVE. cos_o\x27s"
  echo "#   logit_sd settles 9-10 against master\x27s 3.3-3.8 for 50+ epochs at near-identical w_ent,"
  echo "#   and an eval-time logits/2 on the restored best checkpoint is worth +0.0184 test and"
  echo "#   +0.0125 val on YouTube (0.5865 -> 0.6049) -- selectable on val, no test peeking. The"
  echo "#   SAME operation on master gains +0.0001, so master is already calibrated and this is a"
  echo "#   defect of the arm, not the architecture. Optimal divisor 2 ~= the measured 2.7x spread"
  echo "#   ratio: the mis-calibration and the correction are the same number."
  echo "# THE KNOB: features are standardised to unit variance and nothing downstream rescales the"
  echo "#   pooling logits -- geo_temp scales the SCORE, not the weights -- so ||W|| maps directly"
  echo "#   onto sharpness. A smaller pooler lr lets E and geo_temp train at full 1e-3 while the"
  echo "#   sharpness grows 10x slower. NOTE a fixed tau is exactly a 1/tau rescale of the output"
  echo "#   layer, so the net could already have represented the better solution and did not."
  echo "# NOT THE SAME KNOB AS --pooler-wd (feature/pooler-cos-origin-wd, both arms LOST: 1e-6"
  echo "#   0.5836, 1e-4 0.5789). Weight decay pulls toward W=0, i.e. toward uniform pooling, now"
  echo "#   MEASURED at 0.3415 and therefore a bad target; and geoopt RiemannianAdam uses coupled"
  echo "#   L2 so both settings were near no-ops anyway. A lower lr has no shrinkage target at all."
  echo "# PLUMBING VERIFIED by a step with ALL GRADS = 1.0: the 6 MLP tensors (1,249 params) move"
  echo "#   rms 1.000e-04 while geo_temp moves 1.000e-03 and E.weight 1.250e-04, unchanged from the"
  echo "#   single-group run. Split covers 8/8 named params. Default lr_pooler=None keeps ONE group."
  echo "# PRIOR FAILED INTERVENTIONS on this deficit, for reference: pooler wd 1e-6 (0.5836), wd"
  echo "#   1e-4 (0.5789), nl1 at 193 params (0.5658), wpn 10 (0.5637), d_mid/d_max (0.5680)."
  echo "#   Overshoot is RULED OUT: relNN 3.9e-4 (cos_o) vs 3.6e-4 (master) at ep11, both below lr."
  echo "# PROBE v2 -- the full field set, probe.py byte-identical on both diag branches:"
  echo "#   wcol_*        L2 norm of the first Linear weight column per input feature"
  echo "#   lsd_drop_*    logit_sd recomputed with that STANDARDISED column zeroed; the gap from"
  echo "#                 logit_sd is that column\x27s contribution to the spread"
  echo "#   w_seed w_hop2 w_old w_top3   where the softmax mass lands (seed slot, hop 2, oldest"
  echo "#                 third by age, three largest weights)"
  echo "#   w_cos_hi/lo   mass on the top/bottom decile of cos_o within each bag"
  echo "#   r_tail r_head radius of the bottom/top TRAINING-DEGREE decile"
  echo "#   disp_*        geodesic displacement of those nodes from their init position"
  echo "#   gtail         fraction of the E gradient norm landing on bottom-decile nodes"
  echo "#   d_pos d_neg ratio   pooled distance to the positive and to the negatives"
  echo "#   TAU SWEEP at the END, on the restored best weights: logits/tau before the softmax, for"
  echo "#     tau in 1/3, 1/2, 1, 2, 3, 4, inf. tau=inf IS UNIFORM POOLING, measured BAD (0.3415)."
  echo "#     THE KEY READOUT FOR THIS ARM: at lr 1e-3 the sweep peaked at tau=2 (+0.0184 test),"
  echo "#     which says the pooler trained too sharp. If a smaller pooler lr fixes that at source,"
  echo "#     this run\x27s sweep should peak at tau=1 and be FLAT to tau=2, like master is."
  echo "#     Run after training only, so the extra eval passes cannot consume walk RNG"
  echo "#     that training would later have used -- inertness is preserved by construction."
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
