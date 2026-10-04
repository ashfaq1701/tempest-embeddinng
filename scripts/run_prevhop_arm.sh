#!/usr/bin/env bash
# cos_o pooler, KERNEL-2 CAUSAL IN TIME: each token sees itself + the one OLDER hop (index l-1).
#   usage: run_prevhop_arm.sh <CODE> <EXP> <DS> <yes/no bip> <TAG> <PAT> <SEED> <EPOCHS> <WPN> <MWL>
set -u
CODE="${1:?}"; EXPERIMENT="${2:?}"; DS="${3:?}"; BIP="${4:?}"; TAG="${5:?}"; PAT="${6:?}"
SEED="${7:?}"; EPOCHS="${8:?}"; WPN="${9:?}"; MWL="${10:?}"
MASTER=/mnt/nfs2/inf/ms2420/tempest-embeddinng
PY=$MASTER/venv/bin/python
LR=1e-3; HID=32
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CODE"

M=link_property_prediction/model.py; W=link_property_prediction/walk_tokens.py
TR=link_property_prediction/trainer.py; S=scripts/train_link_property_prediction.py
# --- [Q, K, L] tokens, which this arm needs to have a notion of "prior hop" at all ---
grep -q '\[Q, K, L\] fields (walk axis explicit)' $W || { echo "ABORT: walk_tokens is not the [Q,K,L] version" >&2; exit 3; }
grep -q 'Flattened \[Q, T\] fields' $W && { echo "ABORT: walk_tokens is still the flat [Q,T] version" >&2; exit 3; }
grep -q 'shuffle_walk_order=False' link_property_prediction/walks.py || { echo "ABORT: shuffle_walk_order is not False -- walk order would be scrambled and 'prior hop' meaningless" >&2; exit 3; }
grep -q 'Seed sits at row position ``lens-1``' link_property_prediction/walks.py || { echo "ABORT: walk layout contract changed -- re-derive which index is the prior hop" >&2; exit 3; }
# --- the two-linear causal mix, and the DIRECTION ---
grep -q 'self.w_self = nn.Linear(self.hidden, self.hidden)' $M || { echo "ABORT: w_self missing" >&2; exit 3; }
grep -q 'self.w_prev = nn.Linear(self.hidden, self.hidden, bias=False)' $M || { echo "ABORT: w_prev missing, or it has a bias (which breaks the w_prev=0 reduction to master)" >&2; exit 3; }
grep -q 'h_prev = F.pad(h\[:, :, :-1\], (0, 0, 1, 0))' $M || { echo "ABORT: h_prev is not the l-1 shift. Index 0 is the OLDEST hop and lens-1 is the seed (measured: ages 50,40,30,20,10,0 along increasing index), so l-1 is the OLDER hop and l+1 would read the token's own FUTURE" >&2; exit 3; }
grep -q 'self.act(self.w_self(h) + self.w_prev(h_prev))' $M || { echo "ABORT: mix is not act(w_self(h) + w_prev(h_prev))" >&2; exit 3; }
grep -q 'self.stem = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU())' $M || { echo "ABORT: per-token stem missing" >&2; exit 3; }
grep -q 'self.head = nn.Linear(self.hidden, 1)' $M || { echo "ABORT: head missing" >&2; exit 3; }
grep -q 'self.stem(feats) \* m.unsqueeze(-1)' $M || { echo "ABORT: padding not zeroed before the shift" >&2; exit 3; }
grep -q 'nn.Conv1d' $M                     && { echo "ABORT: a Conv1d is present -- this arm is the explicit two-linear form, not the conv" >&2; exit 3; }
grep -q 'kernel_size' $M                   && { echo "ABORT: kernel_size machinery present -- wrong arm" >&2; exit 3; }
# --- the cos_o geometry, unchanged from master ---
grep -q 'torch.stack(\[age, pos, a, cos_o\], dim=-1)' $M || { echo "ABORT: features are not [age, pos, d_mid, cos_o]" >&2; exit 3; }
grep -q 'cos_o = triangle_cos(b, c, a) \* m' $M || { echo "ABORT: cos_o not from triangle_cos(r_tok, r_mid, d_mid)" >&2; exit 3; }
grep -q 'def triangle_cos' $M              || { echo "ABORT: triangle_cos helper missing" >&2; exit 3; }
grep -q 'use_e = (p < small) & (q < small)' $M || { echo "ABORT: no Euclidean small-side branch" >&2; exit 3; }
grep -q 'return out.clamp(-1.0, 1.0)' $M   || { echo "ABORT: triangle_cos does not clamp to [-1,1]" >&2; exit 3; }
grep -q 'self.n_feat = 4' $M               || { echo "ABORT: n_feat is not 4" >&2; exit 3; }
grep -q 'dims = tuple(range(feat.dim() - 1))' $M || { echo "ABORT: standardiser is not rank-agnostic" >&2; exit 3; }
grep -q 'logits.masked_fill(~valid_flat, float("-inf")), dim=-1' $M || { echo "ABORT: not a single softmax over all K*L tokens" >&2; exit 3; }
# --- plumbing: n_layers gone, no stray knobs ---
grep -rq 'n_layers' --include=*.py link_property_prediction $S && { echo "ABORT: n_layers plumbing still present in the package or train script" >&2; exit 3; }
grep -q 'kernel-size-pooler\|lr-pooler\|pooler-wd' $S && { echo "ABORT: a pooler knob from another arm is plumbed" >&2; exit 3; }
# --- nothing from another arm ---
grep -q 'nn.MultiheadAttention\|nn.GRU\|nn.LayerNorm' $M && { echo "ABORT: attention/GRU/LayerNorm present" >&2; exit 3; }
grep -q 'q_off\|n_queries\|logmap0\|log_alpha\|d0_mid\|d0_tok\|theta_o\|angle_at_origin' $M && { echo "ABORT: machinery from another arm is present" >&2; exit 3; }
grep -q 'bag_mean\|_DENOM_FLOOR' $M        && { echo "ABORT: bag-mean normaliser present" >&2; exit 3; }

CELL=d64_k5_wpn${WPN}_mwl${MWL}_lr${LR}_pat${PAT}_h${HID}_prev
OUT=$MASTER/logs/$EXPERIMENT/$CELL/$TAG
mkdir -p "$OUT"; LOG=$OUT/$DS.log
BRANCH=$(git rev-parse --abbrev-ref HEAD); COMMIT=$(git rev-parse --short HEAD)
DIRTY=""; [ -n "$(git status --porcelain -- link_property_prediction scripts)" ] && \
  DIRTY="  *** WORKING TREE DIRTY -- SHA does NOT describe the code that ran ***"
BIPFLAG=""; [ "$BIP" = "yes" ] && BIPFLAG="--is-bipartite"

CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $DS \
--data-root $MASTER/datasets --d-emb 64 --k-train 5 --num-walks-per-node $WPN --max-walk-len $MWL \
--lr $LR --num-epochs $EPOCHS --early-stop-patience $PAT --hidden-dim $HID \
--seed $SEED --use-gpu --use-gpu-tempest $BIPFLAG"

{
  echo "# dataset=$DS bipartite=$BIP (flag: '${BIPFLAG:-none}')"
  echo "# experiment=$EXPERIMENT cell=$CELL tag=$TAG   *** NOT ARCHIVED ***"
  echo "# ARM = $BRANCH $COMMIT: cos_o pooler, KERNEL-2 CAUSAL mix over the walk."
  echo "#   h = stem(feats) * mask                      per token, 0 on padding"
  echo "#   h_prev = F.pad(h[:, :, 1:], (0, 0, 0, 1))   index l+1"
  echo "#   h = GELU(w_self(h) + w_prev(h_prev))"
  echo "#   logits = head(h), ONE softmax over all K*L tokens -> Lorentz midpoint."
  echo "# DIRECTION, MEASURED: walks are stored in TIME ORDER. A real walk (chain 0->1->..->5"
  echo "#   at t=10..60, backward from node 5 at cutoff 60) gives ages 50,40,30,20,10,0 along"
  echo "#   increasing index, so index 0 is the OLDEST hop and lens-1 is the seed, and rising"
  echo "#   index is FORWARD IN TIME. This arm reads l-1, the OLDER hop: each token is"
  echo "#   conditioned on what preceded it, the oldest hop has no prior, and the seed -- the"
  echo "#   token closest to the prediction -- sees the hop just before it."
  echo "#   DO NOT call l+1 'the hop it was reached from'. That is walk-GENERATION order (the"
  echo "#   sampler starts at the seed and steps backward), the REVERSE of time, and that exact"
  echo "#   phrasing made the first version of this arm read every token's own FUTURE."
  echo "# STRICT SUPERSET OF MASTER, AND THAT IS THE POINT. w_prev has no bias, so w_prev = 0"
  echo "#   collapses the pooler to stem -> w_self -> GELU -> head, which IS master's 2-layer"
  echo "#   shape. Measured bitwise equal to a hand-built master reference, max|diff| 0.000e+00."
  echo "#   Pooler params 1,249 -> 2,273, and the delta is EXACTLY w_prev's 1,024 weights:"
  echo "#   stem 160 + w_self 1,056 + w_prev 1,024 + head 33. So the arm's entire hypothesis"
  echo "#   is 'how much does the prior hop matter', with master sitting at w_prev = 0."
  echo "#   Unlike the conv arms, depth and context are NOT confounded here -- the k=1-style"
  echo "#   control is w_prev = 0, which is master itself, already measured."
  echo "# CAUSALITY VERIFIED by perturbing one slot and reading which logits move:"
  echo "#   index 0 -> [0];  index 2 -> [1, 2];  index 3 -> [2, 3];  index 5 -> [4, 5]."
  echo "#   Each logit depends on its own token and l+1, nothing else. Walks meet only in the"
  echo "#   final softmax. Empty batch (Q=0) builds and runs, returning [0, d]."
  echo "# CAVEAT: a missing prior is a ZERO VECTOR, not a flagged channel, so a real token"
  echo "#   whose stem output is near zero is indistinguishable from 'no prior'. With GELU and"
  echo "#   a bias in w_self that is unlikely to bind, but it is untested."
  echo "# --n-layers-pooler IS GONE on this branch (fixed pooler shape), so there is no nl1/nl2"
  echo "#   split: all five datasets run one shape. The WikiLink baselines below were nl1, so"
  echo "#   that one comparison mixes in a pooler-shape change."
  echo "# PRIOR ART: walk-conv-1 (48478ae) is the k=3 symmetric conv on the same [Q,K,L] bag,"
  echo "#   3,297 params. It gave YouTube 0.6105 (TRUNCATED at the ep100 cap, still gaining),"
  echo "#   Flickr 0.6365, ML-20M 0.2567 (a project record), Yelp 0.6475, WikiLink 0.6348."
  echo "#   Its WikiLink val THRASHED (sd 0.093, swings of 0.26 in one epoch) on top of a"
  echo "#   radius blow-up that master and cos_o ALSO have on that dataset -- so the blow-up"
  echo "#   is a WikiLink-at-lr-1e-3 property, not conv-specific, but the val instability was."
  echo "#   Watch for the same thrashing here; it is what cost that arm 0.0204 to drift."
  echo "# ACCEPTANCE TEST IS THE TRAINING LOSS, NOT MRR. Driving train link loss below master's"
  echo "#   widens the val->test gap, monotone 5/5 on YouTube. HEALTHY = loss comparable to"
  echo "#   master with better val AND test. See ATTENTION_ARMS_MESSAGE.md sec 2."
  echo "# BASELINES seed 5, same config. MASTER NOW MEANS cos_o (merged 1ed16df), w_prev = 0:"
  echo "#   cos_o    YouTube 0.5867  Flickr 0.6368  ML-20M 0.2518  Yelp 0.6610  WikiLink 0.6554"
  echo "#   pre-merge master  YouTube 0.6149  Flickr 0.6366  ML-20M 0.2469  Yelp 0.6481*  WikiLink 0.6513"
  echo "#   * hand-killed floor at ep14. YouTube: pre-merge master beats cos_o by 0.0282 there"
  echo "#   and holds that dataset against every arm tried, so YouTube is the one to watch."
  echo "#   cos_o YouTube over 3 seeds: 0.5860 / 0.5867 / 0.5914, mean 0.5880 sd 0.0029."
  echo "#   Record max test AND best_test_mrr -- they diverge, and drift has decided ties."
  echo "# d_emb=64 k_train=5 wpn=$WPN mwl=$MWL lr=$LR hidden=$HID epochs=$EPOCHS patience=$PAT seed=$SEED"
  echo "# code=$CODE (detached git worktree) commit=$COMMIT$DIRTY"
  echo "#   model.py md5=$(md5sum $M | cut -d' ' -f1)  walk_tokens.py md5=$(md5sum $W | cut -d' ' -f1)"
  echo "# host=$(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-none}  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo ""
} > "$LOG"
echo "[$(date '+%F %T')] START $DS tag=$TAG commit=$COMMIT host=$(hostname) job=${SLURM_JOB_ID:-none}" >> "$OUT/DRIVER.log"
$CMD >> "$LOG" 2>&1
rc=$?
echo "=== rc=$rc finished=$(date '+%F %T') ===" >> "$LOG"
MAXT=$(grep -oE 'test [0-9.]+ \(new best\)' "$LOG" | grep -oE '[0-9.]+' | sort -g | tail -1)
echo "[$(date '+%F %T')] DONE  $DS rc=$rc $(grep -oE 'best_test_mrr: *[0-9.]+' "$LOG" | tail -1)  max_test=${MAXT:-none}  eps=$(grep -c '^epoch' "$LOG")" >> "$OUT/DRIVER.log"
