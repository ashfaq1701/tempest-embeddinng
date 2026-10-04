#!/usr/bin/env bash
# cos_o pooler + EMA, nl2: stem -> memory -> master layer 2 -> head. OLDEST -> SEED.
#   usage: run_walkemanl2_arm.sh <CODE> <EXP> <DS> <yes/no bip> <TAG> <PAT> <SEED> <EPOCHS> <WPN> <MWL>
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
# --- [Q, K, L] tokens, which this arm needs to have a notion of hop order at all ---
grep -q '\[Q, K, L\] fields (walk axis explicit)' $W || { echo "ABORT: walk_tokens is not the [Q,K,L] version" >&2; exit 3; }
grep -q 'Flattened \[Q, T\] fields' $W && { echo "ABORT: walk_tokens is still the flat [Q,T] version" >&2; exit 3; }
grep -q 'shuffle_walk_order=False' link_property_prediction/walks.py || { echo "ABORT: shuffle_walk_order is not False -- hop order would be scrambled" >&2; exit 3; }
grep -q 'Seed sits at row position ``lens-1``' link_property_prediction/walks.py || { echo "ABORT: walk layout contract changed -- re-derive the time direction" >&2; exit 3; }
grep -q 'lag = pos.unsqueeze(-2) - pos.unsqueeze(-1)' $M || { echo "ABORT: lag is not hop_j - hop_l. Index 0 is the OLDEST hop and lens-1 is the seed (measured: ages 50,40,30,20,10,0 along increasing index), so hop_j - hop_l >= 0 selects the PAST; the reverse makes every token average its own FUTURE" >&2; exit 3; }
grep -q 'def _walk_memory' $M               || { echo "ABORT: _walk_memory missing" >&2; exit 3; }
grep -q 'alpha = torch.sigmoid(self.log_alpha)' $M || { echo "ABORT: alpha is not sigmoid(log_alpha)" >&2; exit 3; }
grep -q 'self.log_alpha = nn.Parameter' $M  || { echo "ABORT: log_alpha is not a Parameter" >&2; exit 3; }
grep -q 'torch.einsum("qklj,qkjh->qklh", W, h)' $M || { echo "ABORT: tap contraction changed" >&2; exit 3; }
grep -q 'den = W.sum(dim=-1, keepdim=True).clamp_min(_TAP_FLOOR)' $M || { echo "ABORT: tap-sum normaliser missing -- h would grow with depth" >&2; exit 3; }
grep -q 'keep = (lag >= 0) & valid.unsqueeze(-1) & valid.unsqueeze(-2)' $M || { echo "ABORT: tap mask changed" >&2; exit 3; }
grep -q 'self.stem = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU())' $M || { echo "ABORT: per-token stem missing" >&2; exit 3; }
grep -q 'self.head = nn.Linear(self.hidden, 1)' $M || { echo "ABORT: head missing" >&2; exit 3; }
grep -q 'self.mix = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.GELU())' $M || { echo "ABORT: mix (master layer 2) missing -- that is the nl1 arm, not nl2" >&2; exit 3; }
grep -q 'h = self.mix(h)' $M              || { echo "ABORT: mix not applied" >&2; exit 3; }
grep -nq 'h = self._walk_memory(h, pos, valid)' $M && [ "$(grep -n 'h = self._walk_memory' $M | cut -d: -f1)" -lt "$(grep -n 'h = self.mix(h)' $M | cut -d: -f1)" ] || { echo "ABORT: memory must come BEFORE mix (stem -> memory -> layer2 -> head)" >&2; exit 3; }
grep -q 'self.stem(feats) \* m.unsqueeze(-1)' $M || { echo "ABORT: padding not zeroed before the memory" >&2; exit 3; }
grep -q 'Memory: oldest -> seed' $M        || { echo "ABORT: class docstring does not state the oldest->seed direction" >&2; exit 3; }
grep -q 'w_self\|w_prev\|h_prev\|nn.Conv1d' $M && { echo "ABORT: prev-hop or conv machinery present -- wrong arm" >&2; exit 3; }
# --- the cos_o geometry, unchanged from master ---
grep -q 'torch.stack(\[age, pos, a, cos_o\], dim=-1)' $M || { echo "ABORT: features are not [age, pos, d_mid, cos_o]" >&2; exit 3; }
grep -q 'cos_o = triangle_cos(b, c, a) \* m' $M || { echo "ABORT: cos_o not from triangle_cos(r_tok, r_mid, d_mid)" >&2; exit 3; }
grep -q 'def triangle_cos' $M              || { echo "ABORT: triangle_cos helper missing" >&2; exit 3; }
grep -q 'use_e = (p < small) & (q < small)' $M || { echo "ABORT: no Euclidean small-side branch" >&2; exit 3; }
grep -q 'return out.clamp(-1.0, 1.0)' $M   || { echo "ABORT: triangle_cos does not clamp to [-1,1]" >&2; exit 3; }
grep -q 'self.n_feat = 4' $M               || { echo "ABORT: n_feat is not 4" >&2; exit 3; }
grep -q 'dims = tuple(range(feat.dim() - 1))' $M || { echo "ABORT: standardiser is not rank-agnostic" >&2; exit 3; }
grep -q 'logits.masked_fill(~valid_flat, float("-inf")), dim=-1' $M || { echo "ABORT: not a single softmax over all K*L tokens" >&2; exit 3; }
grep -rqE 'n_layers_pooler|self\.n_layers|n_layers: int' --include=*.py link_property_prediction $S && { echo "ABORT: n_layers PLUMBING still present. A prose mention in a docstring is fine -- this matches only the config field, the attribute and the annotation" >&2; exit 3; }
grep -q 'kernel-size-pooler\|lr-pooler\|pooler-wd' $S && { echo "ABORT: a pooler knob from another arm is plumbed" >&2; exit 3; }
grep -q 'nn.MultiheadAttention\|nn.GRU\|nn.LayerNorm' $M && { echo "ABORT: attention/GRU/LayerNorm present" >&2; exit 3; }
grep -q 'q_off\|n_queries\|logmap0\|d0_mid\|d0_tok\|theta_o\|angle_at_origin' $M && { echo "ABORT: machinery from another arm is present" >&2; exit 3; }
grep -q 'bag_mean\|_DENOM_FLOOR' $M        && { echo "ABORT: bag-mean normaliser present" >&2; exit 3; }

CELL=d64_k5_wpn${WPN}_mwl${MWL}_lr${LR}_pat${PAT}_h${HID}_ema_nl2
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
  echo "# ARM = $BRANCH $COMMIT: cos_o pooler + single-rate normalised EMA along each walk, OLDEST -> SEED (forward in time)."
  echo "#   h = stem(feats) * mask                            per token, 0 on padding"
  echo "#   hm_p = sum over q>=p of a^(q-p) h_q, divided by sum over q>=p of a^(q-p)"
  echo "#        = faded average over real hops q at or OLDER than p"
  echo "#   h = mix(hm)   master layer 2, ON the memory -- same slot the prev-hop/conv arms mix at"
  echo "#   logits = head(h), ONE softmax over all K*L tokens -> Lorentz midpoint."
  echo "#   a = sigmoid(log_alpha), ONE rate for every channel, init 0.5. Hop-lag decay matrix, no"
  echo "#   sequential scan; invalid taps are masked so padding rows land at exactly 0."
  echo "# 1,250 PARAMS = master nl2 (1,249) + log_alpha. stem 160, mix 1,056, head 33. alpha reads"
  echo "#   directly as how fast the path fades per hop."
  echo "# VERIFIED NUMERICALLY on a real chain walk (h set to 1..6 so each value names its own slot):"
  echo "#   alpha->0 (1e-9) : hm = [1,2,3,4,5,6] -- every token keeps its own h"
  echo "#   alpha->1 (0.999): hm = [1.000,1.500,2.001,2.501,3.002,3.503]"
  echo "#     oldest hop 1.000000 exactly (no history); seed 3.502918 vs a six-hop mean of 3.5."
  echo "#     The 0.0029 excess is the O(1-alpha) tilt to recent hops, NOT an error: taps a^0..a^5"
  echo "#     on h = 6,5,4,3,2,1 give 20.965035/5.985020 = 3.502918 by hand. alpha 0.99999 gives"
  echo "#     3.500029 and 0.9999999 gives 3.500000."
  echo "#   alpha->0 also reduces the WHOLE pooler to master nl2 BITWISE on real slots (log_alpha"
  echo "#   = -40, max|diff| 0.00e+00), so master NL2 is exactly the alpha->0 limit of this arm."
  echo "# DIRECTION, MEASURED: walks are stored in TIME ORDER. A real walk (chain 0->1->..->5 at"
  echo "#   t=10..60, backward from node 5 at cutoff 60) gives ages 50,40,30,20,10,0 along increasing"
  echo "#   index, so index 0 is the OLDEST hop, lens-1 is the seed, and rising index is FORWARD IN"
  echo "#   TIME. Each token is conditioned on what PRECEDED it."
  echo "#   DO NOT call l+1 the hop it was reached from: that is walk-GENERATION order (the sampler"
  echo "#   starts at the seed and steps backward), the REVERSE of time. That exact phrasing made the"
  echo "#   FIRST versions of these arms read every token own FUTURE, and five runs were launched and"
  echo "#   killed on it. The guards above pin the direction so it cannot come back silently."
  echo "# nl2 FAMILY: matched to cos_o nl2 (1,249) on YouTube/Flickr/ML-20M/Yelp. NOTE there is NO"
  echo "#   cos_o nl2 WikiLink baseline at all, so the WikiLink cell has no matched reference."
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
