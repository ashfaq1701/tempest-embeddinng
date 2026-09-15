#!/bin/bash
# Taobao -> Yelp -> WikiLink, d=64 K=5 seed 5, sequential on the one A6000.
# head: score = w[0]*(-d_H) + w[1]*(d0_u * d0_v),  w init [1, 1]
# loss: softmax cross-entropy (the standard loss; NOT the BCE variant)
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
OUT=$WD/logs/d0ud0v/d64_k5/run_1
cd "$WD" || exit 1
mkdir -p "$OUT"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)

# Cheapest first, so the most finishes soonest. Reference costs per epoch, from
# experiment_logs/geometries/lorentz/*/3.log (commit 07dcc1e6, same loss, baseline head):
#   Taobao   1374s train + 212s eval = 26.4 min/ep, ref run 9 ep  -> ~4.0h
#   Yelp     1789s train + 142s eval = 32.2 min/ep, ref run 19 ep -> ~10.2h
#   WikiLink 2346s train + 286s eval = 43.9 min/ep, ref run 20 ep -> ~14.6h
# ~29h for all three. This does NOT fit one night; it is a two-day chain.
for SPEC in "Taobao:--is-bipartite" "Yelp:--is-bipartite" "WikiLink:"; do
  DS="${SPEC%%:*}"; BIP="${SPEC##*:}"
  LOG="$OUT/$DS.log"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset $DS \
--d-emb 64 --k-train 5 --num-walks-per-node 5 --lr 1e-3 --seed 5 --early-stop-patience 5 \
$BIP --use-gpu --use-gpu-tempest"
  {
    echo "# dataset=$DS  d_emb=64 k_train=5 lr=1e-3 seed=5 patience 5 cap 100"
    echo "# experiment=d0ud0v cell=d64_k5 tag=run_1  (chained; see DRIVER.log)"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
    echo "# head: score = w[0]*(-d_H) + w[1]*(d0_u * d0_v), w init [1,1]. 163 head params."
    echo "#       LOSS: softmax cross-entropy -- the standard loss, NOT the BCE variant."
    echo "#"
    echo "# WHAT THIS IS. The CE control. It changes exactly ONE thing against the seed-5"
    echo "#   reference column: the scorer. Same loss, same flags, same seed. Whatever moves"
    echo "#   here is attributable to d0u*d0v and nothing else."
    echo "#"
    echo "# WHY IT MATTERS. d0u*d0v is the best head on this suite -- 0.5993 on YouTube seed 5"
    echo "#   against the baseline 0.5640, and the only head positive on BOTH the YouTube and"
    echo "#   ML-20M orderings, which otherwise anti-correlate at Spearman -0.68. It has never"
    echo "#   been run on these three. The BCE variant of this same head was launched first and"
    echo "#   aborted at Taobao ep0 precisely because it stacks two changes and this control"
    echo "#   did not exist yet; logs/d0ud0v_bce/ holds that abandoned start."
    echo "#"
    echo "# WHY THESE THREE. The big unexplained deficits against the CRAFT/SGNN-HN bar:"
    echo "#   Taobao -17.55, Yelp -10.65, WikiLink -9.56. Scorer changes have historically moved"
    echo "#   this suite by +/-0.04 at most, so the honest prior is that this does NOT close them."
    echo "#   A +0.01-0.04 move would still be the best scorer evidence available on these sets."
    echo "#"
    echo "# SEED-5 CE REFERENCE (commit 07dcc1e6, baseline head geo_temp*(-d)), the number to beat:"
    echo "#   Taobao   val 0.5536  test 0.5313  stop ep4  (ran 9)   bar 70.68  deficit -17.55"
    echo "#   Yelp     val 0.6633  test 0.6204  stop ep14 (ran 19)  bar 72.69  deficit -10.65"
    echo "#   WikiLink val 0.6682  test 0.6592  stop ep15 (ran 20)  bar 75.48  deficit -9.56"
    echo "#"
    echo "# WATCH: (a) w[1] SIGN -- the radial force rule has held 5/5: positive w[1] compacts"
    echo "#   the table, negative expands it. On YouTube this head ran w[1] NEGATIVE from ep1"
    echo "#   (-0.350, then -0.940) and expanded hard;  (b) w[1] IS INERT AT INIT by construction"
    echo "#   -- d0_u*d0_v ~ 1e-6 at INIT_IRANGE=1e-3, per-row spread grows 244,643x over training."
    echo "#   That earned-range property is the best predictor found so far: every head that"
    echo "#   CAPTURED the score early finished badly;  (c) r_max AGAINST 9.011, the float32"
    echo "#   geometry limit. This matters more here than on YouTube: the CE WikiLink reference"
    echo "#   already reached r_mean 4.03 / r_max 6.23, and this head on YouTube ran r_max to"
    echo "#   12.67 -- past the limit. There is NO projx guard on this branch (179693f1 removed"
    echo "#   it). expmap clamps the geodesic step at 45.055 so a NaN crash is averted, but"
    echo "#   geometry above 9.011 is approximate and numbers from there are suspect;"
    echo "#   (d) VAL/TEST DRIFT -- record peak test AND test@val-checkpoint separately. They"
    echo "#   diverge: the gromov-BCE run lost 0.0021 with val rising while test fell."
    echo "#"
    echo "# WikiLink CAVEAT: CLAUDE.md records the seed-5 run as NOT converged -- stopped at ep15"
    echo "#   by patience on a curve still climbing, while the historical record run reached ep44"
    echo "#   and scored 79.04. Patience is left at 5 for comparability with the reference column,"
    echo "#   so this run may stop early for the same reason. A short WikiLink run is NOT evidence"
    echo "#   against the head."
    echo "#"
    echo "# Taobao CAVEAT: the reference stopped at ep4 of 9. Deficit and early stopping correlate"
    echo "#   across this whole table and it is a confound -- an under-trained run and a"
    echo "#   capacity-limited one look identical. Unlike ML-20M, Taobao has never been checked."
    echo "#"
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  echo "[$(date '+%F %T')] START $DS" >> "$OUT/DRIVER.log"
  $CMD >> "$LOG" 2>&1
  RC=$?
  echo "# finished=$(date '+%F %T')  rc=$RC" >> "$LOG"
  echo "[$(date '+%F %T')] END   $DS rc=$RC" >> "$OUT/DRIVER.log"
done
echo "[$(date '+%F %T')] CHAIN COMPLETE" >> "$OUT/DRIVER.log"
