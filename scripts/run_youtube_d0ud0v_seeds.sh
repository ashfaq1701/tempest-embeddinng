#!/bin/bash
# YouTube d=64 K=5, head = w[0]*(-d_H) + w[1]*(r_u*r_v), THREE SEEDS matching the baseline
# replicates (3, 42, 5). Sequential; each writes its own log.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
cd "$WD" || exit 1
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)

for SEED in 3 42 5; do
  LOG=$WD/logs/d0ud0v_seeds/d64_k5/seed${SEED}/YouTube.log
  mkdir -p "$(dirname "$LOG")"
  CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset YouTube \
--d-emb 64 --k-train 5 --num-walks-per-node 5 --lr 1e-3 --seed ${SEED} --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
  {
    echo "# dataset=YouTube (3.29M edges, NON-bipartite) d_emb=64 k_train=5 lr=1e-3 seed=${SEED} patience 5 cap 100"
    echo "# experiment=d0ud0v_seeds cell=d64_k5 tag=seed${SEED}"
    echo "# branch=$BRANCH commit=$SHA"
    [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
    echo "# head: score = w[0]*(-d_H(P_u,P_v)) + w[1]*(r_u*r_v), r_x = dist0(P_x), w init [1.0, 1.0]"
    echo "#"
    echo "# WHY THIS RUN. d0ud0v is the only head of ~16 tested that is positive on BOTH suites:"
    echo "#   YouTube +0.0367 (0.5993 vs baseline 0.5626) and ML-20M +0.0005 (0.2434 vs 0.2429)."
    echo "#   Every head with a query-INDEPENDENT additive candidate term has lost on YouTube:"
    echo "#   d0v 0.4567, gromov 0.4930, gromov-qnorm 0.5123, gromov-w2 0.5042, gromov b/l 0.4644."
    echo "#   The +0.0367 is SINGLE SEED. This run tests whether it survives three."
    echo "#"
    echo "# WHAT IT WOULD MEAN. 3-seed baseline YouTube is 55.82 +- 0.50 (55.92 / 55.28 / 56.26)."
    echo "#   The bar is CRAFT 58.92 (SGNN-HN 59.64 is excluded as a sequential-recommendation"
    echo "#   method, not a temporal graph model). If +3.67 holds on the mean, YouTube goes"
    echo "#   55.82 -> ~59.5, turning a -3.10 loss into a ~+0.6 win and giving a SECOND column"
    echo "#   alongside GoogleLocal. That projection applies a single-seed delta to a three-seed"
    echo "#   mean, which is exactly what this run exists to check."
    echo "#"
    echo "# BASELINE PER SEED (same flags, geo_temp*(-d_H)): seed3 0.5592 | seed42 0.5528 | seed5 0.5626"
    echo "#"
    echo "# WATCH: (a) THE ESCAPE -- d0ud0v escaped ep16-19 on seed 5 (+0.109, +0.035, +0.009),"
    echo "#   LATER and HIGHER than the baseline's ep14-17. A seed that does not escape is the"
    echo "#   failure mode;  (b) w -- seed 5 converged to [39.9, -14.0], BOTH terms penalising,"
    echo "#   with w[1] crossing zero around ep10-20;  (c) link@5 -- the winners' band on this"
    echo "#   dataset is 0.72-0.77 and seed 5 was 0.757."
    echo "# ~94 s/epoch train + 27 s eval, runs went 19-28 epochs; budget ~50 min per seed."
    echo "# started=$(date '+%F %T')"
    echo "# cmd: $CMD"
    echo
  } > "$LOG"
  $CMD >> "$LOG" 2>&1
  echo "# finished=$(date '+%F %T')" >> "$LOG"
done
