#!/bin/bash
# Patent d=64 K=5 on MASTER's pooler, single lr raised 1e-3 -> 3e-3.
PY=/its/home/ms2420/tempest-embeddinng/venv/bin/python
WD=/its/home/ms2420/tempest-embeddinng
LOG=$WD/logs/patent_lr3e3/d64_k5/run_1/Patent.log
cd "$WD" || exit 1
mkdir -p "$(dirname "$LOG")"
SHA=$(git rev-parse --short HEAD); BRANCH=$(git rev-parse --abbrev-ref HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
CMD="$PY -u scripts/train_link_property_prediction.py --data-suite tgb-seq --dataset Patent \
--d-emb 64 --k-train 5 --lr 3e-3 --num-epochs 50 --early-stop-patience 5 \
--use-gpu --use-gpu-tempest"
{
  echo "# dataset=Patent (NON-bipartite) d_emb=64 k_train=5  lr=3e-3 (SINGLE group)  NO pop bias  seed=42"
  echo "# experiment=patent_lr3e3 cell=d64_k5 tag=run_1"
  echo "# branch=$BRANCH commit=$SHA"
  [ -n "$DIRTY" ] && echo "# WARNING: uncommitted tracked changes -- SHA does not describe the code that ran"
  echo "# head: score = geo_temp * (-d_H), geo_temp LINEAR init 1.0"
  echo "#       pooling = softmax(MLP([log1p(age) | raw pos | rad])), width 32, 162 head params"
  echo "#       nothing standardised; no dataset-derived constant in the pooler"
  echo "#"
  echo "# THE ONE VARIABLE: lr 1e-3 -> 3e-3. Everything else is master as-is."
  echo "#"
  echo "# WHY: lr has NEVER been varied on this suite. All ~30 TGB-Seq runs in logs/ used 1e-3;"
  echo "#   the older tgbl-* era used 1e-4. Meanwhile three separate findings were all fixed by"
  echo "#   speeding SOMETHING up -- log-parameterised geo_temp (+0.0152 YouTube), --lr-temp at"
  echo "#   1e-2 (+0.0080, peak at ep15 not ep25), and the two-group split that took Patent"
  echo "#   0.1630 -> 0.2633. Each addressed a relative rate; the base rate was never questioned."
  echo "#"
  echo "# PATENT-SPECIFIC REASON TO EXPECT lr TO MATTER HERE. The Patent field notes measured the"
  echo "#   original collapse as an INPUT-SCALE defect: rec = -(age/mnia) had cross-token std 0.026"
  echo "#   against pos's 1.417, a 54x imbalance, so the first-layer weight had to grow to ~31.4"
  echo "#   before recency mattered -- 3.6 epochs at 1e-3 with 9,100 steps/epoch, and the run died"
  echo "#   at ep4. At 3e-3 that same growth takes 1.2 epochs."
  echo "#   NOTE master should ALREADY have fixed that imbalance: log1p(age) on Patent spans"
  echo "#   13.3-20.0 (ages one week to 5,518 days), std ~1.5 against pos ~1.4. So if lr still"
  echo "#   helps here it is NOT the same mechanism, and that is the informative outcome."
  echo "#"
  echo "# PATENT REFERENCES (d=64 K=5 seed 42, from logs/*/d64_k5/*/Patent.log):"
  echo "#   split lr, E@1e-3 temp+pooler@1e-2, mnia scalar pooler : 0.2633 @ep4   <- record"
  echo "#   encoded pooler (TimeEncoding+pos-emb, 762p), single lr: 0.2506 @ep5"
  echo "#   parameter-free softmax(rec+pos), 0 params             : 0.2413 @ep13"
  echo "#   [log1p(age),pos,rad] 161p single 1e-3  == THIS CONFIG : 0.2170 @ep4   <- THE BASELINE"
  echo "#   same features, 2 layers 1218p                         : 0.1868 @ep6"
  echo "#   mnia scalar pooler, single 1e-3                       : 0.1630 @ep1, died ep4"
  echo "#"
  echo "# STRUCTURAL CONTEXT: 99.9% of Patent's source queries are cold -- 20 valid tokens from"
  echo "#   51,200 slots -- because every citation carries the citing patent's filing date and the"
  echo "#   cutoff is strictly t_edge < t_query. P_u is never a pooled centroid here; the pooler"
  echo "#   only ever acts on the candidate side. Do not read pooler results from Patent as if"
  echo "#   both sides were live."
  echo "#"
  echo "# WATCH: (a) DO NOT READ BEFORE EPOCH 3 -- on Patent the epoch-1 ranking is close to"
  echo "#   inverted; the run that died had the best ep1 (0.1630) and the record opened at 0.0459."
  echo "#   Separation is reliable by ep4-5;  (b) RECORD BOTH max test AND test@val-checkpoint --"
  echo "#   across the ten best Patent runs, peak-val to peak-test correlation is -0.088, so the"
  echo "#   checkpoint selects close to arbitrarily among good candidates;  (c) |E|mean -- runs"
  echo "#   that worked reached 0.166-0.199, ones that failed stalled at 0.106-0.115;"
  echo "#   (d) divergence -- 3e-3 on a Riemannian ManifoldParameter is untested; a blow-up is a"
  echo "#   real outcome and is itself informative."
  echo "# ~970 s/epoch (889 train + 80 eval). Patent peaks ep4-5 and dies by ep8-13: budget 2-3 h."
  echo "# started=$(date '+%F %T')"
  echo "# cmd: $CMD"
  echo
} > "$LOG"
PYTHONUNBUFFERED=1 $CMD >> "$LOG" 2>&1
echo "[$(date '+%F %T')] DONE rc=$?  $(grep -E 'stopped_at_epoch|best_val_mrr|best_test_mrr' "$LOG" | tr -s ' ' | tr '\n' ' ')" >> "$LOG"
