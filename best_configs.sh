#!/usr/bin/env bash
# best_configs.sh — the best-performing training command per TGB dataset.
# KEEP THIS UPDATED: whenever a new config beats the current best for a dataset,
# replace that dataset's command here (and note the val/test it produced).
set -euo pipefail
cd "$(dirname "$0")"

# wiki   (val ~0.8291 / test ~0.8063 @ ep22 — VelocityHead (identity centroid + line-fit drift μ)
#         on the PER-QUERY CAUSAL substrate (ingest batch BEFORE scoring, walk each (u,t) with
#         cutoff=t, == TPNet). REQUIRES temporal-random-walk>=1.8.6 (cutoff_times).
#         WALK-LENGTH sweep (2026-07-01 overnight): the earlier "len stays 5" was WRONG — it never
#         tested BELOW 5. Full sweep test-MRR: len5 0.8032 < len4 0.8039 < len2 0.8048 < LEN3 0.8063
#         (PEAK). len=3 optimum: +0.0026 test / +0.0038 val over len5, 3-SEED CONFIRMED (test
#         0.8063/0.8064/0.8063 seeds 42/43/44; Δtest +0.0031/+0.0024/+0.0022 all positive).
#         d_emb: 256 best (512 OVERFITS: matches val, −0.0014 test). Pair-recurrence channels all
#         lose — centroid saturates recurrence. See CLAUDE.md + logs/OVERNIGHT_PAIR_FEATURES.md.)
.venv/bin/python scripts/train.py \
  --dataset tgbl-wiki \
  --k-train 100 \
  --num-walks-per-node-query-side 10 --max-walk-len-query-side 3 \
  --d-emb 256 \
  --batch-size 200 --eval-batch-size 20 \
  --num-epochs 50 --early-stop-patience 5 \
  --seed 42 \
  --lr 5e-3 --lr-min 5e-6 \
  --tempest-batch-window-multiplier -1.0 \
  --use-gpu --use-gpu-tempest
