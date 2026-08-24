# Embedding-dimension sweep (d ∈ {8, 16, 32, 64, 128, 256})

Head: master's minimal Poincaré head — pooled gyro-midpoint of the walk-token bag,
score `temperature · (−d_H(P_u, P_v))`. **Two head parameters** (score temperature +
pooling temperature); all remaining parameters are the embedding table `E`.

Protocol: seed 42, `--num-epochs 50 --early-stop-patience 3`, lr 1e-3, K_train 5,
5 walks × len 5, one cell per GPU-run, sequential. Driver: `run_d_sweep.sh`
(commit `3bfe0624`; ML-20M d=128/256 ran at `fd632ac2` — the pull added
`--use-pop-bias`, which defaults off and leaves the default path byte-identical).
Master has since moved to `d370baa4`, which replaces that flag with a fixed causal-degree
`cand_pop` term — also default-off, so master's default head remains the head swept here.

`test@ckpt` = test MRR at the best-val epoch (the reported metric).
`max test` = highest test MRR observed at any epoch — recorded because val and test
diverge late on this pipeline, so `test@ckpt` samples the test curve wherever
patience happened to fire.

## Results

| dataset | d | val | test@ckpt | max test | best ep | ran | stop |
|---|---|---|---|---|---|---|---|
| GoogleLocal | 8 | 0.5874 | 0.5530 | 0.5536 | 21 | 24 | patience |
| GoogleLocal | 16 | 0.6355 | 0.6043 | 0.6044 | 20 | 23 | patience |
| GoogleLocal | 32 | 0.6610 | 0.6301 | 0.6303 | 29 | 32 | patience |
| GoogleLocal | 64 | 0.6732 | 0.6441 | 0.6442 | 30 | 33 | patience |
| GoogleLocal | 128 | 0.6808 | 0.6498 | 0.6498 | 28 | 31 | patience |
| GoogleLocal | 256 | **0.6842** | **0.6550** | 0.6550 | 39 | 42 | patience |
| YouTube | 8 | 0.4456 | 0.3911 | 0.3911 | 50 | 50 | BUDGET |
| YouTube | 16 | 0.5384 | 0.4685 | 0.4685 | 50 | 50 | BUDGET |
| YouTube | 32 | 0.6089 | 0.5302 | 0.5302 | 50 | 50 | BUDGET |
| YouTube | 64 | 0.6465 | 0.5677 | 0.5677 | 50 | 50 | BUDGET |
| YouTube | 128 | 0.6686 | 0.5899 | 0.5899 | 50 | 50 | BUDGET |
| YouTube | 256 | **0.6802** | **0.6031** | 0.6031 | 50 | 50 | BUDGET |
| ML-20M | 8 | 0.2073 | 0.2000 | 0.2000 | 3 | 6 | patience |
| ML-20M | 16 | 0.2390 | 0.2161 | 0.2170 | 21 | 24 | patience |
| ML-20M | 32 | 0.2524 | 0.2224 | 0.2225 | 15 | 18 | patience |
| ML-20M | 64 | 0.2586 | 0.2234 | 0.2241 | 14 | 17 | patience |
| ML-20M | 128 | 0.2619 | 0.2234 | 0.2244 | 14 | 17 | patience |
| ML-20M | 256 | **0.2643** | **0.2245** | 0.2249 | 12 | 15 | patience |

## Reading

**The optimal dimension is dataset-dependent, and orders with the size of the
negative-candidate pool.**

| dataset | unique dst (candidate pool) | Δtest per doubling, 32→64→128→256 | verdict |
|---|---|---|---|
| ML-20M | 9,646 | +0.0010, 0.0000, +0.0011 | saturated by d≈32–64 |
| YouTube | ~94k | +0.0375, +0.0222, +0.0132 | still climbing at 256 |
| GoogleLocal | ~100k+ | +0.0140, +0.0057, +0.0052 | still climbing at 256 |

ML-20M is flat past d=64 — d=128 matched d=64 to four decimals for double the
embedding table. GoogleLocal and YouTube had not turned over by d=256, so **d=512
is an open question on both**; the wiki-era "d=512 overfits" note did not reproduce.

**Two head parameters get close to the 574k-param four-channel head:** GoogleLocal
0.6550 vs 0.6748 (−0.020), YouTube 0.6031 vs 0.6237 (−0.021, and truncated),
ML-20M 0.2245 vs 0.2465 (−0.022). All three beat their TGB-Seq leaderboard #1
except ML-20M, which sits −0.015 under TGN's 0.2399.

**`|E|max` sits at ~0.65 for every GoogleLocal cell, d=8 through d=256.** A 32×
change in dimension does not change the radial regime; capacity, not radius, is
what dimension buys.

## Caveats

1. **YouTube is uniformly budget-truncated** — all six cells hit `stopped_at_epoch:
   50` and were still setting new test bests on the final epoch. The row is a set of
   LOWER BOUNDS. End-of-budget slopes: d=8 +0.00047/ep, d=16 +0.00050, d=32 +0.00108,
   d=64 +0.00074, d=128 +0.00088, d=256 +0.00081 — the *high*-d cells are climbing
   fastest at the cutoff, so truncation understates the benefit of dimension rather
   than manufacturing it. A re-run at `--num-epochs 150` would widen the d=8→256 gap,
   not narrow it.
2. **ML-20M d=8 collapses** — peaked at epoch 3 and early-stopped at epoch 6, the
   one-epoch phenomenon. It is the only cell in the sweep that fell over, and it does
   so only at the smallest dimension.
3. **Run-to-run noise is ~0.0002 test, not the ±0.01 wiki band.** Two independent
   replicates of the same config: GoogleLocal d=32 gave 0.6303 (2026-08-23) and
   0.6301 (this sweep); ML-20M d=32 gave 0.2222 and 0.2224. Single-seed deltas on
   these datasets are readable at the third decimal.
4. Single seed (42) throughout. Patent and Flickr were dropped from the sweep.
