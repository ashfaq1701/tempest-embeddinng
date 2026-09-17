# ML-20M: the bag under-samples, and it is BREADTH that fixes it, not depth

Run 2026-09-17, ML-20M d=64 K=5 lr=1e-3 seed 5 patience 8 `--is-bipartite`, commit `44413a26`
(master plus telemetry; see `../pooler_capacity/README.md` for the equivalence hashes).

## Why this was run

ML-20M is **the only one of the eight datasets that cannot fit its own training objective.**
From the shipped seed-5 logs, `link` at the last epoch run:

| googlelocal | flickr | youtube | yelp | wikilink | taobao | **ml-20m** |
|---|---|---|---|---|---|---|
| 0.0086 | 0.0766 | 0.0776 | 0.0685 | 0.1673 | 0.0966 | **0.4320** |

2.6x the next worst and 50x GoogleLocal, still falling 1.2%/epoch when patience stops it at
ep12, and 0.4660 at the val peak. Not stopped early, not over-fitted — it never fits.
ML-20M has 127 edges/node (Taobao 8.6, Yelp 9.8) and mean reachable walk length 47.4, but the
default `wpn=5 x mwl=5` draws ~20 tokens: roughly an 11% sample of the neighbourhood.

## The arms

`bag ~= num_walks_per_node x max_walk_len`, so `deep` and `wide` quadruple the bag by
orthogonal routes at the same number of walk steps.

| arm | wpn | mwl | bag_n | pool_eff | link | best val | best test |
|---|---|---|---|---|---|---|---|
| control (shipped) | 5 | 5 | ~20 | — | 0.4320 floor | 0.2826 | 0.2429 |
| **wide** | 20 | 5 | 99.8 | **41** | 0.3866 @ep16 | **0.3087** @ep8 | **0.2601** (max 0.2637) |
| deep | 5 | 20 | 96.6 | **6.9-10.1** | 0.6394 @ep2 | 0.2612 @ep2 | killed ep2 |
| wider | 40 | 5 | 199.5 | 69-78 | — | in flight | in flight |

## What it shows

**Breadth and depth are not interchangeable at matched bag size.** Given ~100 tokens, the
pooler keeps ~41 effective when they come from 20 short walks and only ~7-10 when they come
from 5 long walks. It discards depth *at the point of pooling* — which is what the hop-index
and `log1p(age)` channels would do to distant hops.

**Breadth is a real lever.** `wide` drove `link` to 0.4308 by ep7, below the control's entire
12-epoch floor, five epochs early — so the bag genuinely was the binding constraint on fitting.
Test 0.2601 val-selected (max 0.2637) against 0.2429: **+0.0172**, moving the CRAFT deficit
-11.62 -> -9.90. Real, but nowhere near closing it.

**`deep` was killed at ep2** after two epochs of exact val parity with the control (0.2297 vs
0.2303, then 0.2612 vs 0.2612) while fitting *worse* (link 0.6394 vs 0.5644) at 11.3x the
control's cost. Stated honestly: I had said ep3 and moved my own line forward by one epoch.
The caveat that survives is that `pool_eff` was rising (6.93 -> 10.08), so this rules out depth
being competitive with breadth **at equal bag size and equal wall-clock**, not that depth can
never be exploited by a much longer run or a pooler that does not see hop index.

## Do not re-run Taobao long

CLAUDE.md lists Taobao's convergence as unchecked. Its own shipped log answers it: val peaks at
ep4 with `link` still at 0.3332 and falling steeply, then over ep5-9 the loss collapses 93% to
0.0966 while val sits flat at 0.546-0.552. That is overfit onset, not patience truncating a
climbing curve. A long-patience re-run costs ~5.6 GPU-hours (1576 s/epoch) to confirm a flat
line. **WikiLink is the one where "run it long" is still genuinely open** — its log ends at
ep20 with `link` *rising*, 0.1620 -> 0.1673.
