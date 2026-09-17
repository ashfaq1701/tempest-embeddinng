# Pooler capacity on YouTube: width is a null, depth is feature-dependent and loses on master

Run 2026-09-17, YouTube d=64 K=5 lr=1e-3 seed 5 patience 8, commit `44413a26` on
`exp/escape-instrumentation`. That commit is **master plus telemetry**: with
`TEMPEST_POOL_{FEAT,LAYERS,F4}` unset it is bit-identical to master — forward hash
`f62e1260cdbfb833`, `grad_E` `4fbd9a1e0ec7a291`, pooler init `6673dae38ebc39d1` on both refs.
The extra code is a `@torch.no_grad()` stats stash and an opt-in checkpoint path.

`E` is drawn **before** the pooler in `LinkPredHead.__init__`, so every arm here shares a
bit-identical embedding table and only the pooler weights differ. The pooler-only init draw
does differ between arms, so a gap under ~0.01 is not separable from init luck.

| third feature | pooler | params | max test | val-selected | escape |
|---|---|---|---|---|---|
| `rad` (master) | 1 hidden layer | 161 | **0.5590** @ep26 | **0.5590** | ep14-15 |
| `rad` | 2 hidden layers | 1218 | 0.5508 @ep17 | 0.5480 | ep13-14 |
| `sdist` | 1 hidden layer | 161 | 0.4067 @ep24 | 0.4067 | never |
| `sdist` | 2 hidden layers | 1217 | 0.4315 @ep29 | (truncated, still climbing) | never |
| `sdist` | hidden 64, 1 layer | 321 | (null, killed ep15) | — | never |

## Three findings

**Width is a clean null.** hidden 32 -> 64 stayed inside `[-0.0071, +0.0010]` of baseline val
for 15 consecutive epochs with `pool_eff` within 0.1 of it throughout. Killed at ep15.

**Depth flips sign with the feature.** +0.025 max test on `sdist`, **-0.011 on `rad`**. Same
code change, opposite conclusions. Do not add pooler depth: it gains on a feature that already
loses by 0.15 and loses on the one master ships. What it does consistently is pull the curve
~2 epochs earlier; on `rad` that means escaping sooner and plateauing *lower* (link 0.047 at
stop against the control's 0.063).

**Neither restores the escape, so the escape is not a capacity limit.** Both `sdist` arms held
`pool_eff` at 4.88-5.27 through ep13-15, exactly where the `rad` control lifts
6.66 -> 7.21 -> 7.53 and val jumps 0.4386 -> 0.5287. The escape is a property of the pooler's
FEATURE SET, not of how much capacity sits on top of it.

## The process lesson, which matters more than the numbers

The `rad`+2-layer arm was called a win six times between ep8 and ep15 on a lead that peaked at
**+0.104** and then decayed monotonically to an inversion:

| ep | 13 | 14 | 15 | 16 | 17 | 18 | ... | final |
|---|---|---|---|---|---|---|---|---|
| Δ test vs 1-layer | +0.076 | +0.088 | +0.067 | +0.025 | +0.014 | +0.008 | | **-0.011** |

CLAUDE.md already says warmup ordering inverts here and that a pooler A/B must be read by when
the escape fires and from what plateau. **An epoch-matched lead taken before both arms have
passed their own peak carries no information about the peak.** The `rad` control does not peak
until ep26; nothing before ep26 was callable. Report the delta *and its trend*, and wait.
