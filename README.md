# tempest-embedding

Walks-supervised temporal link prediction with Tempest. Architecture
notes in `CLAUDE.md`.

The pre-rewrite design and its development history (35 lessons across
7 stages of diagnostics) are preserved on branch
`backup/important-walk-embedding`.

## Layout

```
tempest_walks/
  data.py        TGB loader + Batch dataclass + batcher
  evaluator.py   TGB Evaluator wrapper (architecture-agnostic)
  negatives.py   Uniform / Historical (Vitter R) / TGB samplers
  walks.py       Tempest walk-sampler wrapper
  model.py       EmbeddingTable + ProjectionHead + LinkHead
  losses.py      alignment_loss (InfoNCE, per-chunk backward)
  trainer.py     Strict-causal train + eval loop
  utils.py       seeding, dataset derivation, chunk auto-sizer, LR λ
scripts/
  train.py       CLI entry point
tests/
  test_chunked_infonce.py    chunked-vs-full + vs naive triple-loop
  test_vitter_r_uniformity.py
```
