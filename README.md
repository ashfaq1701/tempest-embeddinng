# tempest-embedding

Walks-supervised temporal link prediction with Tempest. Architecture is
currently being rebuilt from first principles — see `CLAUDE.md`.

The pre-rewrite design and its full development history (35 lessons
across 7 stages of diagnostics) are preserved on branch
`backup/important-walk-embedding`.

## Layout

```
tempest_walks/
  data.py        TGB loader + Batch dataclass + batcher
  evaluator.py   TGB Evaluator wrapper (architecture-agnostic)
  negatives.py   Uniform / Historical (Vitter R) / TGB samplers
  walks.py       Tempest walk-sampler wrapper                 (stub)
  model.py       EmbeddingTable + ProjectionHead + LinkHead   (stub)
  losses.py      alignment_loss + uniformity_loss             (stub)
  trainer.py     Strict-causal train + eval loop              (stub)
scripts/
  train.py       CLI entry point                              (stub)
tests/
  test_vitter_r_uniformity.py
```
