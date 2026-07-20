# tempest-embedding

Walks-supervised temporal link prediction with Tempest. Architecture
notes in `CLAUDE.md`.

The pre-rewrite design and its development history (35 lessons across
7 stages of diagnostics) are preserved on branch
`backup/important-walk-embedding`.

## Layout

```
link_property_prediction/
  data.py        TGB loader + Batch dataclass + batcher
  evaluator.py   TGB Evaluator wrapper (architecture-agnostic)
  negatives.py   Uniform / Historical (Vitter R) / TGB samplers
  walks.py       Tempest walk-sampler wrapper
  model.py       EmbeddingTable + ProjectionHead + LinkHead
  losses.py      alignment_loss (InfoNCE with sampled negatives)
  trainer.py     Strict-causal train + eval loop
  utils.py       seeding, dataset derivation, LR schedule λ
scripts/
  train.py       CLI entry point
tests/
  test_walk_contract.py        shape + alignment contract for Tempest walks
  test_vitter_r_uniformity.py  χ² check on Historical (Vitter R) reservoir
```
