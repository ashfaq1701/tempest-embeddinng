# tempest-embedding

Walks-supervised temporal link prediction with Tempest, evaluated on TGB-Seq.
Dataset and eval notes in `CLAUDE.md`.

## Layout

```
link_property_prediction/
  data.py           SplitData / Loaded / Batch + chronological batcher
  evaluator.py      Evaluator + DataSuite ABCs, make_suite
  tgb_seq_eval.py   TGB-Seq loader, fixed eval negatives, TGB-Seq evaluator
  negatives.py      uniform negative sampler
  walks.py          Tempest walk-sampler wrapper
  walk_tokens.py    per-query walk token bags
  model.py          Poincare embedding table + link-pred head
  trainer.py        strict-causal train + eval loop
  utils.py          seeding
scripts/
  train_link_property_prediction.py   CLI entry point
tests/
  test_create_batches.py   batch-iterator contract
  test_walk_edge_feats.py  walk edge-feature pairing
experiment_logs/            versioned run logs for the paper
```
