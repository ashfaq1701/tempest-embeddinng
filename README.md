# tempest-embedding

Walk-supervised temporal graph learning with Tempest. Two arms share one encoder:

- **Link prediction** trains a Lorentz embedding table and a walk-bag pooler on TGB-Seq, TGB and
  DyGLib, scored by MRR.
- **Dynamic node classification** freezes a link-prediction checkpoint and trains a small
  classifier on features of its walks (DyGLib Wikipedia / Reddit, ROC-AUC).

Dataset, protocol and result notes are in `CLAUDE.md`; the node-classification write-up is
`reports/node_classification_2026-10-08.md`.

## Layout

```
link_property_prediction/      encoder + link-prediction arm
  data.py           SplitData / Loaded / Batch + chronological batcher
  evaluator.py      Evaluator + DataSuite ABCs, make_suite
  tgb_seq_eval.py   TGB-Seq suite: loader, fixed eval negatives, MRR evaluator
  tgb_eval.py       TGB suite: loader (auto-download), TGB's negatives and evaluator
  dyglib_eval.py    DyGLib suite: Zenodo auto-download, split, per-interaction labels
  negatives.py      uniform negative sampler
  walks.py          Tempest wrapper: walks + per-node history (recency, popularity)
  walk_tokens.py    per-query walk token bags
  lorentz.py        Lorentz (hyperboloid) manifold
  model.py          embedding table + walk-bag pooler + link-pred head
  trainer.py        strict-causal train + eval loop
  utils.py          seeding
node_classification/           node-classification arm, on a frozen checkpoint
  encoder.py        FrozenEncoder: rebuild + freeze a checkpoint, per-interaction features
  model.py          NodeClassifier (small MLP)
  trainer.py        classifier train + eval loop (ROC-AUC, early stopping)
scripts/
  train_link_property_prediction.py   link-prediction entry point (--save-checkpoint)
  train_node_classification.py        node-classification entry point (--checkpoint)
tests/
  test_create_batches.py   batch-iterator contract
  test_walk_edge_feats.py  walk edge-feature pairing
  test_lorentz.py          manifold numerics
experiment_logs/            versioned run logs for the paper
```

## Running

```
# link prediction (saves the frozen encoder for node classification)
scripts/train_link_property_prediction.py --data-suite dyglib --dataset wikipedia \
  --is-bipartite --k-train 5 --k-eval 5 --seed 42 --use-gpu --use-gpu-tempest \
  --save-checkpoint <ckpt.pt>

# node classification on that checkpoint
scripts/train_node_classification.py --dataset wikipedia --checkpoint <ckpt.pt> \
  --seed 42 --use-gpu --use-gpu-tempest
```
