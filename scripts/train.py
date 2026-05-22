"""CLI entry point.

Argparse → Config dataclass → Trainer → train → print results.

Hyperparameters exposed at CLI:
  Loss-formulation:    η (uniform weight), t (uniform temp),
                       β (time decay), M (uniform pair count)
  Walks:               num_walks_per_node, max_walk_len, walk_bias,
                       start_bias
  Negatives:           num_neg_per_pos, hist_neg_ratio
  Optimisation:        lr, weight_decay, batch_size
  Training:            num_epochs, early_stop_patience, seed
  Dataset:             dataset_name (tgbl-wiki-v2, tgbl-review-v2, ...)
  System:              use_gpu, tgb_root

τ for time normalisation is derived from dataset training span at
load time, not exposed.

Single root seed → all RNGs (numpy, torch, sampler internals).
"""
