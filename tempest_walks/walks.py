"""Tempest walk sampler wrapper.

Responsibility:
  - Construct a TemporalRandomWalk instance with the right config.
  - Provide walks_for_nodes(seed_nodes) returning the standard
    (nodes, timestamps, lens, edge_feats) tuple with seed at
    position lens-1.
  - Provide add_edges(...) for post-batch ingest (strict-causal).
  - Provide reset() for epoch-boundary state clear.

Walk layout (Tempest convention):
  nodes:      [n_0, n_1, ..., n_{lens-1}, padding...]
  timestamps: [t_1, t_2, ..., sentinel, padding...]
              where t_k = time of edge between nodes[k-1] and nodes[k]
  seed:       nodes[lens-1]
  Edge feature attached to context at position p (under convention β,
  edge-toward-seed): timestamps and edge_feats index p+1, i.e. the
  edge LEAVING that context toward the seed.

CLI-exposed knobs (passed through from train.py):
  num_walks_per_node, max_walk_len, walk_bias, start_bias.
"""
