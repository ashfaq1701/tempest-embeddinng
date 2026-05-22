"""Strict-causal training + eval loop.

Single Trainer class. Per-batch ordering:

  TRAINING:
    1. walks = walk_gen.walks_for_nodes(seeds)       ← pre-ingest state
    2. unif_pairs = sample uniformity pairs
    3. L = L_align + η·L_uniform                     ← table loss
    4. neg = neg_sampler.sample(batch)               ← pre-observe state
    5. logits = link_head(E[u].detach(), E[v].detach()) for pos+neg
    6. L += L_bce
    7. L.backward(); optimizer.step()
    8. neg_sampler.observe(batch.src, batch.tgt)     ← post-scoring
    9. walk_gen.add_edges(batch)                     ← post-scoring, last

  EVAL (within torch.no_grad()):
    1. logits = link_head(E[u], E[v]) for pos + TGB-supplied negs
    2. evaluator.update(logits, labels)
    3. walk_gen.add_edges(batch)                     ← Tempest state
                                                       carries forward
    NOTE: reservoir not updated, walks not sampled at eval. Model
          parameters frozen.

Epoch boundary:
  - walk_gen.reset()
  - neg_sampler.reset() (if Historical)
  - Model parameters and optimiser state are NOT reset.

Final test eval:
  - walk_gen.reset() then re-ingest training edges so Tempest reflects
    "all of training" before val/test scoring begins.

Early stop:
  - Snapshot best-val model + projection + head state_dicts.
  - Restore before final eval.
  - Optimiser state not snapshotted (not needed after training stops).
"""
