"""Model components: embedding table, link head.

Two classes, no shared state:

EmbeddingTable
  - Single nn.Embedding(num_nodes, d_emb).
  - Lookup-only. Trained DIRECTLY by InfoNCE contrastive alignment
    on raw rows — there is no projection-head wrapper between E and
    the loss. Both losses operate on the same embedding so there is
    no asymmetry between what L_align optimises and what L_link
    consumes (see the no-projection-branch result on wiki, and the
    review/coin failure-mode analysis: a learned projection only
    optimised by L_align introduces a representation seam against
    a link head that reads raw E).

LinkHead
  - score(u, v) = bilinear(E(u), E(v)) + small_MLP(pair_features(u, v))
    bilinear  = (W·E(u)) · E(v) + b   (Linear(d, d)·v inner product
                                        plus scalar bias — same
                                        expressivity as nn.Bilinear
                                        but with a backward graph
                                        that materialises [B, 1+K, d]
                                        instead of [B, 1+K, d, d];
                                        nn.Bilinear OOMs at our batch
                                        × K_train when E is not
                                        detached on the link path)
    MLP input = 6-channel pair features
                [E(u), E(v), E(u)*E(v), |E(u)-E(v)|,
                 (E(u)-E(v))^2, E(u)+E(v)]
  - Per-query batched: inputs are [B, 1+K, d_emb], output is
    [B, 1+K] logits. Column 0 holds the positive candidate at
    training; columns 1..K are negatives sharing the same query
    source.
  - E is NOT detached on the call site — L_link's gradient flows
    back through this head into E. Joint training of E by both
    L_align and L_link.
  - No node features, no edge features, no time features at scoring.
  - Asymmetric by construction. For undirected datasets the caller
    symmetrises by averaging forward(e_u, e_v) with
    forward(e_v, e_u); applied at both training and eval so the
    two paths share the exact same scoring rule.
"""

from typing import Optional

import torch
import torch.nn as nn


class EmbeddingTable(nn.Module):
    """Lookup-only node embedding table."""

    def __init__(self, num_nodes: int, d_emb: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.E = nn.Embedding(num_nodes, d_emb)
        # Small Gaussian init; downstream L2-norm in projections handles scale.
        nn.init.normal_(self.E.weight, mean=0.0, std=0.02)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """node_ids: any shape of long; returns shape + [d_emb]."""
        return self.E(node_ids)


# NOTE: the previous bilinear + 6-channel pair_mlp `LinkHead` has been
# removed. Its replacement is `link_pred_head_v2.LinkPredHeadV2`, a
# walk-mediated head that consumes per-position primitives + K + t
# features + a direct (E[u], E[v]) bypass. See
# analysis/REPORT.md §9 for the design rationale.
