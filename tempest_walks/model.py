"""Embedding table.

Single nn.Embedding(num_nodes, d_emb), lookup-only. Trained directly
by InfoNCE contrastive alignment on raw rows — there is no projection
head between E and L_align. The link-prediction head
(`link_pred_head_v2.LinkPredHeadV2`) consumes E.detach(), so L_link
trains only the head's parameters; L_align is the sole gradient path
into E.
"""

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


