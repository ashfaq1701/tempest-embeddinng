"""Node embedding table.

Single ``nn.Embedding(num_nodes, d_emb)``, link-trained (the link loss is the
only gradient path into E). Rows live on the unit sphere as a
``geoopt.ManifoldParameter`` on ``geoopt.Sphere()`` and are kept unit-norm by
RiemannianAdam (see trainer.py). E is the only manifold parameter; the GRU/head
weights are Euclidean.
"""
import math

import geoopt
import torch
import torch.nn as nn


class EmbeddingTable(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb

        self.E = nn.Embedding(num_nodes, d_emb)
        nn.init.normal_(self.E.weight, mean=0.0, std=1.0 / math.sqrt(d_emb))
        # Feasible init: project every row onto the sphere (RiemannianAdam
        # assumes the parameter starts on the manifold).
        with torch.no_grad():
            w = self.E.weight.data
            w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.E.weight = geoopt.ManifoldParameter(w, manifold=geoopt.Sphere())

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.E(node_ids)
