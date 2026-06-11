"""Node embedding table.

Single ``nn.Embedding(num_nodes, d_emb)``, link-trained (the link loss is the
only gradient path into E — there is no alignment loss). Geometry depends on
the optimizer variant:

  * ``sphere=False`` (AdamW / Prodigy) — plain Euclidean rows; the link head
    cosine-normalises in the score, so magnitude is a free degree of freedom.
  * ``sphere=True`` (geometric / RiemannianAdam) — rows are a
    ``geoopt.ManifoldParameter`` on ``geoopt.Sphere()`` kept unit-norm by the
    Riemannian optimizer.
"""
import math

import geoopt
import torch
import torch.nn as nn


class EmbeddingTable(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int, sphere: bool = False):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.sphere = sphere

        self.E = nn.Embedding(num_nodes, d_emb)
        nn.init.normal_(self.E.weight, mean=0.0, std=1.0 / math.sqrt(d_emb))
        if sphere:
            with torch.no_grad():
                w = self.E.weight.data
                w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            self.E.weight = geoopt.ManifoldParameter(w, manifold=geoopt.Sphere())

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.E(node_ids)
