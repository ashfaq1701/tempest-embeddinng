"""Embedding table.

Single nn.Embedding(num_nodes, d_emb), lookup-only. Trained directly
by InfoNCE contrastive alignment on raw rows — there is no projection
head between E and L_align. The link-prediction head
(`link_pred_head.LinkPredHead`) consumes E.detach(), so L_link
trains only the head's parameters; L_align is the sole gradient path
into E.

Geometry: every row is constrained to the unit hypersphere
(‖E[i]‖ = 1). `E.weight` is a `geoopt.ManifoldParameter` on a
`geoopt.Sphere()`, and E is optimised on that manifold by a
RiemannianAdam optimiser (see trainer.py) — tangent-space gradient,
retraction, transported momentum — so the parameter stays ON the
sphere at every step. There is NO read-time normalisation: rows are
unit-norm between steps because the optimiser maintains the
constraint, and `forward` is a raw lookup. (Normalising at read
would be redundant and would silently mask any optimiser bug that
let rows drift off the manifold.)

On the sphere the squared-L2 alignment similarity reduces to a
cosine objective: ‖a − b‖² = 2 − 2⟨a,b⟩ when ‖a‖ = ‖b‖ = 1, and the
constant −2/τ cancels in the per-seed softmax. So the existing
squared-L2 loss code computes cosine-at-temperature-τ/2 unchanged.
"""

import geoopt
import torch
import torch.nn as nn


class EmbeddingTable(nn.Module):
    """Lookup-only node embedding table, rows constrained to the unit sphere."""

    def __init__(self, num_nodes: int, d_emb: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.manifold = geoopt.Sphere()

        self.E = nn.Embedding(num_nodes, d_emb)
        # Gaussian rows give uniform directions on the sphere; the std value
        # is irrelevant once rows are normalised (scale is removed).
        nn.init.normal_(self.E.weight, mean=0.0, std=0.02)
        # Feasible init: project every row onto the manifold. Mandatory with
        # a Riemannian optimiser — it assumes the parameter starts on it.
        with torch.no_grad():
            w = self.E.weight.data
            w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.E.weight = geoopt.ManifoldParameter(w, manifold=self.manifold)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """node_ids: any shape of long; returns shape + [d_emb]. Raw lookup —
        rows are kept unit-norm by the Riemannian optimiser, not here."""
        return self.E(node_ids)
