"""Cross walk-encoder link head (geodesic, on the unit sphere).

Per batch the head is given, for every UNIQUE node that appears (as a source or
a candidate), K Tempest walks of length L. It looks up E on the walk nodes,
encodes each node's walks with a GRU (final state, pooled over the K walks) into
a context vector h[node], projects h onto the unit sphere, and scores a
candidate pair (u, v) by the symmetric cross GEODESIC distance:

    logit(u, v) = -scale * ( d_g(E[u], ĥ[v]) + d_g(E[v], ĥ[u]) )
    d_g(a, b)   = arccos⟨a, b⟩          (great-circle arc length on S^{d-1})

E lives on the sphere; ĥ = h/‖h‖ projects the GRU output onto the same sphere so
both operands are manifold points. E is link-trained (no detach): the loss flows
into E directly (the E[u]/E[v] terms) and through the GRU (E on the walk nodes).
The GRU weights are Euclidean. ``scale`` is the only temperature.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossWalkGRUHead(nn.Module):
    def __init__(self, d_emb: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(d_emb, d_emb, num_layers=num_layers, batch_first=True)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))

    def encode(self, walk_emb: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """walk_emb [M, K, L, d], valid [M, K, L] bool -> h [M, d].

        GRU each of the M*K walks, take the hidden state at the last valid
        position, mean-pool over the K walks of each node."""
        M, K, L, d = walk_emb.shape
        out, _ = self.gru(walk_emb.reshape(M * K, L, d))          # [M*K, L, d]
        last_idx = valid.reshape(M * K, L).sum(dim=1).clamp_min(1) - 1  # [M*K]
        rows = torch.arange(M * K, device=out.device)
        last = out[rows, last_idx]                                # [M*K, d]
        return last.reshape(M, K, d).mean(dim=1)                  # [M, d]

    def forward(self, E_u: torch.Tensor, E_v: torch.Tensor,
                h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        """E_u/h_u [B, d]; E_v/h_v [B, C, d] -> [B, C] logits.

        Both operands are projected onto the unit sphere; the geodesic arc
        length arccos⟨·,·⟩ is the distance (clamp keeps the acos gradient
        finite at ±1). Closer => higher logit."""
        eu = F.normalize(E_u, dim=-1).unsqueeze(1)               # [B, 1, d]
        hu = F.normalize(h_u, dim=-1).unsqueeze(1)
        ev = F.normalize(E_v, dim=-1)                            # [B, C, d]
        hv = F.normalize(h_v, dim=-1)
        eps = 1e-6
        c1 = (eu * hv).sum(dim=-1).clamp(-1 + eps, 1 - eps)      # ⟨E[u], ĥ[v]⟩
        c2 = (ev * hu).sum(dim=-1).clamp(-1 + eps, 1 - eps)      # ⟨E[v], ĥ[u]⟩
        g = torch.arccos(c1) + torch.arccos(c2)                  # [B, C]
        return -self.logit_scale.clamp_min(1e-3) * g
