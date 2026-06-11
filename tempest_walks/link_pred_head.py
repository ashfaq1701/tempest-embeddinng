"""Cross walk-encoder link head.

Per batch the head is given, for every UNIQUE node that appears (as a source or
a candidate), K Tempest walks of length L. It looks up E on the walk nodes,
encodes each node's walks with a GRU (final state, pooled over the K walks) into
a context vector h[node], and scores a candidate pair (u, v) by the symmetric
cross comparison of E[u] against h[v] and E[v] against h[u]:

    dist:  logit(u, v) = -scale * ( D(E[u], h[v]) + D(E[v], h[u]) )    D = l2² or l1
    cos:   logit(u, v) =  scale * ( cos(E[u], h[v]) + cos(E[v], h[u]) )

E is link-trained (no detach, no alignment): the link loss flows into E both
directly (the E[u] / E[v] terms) and through the GRU (E on the walk nodes).
``scale`` is the only temperature. For the distance forms E / h are compared
RAW (no normalisation) so magnitude is part of the signal; training aligns the
two spaces.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossWalkGRUHead(nn.Module):
    def __init__(self, d_emb: int, dist: str = "l2", num_layers: int = 1):
        super().__init__()
        if dist not in ("l2", "l1", "cos", "geodesic"):
            raise ValueError(f"dist must be l2 / l1 / cos / geodesic, got {dist!r}")
        self.dist = dist
        self.gru = nn.GRU(d_emb, d_emb, num_layers=num_layers, batch_first=True)
        # cos/geodesic operate on the unit sphere (bounded); raw distances are
        # O(1-10) — start the temperature accordingly.
        self.logit_scale = nn.Parameter(
            torch.tensor(10.0 if dist in ("cos", "geodesic") else 1.0))

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
        """E_u/h_u [B, d]; E_v/h_v [B, C, d] -> [B, C] logits."""
        if self.dist in ("cos", "geodesic"):
            # Sphere forms: project BOTH operands (incl. the GRU output h) onto
            # the unit sphere so E and h are comparable manifold points.
            eu = F.normalize(E_u, dim=-1).unsqueeze(1)           # [B, 1, d]
            hu = F.normalize(h_u, dim=-1).unsqueeze(1)
            ev = F.normalize(E_v, dim=-1)                        # [B, C, d]
            hv = F.normalize(h_v, dim=-1)
            c1 = (eu * hv).sum(dim=-1)                           # cos(E[u], ĥ[v])
            c2 = (ev * hu).sum(dim=-1)                           # cos(E[v], ĥ[u])
            if self.dist == "cos":
                return self.logit_scale.clamp(1.0, 100.0) * (c1 + c2)
            # geodesic: arc length d(a,b)=arccos⟨a,b⟩ (clamp keeps the acos
            # gradient finite at ±1); closer => higher logit.
            eps = 1e-6
            g = (torch.arccos(c1.clamp(-1 + eps, 1 - eps))
                 + torch.arccos(c2.clamp(-1 + eps, 1 - eps)))
            return -self.logit_scale.clamp_min(1e-3) * g

        diff1 = E_u.unsqueeze(1) - h_v                           # [B, C, d]  E[u] vs h[v]
        diff2 = E_v - h_u.unsqueeze(1)                           # [B, C, d]  E[v] vs h[u]
        if self.dist == "l1":
            d = diff1.abs().sum(dim=-1) + diff2.abs().sum(dim=-1)        # [B, C]
        else:  # l2 (squared)
            d = diff1.pow(2).sum(dim=-1) + diff2.pow(2).sum(dim=-1)      # [B, C]
        return -self.logit_scale.clamp_min(1e-3) * d
