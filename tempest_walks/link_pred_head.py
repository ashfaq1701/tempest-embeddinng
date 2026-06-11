"""Cross walk-encoder link head (chord distance on the unit sphere) with time.

Per batch the head is given, for every UNIQUE node, K Tempest walks of length L,
plus the per-position inter-event Δt (log1p, query-INDEPENDENT) and, at scoring
time, the query-dependent recency of each candidate.

  encode:  GRU over [E(walk node) ‖ Time2Vec(Δt)] -> h[node], projected to the
           unit sphere (deduped per node). The seed (rightmost) position carries
           Δt=0, a no-delta marker; its real next edge is the scoring edge
           (query-dependent), injected below.
  score:   logit(u, v) = -scale*( ‖E[u]-ĥ[v]‖ + ‖E[v]-ĥ[u]‖ )
                         + rec_head( Time2Vec( log1p(t_query - t_last[v]) ) )
           ‖a-b‖ = √(2-2⟨a,b⟩) is the chord distance (both operands on the sphere;
           a sweep found chord ≥ geodesic > cosine). The recency term carries the
           query time the GRU is blind to.

E is link-trained (no detach); GRU/Time2Vec/rec_head are Euclidean. E is the
only manifold parameter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    """Time2Vec (Kazemi et al. 2019): scalar τ -> [linear, sin(ω₁τ+φ₁), …].

    The first channel is linear (ω₀τ+φ₀); the rest are periodic. τ is expected
    pre-normalised (we feed log1p of a delta)."""

    def __init__(self, dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(dim - 1))
        self.b = nn.Parameter(torch.rand(dim - 1) * 2 * torch.pi)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.unsqueeze(-1)                                   # [..., 1]
        lin = self.w0 * tau + self.b0                            # [..., 1]
        per = torch.sin(tau * self.w + self.b)                  # [..., dim-1]
        return torch.cat([lin, per], dim=-1)                    # [..., dim]


class CrossWalkGRUHead(nn.Module):
    def __init__(self, d_emb: int, d_time: int = 16, num_layers: int = 1):
        super().__init__()
        self.t2v_walk = Time2Vec(d_time)                        # within-walk Δt
        self.gru = nn.GRU(d_emb + d_time, d_emb,
                          num_layers=num_layers, batch_first=True)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        # Query-dependent recency injected at scoring time.
        self.t2v_rec = Time2Vec(d_time)
        self.rec_head = nn.Linear(d_time, 1)

    def encode(self, walk_emb: torch.Tensor, dt_log: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
        """walk_emb [M, K, L, d], dt_log [M, K, L] (log1p Δt; 0 at seed/padding),
        valid [M, K, L] bool -> h [M, d]."""
        M, K, L, d = walk_emb.shape
        tfeat = self.t2v_walk(dt_log) * valid.unsqueeze(-1).float()   # [M,K,L,d_time]
        x = torch.cat([walk_emb, tfeat], dim=-1)                      # [M,K,L,d+d_time]
        out, _ = self.gru(x.reshape(M * K, L, x.shape[-1]))          # [M*K, L, d]
        last_idx = valid.reshape(M * K, L).sum(dim=1).clamp_min(1) - 1
        rows = torch.arange(M * K, device=out.device)
        last = out[rows, last_idx]
        pooled = last.reshape(M, K, d).mean(dim=1)                   # [M, d]
        return F.normalize(pooled, dim=-1)                          # h on the unit sphere

    def forward(self, E_u: torch.Tensor, E_v: torch.Tensor,
                h_u: torch.Tensor, h_v: torch.Tensor,
                rec_v_log: torch.Tensor) -> torch.Tensor:
        """E_u/h_u [B, d]; E_v/h_v [B, C, d]; rec_v_log [B, C] -> [B, C]."""
        eu = F.normalize(E_u, dim=-1).unsqueeze(1)
        hu = F.normalize(h_u, dim=-1).unsqueeze(1)
        ev = F.normalize(E_v, dim=-1)
        hv = F.normalize(h_v, dim=-1)
        eps = 1e-6
        c1 = (eu * hv).sum(dim=-1).clamp(-1 + eps, 1 - eps)
        c2 = (ev * hu).sum(dim=-1).clamp(-1 + eps, 1 - eps)
        # chord distance on the sphere: ‖a-b‖ = √(2-2⟨a,b⟩) (clamp keeps the
        # sqrt gradient finite at coincidence). Closer => higher logit.
        d = torch.sqrt(2.0 - 2.0 * c1) + torch.sqrt(2.0 - 2.0 * c2)  # [B, C]
        rec = self.rec_head(self.t2v_rec(rec_v_log)).squeeze(-1)    # [B, C]
        return -self.logit_scale.clamp_min(1e-3) * d + rec
