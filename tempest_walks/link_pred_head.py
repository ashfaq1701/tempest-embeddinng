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
    def __init__(self, d_emb: int, d_time: int = 16, num_layers: int = 2,
                 use_pair_recency: bool = False, use_pair_history: bool = False,
                 use_ctx_term: bool = False):
        super().__init__()
        self.t2v_walk = Time2Vec(d_time)                        # within-walk Δt
        self.gru = nn.GRU(d_emb + d_time, d_emb,
                          num_layers=num_layers, batch_first=True)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        # Query-dependent recency injected at scoring time.
        self.t2v_rec = Time2Vec(d_time)
        self.rec_head = nn.Linear(d_time, 1)

        # Feature #1/#2: exact pairwise (u,v) recurrence from the streaming store.
        # An additive logit term (mirrors rec_head) keyed on the candidate pair,
        # not the candidate alone. use_pair_history extends it with the ever-bit
        # + decayed log-count (#2). Zero new params / identical logits when off.
        self.use_pair_recency = use_pair_recency
        self.use_pair_history = use_pair_history
        if use_pair_recency:
            self.t2v_pair = Time2Vec(d_time)
            extra = 2 if use_pair_history else 0
            self.pair_head = nn.Linear(d_time + extra, 1)

        # Feature #5: context-context chord term h[u]<->h[v] (free; the missing
        # explicit common-neighbour channel). Own learnable scale so it can
        # vanish if unhelpful.
        self.use_ctx_term = use_ctx_term
        if use_ctx_term:
            self.logit_scale_ctx = nn.Parameter(torch.tensor(1.0))

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
                rec_v_log: torch.Tensor,
                pair_rec_log: torch.Tensor = None,
                pair_ever: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """E_u/h_u [B, d]; E_v/h_v [B, C, d]; rec_v_log [B, C] -> [B, C].
        pair_* [B, C] are the streaming-store features (used iff the matching
        flag is on)."""
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
        logit = -self.logit_scale.clamp_min(1e-3) * d + rec

        if self.use_ctx_term:
            c3 = (hu * hv).sum(dim=-1).clamp(-1 + eps, 1 - eps)     # [B, C]
            logit = logit - self.logit_scale_ctx.clamp_min(1e-3) * torch.sqrt(
                2.0 - 2.0 * c3)

        if self.use_pair_recency:
            feat = self.t2v_pair(pair_rec_log)                     # [B, C, d_time]
            if self.use_pair_history:
                feat = torch.cat(
                    [feat, pair_ever.unsqueeze(-1), pair_count_log.unsqueeze(-1)],
                    dim=-1)
            logit = logit + self.pair_head(feat).squeeze(-1)       # [B, C]

        return logit
