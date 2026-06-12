"""Source-side walk-encoder link head (chord distance on the unit sphere) with time.

SOURCE-SIDE ONLY variant: walks are sampled for the SOURCE u alone (not the
candidates). The source's walk context is GRU-encoded to h[u]; each candidate is its
raw embedding E[v]. The score is the chord distance between h[u] and E[v] — "does the
candidate's embedding match the source's recent walk context".

Per batch the head is given, for every unique SOURCE, K Tempest walks of length L,
plus the per-position inter-event Δt (log1p, query-INDEPENDENT) and, at scoring time,
the query-dependent recency of each candidate (from a per-node last-seen store, since
candidate walks are no longer sampled).

  encode:  GRU over [E(walk node) ‖ Time2Vec(Δt)] -> h[u], projected to the unit
           sphere (deduped per source). The seed (rightmost) position carries Δt=0.
  score:   logit(u, v) = -scale*‖ĥ[u] - E[v]‖
                         + rec_head( Time2Vec( log1p(t_query - t_last[v]) ) )
                         [ + pair_head( pair features )  if use_pair_features ]
           ‖a-b‖ = √(2-2⟨a,b⟩) is the chord distance (both operands on the sphere).
           The recency term carries the query time the GRU is blind to.

Pair features (optional, `use_pair_features`): exact pairwise (u,v) recurrence +
history from the streaming PairRecencyStore — Time2Vec(time-since-last (u,v)
interaction) ‖ ever-interacted bit ‖ decayed log interaction-count — added as one
extra logit term (additive, keyed on the candidate PAIR rather than the candidate
alone like rec_head). Multi-seed confirmed +~0.02 test on tgbl-wiki; byte-identical
to the baseline when off.

E is link-trained (no detach); GRU/Time2Vec/heads are Euclidean. E is the only
manifold parameter.
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
                 use_pair_features: bool = False):
        super().__init__()
        self.t2v_walk = Time2Vec(d_time)                        # within-walk Δt
        self.gru = nn.GRU(d_emb + d_time, d_emb,
                          num_layers=num_layers, batch_first=True)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
        # Query-dependent recency injected at scoring time.
        self.t2v_rec = Time2Vec(d_time)
        self.rec_head = nn.Linear(d_time, 1)

        # Exact pairwise (u,v) recurrence + history from the streaming store, as a
        # single additive logit term: Time2Vec(log1p(t_query - last_ts[u,v])) ‖
        # ever-interacted bit ‖ log1p(count[u,v]). Byte-identical to the baseline
        # when off.
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.t2v_pair = Time2Vec(d_time)
            self.pair_head = nn.Linear(d_time + 2, 1)

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

    def forward(self, h_u: torch.Tensor, E_v: torch.Tensor,
                rec_v_log: torch.Tensor,
                pair_rec_log: torch.Tensor = None,
                pair_ever: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """h_u [B, d] (source walk encoding); E_v [B, C, d] (candidate embeddings);
        rec_v_log [B, C] -> [B, C]. Source-side-only: score the chord distance between
        the source's walk context h[u] and each candidate's raw embedding E[v].
        pair_* [B, C] are the streaming-store features (used iff use_pair_features)."""
        hu = F.normalize(h_u, dim=-1).unsqueeze(1)                 # [B, 1, d]
        ev = F.normalize(E_v, dim=-1)                              # [B, C, d]
        eps = 1e-6
        c = (hu * ev).sum(dim=-1).clamp(-1 + eps, 1 - eps)         # [B, C]
        # chord distance on the sphere: ‖a-b‖ = √(2-2⟨a,b⟩) (clamp keeps the
        # sqrt gradient finite at coincidence). Closer => higher logit.
        d = torch.sqrt(2.0 - 2.0 * c)                              # [B, C]
        rec = self.rec_head(self.t2v_rec(rec_v_log)).squeeze(-1)    # [B, C]
        logit = -self.logit_scale.clamp_min(1e-3) * d + rec

        if self.use_pair_features:
            feat = torch.cat(
                [self.t2v_pair(pair_rec_log),
                 pair_ever.unsqueeze(-1), pair_count_log.unsqueeze(-1)],
                dim=-1)                                             # [B, C, d_time+2]
            logit = logit + self.pair_head(feat).squeeze(-1)       # [B, C]

        return logit
