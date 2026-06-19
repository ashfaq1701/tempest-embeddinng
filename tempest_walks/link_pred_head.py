"""Symmetric point head — recency-weighted predictions on both sides, four grad-paths.

Four geometric grad-paths, each a clean inner product of unit/ambient vectors:

  μ_u = recency-weighted tangent mean of u's walk tokens, in T_{E[u]};  P_u = Exp_{E[u]}(μ_u)
  μ_v = recency-weighted tangent mean of v's walk tokens, in T_{E[v]};  P_v = Exp_{E[v]}(μ_v)
  S_u = recency-weighted EXTRINSIC sum of u's token embeddings  (ambient, Σ w_i·E[w_i])
  S_v = recency-weighted EXTRINSIC sum of v's token embeddings  (ambient, Σ w_j·E[w_j])

  T1 = ⟨P_u, E_v⟩                 grad → E[v] identity      (+ u's tokens via P_u)
  T2 = ⟨P_v, E_u⟩                 grad → E[u] identity      (+ v's tokens via P_v)
  T3 = ⟨P_u, S_v⟩                 grad → v's tokens, INDIVIDUALLY, age-weighted (w_j·P_u)
  T4 = ⟨P_v, S_u⟩                 grad → u's tokens, INDIVIDUALLY, age-weighted (w_i·P_v)
  geo = α·(coef_t1·T1 + coef_t2·T2 + coef_t3·T3 + coef_t4·T4)
  logit = coef_geo·geo + coef_rec·rec(v) [+ coef_pair·pair(u,v) + coef_pair_count·log1p(count)]

KEY IDENTITY (why this is cheap AND exact): sum-pooling commutes through the inner
product — Σ_j w_j⟨P_u, E[w_j]⟩ = ⟨P_u, Σ_j w_j E[w_j]⟩ = ⟨P_u, S_v⟩. So "compare the
prediction against every one of the other side's tokens, individually, age-weighted"
equals "compare against the weighted token SUM" — one per-node vector, no [B,C,M,d]
walk axis, no chunking. Per-token gradient is preserved exactly: ∂T3/∂E[w_j] = w_j·P_u.
(Holds ONLY for linear/sum pooling; logsumexp/max would NOT commute and would force the
walk axis. S_* is the RAW weighted sum — do not normalize it, or the equality breaks.)

DEDUP: μ_v, P_v, S_v are per-node (v's frame, v's tokens only). The recency weights use
the SHIFT-INVARIANT edge-time form so they are query-independent (t_query cancels);
combined with one pre-ingest snapshot per batch, they are computed ONCE per unique
candidate and scattered via v_inv. No [B,C,M,d] tensor is ever materialized.

COLD NODE: no tokens ⇒ μ=0 ⇒ P=E[node]; S=0. geo degrades to the identity terms
(T1,T2 → ⟨E_u,E_v⟩) with the token terms (T3,T4) contributing 0. Graceful.

E is the single sphere parameter (geoopt ManifoldParameter, link-trained, no detach).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpDecayBasis(nn.Module):
    """Multi-rate exponential-decay staleness encoder: φ(Δt) = [exp(−ρ_k·Δt)]_{k=1..K}.
    ρ_k = exp(log_rates_k) > 0 learnable, log-spaced init from 1/t_train to 1. Scale-free
    on RAW Δt, bounded [0,1], monotone, multi-timescale. Never-seen event (Δt→+inf) ⇒ 0."""

    def __init__(self, dim: int, t_train: float):
        super().__init__()
        r_lo = -math.log(max(float(t_train), 1.0))
        r_hi = 0.0
        self.log_rates = nn.Parameter(torch.linspace(r_lo, r_hi, dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.exp(self.log_rates) * dt.unsqueeze(-1))


class SymmetricPointHead(nn.Module):

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────

    def __init__(self, d_emb: int, d_time: int = 16,
                 use_pair_features: bool = False, t_train: float = 1.0,
                 token_terms_start_off: bool = True) -> None:
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6

        # shared recency rate λ = softplus(log_lambda); init ≈ C/t_train (C=10) so
        # λ·age ~ O(1) → softmax unsaturated → λ trains (zero-init would argmax-pin it).
        lam0 = 10.0 / max(float(t_train), 1.0)
        self.log_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(10.0))          # geo → logit scale

        # per-term mix gains: identity terms on (init 1), token terms off (init 0)
        self.coef_t1 = nn.Parameter(torch.ones(1))             # ⟨P_u, E_v⟩
        self.coef_t2 = nn.Parameter(torch.ones(1))             # ⟨P_v, E_u⟩
        t_init = 0.0 if token_terms_start_off else 1.0
        self.coef_t3 = nn.Parameter(torch.full((1,), t_init))  # ⟨P_u, S_v⟩
        self.coef_t4 = nn.Parameter(torch.full((1,), t_init))  # ⟨P_v, S_u⟩

        # additive channels
        self.basis_rec = ExpDecayBasis(d_time, t_train)
        self.rec_head = nn.Linear(d_time, 1)
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.basis_pair = ExpDecayBasis(d_time, t_train)
            self.pair_head = nn.Linear(d_time, 1)

        self.coef_geo = nn.Parameter(torch.ones(1))
        self.coef_rec = nn.Parameter(torch.ones(1))
        if use_pair_features:
            self.coef_pair = nn.Parameter(torch.ones(1))
            self.coef_pair_count = nn.Parameter(torch.zeros(1))

    # ──────────────────────────────────────────────────────────────────
    # Sphere primitives
    # ──────────────────────────────────────────────────────────────────

    def _logmap(self, base: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        c = (base * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * base
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _expmap(self, base: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        vn = v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.cos(vn) * base + torch.sin(vn) * (v / vn)

    # ──────────────────────────────────────────────────────────────────
    # Per-node aggregates
    # ──────────────────────────────────────────────────────────────────

    def _recency_weights(self, edge_t: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """w = softmax(+λ·edge_t) over the token axis, query-independent (shift-invariant).
           Mask to −inf BEFORE softmax so the internal max is over valid tokens only;
           a node with no valid tokens → all-0 (cold)."""
        lam = F.softplus(self.log_lambda)
        wlog = (lam * edge_t).masked_fill(~mask, float("-inf"))      # [..., M]
        return torch.nan_to_num(torch.softmax(wlog, dim=-1), nan=0.0)

    def _node_aggregates(self, base_emb: torch.Tensor, tok_emb: torch.Tensor,
                         tok_edge_t: torch.Tensor, tok_mask: torch.Tensor):
        """P = Exp_base(Σ w·Log_base(tok))  [G,d] unit ;  S = Σ w·tok  [G,d] ambient (raw).
           Same w drives both. Cold ⇒ μ=0 ⇒ P=base, S=0."""
        base = F.normalize(base_emb, dim=-1)                         # [G, d]
        tok = F.normalize(tok_emb, dim=-1)                          # [G, M, d]
        w = self._recency_weights(tok_edge_t, tok_mask)            # [G, M]
        # P: tangent mean → sphere
        g = self._logmap(base.unsqueeze(-2), tok)                  # [G, M, d]
        mu = (w.unsqueeze(-1) * g).sum(dim=-2)                     # [G, d]
        P = self._expmap(base, mu)                                 # [G, d]
        # S: raw ambient weighted sum (NOT normalized)
        S = (w.unsqueeze(-1) * tok).sum(dim=-2)                    # [G, d]
        return P, S

    # ──────────────────────────────────────────────────────────────────
    # Geometric score
    # ──────────────────────────────────────────────────────────────────

    def _geo(self, P_u, S_u, E_u, P_v, S_v, E_v) -> torch.Tensor:
        """α·(c1·⟨P_u,E_v⟩ + c2·⟨P_v,E_u⟩ + c3·⟨P_u,S_v⟩ + c4·⟨P_v,S_u⟩)."""
        eu = F.normalize(E_u, dim=-1).unsqueeze(1)                  # [B,1,d]
        ev = F.normalize(E_v, dim=-1)                              # [B,C,d]
        Pu = P_u.unsqueeze(1)                                      # [B,1,d]
        Su = S_u.unsqueeze(1)                                      # [B,1,d]
        t1 = (Pu * ev).sum(-1)                                     # ⟨P_u, E_v⟩
        t2 = (P_v * eu).sum(-1)                                    # ⟨P_v, E_u⟩
        t3 = (Pu * S_v).sum(-1)                                    # ⟨P_u, S_v⟩
        t4 = (P_v * Su).sum(-1)                                    # ⟨P_v, S_u⟩
        alpha = self.alpha.clamp_min(1e-3)
        return alpha * (self.coef_t1 * t1 + self.coef_t2 * t2
                        + self.coef_t3 * t3 + self.coef_t4 * t4)   # [B,C]

    # ──────────────────────────────────────────────────────────────────
    # Additive channels
    # ──────────────────────────────────────────────────────────────────

    def _rec(self, rec_v_dt: torch.Tensor) -> torch.Tensor:
        return self.rec_head(self.basis_rec(rec_v_dt)).squeeze(-1)

    def _pair(self, pair_dt: torch.Tensor, pair_count_log: torch.Tensor) -> torch.Tensor:
        pair = self.pair_head(self.basis_pair(pair_dt)).squeeze(-1)
        return self.coef_pair * pair + self.coef_pair_count * pair_count_log

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(self,
                E_u: torch.Tensor,
                tok_u_emb: torch.Tensor, tok_u_edge_t: torch.Tensor,
                tok_u_mask: torch.Tensor,
                uniq_v_ids: torch.Tensor, v_inv: torch.Tensor,
                tok_v_ids: torch.Tensor, tok_v_edge_t: torch.Tensor,
                tok_v_mask: torch.Tensor, e_weight: torch.Tensor,
                rec_v_dt: torch.Tensor,
                pair_dt: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        B, C = rec_v_dt.shape

        # source side (per query)
        P_u, S_u = self._node_aggregates(E_u, tok_u_emb, tok_u_edge_t, tok_u_mask)  # [B,d]×2

        # candidate side (per UNIQUE node, then scatter)
        ev_u = F.embedding(uniq_v_ids, e_weight)                              # [Mv,d]
        tok_v_emb = F.embedding(tok_v_ids.clamp_min(0), e_weight)             # [Mv,M,d]
        P_v_u, S_v_u = self._node_aggregates(ev_u, tok_v_emb, tok_v_edge_t, tok_v_mask)
        E_v = F.normalize(ev_u, dim=-1)[v_inv].view(B, C, -1)                # [B,C,d]
        P_v = P_v_u[v_inv].view(B, C, -1)                                    # [B,C,d]
        S_v = S_v_u[v_inv].view(B, C, -1)                                    # [B,C,d]

        geo = self._geo(P_u, S_u, E_u, P_v, S_v, E_v)                        # [B,C]
        logit = self.coef_geo * geo + self.coef_rec * self._rec(rec_v_dt)
        if self.use_pair_features:
            logit = logit + self._pair(pair_dt, pair_count_log)
        return logit
