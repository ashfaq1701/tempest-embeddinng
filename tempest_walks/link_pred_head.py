"""Unified symmetric witness head — symmetric predictions + soft-MAX witness terms.

The single-model "best of both": symmetric predictions on BOTH sides, with the
candidate/source token comparisons done as recency-weighted soft-MAX WITNESS terms
(logsumexp) rather than a diluting sum — so the discriminative "v has one exact
connector matching u's prediction" signal is representable, while a learnable sharpness
β lets the witness soften back toward the average when that is better.

  μ_u = Σ_i softmax(+λ·edge_t_i^u)·Log_{E[u]}(E[w_i^u]) ;  P_u = Exp_{E[u]}(μ_u)
  μ_v = Σ_j softmax(+λ·edge_t_j^v)·Log_{E[v]}(E[w_j^v]) ;  P_v = Exp_{E[v]}(μ_v)

  T1 = ⟨P_u, E_v⟩                                         identity  (cheap, [B,C])
  T2 = ⟨P_v, E_u⟩                                         identity  (cheap, [B,C])
  T3 = logsumexp_j ( β·⟨P_u, E[w_j^v]⟩ − ρ·age_j^v )     candidate witness  ([B,C,M])
  T4 = logsumexp_i ( β·⟨P_v, E[w_i^u]⟩ − ρ·age_i^u )     source witness     ([B,C,M])
  geo   = α·(c1·T1 + c2·T2 + c3·T3 + c4·T4)
  logit = coef_geo·geo + coef_rec·rec(v) [+ coef_pair·pair(u,v) + coef_pair_count·log1p(count)]

WHY WITNESS (logsumexp), NOT SUM: on recurrence-heavy wiki the discriminative question is
"does v have ONE token landing on u's prediction" — a max/argmax signal. A linear sum
(Σ w·⟨P_u,E[w]⟩ = ⟨P_u, Σ w·E[w]⟩) dilutes that single witness among many tokens and
structurally cannot represent it. logsumexp surfaces it. The price is that logsumexp does
NOT commute through the inner product, so the per-token [B,C,M,d] axis is real and there
is NO candidate-side dedup (deliberate — dropped to buy back the witness). Single pass, no
chunking; peak memory ~ B·C·M·d, bounded by walk length M (run M≈50 where it fits).

β — THE SUM↔MIN DIAL: large β → hard witness (single best connector); small β → softens
toward the average. One learnable β (shared T3/T4) subsumes both prior heads. Init ~1.

TWO RECENCY-RATE ROLES (cannot merge):
  • μ-λ  = softplus(log_lambda)  — SOFTMAX (scale-invariant). softmax(+λ·edge_t):
    t_query cancels, query-independent. ONE shared λ for μ_u, μ_v.
  • witness-ρ = exp(log_rate)    — LOGSUMEXP (NOT scale-invariant). raw AGE, init
    −log(t_train) ⇒ ρ·age~O(1). ONE shared ρ for T3, T4.
  μ uses edge_t (shift-inv); witness uses age (bounded ρ). The trainer supplies both.

COEFS: c1,c2 init 1 ; c3,c4 init 0 (witness terms earn weight; baseline = symmetric
identity-cosine head). Plain cosine identity (no ellipse to start).

COLD NODE: no tokens ⇒ μ=0 ⇒ P=E[node]; an all-masked witness returns NEUTRAL 0
(mask −inf before logsumexp, then where(any_valid, ·, 0)). geo degrades to identity terms.

E is the single sphere parameter (geoopt ManifoldParameter, link-trained, no detach).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpDecayBasis(nn.Module):
    """φ(Δt) = [exp(−ρ_k·Δt)]_{k=1..K}, ρ_k = exp(log_rates_k) log-spaced 1/t_train→1.
       Scale-free on RAW Δt, bounded [0,1], monotone; Δt→∞ ⇒ 0."""

    def __init__(self, dim: int, t_train: float) -> None:
        super().__init__()
        r_lo = -math.log(max(float(t_train), 1.0))
        r_hi = 0.0
        self.log_rates = nn.Parameter(torch.linspace(r_lo, r_hi, dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.exp(self.log_rates) * dt.unsqueeze(-1))


class UnifiedSymmetricWitnessHead(nn.Module):

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────

    def __init__(self, d_emb: int, d_time: int = 16,
                 use_pair_features: bool = False, t_train: float = 1.0,
                 witness_terms_start_off: bool = True) -> None:
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6

        # μ-recency λ (softmax, shift-invariant): init ≈ C/t_train so λ·age~O(1) → λ trains
        lam0 = 10.0 / max(float(t_train), 1.0)
        self.log_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))
        # witness-recency ρ (logsumexp, bounded): init −log(t_train) ⇒ ρ·age~O(1)
        self.log_rate = nn.Parameter(
            torch.tensor([-math.log(max(float(t_train), 1.0))], dtype=torch.float32))
        # witness sharpness β (sum↔min dial): init moderate β≈1 ; softplus⁻¹(1)
        self.log_beta = nn.Parameter(
            torch.tensor([math.log(math.expm1(1.0))], dtype=torch.float32))

        self.alpha = nn.Parameter(torch.tensor(10.0))            # geo → logit scale

        # per-term mix: identity on, witness off (earn weight)
        self.coef_t1 = nn.Parameter(torch.ones(1))               # ⟨P_u, E_v⟩
        self.coef_t2 = nn.Parameter(torch.ones(1))               # ⟨P_v, E_u⟩
        c0 = 0.0 if witness_terms_start_off else 1.0
        self.coef_t3 = nn.Parameter(torch.full((1,), c0))        # candidate witness
        self.coef_t4 = nn.Parameter(torch.full((1,), c0))        # source witness

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
    # Prediction
    # ──────────────────────────────────────────────────────────────────

    def _mu_recency_weights(self, edge_t: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """softmax(+λ·edge_t) over the last (token) axis; shift-invariant.
           Mask to −inf BEFORE the softmax max-subtraction; no tokens ⇒ all-0."""
        lam = F.softplus(self.log_lambda)
        wlog = (lam * edge_t).masked_fill(~mask, float("-inf"))
        return torch.nan_to_num(torch.softmax(wlog, dim=-1), nan=0.0)

    def _predict(self, base_emb: torch.Tensor, tok_emb: torch.Tensor,
                 tok_edge_t: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
        """P = Exp_base(Σ w·Log_base(tok)). Shape-agnostic over leading axes:
           base_emb [..., d] ; tok_emb [..., M, d] ; tok_edge_t/tok_mask [..., M]."""
        base = F.normalize(base_emb, dim=-1)
        tok = F.normalize(tok_emb, dim=-1)
        w = self._mu_recency_weights(tok_edge_t, tok_mask)        # [..., M]
        g = self._logmap(base.unsqueeze(-2), tok)                # [..., M, d]
        mu = (w.unsqueeze(-1) * g).sum(dim=-2)                   # [..., d]
        return self._expmap(base, mu)                            # [..., d]

    # ──────────────────────────────────────────────────────────────────
    # Witness
    # ──────────────────────────────────────────────────────────────────

    def _witness(self, P: torch.Tensor, tok_emb: torch.Tensor,
                 tok_age: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
        """logsumexp_m( β·⟨P, tok_m⟩ − ρ·age_m ) over the token axis; neutral 0 if empty.
           P [B,C,d] ; tok_emb [B,C,M,d] (unit) ; tok_age/tok_mask [B,C,M]  -> [B,C]."""
        tok = F.normalize(tok_emb, dim=-1)                       # [B,C,M,d]
        sim = (P.unsqueeze(-2) * tok).sum(-1)                    # [B,C,M]  ⟨P, tok_m⟩
        beta = F.softplus(self.log_beta)
        rho = torch.exp(self.log_rate)
        lw = (beta * sim - rho * tok_age).masked_fill(~tok_mask, float("-inf"))  # [B,C,M]
        wit = torch.logsumexp(lw, dim=-1)                        # [B,C]
        return torch.where(tok_mask.any(dim=-1), wit, torch.zeros_like(wit))

    # ──────────────────────────────────────────────────────────────────
    # Geometric score
    # ──────────────────────────────────────────────────────────────────

    def _geo(self, P_u: torch.Tensor, E_u: torch.Tensor,
             P_v: torch.Tensor, E_v: torch.Tensor,
             wit_cand: torch.Tensor, wit_src: torch.Tensor) -> torch.Tensor:
        eu = F.normalize(E_u, dim=-1).unsqueeze(1)               # [B,1,d]
        ev = F.normalize(E_v, dim=-1)                           # [B,C,d]
        t1 = (P_u.unsqueeze(1) * ev).sum(-1)                    # ⟨P_u, E_v⟩   [B,C]
        t2 = (P_v * eu).sum(-1)                                 # ⟨P_v, E_u⟩   [B,C]
        alpha = self.alpha.clamp_min(1e-3)
        return alpha * (self.coef_t1 * t1 + self.coef_t2 * t2
                        + self.coef_t3 * wit_cand + self.coef_t4 * wit_src)

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
                # ── source side (per query) ──
                E_u: torch.Tensor,             # [B, d]
                tok_u_emb: torch.Tensor,       # [B, n, d]
                tok_u_edge_t: torch.Tensor,    # [B, n]   → μ_u softmax (shift-invariant)
                tok_u_age: torch.Tensor,       # [B, n]   → T4 witness (bounded ρ)
                tok_u_mask: torch.Tensor,      # [B, n]
                # ── candidate side (per UNIQUE node + scatter) ──
                uniq_v_ids: torch.Tensor,      # [Mv, d]  unique candidate embeddings → E_v / P_v base
                v_inv: torch.Tensor,           # [B*C]    scatter index (unique → [B,C] grid)
                tok_v_emb_u: torch.Tensor,     # [Mv, M, d]  unique candidate tokens (_predict runs here)
                tok_v_edge_t_u: torch.Tensor,  # [Mv, M]     → μ_v softmax (shift-invariant, dedupable)
                tok_v_mask_u: torch.Tensor,    # [Mv, M]
                tok_v_age: torch.Tensor,       # [B, C, M]   → T3 witness; query-dependent, GENUINELY grid
                # ── additive channels ──
                rec_v_dt: torch.Tensor,        # [B, C]
                pair_dt: torch.Tensor = None,         # [B, C]
                pair_count_log: torch.Tensor = None   # [B, C]
                ) -> torch.Tensor:
        B, C = rec_v_dt.shape
        M = tok_v_emb_u.shape[1]
        d = self.d_emb

        # source prediction (per query)
        P_u = self._predict(E_u, tok_u_emb, tok_u_edge_t, tok_u_mask)          # [B,d]

        # candidate prediction PER UNIQUE v (v's own frame/tokens; shift-invariant +λ·edge_t
        # weights → query-independent), then scatter. Bit-exact to the full-grid P_v in both
        # forward and backward (scatter-add adjoint), at [Mv,M,d] instead of [B,C,M,d].
        P_v_u = self._predict(uniq_v_ids, tok_v_emb_u, tok_v_edge_t_u, tok_v_mask_u)  # [Mv,d]
        P_v = P_v_u[v_inv].view(B, C, d)                                       # [B,C,d]
        E_v = F.normalize(uniq_v_ids, dim=-1)[v_inv].view(B, C, d)            # [B,C,d]

        # T3 candidate witness: P_u (broadcast over C) vs v's tokens. The ONLY candidate-side
        # [B,C,M,d] block — scatter the unique tokens/mask to the grid here (witness can't dedup).
        tok_v_emb = tok_v_emb_u[v_inv].view(B, C, M, d)                       # [B,C,M,d]
        tok_v_mask = tok_v_mask_u[v_inv].view(B, C, M)                        # [B,C,M]
        P_u_bc = P_u.unsqueeze(1).expand(B, C, d)                             # [B,C,d]
        wit_cand = self._witness(P_u_bc, tok_v_emb, tok_v_age, tok_v_mask)     # [B,C]

        # T4 source witness: P_v vs u's tokens, broadcast u's [B,n,*] over C → [B,C,n,*]
        n = tok_u_emb.shape[1]
        tok_u_emb_bc = tok_u_emb.unsqueeze(1).expand(B, C, n, d)               # [B,C,n,d]
        tok_u_age_bc = tok_u_age.unsqueeze(1).expand(B, C, n)                  # [B,C,n]
        tok_u_mask_bc = tok_u_mask.unsqueeze(1).expand(B, C, n)               # [B,C,n]
        wit_src = self._witness(P_v, tok_u_emb_bc, tok_u_age_bc, tok_u_mask_bc)  # [B,C]

        geo = self._geo(P_u, E_u, P_v, E_v, wit_cand, wit_src)                 # [B,C]
        logit = self.coef_geo * geo + self.coef_rec * self._rec(rec_v_dt)
        if self.use_pair_features:
            logit = logit + self._pair(pair_dt, pair_count_log)
        return logit
