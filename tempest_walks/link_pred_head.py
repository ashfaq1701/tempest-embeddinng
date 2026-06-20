"""Geometric link head — ASYMMETRIC CSR version (μ_u + query co-reach only).

This is the SYMMETRIC head (SymmetricGeometricHead) restricted to the QUERY side: it builds
a prediction μ_u for the source only and scores the candidate against it. It is written to
MIRROR the symmetric head exactly — same helpers (_logmap, _ellipse_dist_sq, _heading,
_mu_from_csr, _identity, _coreach), same shared parameters and NAMES, same forward signature
— so the transition to the full symmetric head is a clean superset: add μ_v, the two
candidate-side terms, and their two coefficients; nothing else changes. An asymmetric
checkpoint loads into the symmetric head directly (the candidate coefs init 0 ⇒ symmetric
starts identical to asymmetric).

  μ_u = Σ_w softmax(logsumexp_occ(−λ·age)+γμ·log(1+k))·Log_{E[u]}(E[w∈u-CSR]) ; r_u = ĝate(μ_u)

  query_identity = −α·ellipse( Log_{E[u]}(E_v) − μ_u ; r_u )         is v in u's region?   [B,C]
  query_coreach  = logsumexp_w(−α·ellipse(Log_{E[u]}(E[v-conn_w])−μ_u; r_u)
                               + γc·log(1+k_w) + logsumexp_occ(−ρ·age))                     [B,C]
                   does v have ONE connector in u's region?  (candidate CSR in u's frame)
  geo   = coef_query_identity·query_identity + coef_query_coreach·query_coreach
  logit = geo + coef_staleness·staleness(v) [+ coef_pair·pair + coef_pair_count·log1p(cnt)]

The candidate CSR is used ONLY as the connector witnesses for query_coreach (it is NOT used
to build a μ_v here — that is the symmetric head's addition). Baseline at init: coef_query_
identity=1, coef_query_coreach=0 ⇒ the proven identity term and nothing else.

μ AS A COUNT CENTER OF MASS, COUNT IN TWO ROLES, TIME UNITS, COLD-NODE GATING — all identical
to the symmetric head; see SymmetricGeometricHead for the full discussion. γμ=0 recovers the
proven recency μ exactly; γc=0 recovers the proven co-reach. E is the single sphere parameter.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpDecayBasis(nn.Module):
    """Multi-rate exp-decay staleness encoder: φ(Δt) = [exp(−ρ_k·Δt)]_{k=1..K}, ρ_k =
    exp(log_rates_k) log-spaced 1/t_train→1. Scale-free on RAW Δt, bounded [0,1]; Δt→∞ ⇒ 0."""

    def __init__(self, dim: int, t_train: float):
        super().__init__()
        r_lo = -math.log(max(float(t_train), 1.0))
        r_hi = 0.0
        self.log_rates = nn.Parameter(torch.linspace(r_lo, r_hi, dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.exp(self.log_rates) * dt.unsqueeze(-1))


class GeometricPointHead(nn.Module):
    def __init__(self, d_emb: int, d_time: int = 16,
                 use_pair_features: bool = False, t_train: float = 1.0):
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6

        # Shared μ recency λ (softmax, scale-invariant), init λ ≈ C/t_train so λ·age~O(1).
        lam0 = 10.0 / max(float(t_train), 1.0)
        self.log_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(10.0))     # shared distance weight
        # Shared anisotropic ellipse (a,b ≥ 0).
        self.log_a = nn.Parameter(torch.zeros(1))
        self.log_b = nn.Parameter(torch.zeros(1))
        # Shared μ count emphasis (relative), init 0 ⇒ recovers recency μ.
        self.gamma_mu = nn.Parameter(torch.zeros(1))

        # --- candidate staleness channel ---------------------------------------
        self.basis_staleness = ExpDecayBasis(d_time, t_train)
        self.staleness_head = nn.Linear(d_time, 1)

        # --- pair channel (FLAGGED) --------------------------------------------
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.basis_pair = ExpDecayBasis(d_time, t_train)
            self.pair_head = nn.Linear(d_time, 1)

        # --- geometric mix coefficients (QUERY side only) ----------------------
        # Names mirror the symmetric head; the candidate-side coefs are simply absent here.
        # query_identity is the proven baseline (init 1); query_coreach earns weight (init 0).
        self.coef_query_identity = nn.Parameter(torch.ones(1))
        self.coef_query_coreach = nn.Parameter(torch.zeros(1))
        self.coef_staleness = nn.Parameter(torch.ones(1))
        if use_pair_features:
            self.coef_pair = nn.Parameter(torch.ones(1))
            self.coef_pair_count = nn.Parameter(torch.zeros(1))

        # --- shared co-reach recency ρ + absolute count γc ---------------------
        self.log_rate_coreach = nn.Parameter(
            torch.tensor([-math.log(max(float(t_train), 1.0))], dtype=torch.float32))
        self.gamma_coreach = nn.Parameter(torch.zeros(1))

    # ──────────────────────────────────────────────────────────────────
    # Geometry primitives (IDENTICAL to SymmetricGeometricHead)
    # ──────────────────────────────────────────────────────────────────

    def _logmap(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _ellipse_dist_sq(self, delta: torch.Tensor, r: torch.Tensor,
                         a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Ellipse distance² a‖δ∥‖²+b‖δ⊥‖² in heading frame r. delta [...,d]; r broadcastable
           to delta (caller unsqueezes the U axis where needed) -> [...]."""
        dpar2 = (delta * r).sum(-1).pow(2)
        dperp2 = ((delta * delta).sum(-1) - dpar2).clamp_min(0.0)
        return a * dpar2 + b * dperp2

    def _heading(self, mu: torch.Tensor) -> torch.Tensor:
        """Gated heading r = g(‖μ‖)·μ/‖μ‖, g=‖μ‖²/(‖μ‖²+m0²) → 0 cold (isotropic), 1 warm."""
        mu_norm = mu.norm(dim=-1, keepdim=True)
        m0 = 0.05
        gate = (mu_norm * mu_norm) / (mu_norm * mu_norm + m0 * m0)
        return gate * mu / mu_norm.clamp_min(self.eps)

    # ──────────────────────────────────────────────────────────────────
    # μ — count center of mass over a CSR (IDENTICAL to SymmetricGeometricHead)
    # ──────────────────────────────────────────────────────────────────

    def _mu_from_csr(self, base_emb: torch.Tensor, ids: torch.Tensor, nmask: torch.Tensor,
                     ages: torch.Tensor, amask: torch.Tensor,
                     e_weight: torch.Tensor) -> torch.Tensor:
        """μ = Σ_w softmax_w(logsumexp_occ(−λ·age)+γμ·log(1+k))·Log_base(E[node_w]).
           base [...,d] ; ids/nmask [...,U] ; ages/amask [...,U,kmax]  ->  μ [...,d]."""
        base = F.normalize(base_emb, dim=-1)
        ew = F.normalize(F.embedding(ids.clamp_min(0), e_weight), dim=-1)   # [...,U,d]
        g = self._logmap(base.unsqueeze(-2), ew)                           # [...,U,d]
        lam = F.softplus(self.log_lambda)
        occ = (-lam * ages).masked_fill(~amask, float("-inf"))            # [...,U,kmax]
        R = torch.logsumexp(occ, dim=-1)                                  # [...,U]
        k = amask.sum(-1).to(ages.dtype)
        ell = (R + self.gamma_mu * torch.log1p(k)).masked_fill(~nmask, float("-inf"))
        w = torch.nan_to_num(torch.softmax(ell, dim=-1), nan=0.0)
        return (w.unsqueeze(-1) * g).sum(dim=-2)                          # [...,d]

    # ──────────────────────────────────────────────────────────────────
    # identity — probe an embedding against a prediction (IDENTICAL)
    # ──────────────────────────────────────────────────────────────────

    def _identity(self, frame: torch.Tensor, probe: torch.Tensor, mu: torch.Tensor,
                  r: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                  alpha: torch.Tensor) -> torch.Tensor:
        """−α·ellipse( Log_frame(probe) − μ ; r ). All [...,d] (caller broadcasts) -> [...]."""
        nu = self._logmap(frame, probe)
        dist = self._ellipse_dist_sq(nu - mu, r, a, b).clamp_min(self.eps).sqrt()
        return -alpha * dist

    # ──────────────────────────────────────────────────────────────────
    # co-reach — count-aware ∃-witness of a CSR against a prediction (IDENTICAL)
    # ──────────────────────────────────────────────────────────────────

    def _coreach(self, frame: torch.Tensor, mu: torch.Tensor, r: torch.Tensor,
                 ids: torch.Tensor, nmask: torch.Tensor, ages: torch.Tensor,
                 amask: torch.Tensor, e_weight: torch.Tensor,
                 a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """logsumexp_w(−α·ellipse(Log_frame(E[w])−μ; r) + γc·log(1+k_w) + logsumexp_occ(−ρ·age)).
           frame/mu/r [...,d] ; ids/nmask [...,U] ; ages/amask [...,U,kmax]  ->  [...]. Neutral 0."""
        ec = F.normalize(F.embedding(ids.clamp_min(0), e_weight), dim=-1)   # [...,U,d]
        gc = self._logmap(frame.unsqueeze(-2), ec)                         # [...,U,d]
        delta = gc - mu.unsqueeze(-2)                                      # [...,U,d]
        d = self._ellipse_dist_sq(delta, r.unsqueeze(-2), a, b).clamp_min(self.eps).sqrt()  # [...,U]
        rho = torch.exp(self.log_rate_coreach)
        A = torch.logsumexp((-rho * ages).masked_fill(~amask, -1e9), dim=-1)   # [...,U]
        k = amask.sum(-1).to(ages.dtype)
        lw = (-alpha * d + self.gamma_coreach * torch.log1p(k) + A).masked_fill(~nmask, -1e9)
        wit = torch.logsumexp(lw, dim=-1)                                 # [...]
        return torch.where(nmask.any(dim=-1), wit, torch.zeros_like(wit))

    # ──────────────────────────────────────────────────────────────────
    # Forward — QUERY side only (mirror of the symmetric forward, candidate terms dropped)
    # ──────────────────────────────────────────────────────────────────

    def forward(self,
                # ── source CSR (u side) ──
                E_u: torch.Tensor,             # [B, d]
                src_ids: torch.Tensor,         # [B, Us]
                src_nmask: torch.Tensor,       # [B, Us]
                src_ages: torch.Tensor,        # [B, Us, ks]
                src_amask: torch.Tensor,       # [B, Us, ks]
                # ── candidate CSR (v side) — used here ONLY as query_coreach connectors ──
                E_v: torch.Tensor,             # [B, C, d]
                cand_ids: torch.Tensor,        # [B, C, Uv]
                cand_nmask: torch.Tensor,      # [B, C, Uv]
                cand_ages: torch.Tensor,       # [B, C, Uv, kv]
                cand_amask: torch.Tensor,      # [B, C, Uv, kv]
                # ── shared table + additive channels ──
                e_weight: torch.Tensor,        # [N, d]
                t_query_t: torch.Tensor,       # [B]  reserved (ages precomputed)
                staleness_dt: torch.Tensor,    # [B, C]
                pair_dt: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """-> logits [B, C]."""
        B, C = E_v.shape[0], E_v.shape[1]
        d = self.d_emb
        eu = F.normalize(E_u, dim=-1)                              # [B, d]
        ev = F.normalize(E_v, dim=-1)                              # [B, C, d]
        a = F.softplus(self.log_a)
        b = F.softplus(self.log_b)
        alpha = self.alpha.clamp_min(1e-3)

        # --- prediction on the QUERY side only ---
        mu_u = self._mu_from_csr(eu, src_ids, src_nmask, src_ages, src_amask, e_weight)  # [B,d]
        r_u = self._heading(mu_u)                                 # [B,d]

        eu_bc = eu.unsqueeze(1).expand(B, C, d)                   # [B,C,d]
        mu_u_bc = mu_u.unsqueeze(1).expand(B, C, d)
        r_u_bc = r_u.unsqueeze(1).expand(B, C, d)

        # --- query_identity: is v in u's region? ---
        q_ident = self._identity(eu_bc, ev, mu_u_bc, r_u_bc, a, b, alpha)   # [B,C]

        # --- query_coreach: does v have a connector in u's region? (candidate CSR in u's frame) ---
        q_coreach = self._coreach(eu_bc, mu_u_bc, r_u_bc,
                                  cand_ids, cand_nmask, cand_ages, cand_amask,
                                  e_weight, a, b, alpha)           # [B,C]

        geo = (self.coef_query_identity * q_ident
               + self.coef_query_coreach * q_coreach)

        # --- candidate STALENESS channel ---
        staleness = self.staleness_head(self.basis_staleness(staleness_dt)).squeeze(-1)  # [B,C]
        logit = geo + self.coef_staleness * staleness

        # --- PAIR channel (FLAGGED) ---
        if self.use_pair_features:
            pair = self.pair_head(self.basis_pair(pair_dt)).squeeze(-1)
            logit = (logit + self.coef_pair * pair
                     + self.coef_pair_count * pair_count_log)

        return logit
