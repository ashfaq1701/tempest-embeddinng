"""Unified symmetric witness head — ONE tangent geometry (master's ellipse, symmetric).

Everything is measured in the tangent: each node builds a recency-weighted tangent-mean
prediction μ and a heading r = μ/‖μ‖ IN ITS OWN FRAME (at E[node]); both the identity and
the witness are anisotropic ELLIPSE distances from the OTHER side's tangent-mapped points
to μ. No Exp, no sphere point P — μ stays in the tangent (exactly master's _cross/geo).

  μ_u = Σ_i softmax(+λ·edge_t_i^u)·Log_{E[u]}(E[w_i^u])  ;  r_u = μ_u/‖μ_u‖   (u's frame)
  μ_v = Σ_j softmax(+λ·edge_t_j^v)·Log_{E[v]}(E[w_j^v])  ;  r_v = μ_v/‖μ_v‖   (v's frame)
  ellipse_d(δ; r) = √( a·⟨δ,r⟩² + b·(‖δ‖²−⟨δ,r⟩²) )                a,b ≥ 0, shared

  T1 = −α·ellipse_d( Log_{E[u]}(E_v) − μ_u ; r_u )       candidate identity, u's frame  [B,C]
  T2 = −α·ellipse_d( Log_{E[v]}(E_u) − μ_v ; r_v )       source identity, v's frame      [B,C]
  T3 = logsumexp_j( −α·ellipse_d(Log_{E[u]}(E[w_j^v])−μ_u ; r_u) − ρ·age_j )   cand witness [B,C,M]
  T4 = logsumexp_i( −α·ellipse_d(Log_{E[v]}(E[w_i^u])−μ_v ; r_v) − ρ·age_i )   src  witness [B,C,M]
  geo   = coef_geo·( c1·T1 + c2·T2 + c3·T3 + c4·T4 )
  logit = geo + coef_rec·rec(v) [+ coef_pair·pair(u,v) + coef_pair_count·log1p(count)]

ELLIPSE (anisotropy): the distance is an ELLIPSE oriented along each side's heading r=μ/‖μ‖,
not a circle — being off ALONG the heading (a·‖δ∥‖²) is weighted differently from sideways
(b·‖δ⊥‖²). a=b → isotropic circle √a·‖δ‖ (init log_a=log_b=0 ⇒ a=b=softplus(0)=0.69; α
absorbs the scale). a,b = softplus(·) (master's parameterization). Shared (a,b) across the
u-side and v-side keeps the head symmetric. Cold μ→0 ⇒ ‖r‖=‖μ‖/eps→0 ⇒ d → √b·‖ν‖ (geodesic,
verified graceful: r's MAGNITUDE shrinks, not a unit direction — no noise amplification).

WITNESS = soft-MIN (logsumexp of −α·d) over the OTHER side's tokens — "does v have ONE
token whose tangent image lands on u's prediction", existential not averaged. c3,c4 init 0
(earn weight; baseline = the symmetric identity ellipse). NON-dedupable (the Log into the
other frame is pair-dependent → the [B,C,M,d] axis is real). μ_v/r_v ARE per-v and stay
deduped: computed once per unique v on [Mv,M,d] and scattered via v_inv (bit-exact).

TWO RECENCY-RATE ROLES (cannot merge):
  • μ-λ  = softplus(log_lambda) — SOFTMAX over edge_t (shift-invariant, query-independent →
    keeps μ_v dedupable). ONE shared λ for μ_u, μ_v.
  • witness-ρ = exp(log_rate)   — LOGSUMEXP over raw age (NOT scale-invariant), init
    −log(t_train) ⇒ ρ·age~O(1). ONE shared ρ for T3, T4.
α is the shared distance weight (init 10) for identity AND witness (master's single scale).

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

        # α: shared distance weight for identity (−α·d) AND witness soft-min (master's single scale)
        self.alpha = nn.Parameter(torch.tensor(10.0))
        # anisotropic ellipse (shared u/v-side): a=along-heading, b=off-heading; init a=b=1 (circle)
        self.log_a = nn.Parameter(torch.zeros(1))
        self.log_b = nn.Parameter(torch.zeros(1))

        # per-term mix: identity on, witness off (earn weight)
        self.coef_t1 = nn.Parameter(torch.ones(1))               # candidate identity (u frame)
        self.coef_t2 = nn.Parameter(torch.ones(1))               # source identity (v frame)
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
    # Tangent primitives
    # ──────────────────────────────────────────────────────────────────

    def _logmap(self, base: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Sphere log-map at unit `base`: Log_base(x) ∈ T_base. base broadcasts over x.
           base [...,d] unit ; x [...,d] unit  ->  [...,d] tangent."""
        c = (base * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * base
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _ellipse_dist_sq(self, delta: torch.Tensor, r: torch.Tensor,
                         a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Anisotropic ellipse distance² in the heading frame: a·⟨δ,r⟩² + b·‖δ⊥‖².
           delta [...,d] ; r [...,d] (broadcastable, heading)  ->  [...] (no d)."""
        dpar2 = (delta * r).sum(-1).pow(2)
        dperp2 = ((delta * delta).sum(-1) - dpar2).clamp_min(0.0)
        return a * dpar2 + b * dperp2

    def _ellipse_d(self, delta: torch.Tensor, r: torch.Tensor,
                   a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Ellipse distance (sqrt of dist²), eps-clamped."""
        return self._ellipse_dist_sq(delta, r, a, b).clamp_min(self.eps).sqrt()

    # ──────────────────────────────────────────────────────────────────
    # Prediction (tangent mean μ — NO Exp; shape-agnostic over leading axes)
    # ──────────────────────────────────────────────────────────────────

    def _predict_mu(self, base_emb: torch.Tensor, tok_emb: torch.Tensor,
                    tok_edge_t: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
        """μ = Σ softmax(+λ·edge_t)·Log_base(tok)  in base's tangent. Cold ⇒ μ=0.
           base_emb [...,d] ; tok_emb [...,M,d] ; tok_edge_t/tok_mask [...,M]  ->  μ [...,d]."""
        base = F.normalize(base_emb, dim=-1)
        tok = F.normalize(tok_emb, dim=-1)
        lam = F.softplus(self.log_lambda)
        wlog = (lam * tok_edge_t).masked_fill(~tok_mask, float("-inf"))
        w = torch.nan_to_num(torch.softmax(wlog, dim=-1), nan=0.0)
        g = self._logmap(base.unsqueeze(-2), tok)               # [...,M,d]
        return (w.unsqueeze(-1) * g).sum(dim=-2)                # [...,d]

    # ──────────────────────────────────────────────────────────────────
    # Witness (tangent ellipse soft-min over the OTHER side's tokens)
    # ──────────────────────────────────────────────────────────────────

    def _witness(self, base_pt: torch.Tensor, mu: torch.Tensor, r: torch.Tensor,
                 tok_emb: torch.Tensor, tok_age: torch.Tensor, tok_mask: torch.Tensor,
                 a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor,
                 rho: torch.Tensor) -> torch.Tensor:
        """logsumexp_m( −α·ellipse_d(Log_base(tok_m) − μ ; r) − ρ·age_m ); neutral 0 if empty.
           base_pt/mu/r [...,1,d] (broadcast over the token axis) ; tok_emb [...,M,d]  -> [...]."""
        tok = F.normalize(tok_emb, dim=-1)                      # [...,M,d]
        gc = self._logmap(base_pt, tok)                         # [...,M,d]  Log_base(tok)
        dist = self._ellipse_d(gc - mu, r, a, b)                # [...,M]
        lw = (-alpha * dist - rho * tok_age).masked_fill(~tok_mask, float("-inf"))
        wit = torch.logsumexp(lw, dim=-1)                       # [...]
        return torch.where(tok_mask.any(dim=-1), wit, torch.zeros_like(wit))

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
                tok_u_edge_t: torch.Tensor,    # [B, n]   → μ_u softmax
                tok_u_age: torch.Tensor,       # [B, n]   → T4 witness
                tok_u_mask: torch.Tensor,      # [B, n]
                # ── candidate side (per UNIQUE node + scatter) ──
                uniq_v_ids: torch.Tensor,      # [Mv, d]  unique candidate embeddings
                v_inv: torch.Tensor,           # [B*C]    scatter index
                tok_v_emb_u: torch.Tensor,     # [Mv, M, d]  unique candidate tokens
                tok_v_edge_t_u: torch.Tensor,  # [Mv, M]     → μ_v softmax (dedupable)
                tok_v_mask_u: torch.Tensor,    # [Mv, M]
                tok_v_age: torch.Tensor,       # [B, C, M]   → T3 witness; query-dependent grid
                # ── additive channels ──
                rec_v_dt: torch.Tensor,        # [B, C]
                pair_dt: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        B, C = rec_v_dt.shape
        d = self.d_emb
        M = tok_v_emb_u.shape[1]
        a = F.softplus(self.log_a)         # ellipse weights > 0 (master's parameterization)
        b = F.softplus(self.log_b)
        alpha = self.alpha.clamp_min(1e-3)
        rho = torch.exp(self.log_rate)

        eu = F.normalize(E_u, dim=-1)                                   # [B,d]   source identity (unit)
        ev_u = F.normalize(uniq_v_ids, dim=-1)                          # [Mv,d]  unique candidate identity

        # tangent predictions μ (candidate side deduped, then scattered) + headings r = μ/‖μ‖
        mu_u = self._predict_mu(E_u, tok_u_emb, tok_u_edge_t, tok_u_mask)               # [B,d]
        mu_v_u = self._predict_mu(uniq_v_ids, tok_v_emb_u, tok_v_edge_t_u, tok_v_mask_u)  # [Mv,d]
        mu_v = mu_v_u[v_inv].view(B, C, d)                             # [B,C,d]
        ev = ev_u[v_inv].view(B, C, d)                                # [B,C,d]
        ru = mu_u / mu_u.norm(dim=-1, keepdim=True).clamp_min(self.eps)        # [B,d]
        rv = mu_v / mu_v.norm(dim=-1, keepdim=True).clamp_min(self.eps)        # [B,C,d]

        # IDENTITY — tangent ellipse distance to μ, both frames → −α·d  (master's geo channel)
        nu_uv = self._logmap(eu.unsqueeze(1), ev)                     # [B,C,d]  Log_{E[u]}(E_v)
        d1 = self._ellipse_d(nu_uv - mu_u.unsqueeze(1), ru.unsqueeze(1), a, b)   # [B,C]
        nu_vu = self._logmap(ev, eu.unsqueeze(1).expand(B, C, d))     # [B,C,d]  Log_{E[v]}(E_u)
        d2 = self._ellipse_d(nu_vu - mu_v, rv, a, b)                  # [B,C]
        t1 = -alpha * d1
        t2 = -alpha * d2

        # WITNESS — tangent ellipse soft-min over the other side's tokens (the [B,C,M,d] blocks)
        tok_v_emb = tok_v_emb_u[v_inv].view(B, C, M, d)              # [B,C,M,d]
        tok_v_mask = tok_v_mask_u[v_inv].view(B, C, M)              # [B,C,M]
        wit_cand = self._witness(                                    # T3, u's frame [B,C]
            eu[:, None, None, :], mu_u[:, None, None, :], ru[:, None, None, :],
            tok_v_emb, tok_v_age, tok_v_mask, a, b, alpha, rho)
        n = tok_u_emb.shape[1]
        wit_src = self._witness(                                     # T4, v's frame [B,C]
            ev[:, :, None, :], mu_v[:, :, None, :], rv[:, :, None, :],
            tok_u_emb[:, None, :, :].expand(B, C, n, d),
            tok_u_age[:, None, :].expand(B, C, n),
            tok_u_mask[:, None, :].expand(B, C, n), a, b, alpha, rho)

        geo = self.coef_geo * (self.coef_t1 * t1 + self.coef_t2 * t2
                               + self.coef_t3 * wit_cand + self.coef_t4 * wit_src)
        logit = geo + self.coef_rec * self._rec(rec_v_dt)
        if self.use_pair_features:
            logit = logit + self._pair(pair_dt, pair_count_log)
        return logit
