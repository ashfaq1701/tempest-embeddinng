"""Geometric link head — SYMMETRIC-MEET (μ on BOTH sides), COUNT-FREE.

Builds a displacement μ_u for the source from its walk tokens, μ_v for each candidate from ITS
walk tokens, and scores with the query-anchored terms PLUS a symmetric MEET term that compares
the two drifted positions:

  μ_u = drift in T_{E[u]} ;  μ_v = drift in T_{E[v]}  (same routine, candidate frame + tokens)
  q_u = exp_{E[u]}(μ_u) ;  q_v = exp_{E[v]}(μ_v)      (both back ON the sphere)
  meet(u,v) = ⟨q_u, q_v⟩                              do u's and v's drifted positions coincide?

μ_u and μ_v live in DIFFERENT tangent spaces, so the exp-map sends both to points on the SAME
sphere FIRST — ⟨q_u, q_v⟩ is then parallel-transport-free. meet rises when u and v drift toward a
shared neighbourhood even if E[u], E[v] are structurally far apart (co-reachability as centroid
convergence). coef_meet init 0 ⇒ the head starts as the proven query-anchored baseline and meet
earns its weight. Age stays t_query − t_edge throughout (the trainer forms it; the head never
re-defines it). The asymmetric query_coreach ∃-witness is DROPPED — the symmetric meet replaces
it. The score is identity + meet:

  TOKEN BASIS — each seed carries a COUNT-FREE bag of walk-reached positions p (one token per
  reached position; a node recurring k times is k tokens). μ sums the softmax over those tokens
  directly — multiplicity is implicit in token repetition, no explicit count. age_p =
  t_query − t_edge(p).

  μ_u = exp(γ·C_u) · Σ_p softmax_p(−λ·age_p)·Log_{E[u]}(E[node_p]) ; r_u = ĝate(μ_u)   (μ_v sym.)
        — recency softmax sets the DIRECTION; the COUNT magnitude factor exp(γ·C_u),
          C_u = Σ_w log1p(count_w) over distinct neighbours (walk multiplicity), sets how FAR it
          drifts: a strongly/repeatedly-connected seed drifts further. γ init 0 ⇒ no-op baseline.

  identity = −α·ellipse( Log_{E[u]}(E_v) − μ_u ; r_u )          is v in u's region?     [B,C]
  meet     = ⟨ exp_{E[u]}(μ_u), exp_{E[v]}(μ_v) ⟩              do drifted positions meet? [B,C]
  geo   = coef_identity·identity + coef_meet·meet
  logit = geo + coef_staleness·staleness(v) [+ coef_pair·pair + coef_pair_count·log1p(cnt)]

Both sides build μ: the source tokens drive μ_u (identity + the u half of meet), the candidate
tokens drive μ_v (the v half of meet).

BASELINE AT INIT — coef_identity=1, coef_meet=0 ⇒ at init the head IS the proven identity term
and nothing else; meet earns its weight from zero.

COUNT-FREE — a node reached k times appears as k tokens, so its μ softmax-mass is summed across
them; the old γμ·log(1+k) emphasis (learned ≈ −0.4, suppressed) is gone. The pair channel still
carries its own recency+count.

TIME UNITS — raw ages; μ softmax scale-invariant (λ self-scales). E stays the single sphere
parameter (link-trained, no detach).
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
        self.log_a = nn.Parameter(torch.zeros(1))         # anisotropic ellipse (a,b ≥ 0)
        self.log_b = nn.Parameter(torch.zeros(1))

        # --- candidate staleness channel ---------------------------------------
        self.basis_staleness = ExpDecayBasis(d_time, t_train)
        self.staleness_head = nn.Linear(d_time, 1)

        # --- pair channel (FLAGGED) --------------------------------------------
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.basis_pair = ExpDecayBasis(d_time, t_train)
            self.pair_head = nn.Linear(d_time, 1)

        # --- geometric mix coefficients ----------------------------------------
        # identity is the proven baseline (init 1); meet earns weight (init 0).
        self.coef_identity = nn.Parameter(torch.ones(1))
        # SYMMETRIC MEET — build μ on BOTH sides and compare the two drifted positions. coef init 0.
        self.coef_meet = nn.Parameter(torch.zeros(1))
        self.coef_staleness = nn.Parameter(torch.ones(1))
        # COUNT MAGNITUDE FACTOR — scale μ by exp(γ·C), C = Σ_w log1p(count_w) (walk multiplicity).
        # γ init 0 ⇒ exp(0)=1 ⇒ no-op (recovers the recency-only μ); the count-magnitude earns its
        # weight. γ>0: strongly/repeatedly-connected seeds drift further (larger ‖μ‖, capped ≤ π).
        self.gamma_count = nn.Parameter(torch.zeros(1))
        if use_pair_features:
            self.coef_pair = nn.Parameter(torch.ones(1))
            self.coef_pair_count = nn.Parameter(torch.zeros(1))

    # ──────────────────────────────────────────────────────────────────
    # Geometry primitives (shape-agnostic over leading axes)
    # ──────────────────────────────────────────────────────────────────

    def _logmap(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _expmap(self, p: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """exp_p(δ) = cos‖δ‖·p + sin‖δ‖·δ/‖δ‖ ∈ S^{d-1}. δ = μ is a sum of Log_p's (all ⊥ p),
        so the result is unit-norm; ‖δ‖ capped to the injectivity radius (<π); δ→0 ⇒ exp_p(δ)=p
        (a cold node drifts nowhere). Final normalize is a numeric belt. Sends a tangent
        displacement back ONTO the sphere so two drifted positions q_u, q_v live on the SAME
        manifold and ⟨q_u, q_v⟩ is parallel-transport-free."""
        norm = delta.norm(dim=-1, keepdim=True)                     # ‖μ‖ = drift angle
        theta = norm.clamp(max=math.pi - self.eps)
        coef = torch.sin(theta) / norm.clamp_min(self.eps)         # → 0 as norm→0 (q→p)
        q = torch.cos(theta) * p + coef * delta
        return F.normalize(q, dim=-1)

    def _ellipse_dist_sq(self, delta: torch.Tensor, r: torch.Tensor,
                         a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Ellipse distance² a‖δ∥‖²+b‖δ⊥‖² in the heading frame r. delta [...,d];
           r broadcastable to delta (caller unsqueezes the U axis where needed) -> [...]."""
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
    # μ — recency center of mass over a token bag (frame-agnostic)
    # ──────────────────────────────────────────────────────────────────

    def _mu_from_csr(self, base_emb: torch.Tensor, ids: torch.Tensor, nmask: torch.Tensor,
                     ages: torch.Tensor, e_weight: torch.Tensor) -> torch.Tensor:
        """μ = Σ_p softmax_p(−λ·age_p)·Log_base(E[node_p]) over the seed's token positions.
           base [...,d] ; ids/nmask/ages [...,U]  ->  μ [...,d]. Count-free: a node recurring
           k times is k tokens, so its softmax mass is summed across them automatically."""
        base = F.normalize(base_emb, dim=-1)
        ew = F.normalize(F.embedding(ids.clamp_min(0), e_weight), dim=-1)   # [...,U,d]
        g = self._logmap(base.unsqueeze(-2), ew)                           # [...,U,d]
        lam = F.softplus(self.log_lambda)
        ell = (-lam * ages).masked_fill(~nmask, float("-inf"))            # [...,U]
        w = torch.nan_to_num(torch.softmax(ell, dim=-1), nan=0.0)
        return (w.unsqueeze(-1) * g).sum(dim=-2)                          # [...,d]

    # ──────────────────────────────────────────────────────────────────
    # identity — probe an embedding against a prediction (frame-agnostic)
    # ──────────────────────────────────────────────────────────────────

    def _identity(self, frame: torch.Tensor, probe: torch.Tensor, mu: torch.Tensor,
                  r: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                  alpha: torch.Tensor) -> torch.Tensor:
        """−α·ellipse( Log_frame(probe) − μ ; r ). All [...,d] (caller broadcasts) -> [...]."""
        nu = self._logmap(frame, probe)                                  # [...,d]
        dist = self._ellipse_dist_sq(nu - mu, r, a, b).clamp_min(self.eps).sqrt()
        return -alpha * dist

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(self,
                # ── source tokens (u side) ──
                E_u: torch.Tensor,             # [B, d]
                src_ids: torch.Tensor,         # [B, Us]
                src_nmask: torch.Tensor,       # [B, Us]
                src_ages: torch.Tensor,        # [B, Us]
                src_count: torch.Tensor,       # [B]     Σ_w log1p(count_w) — count magnitude factor
                # ── candidate tokens (v side) — drive μ_v for the meet term ──
                E_v: torch.Tensor,             # [B, C, d]
                cand_ids: torch.Tensor,        # [B, C, Uv]
                cand_nmask: torch.Tensor,      # [B, C, Uv]
                cand_ages: torch.Tensor,       # [B, C, Uv]
                cand_count: torch.Tensor,      # [B, C]  Σ_w log1p(count_w) — count magnitude factor
                # ── shared table + additive channels ──
                e_weight: torch.Tensor,        # [N, d]
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

        # --- prediction μ_u, with COUNT MAGNITUDE FACTOR, broadcast to the [B,C] grid ---
        mu_u = self._mu_from_csr(eu, src_ids, src_nmask, src_ages, e_weight)   # [B,d]
        mu_u = mu_u * torch.exp(self.gamma_count * src_count).unsqueeze(-1)    # ·exp(γ·C_u)
        r_u = self._heading(mu_u)                                 # [B,d]
        eu_bc = eu.unsqueeze(1).expand(B, C, d)                   # [B,C,d]
        mu_u_bc = mu_u.unsqueeze(1).expand(B, C, d)
        r_u_bc = r_u.unsqueeze(1).expand(B, C, d)

        # --- identity: is v in u's region? (proven baseline) ---
        q_ident = self._identity(eu_bc, ev, mu_u_bc, r_u_bc, a, b, alpha)   # [B,C]

        # --- MEET: build μ_v in v's OWN tangent space, exp BOTH displacements back to the
        # sphere, and compare the two drifted positions. μ_u, μ_v live in different tangent
        # spaces (T_{E[u]} vs T_{E[v]}); the exp-map sends them to points on the SAME sphere so
        # ⟨q_u, q_v⟩ is transport-free. Rises when u and v drift toward a shared neighbourhood,
        # even if E[u], E[v] are structurally far apart — symmetric co-reachability. Age stays
        # t_query − t_edge (cand_ages already carry it). coef_meet init 0 ⇒ no-op at init.
        mu_v = self._mu_from_csr(ev, cand_ids, cand_nmask, cand_ages, e_weight)  # [B,C,d]
        mu_v = mu_v * torch.exp(self.gamma_count * cand_count).unsqueeze(-1)     # ·exp(γ·C_v)
        q_u = self._expmap(eu, mu_u)                              # [B,d]   drifted source pos
        q_v = self._expmap(ev, mu_v)                             # [B,C,d] drifted candidate pos
        meet = (q_u.unsqueeze(1) * q_v).sum(-1)                  # [B,C]   ⟨q_u, q_v⟩

        geo = (self.coef_identity * q_ident
               + self.coef_meet * meet)

        # --- candidate STALENESS channel ---
        staleness = self.staleness_head(self.basis_staleness(staleness_dt)).squeeze(-1)  # [B,C]
        logit = geo + self.coef_staleness * staleness

        # --- PAIR channel (FLAGGED) ---
        if self.use_pair_features:
            pair = self.pair_head(self.basis_pair(pair_dt)).squeeze(-1)
            logit = (logit + self.coef_pair * pair
                     + self.coef_pair_count * pair_count_log)

        return logit
