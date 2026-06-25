"""Geometric link head — REACH (one-sided drift), COUNT-FREE.

Builds a displacement μ_u for the source ONLY, from its walk tokens, and scores each candidate
by how close its STATIC embedding E[v] is to u's drifted position:

  μ_u = drift in T_{E[u]}                             (recency centroid of u's Log vectors)
  q_u = exp_{E[u]}(μ_u)                               u's drifted position, back ON the sphere
  μ_v, q_v = the SAME, built from the candidate's own walk bag (symmetric)
  reach(u,v) = ⟨q_u, E[v]⟩ + ⟨E[u], q_v⟩             does u reach v AND v reach u?

SYMMETRIC: both sides sample walks. q_u is u's neighbourhood drifted off E[u]; q_v is v's
neighbourhood drifted off E[v]. The first inner product asks "does v sit where u is heading?",
the second "does u sit where v is heading?". reach rises when each lands in the other's drift
region, even if E[u], E[v] are structurally far apart. coef_reach init 0 ⇒ the head starts as
the proven query-anchored baseline and reach earns its weight. Age stays t_query − t_edge
throughout (the trainer forms it; the head never re-defines it). The score is identity + reach:

  TOKEN BASIS — the source seed carries a COUNT-FREE bag of walk-reached positions p (one token
  per reached position; a node recurring k times is k tokens). μ sums the softmax over those
  tokens directly — multiplicity is implicit in token repetition, no explicit count. age_p =
  t_query − t_edge(p).

  μ_u = Σ_p softmax_p(−λ·age_p)·Log_{E[u]}(E[node_p ∈ u-tokens]) ; r_u = ĝate(μ_u)

  identity = −α·ellipse( Log_{E[u]}(E_v) − μ_u ; r_u )          is v in u's region?     [B,C]
  reach    = ⟨ exp_{E[u]}(μ_u), E_v ⟩ + ⟨ E_u, exp_{E[v]}(μ_v) ⟩   symmetric drift     [B,C]
  geo   = coef_identity·identity + coef_reach·reach
  logit = geo + coef_staleness·staleness(v) [+ coef_pair·pair + coef_pair_count·log1p(cnt)]

identity stays ASYMMETRIC (anchored at the source u); reach is SYMMETRIC — both u and v build a
μ from their own candidate/source walk bag. The candidate side now samples walks (cand_tokens).

BASELINE AT INIT — coef_identity=1, coef_reach=0 ⇒ at init the head IS the proven identity term
and nothing else; reach earns its weight from zero.

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

from .walk_tokens import WalkTokens


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
        # identity is the proven baseline (init 1); reach earns weight (init 0).
        self.coef_identity = nn.Parameter(torch.ones(1))
        # REACH — ⟨exp_{E[u]}(μ_u), E_v⟩: does u's one-sided drift reach v? coef init 0.
        self.coef_reach = nn.Parameter(torch.zeros(1))
        self.coef_staleness = nn.Parameter(torch.ones(1))
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
        (a cold node drifts nowhere). Final normalize is a numeric belt. Sends u's tangent
        displacement back ONTO the sphere so q_u = exp_{E[u]}(μ_u) lives on the SAME manifold as
        the candidate's unit E[v] and ⟨q_u, E[v]⟩ is a plain inner product."""
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
                e_weight: torch.Tensor,        # [N, d]  the whole node-embedding table
                src_tokens: WalkTokens,        # source walk tokens — self-contained: `seeds`
                                               # ARE the sources, `cutoffs` ARE the query times
                cand_tokens: WalkTokens,       # candidate walk tokens (Q = B*C rows). `seeds`
                                               # ARE the candidate ids (row-major over [B, C]);
                                               # `cutoffs` ARE the query times. No separate
                                               # cand_ids — it is just seeds reshaped to [B, C].
                staleness_dt: torch.Tensor,    # [B, C]
                pair_dt: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """-> logits [B, C]. The head owns all embedding lookups and timing: E_u, E_v and the
        token embeddings all come from `e_weight`. The candidate ids are recovered from the
        candidate bag (`cand_tokens.seeds` reshaped to [B, C]); B = #source seeds, C = (B*C)/B.
        Token ages come from each bag's cutoffs − pos_ts. The trainer hands over only the table,
        the source bag, the candidate bag, and the store-derived staleness / pair channels."""
        B = src_tokens.seeds.shape[0]
        C = cand_tokens.seeds.shape[0] // B
        d = self.d_emb
        cand_ids = cand_tokens.seeds.view(B, C)                   # [B, C]  recovered, not passed
        E_u = F.embedding(src_tokens.seeds, e_weight)             # [B, d]
        E_v = F.embedding(cand_ids, e_weight)                     # [B, C, d]
        eu = F.normalize(E_u, dim=-1)                             # [B, d]
        ev = F.normalize(E_v, dim=-1)                             # [B, C, d]
        a = F.softplus(self.log_a)
        b = F.softplus(self.log_b)
        alpha = self.alpha.clamp_min(1e-3)

        # --- prediction μ_u (dense per-row softmax over the source token bag) ---
        src_ages = (src_tokens.cutoffs.unsqueeze(-1)
                    - src_tokens.pos_ts).clamp_min(0).to(eu.dtype)   # [B, U]
        mu_u = self._mu_from_csr(
            E_u, src_tokens.node_ids, src_tokens.node_mask, src_ages, e_weight)  # [B,d]
        r_u = self._heading(mu_u)                                 # [B,d]
        eu_bc = eu.unsqueeze(1).expand(B, C, d)                   # [B,C,d]
        mu_u_bc = mu_u.unsqueeze(1).expand(B, C, d)
        r_u_bc = r_u.unsqueeze(1).expand(B, C, d)

        # --- identity: is v in u's region? (proven baseline) ---
        q_ident = self._identity(eu_bc, ev, mu_u_bc, r_u_bc, a, b, alpha)   # [B,C]

        # --- SYMMETRIC REACH: both sides drift their neighbourhood off their own embedding and
        # inner-product against the OTHER side's static embedding, so the score is symmetric in
        # (u, v):   reach(u,v) = ⟨q_u, E_v⟩ + ⟨E_u, q_v⟩
        # q_u = exp_{E[u]}(μ_u) drifts u's neighbourhood off E[u]; q_v = exp_{E[v]}(μ_v) drifts
        # each candidate's neighbourhood off E[v] (μ_v from the candidate walk bag). The first
        # term asks "does v sit where u is heading?", the second "does u sit where v is heading?".
        # coef_reach init 0 ⇒ no-op at init.
        q_u = self._expmap(eu, mu_u)                              # [B,d]   drifted source pos
        # candidate-side μ_v from the candidate token bag (Q = B*C rows; base frame = E[v]).
        E_v_flat = E_v.reshape(B * C, d)                          # [B*C,d]
        ev_flat = ev.reshape(B * C, d)                            # [B*C,d] unit base frame
        cand_ages = (cand_tokens.cutoffs.unsqueeze(-1)
                     - cand_tokens.pos_ts).clamp_min(0).to(eu.dtype)   # [B*C, U_v]
        mu_v = self._mu_from_csr(
            E_v_flat, cand_tokens.node_ids, cand_tokens.node_mask, cand_ages, e_weight)  # [B*C,d]
        q_v = self._expmap(ev_flat, mu_v).reshape(B, C, d)       # [B,C,d] drifted candidate pos
        reach = ((q_u.unsqueeze(1) * ev).sum(-1)                 # ⟨q_u, E_v⟩  [B,C]
                 + (eu.unsqueeze(1) * q_v).sum(-1))              # ⟨E_u, q_v⟩  [B,C]

        geo = (self.coef_identity * q_ident
               + self.coef_reach * reach)

        # --- candidate STALENESS channel ---
        staleness = self.staleness_head(self.basis_staleness(staleness_dt)).squeeze(-1)  # [B,C]
        logit = geo + self.coef_staleness * staleness

        # --- PAIR channel (FLAGGED) ---
        if self.use_pair_features:
            pair = self.pair_head(self.basis_pair(pair_dt)).squeeze(-1)
            logit = (logit + self.coef_pair * pair
                     + self.coef_pair_count * pair_count_log)

        return logit
