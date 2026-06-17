"""Link head — DualMu version (symmetric walk-mean, geodesic on the hypersphere).

Both endpoints build a recency-weighted PREDICTED NEIGHBOUR POSITION from their own
temporal walks, then the geometric channel scores the link by the geodesic between the
two predicted points:

  μ_u = Σ softmax(−λ·age)·Log_{E[u]}(E[w])   over u's walk-neighbours w   (tangent at E[u])
  P_u = Exp_{E[u]}(μ_u)                        a point on the sphere
  μ_v = Σ softmax(−λ·age)·Log_{E[v]}(E[w])   over v's walk-neighbours w   (tangent at E[v])
  P_v = Exp_{E[v]}(μ_v)
  geo = −α · arccos(⟨P_u, P_v⟩)               geodesic on the hypersphere

  logit = coef_geo·geo + coef_rec·rec(v) [+ coef_pair·pair(u,v) + coef_pair_count·log1p(count)]

The geometry is just TWO scalars (one shared recency rate λ for both μ's — the source and
candidate sides are symmetric — and one distance scale α): no ellipse, no heading frame,
no co-reachability channel. μ_u and μ_v ARE the geometric model. μ_u, μ_v live in DIFFERENT
tangent spaces (T_{E[u]} vs T_{E[v]}), so Exp-mapping each to the sphere first is what makes
the geodesic well-defined. A cold node with no walk-neighbours has μ ≈ 0 ⇒ P = E[node], so
geo degrades gracefully to −α·geodesic(E[u], E[v]) (the plain embedding distance).

Additive non-geometric channels (carried over from the proven point head, each with its own
learnable per-channel coefficient so the model can rebalance):
  - rec  : candidate v's own staleness, ExpDecayBasis recency of Δt_v = t_query − t_last[v].
  - pair : (FLAGGED, --use-pair-features) exact (u,v) recurrence — ExpDecayBasis recency of
           the last (u,v) interaction Δt_uv, plus log1p(interaction count) on its own coef
           (init 0 ⇒ off, earns its weight). The proven recurrence win; a no-grad-to-E
           additive logit bias that sharpens scores without touching the sphere geometry.

TIME UNITS — ages are RAW (t_query − t_edge), never normalized at runtime. The μ softmax
is scale-invariant in the LIMIT, but its INIT temperature is not: λ is initialised ≈ 1/t_train
so λ·(typical raw age) ~ O(1) and λ actually trains — at the naive λ≈O(1) init the softmax
saturates to a hard argmax and ∂loss/∂λ → 0 (λ pinned, μ frozen as "the single most-recent
neighbour"). The rec/pair ExpDecayBasis is scale-free on raw Δt by construction (log-spaced
rates from 1/t_train to 1; a never-seen event Δt→∞ maps to φ=0 for free). t_train sets these
inits only — never a per-step scaler.

E stays the single sphere parameter (link-trained, no detach).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpDecayBasis(nn.Module):
    """Multi-rate exponential-decay staleness encoder: φ(Δt) = [exp(−ρ_k·Δt)]_{k=1..K}.
    ρ_k = exp(log_rates_k) > 0 learnable, log-spaced init from 1/t_train (train-span
    scale) to 1 (most-recent scale). The Hawkes/TPP recency feature — scale-free on RAW
    Δt, bounded [0,1], monotone, multi-timescale — feeding the rec / pair channels. A
    never-seen event (Δt → +inf, encoded as a huge value) maps to φ = 0 for free."""

    def __init__(self, dim: int, t_train: float):
        super().__init__()
        r_lo = -math.log(max(float(t_train), 1.0))    # ρ ≈ 1/t_train
        r_hi = 0.0                                    # ρ ≈ 1
        self.log_rates = nn.Parameter(torch.linspace(r_lo, r_hi, dim))   # [K]

    def forward(self, dt: torch.Tensor) -> torch.Tensor:   # dt [...]  raw Δt ≥ 0
        return torch.exp(-torch.exp(self.log_rates) * dt.unsqueeze(-1))  # [..., K]


class DualMuHead(nn.Module):
    """Symmetric walk-mean head (see module docstring). Geometric channel = geodesic
    between the two predicted points P_u, P_v; plus the rec channel (v's staleness) and
    an optional flagged pair channel ((u,v) recurrence + count). Per-channel learnable
    coefficients (init 1, count init 0) let the model rebalance the terms."""

    def __init__(self, d_emb: int, d_time: int = 16,
                 use_pair_features: bool = False, t_train: float = 1.0):
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6

        # --- geometric channel: μ_u ↔ μ_v geodesic --------------------------------
        # λ = softplus(log_lambda) ≥ 0, shared by both μ's (source/candidate symmetric),
        # on RAW age inside the μ softmax. The softmax is scale-invariant in the LIMIT,
        # but its INIT temperature is NOT: with λ·age ≫ 1 the softmax saturates to a hard
        # argmax and ∂softmax/∂λ → 0, so λ stays pinned and never trains. Init λ ≈ 1/t_train
        # so λ·(typical raw age) ~ O(1) — unsaturated, so λ actually adapts. (raw age kept
        # everywhere; this only fixes the init temperature.) softplus⁻¹(x)=log(expm1(x)).
        lam0 = 1.0 / max(float(t_train), 1.0)
        self.log_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(10.0))     # geodesic → logit scale

        # --- rec channel: candidate v's own staleness -----------------------------
        self.basis_rec = ExpDecayBasis(d_time, t_train)
        self.rec_head = nn.Linear(d_time, 1)

        # --- pair channel (FLAGGED): exact (u,v) recurrence -----------------------
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.basis_pair = ExpDecayBasis(d_time, t_train)   # recency φ(Δt_uv)
            self.pair_head = nn.Linear(d_time, 1)

        # --- learnable per-channel mix coefficients (init 1 = plain sum) ----------
        self.coef_geo = nn.Parameter(torch.ones(1))
        self.coef_rec = nn.Parameter(torch.ones(1))
        if use_pair_features:
            self.coef_pair = nn.Parameter(torch.ones(1))
            # (u,v) interaction-COUNT term: coef_pair_count · log1p(count). Init 0 ⇒ off,
            # earns its weight. Never-seen pair → count=0 → log1p=0 → no contribution.
            self.coef_pair_count = nn.Parameter(torch.zeros(1))

    # ──────────────────────────────────────────────────────────────────
    # Sphere geometry helpers
    # ──────────────────────────────────────────────────────────────────

    def _logmap(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Sphere log-map at base point p (closed form = geoopt.Sphere().logmap)."""
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _expmap(self, p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sphere exp-map at p: Exp_p(v) = cos‖v‖·p + sin‖v‖·v/‖v‖. v=0 ⇒ p (graceful)."""
        vn = v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.cos(vn) * p + torch.sin(vn) * (v / vn)

    def _mu(self, base: torch.Tensor, ew: torch.Tensor,
            age: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Recency-weighted mean of Log_base(neighbour), in the tangent space at base.
           base [..., d] unit ; ew [..., M, d] unit ; age [..., M] ; mask [..., M] bool
           -> μ [..., d] (tangent at base). Masked / cold ⇒ μ = 0."""
        g = self._logmap(base.unsqueeze(-2), ew)                       # [..., M, d]
        lam = F.softplus(self.log_lambda)
        wlog = (-lam * age).masked_fill(~mask, float("-inf"))          # [..., M]
        w = torch.nan_to_num(torch.softmax(wlog, dim=-1), nan=0.0)     # cold ⇒ all 0
        return (w.unsqueeze(-1) * g).sum(dim=-2)                       # [..., d]

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(self, tok_emb: torch.Tensor, tok_age: torch.Tensor,
                tok_mask: torch.Tensor, E_u: torch.Tensor, E_v: torch.Tensor,
                rec_v_dt: torch.Tensor,
                cand_ids: torch.Tensor, cand_age: torch.Tensor,
                cand_mask: torch.Tensor, e_weight: torch.Tensor,
                pair_dt: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """Source side (μ_u): tok_emb [B,n,d], tok_age [B,n], tok_mask [B,n], E_u [B,d].
           Candidate side (μ_v): cand_ids/cand_age/cand_mask [B,C,M] (v's walk-neighbour
           node ids gathered from e_weight [N,d]), E_v [B,C,d].
           rec_v_dt [B,C] RAW Δt_v candidate staleness (→ basis_rec).
           pair_dt [B,C] RAW Δt_uv, pair_count_log [B,C] log1p(count) — when pair on.
           -> logits [B, C]."""
        eu = F.normalize(E_u, dim=-1)                                  # [B, d]
        ev = F.normalize(E_v, dim=-1)                                  # [B, C, d]

        # --- geometric channel: P_u (u's walks) vs P_v (v's walks) ----------------
        ew_u = F.normalize(tok_emb, dim=-1)                            # [B, n, d]
        mu_u = self._mu(eu, ew_u, tok_age, tok_mask)                   # [B, d]
        P_u = self._expmap(eu, mu_u)                                   # [B, d]

        ew_v = F.normalize(F.embedding(cand_ids.clamp_min(0), e_weight), dim=-1)  # [B,C,M,d]
        mu_v = self._mu(ev, ew_v, cand_age, cand_mask)                 # [B, C, d]
        P_v = self._expmap(ev, mu_v)                                   # [B, C, d]

        cos = (P_u.unsqueeze(1) * P_v).sum(-1).clamp(-1 + self.eps, 1 - self.eps)  # [B,C]
        geo = -self.alpha.clamp_min(1e-3) * torch.arccos(cos)         # [B, C] = −α·geodesic

        # --- channels combined with learnable per-channel coefficients ------------
        rec = self.rec_head(self.basis_rec(rec_v_dt)).squeeze(-1)     # [B, C]
        logit = self.coef_geo * geo + self.coef_rec * rec
        if self.use_pair_features:
            pair = self.pair_head(self.basis_pair(pair_dt)).squeeze(-1)  # [B, C]
            logit = (logit + self.coef_pair * pair
                     + self.coef_pair_count * pair_count_log)
        return logit
