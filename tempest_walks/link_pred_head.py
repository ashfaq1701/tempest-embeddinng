"""Geometric link head — REACH (one-sided drift), vMF NEIGHBOURHOOD SUMMARY.

The source's walk tokens are summarised by the WEIGHTED SPHERICAL RESULTANT — the von
Mises–Fisher (vMF) summary of the neighbourhood — which keeps the COUNT and AGE that the old
softmax centroid normalized away:

  R_u       = Σ_p w_p · Ê[node_p] ,  w_p = exp(−λ·age_p)        (ambient, UN-normalized)
  center_u  = R_u / ‖R_u‖                                        representative point ON the sphere
  mass_u    = Σ_p w_p                                            recency-weighted COUNT (age+count)
  coh_u     = ‖R_u‖ / mass_u ∈ [0,1]                             COHERENCE (1 = agree, 0 = dispersed)

One arrow R_u carries everything. The DIRECTION (center_u) is the consensus point of the
neighbourhood — age and relative multiplicity bend it (recent / repeated neighbours pull
harder). The LENGTH splits into mass_u (how many recent neighbours — the absolute count×recency
softmax destroyed by dividing by Σw) and coh_u (how tightly they agree). A single sphere point
is a pure direction and cannot store an absolute count, so the count lives in the companion
scalar mass_u — that is the correct factorisation, not a workaround: direction = WHERE,
concentration = HOW SURE.

  μ_u   = Log_{E[u]}(center_u)            tangent drift to the centre (feeds identity)
  r_u   = ĝate(coh_u) · μ̂_u              heading, gated by COHERENCE (not ‖μ‖)
  q_u   = center_u                        u's drifted position is the centre itself
  reach(u,v) = κ(mass_u, coh_u) · ⟨ center_u , E_v ⟩    sharpened by count + coherence

The token bag now KEEPS self-recurrences (u re-entering its own neighbourhood) — in this
ambient-resultant construction a self-token contributes w_self·Ê[u] (a real unit vector), not
the zero tangent vector the old softmax-Log centroid produced, so it is legitimate evidence of
u's self-activity that bends the centre toward E[u] and feeds mass/coherence.

The score is identity + reach:

  identity = −α·ellipse( Log_{E[u]}(E_v) − μ_u ; r_u )          is v in u's region?     [B,C]
  reach    = κ_u · ⟨ center_u , E_v ⟩                          does v match u's centre, [B,C]
                                                                how confidently?
  geo   = coef_identity·identity + coef_reach·reach
  logit = geo + coef_staleness·staleness(v) [+ coef_pair·pair + coef_pair_count·log1p(cnt)]

κ(mass, coh) = softplus(W·[log1p(mass), coh] + b) is the vMF CONCENTRATION: many recent agreeing
neighbours ⇒ sharp, trustworthy centre ⇒ strong reach; few / old / scattered ⇒ flat ⇒ weak. This
is the native way a spherical centre expresses how much evidence backs it — and the place AGE and
COUNT do real work in the score.

SPHERE-VALID — center_u is a unit vector by construction (normalize never leaves the sphere), so
μ_u = Log_{E[u]}(center_u) has ‖μ‖ = angle(E[u], center_u) ≤ π (no exp-map wrap, no clamp).

BASELINE AT INIT — coef_identity=1, coef_reach=0 ⇒ at init the head is the identity term over the
vMF centre; reach (where the absolute count enters via κ) earns its weight from zero. Age and
RELATIVE count already act at init through the centre direction.

WHY THIS CENTRE — center_u is the maximum-likelihood mean direction of a vMF distribution: the
textbook "average of points on a sphere". The softmax tangent centroid only approximated it (a
tangent-space average) and, by summing weights to 1, discarded mass_u and coh_u entirely.

Only the source builds the summary — the candidate enters via its static E[v] (identity + reach)
and the staleness / pair channels. E stays the single sphere parameter (link-trained, no detach).
Ages are raw t_query − t_edge (cutoffs − pos_ts).
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

        # Recency λ for the resultant weights w_p = exp(−λ·age_p). init λ ≈ 10/t_train.
        lam0 = 10.0 / max(float(t_train), 1.0)
        self.log_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))
        # Adaptive sharpening strength γ = softplus(log_gamma). The CENTER direction uses
        # w_p^{1+γ·r̄} (r̄ = detached unsharpened coherence), so a coherent bag sharpens toward
        # its most-recent member (pointer, good for recurrence) while a dispersed bag stays a
        # democratic summary (good for cold-start). init γ≈0.1 (near-democratic; at γ=0 this is
        # byte-equivalent to the plain vMF). mass/coh stay on the UNSHARPENED weights.
        self.log_gamma = nn.Parameter(
            torch.tensor([math.log(math.expm1(0.1))], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(10.0))     # shared distance weight
        self.log_a = nn.Parameter(torch.zeros(1))         # anisotropic ellipse (a,b ≥ 0)
        self.log_b = nn.Parameter(torch.zeros(1))

        # vMF concentration κ(mass, coh) = softplus(W·[log1p(mass), coh] + b).
        # init: W=0, b→softplus≈1 ⇒ reach starts as plain ⟨center, E_v⟩; count/coherence
        # sharpening earns its weight.
        self.kappa_head = nn.Linear(2, 1)
        nn.init.zeros_(self.kappa_head.weight)
        nn.init.constant_(self.kappa_head.bias, math.log(math.expm1(1.0)))

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

    def _ellipse_dist_sq(self, delta: torch.Tensor, r: torch.Tensor,
                         a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Ellipse distance² a‖δ∥‖²+b‖δ⊥‖² in the heading frame r. delta [...,d];
           r broadcastable to delta (caller unsqueezes the U axis where needed) -> [...]."""
        dpar2 = (delta * r).sum(-1).pow(2)
        dperp2 = ((delta * delta).sum(-1) - dpar2).clamp_min(0.0)
        return a * dpar2 + b * dperp2

    def _heading(self, mu: torch.Tensor, coh: torch.Tensor) -> torch.Tensor:
        """Gated heading r = g(coh)·μ̂, g = coh²/(coh²+m0²) → 0 dispersed (isotropic), 1 coherent.
        Gated by neighbourhood COHERENCE — under the resultant centre ‖μ‖ is distance-to-centre,
        not agreement, so coh (the mean resultant length) is the right anisotropy signal."""
        c = coh.unsqueeze(-1)
        m0 = 0.3
        gate = (c * c) / (c * c + m0 * m0)
        return gate * mu / mu.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    # ──────────────────────────────────────────────────────────────────
    # Neighbourhood summary — weighted spherical resultant (vMF MLE)
    # ──────────────────────────────────────────────────────────────────

    def _directional_mean(self, ids: torch.Tensor, nmask: torch.Tensor,
                          ages: torch.Tensor, e_weight: torch.Tensor):
        """Weighted spherical resultant of the token bag → (center, mass, coherence).
           Honest summary stats (mass, coh) come from the UNSHARPENED weights w_p = exp(−λ·age_p);
           the DIRECTION is sharpened by w_p^{1+γ·r̄} (r̄ = detached unsharpened coherence), so a
           coherent bag concentrates toward its most-recent member (pointer) while a dispersed bag
           stays democratic (summary).
             center    [...,d]  = normalize(Σ w_p^{1+γ·r̄}·Ê[node_p])   sharpened mean direction
             mass      [...]     = Σ_p w_p                            recency-weighted COUNT
             coherence [...]     = ‖Σ w_p Ê‖/mass ∈[0,1]              unsharpened peakedness r̄
           ids/nmask/ages [...,U]. Cold bag (all-pad) ⇒ center ≈ 0, mass = 0, coherence = 0.
           At γ=0 this is byte-equivalent to the plain vMF (w^1, center = R/‖R‖)."""
        ew = F.normalize(F.embedding(ids.clamp_min(0), e_weight), dim=-1)   # [...,U,d]
        lam = F.softplus(self.log_lambda)
        logw = -lam * ages + torch.where(nmask, 0.0, -1e9)                 # [...,U] log-w, pad→−inf

        # --- honest summary stats from the UNSHARPENED weights (exponent 1) ---
        w = torch.exp(logw)                                              # [...,U] (0 on pad)
        R0 = (w.unsqueeze(-1) * ew).sum(dim=-2)                          # [...,d]
        mass = w.sum(dim=-1)                                             # [...]
        r_bar = R0.norm(dim=-1) / mass.clamp_min(self.eps)              # [...] coherence/peakedness
        coherence = r_bar

        # --- sharpen ONLY the direction, by the DETACHED peakedness (no feedback loop) ---
        gamma = F.softplus(self.log_gamma)
        expo = (1.0 + gamma * r_bar.detach()).unsqueeze(-1)            # [...,1]
        ws = torch.exp(logw * expo)                                    # [...,U] w_p^{1+γ·r̄}
        Rs = (ws.unsqueeze(-1) * ew).sum(dim=-2)                       # [...,d]
        center = Rs / Rs.norm(dim=-1, keepdim=True).clamp_min(self.eps)  # [...,d] sphere-safe
        return center, mass, coherence

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
                cand_ids: torch.Tensor,        # [B, C]  candidate node ids
                staleness_dt: torch.Tensor,    # [B, C]
                pair_dt: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """-> logits [B, C]. E_u = e_weight[src_tokens.seeds], E_v = e_weight[cand_ids];
        token ages = src_tokens.cutoffs − src_tokens.pos_ts."""
        B, C = cand_ids.shape[0], cand_ids.shape[1]
        d = self.d_emb
        E_u = F.embedding(src_tokens.seeds, e_weight)             # [B, d]
        E_v = F.embedding(cand_ids, e_weight)                     # [B, C, d]
        eu = F.normalize(E_u, dim=-1)                             # [B, d]
        ev = F.normalize(E_v, dim=-1)                             # [B, C, d]
        a = F.softplus(self.log_a)
        b = F.softplus(self.log_b)
        alpha = self.alpha.clamp_min(1e-3)

        # --- neighbourhood summary: vMF resultant (centre + count/age mass + coherence) ---
        src_ages = (src_tokens.cutoffs.unsqueeze(-1)
                    - src_tokens.pos_ts).clamp_min(0).to(eu.dtype)   # [B, U]
        center, mass, coh = self._directional_mean(
            src_tokens.node_ids, src_tokens.node_mask, src_ages, e_weight)   # [B,d],[B],[B]

        mu_u = self._logmap(eu, center)                          # [B,d] drift to the centre
        r_u = self._heading(mu_u, coh)                           # [B,d] coherence-gated heading
        eu_bc = eu.unsqueeze(1).expand(B, C, d)                  # [B,C,d]
        mu_u_bc = mu_u.unsqueeze(1).expand(B, C, d)
        r_u_bc = r_u.unsqueeze(1).expand(B, C, d)

        # --- identity: is v in u's region? (proven baseline) ---
        q_ident = self._identity(eu_bc, ev, mu_u_bc, r_u_bc, a, b, alpha)   # [B,C]

        # --- reach: κ(mass, coh)·⟨centre, E_v⟩ — count & age set the centre's confidence ---
        kappa = F.softplus(self.kappa_head(
            torch.stack([torch.log1p(mass), coh], dim=-1))).squeeze(-1)     # [B]
        reach = kappa.unsqueeze(-1) * (center.unsqueeze(1) * ev).sum(-1)    # [B,C]

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
