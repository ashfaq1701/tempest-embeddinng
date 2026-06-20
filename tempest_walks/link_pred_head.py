"""Geometric link head — CSR point version (count center-of-mass μ + anisotropic ellipse)
+ co-reachability (∃-witness) channel, count-aware, over a deduplicated CSR.

Both the source and candidate walk-neighbourhoods arrive as a deduplicated CSR from the
trainer (one routine prepares both, symmetrically):
    node_ids  [.,U]        distinct neighbour node ids (−1 padded)
    node_mask [.,U]        valid distinct node
    ages      [.,U,kmax]   each node's OCCURRENCE ages (raw; all kept ⇒ recency exact)
    age_mask  [.,U,kmax]   valid occurrence (count = age_mask.sum(-1))

  base point   p = E[u]
  μ (source CSR, COUNT CENTER OF MASS):
      g_w = Log_{E[u]}(E[node_w])                         distinct-node tangent vectors
      ℓ_w = logsumexp_{i∈occ(w)}(−λ·age_i) + γμ·log(1+k_w)
      w_w = softmax_w(ℓ_w) ;  μ = Σ_w w_w·g_w            weighted Karcher mean (center of mass)
  heading      r   = μ/‖μ‖, GATED by ‖μ‖ (cold μ ⇒ isotropic)
  identity (candidate vs μ):
      ν   = Log_{E[u]}(E[v]) ;  d = √(a‖δ∥‖² + b‖δ⊥‖²), δ=ν−μ ;  identity = −α·d
  co-reach (candidate CSR ∃-witness, count-aware):
      c_w = Log_{E[u]}(E[conn_w]) ; d_w = ellipse(c_w−μ)
      coreach = logsumexp_w( −α·d_w + γc·log(1+k_w) + logsumexp_{i∈occ(w)}(−ρ·age_i) )
  geo   = coef_identity·identity + coef_coreach·coreach
  logit = geo + coef_staleness·staleness(v) [+ coef_pair·pair(u,v) + coef_pair_count·log1p(cnt)]

μ AS A CENTER OF MASS — μ is the first-order weighted Fréchet/Karcher mean of u's neighbours;
each distinct node's MASS w_w is its recency-discounted, count-emphasized occurrence weight.
High count → more mass → centroid pulled toward that node (closer); high age → less mass →
pulled less (away). The recency combine is sum-OUTSIDE-exp (logsumexp over a node's ages),
NOT exp-of-sum (which would make count push away). γμ=0 recovers the proven recency μ EXACTLY
(softmax over nodes of the per-node recency mass == flat slot-softmax μ); γμ>0 adds an explicit
RELATIVE count emphasis (which node μ points toward). ABSOLUTE count cannot live in μ (the
softmax normalizes it away) — it lives in the co-reach witness below.

COUNT, TWO PLACES, TWO ROLES:
  • μ: γμ·log(1+k) INSIDE the softmax ⇒ RELATIVE count (biases the prediction DIRECTION).
  • co-reach: γc·log(1+k) inside the (unnormalized) logsumexp ⇒ ABSOLUTE count (connection
    STRENGTH). Applied once per node via the logsumexp-shift identity. Together with the
    witness recency (−ρ·age) and existence (it fires), the self-witness case (v itself a
    recent, frequent connector) reproduces pair_ever / pair_rec / pair_count geometrically —
    the in-geometry replacement aiming to make the flagged pair channel droppable.

TIME UNITS — ages are RAW (t_query − t_edge), precomputed by the trainer.
  • μ recency is a SOFTMAX (scale-invariant) → λ self-scales; γμ=0 = proven baseline.
  • co-reach recency is a LOGSUMEXP (NOT scale-invariant) → own ρ = exp(log_rate_coreach),
    init −log(t_train) ⇒ ρ·age ~ O(1) at init (bounded). The count bonus is log of a small
    integer ⇒ always bounded.

E stays the single sphere parameter (link-trained, no detach).
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

        # μ recency λ (softmax, scale-invariant). Init λ ≈ C/t_train (C=10) so λ·age~O(1)
        # and λ trains (zero-init saturates the softmax → argmax → λ pinned).
        lam0 = 10.0 / max(float(t_train), 1.0)
        self.log_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(10.0))     # distance weight
        # Intrinsic-frame anisotropy (a,b ≥ 0): ellipse d² = a‖δ∥‖² + b‖δ⊥‖² along heading r.
        # a=b ⇒ isotropic ‖ν−μ‖. +0.0025 test wiki / +0.0095 test review (2-seed).
        self.log_a = nn.Parameter(torch.zeros(1))         # radial (along-heading)
        self.log_b = nn.Parameter(torch.zeros(1))         # tangential (off-heading)
        # μ COUNT emphasis (relative): γμ·log(1+k) inside the μ softmax. init 0 ⇒ count off,
        # recovers the proven recency μ exactly; earns its weight.
        self.gamma_mu = nn.Parameter(torch.zeros(1))

        # --- candidate staleness channel ---------------------------------------
        self.basis_staleness = ExpDecayBasis(d_time, t_train)
        self.staleness_head = nn.Linear(d_time, 1)

        # --- pair channel (FLAGGED) — to be removed once co-reach count absorbs it ---
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.basis_pair = ExpDecayBasis(d_time, t_train)
            self.pair_head = nn.Linear(d_time, 1)

        # --- learnable per-channel mix coefficients ----------------------------
        self.coef_identity = nn.Parameter(torch.ones(1))     # 1st-order candidate fit
        self.coef_staleness = nn.Parameter(torch.ones(1))    # candidate staleness
        if use_pair_features:
            self.coef_pair = nn.Parameter(torch.ones(1))
            self.coef_pair_count = nn.Parameter(torch.zeros(1))

        # --- co-reachability (∃-witness) channel -------------------------------
        # coef_coreach     : channel gain, init 0 ⇒ starts as the proven baseline.
        # log_rate_coreach : connector-recency ρ = exp(·), init −log(t_train) ⇒ ρ·age~O(1).
        # gamma_coreach    : ABSOLUTE in-geometry count, +γc·log(1+k) inside the witness; init 0.
        self.coef_coreach = nn.Parameter(torch.zeros(1))
        self.log_rate_coreach = nn.Parameter(
            torch.tensor([-math.log(max(float(t_train), 1.0))], dtype=torch.float32))
        self.gamma_coreach = nn.Parameter(torch.zeros(1))

    # ──────────────────────────────────────────────────────────────────
    # Geometry primitives
    # ──────────────────────────────────────────────────────────────────

    def _logmap(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Sphere log-map at base point p (closed form = geoopt.Sphere().logmap)."""
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _ellipse_dist_sq(self, delta: torch.Tensor, r: torch.Tensor,
                         a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Anisotropic ellipse distance² in the per-source heading frame.
           delta [B, ..., d] ; r [B, d] heading (broadcast over axes after B) -> [B, ...]."""
        extra = delta.dim() - 2
        r_b = r.view(r.shape[0], *([1] * extra), r.shape[-1])
        dpar2 = (delta * r_b).sum(-1).pow(2)
        dperp2 = ((delta * delta).sum(-1) - dpar2).clamp_min(0.0)
        return a * dpar2 + b * dperp2

    # ──────────────────────────────────────────────────────────────────
    # μ — count center of mass over the source CSR
    # ──────────────────────────────────────────────────────────────────

    def _mu_from_csr(self, eu: torch.Tensor, ids: torch.Tensor, nmask: torch.Tensor,
                     ages: torch.Tensor, amask: torch.Tensor,
                     e_weight: torch.Tensor) -> torch.Tensor:
        """μ = Σ_w softmax_w(logsumexp_occ(−λ·age) + γμ·log(1+k))·Log_{E[u]}(E[node_w]).
           Weighted Karcher mean (center of mass); γμ=0 = proven recency μ. Cold ⇒ μ=0.
           eu [B,d] ; ids/nmask [B,U] ; ages/amask [B,U,kmax]  ->  μ [B,d]."""
        ew = F.normalize(F.embedding(ids.clamp_min(0), e_weight), dim=-1)   # [B,U,d]
        g = self._logmap(eu.unsqueeze(1), ew)                              # [B,U,d]
        lam = F.softplus(self.log_lambda)
        # per-node recency mass (log) = logsumexp over the node's occurrences of −λ·age
        occ = (-lam * ages).masked_fill(~amask, float("-inf"))            # [B,U,kmax]
        R = torch.logsumexp(occ, dim=-1)                                  # [B,U]
        k = amask.sum(-1).to(ages.dtype)                                  # [B,U] count
        ell = (R + self.gamma_mu * torch.log1p(k)).masked_fill(~nmask, float("-inf"))
        w = torch.nan_to_num(torch.softmax(ell, dim=-1), nan=0.0)         # [B,U] cold→0
        return (w.unsqueeze(-1) * g).sum(dim=1)                           # [B,d]

    # ──────────────────────────────────────────────────────────────────
    # co-reach — count-aware ∃-witness over the candidate CSR
    # ──────────────────────────────────────────────────────────────────

    def _coreach_from_csr(self, eu: torch.Tensor, mu: torch.Tensor, r: torch.Tensor,
                          a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor,
                          ids: torch.Tensor, nmask: torch.Tensor,
                          ages: torch.Tensor, amask: torch.Tensor,
                          e_weight: torch.Tensor) -> torch.Tensor:
        """coreach = logsumexp_w( −α·d_w + γc·log(1+k_w) + logsumexp_occ(−ρ·age) ),
           d_w = ellipse(Log_{E[u]}(E[conn_w]) − μ). One log-map per distinct connector
           (U ≤ flat M); ages enter only the cheap per-node recency reduction (no d axis).
           Bit-exact to the flat per-occurrence witness via the logsumexp-shift identity.
           ids/nmask [B,C,U] ; ages/amask [B,C,U,kmax]  ->  [B,C]. Neutral 0 if empty."""
        ec = F.normalize(F.embedding(ids.clamp_min(0), e_weight), dim=-1)   # [B,C,U,d]
        gc = self._logmap(eu[:, None, None, :], ec)                        # [B,C,U,d]
        d = self._ellipse_dist_sq(gc - mu[:, None, None, :], r, a, b).clamp_min(self.eps).sqrt()  # [B,C,U]
        rho = torch.exp(self.log_rate_coreach)
        A = torch.logsumexp((-rho * ages).masked_fill(~amask, -1e9), dim=-1)   # [B,C,U] combined recency
        k = amask.sum(-1).to(ages.dtype)                                   # [B,C,U] count
        lw = (-alpha * d + self.gamma_coreach * torch.log1p(k) + A).masked_fill(~nmask, -1e9)
        coreach = torch.logsumexp(lw, dim=-1)                             # [B,C]
        return torch.where(nmask.any(dim=-1), coreach, torch.zeros_like(coreach))

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(self,
                # ── source CSR (μ side) ──
                E_u: torch.Tensor,             # [B, d]      source base point
                src_ids: torch.Tensor,         # [B, Us]     distinct source-walk nodes
                src_nmask: torch.Tensor,       # [B, Us]
                src_ages: torch.Tensor,        # [B, Us, ks] per-node occurrence ages
                src_amask: torch.Tensor,       # [B, Us, ks]
                # ── candidate CSR (identity + connectors side) ──
                E_v: torch.Tensor,             # [B, C, d]   candidate identity
                cand_ids: torch.Tensor,        # [B, C, Uv]  distinct candidate-walk connectors
                cand_nmask: torch.Tensor,      # [B, C, Uv]
                cand_ages: torch.Tensor,       # [B, C, Uv, kv]
                cand_amask: torch.Tensor,      # [B, C, Uv, kv]
                # ── shared table + additive channels ──
                e_weight: torch.Tensor,        # [N, d]      embedding table (head gathers)
                t_query_t: torch.Tensor,       # [B]         reserved (ages are precomputed)
                staleness_dt: torch.Tensor,    # [B, C]      raw Δt_v
                pair_dt: torch.Tensor = None,        # [B, C]  (flagged)
                pair_count_log: torch.Tensor = None  # [B, C]  (flagged)
                ) -> torch.Tensor:
        """-> logits [B, C].  IDs in; embeddings gathered here for both CSR sides."""
        eu = F.normalize(E_u, dim=-1)                              # [B, d]
        ev = F.normalize(E_v, dim=-1)                              # [B, C, d]

        # --- μ: count center of mass over the source CSR -----------------------
        mu = self._mu_from_csr(eu, src_ids, src_nmask, src_ages, src_amask, e_weight)  # [B,d]

        # --- heading r, GATED for cold μ (vanishes as ‖μ‖→0 ⇒ isotropic geodesic) ----
        mu_norm = mu.norm(dim=-1, keepdim=True)                    # [B,1]
        m0 = 0.05
        mu_gate = (mu_norm * mu_norm) / (mu_norm * mu_norm + m0 * m0)
        r = mu_gate * mu / mu_norm.clamp_min(self.eps)            # [B,d]
        a = F.softplus(self.log_a)
        b = F.softplus(self.log_b)
        alpha = self.alpha.clamp_min(1e-3)

        # --- identity: candidate v vs μ (ellipse) ------------------------------
        nu = self._logmap(eu.unsqueeze(1), ev)                    # [B,C,d]
        d = self._ellipse_dist_sq(nu - mu.unsqueeze(1), r, a, b).clamp_min(self.eps).sqrt()  # [B,C]
        identity = -alpha * d                                     # [B,C]

        # --- co-reach: candidate CSR connectors vs μ (count-aware ∃-witness) ----
        coreach = self._coreach_from_csr(eu, mu, r, a, b, alpha,
                                         cand_ids, cand_nmask, cand_ages, cand_amask,
                                         e_weight)                # [B,C]
        geo = self.coef_identity * identity + self.coef_coreach * coreach

        # --- candidate STALENESS channel ---------------------------------------
        staleness = self.staleness_head(self.basis_staleness(staleness_dt)).squeeze(-1)  # [B,C]
        logit = geo + self.coef_staleness * staleness

        # --- PAIR channel (FLAGGED) — removable once count absorbs it -----------
        if self.use_pair_features:
            pair = self.pair_head(self.basis_pair(pair_dt)).squeeze(-1)  # [B,C]
            logit = (logit + self.coef_pair * pair
                     + self.coef_pair_count * pair_count_log)

        return logit
