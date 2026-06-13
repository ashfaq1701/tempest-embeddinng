"""Geometric link head — soft / probabilistic (tangent-space Gaussian + Mahalanobis).

The robust member of the family. Same crafted idea as the point version
(base point E[u]; u's temporal-walk neighbours log-mapped into the flat tangent
space T_{E[u]}; a recency-weighted prediction of "where v should belong"), but
instead of a single point μ scored by separate distance + angle, we fit a
recency-weighted GAUSSIAN N(μ, C) over the neighbour tangents and score the
candidate by its MAHALANOBIS distance to it:

  base point   p = E[u]
  neighbours   g_i = Log_{E[u]}(E[node_i])          tangent vectors at E[u]
  candidate    ν   = Log_{E[u]}(E[v])
  recency      w_i = softmax_i(−λ·age_i)            (λ ≥ 0 learnable; Σ w_i = 1)
  mean         μ   = Σ_i w_i g_i                    predicted position
  covariance   C   = Σ_i w_i (g_i−μ)(g_i−μ)ᵀ        recency-weighted spread
  score        m²  = (ν−μ)ᵀ (C + τI)⁻¹ (ν−μ)        Mahalanobis (τ = shrinkage ≥ 0)
  logit        = −scale · √m²

Why Mahalanobis (vs the point head's −α‖ν−μ‖ − β·angle):
  - It is direction-aware distance: deviations along axes where u's neighbourhood
    is SPREAD are cheap, deviations along axes where it is TIGHT are expensive. So
    it fuses "distance" and "angle" into one quantity rather than two scalars.
  - A near-degenerate "narrow line" neighbourhood is just a high-variance axis +
    tight perpendicular axes — the ellipse handles it smoothly, with NO unstable
    eigenvector/line fit. That is the whole reason to prefer this variant.

The shrinkage τ is also the dial back to the simpler heads:
  τ → ∞  ⇒  (C+τI)⁻¹ → (1/τ)I  ⇒  m² → ‖ν−μ‖²/τ  ⇒  the plain mean-distance head.
  τ → 0  ⇒  full Mahalanobis (most direction-aware, least regularised).

FAST: C is rank ≤ n (n = #neighbours ≪ d), so we NEVER form or invert the [d,d]
covariance. By Woodbury, (C+τI)⁻¹ needs only an [n,n] inverse:
  m² = (1/τ)[ ‖δ‖² − (Aδ)ᵀ G (Aδ) ],   δ = ν−μ,  A_i = √w_i (g_i−μ),  G = (τI_n + AAᵀ)⁻¹
G and μ, A are per-source (candidate-independent); only the [n]-vectors Aδ are
per-candidate. The [d,d] object is never built.

E stays the single sphere parameter (link-trained, no detach). λ, τ, scale are the
only geometric parameters (plus the optional proven recency/pair terms).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    """Time2Vec (Kazemi et al. 2019): scalar τ -> [linear, sin(ω₁τ+φ₁), …]."""

    def __init__(self, dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(dim - 1))
        self.b = nn.Parameter(torch.rand(dim - 1) * 2 * math.pi)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.unsqueeze(-1)
        lin = self.w0 * tau + self.b0
        per = torch.sin(tau * self.w + self.b)
        return torch.cat([lin, per], dim=-1)


class GeometricGaussianHead(nn.Module):
    def __init__(self, d_emb: int, d_time: int = 16,
                 use_pair_features: bool = False):
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6

        # --- geometric channel (the whole model) -------------------------------
        # u enters as the BASE POINT (everything is relative to E[u]); no separate
        # u-vs-v term is needed. Three scalars: recency decay, covariance shrinkage,
        # logit scale.
        self.log_lambda = nn.Parameter(torch.zeros(1))   # λ = softplus(·) ≥ 0
        self.log_tau = nn.Parameter(torch.zeros(1))      # τ = softplus(·) ≥ 0  (shrinkage / dial)
        self.scale = nn.Parameter(torch.tensor(10.0))    # logit = −scale·√m²

        # --- proven extra terms (optional, additive) ---------------------------
        self.t2v_rec = Time2Vec(d_time)
        self.rec_head = nn.Linear(d_time, 1)
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.t2v_pair = Time2Vec(d_time)
            self.pair_head = nn.Linear(d_time + 2, 1)

    # ----------------------------------------------------------------------
    # Sphere log-map at base point p (closed form; equals geoopt.Sphere().logmap,
    # validated). p [.,d] unit, x [.,d] unit -> tangent vector at p [.,d].
    # ----------------------------------------------------------------------
    def _logmap(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    # ----------------------------------------------------------------------
    def forward(self, tok_emb: torch.Tensor, tok_age: torch.Tensor,
                tok_mask: torch.Tensor, E_u: torch.Tensor, E_v: torch.Tensor,
                rec_v_log: torch.Tensor,
                pair_rec_log: torch.Tensor = None,
                pair_ever: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """tok_emb [B,n,d]  source walk-neighbour embeddings (context only; seed +
                            padding excluded, marked by tok_mask).
           tok_age [B,n]    age = t_query − t_node per token (≥0; 0 where masked).
           tok_mask[B,n]    bool, True at real neighbour positions.
           E_u     [B,d]    source embeddings (the tangent-space BASE POINT).
           E_v     [B,C,d]  candidate embeddings.
           rec_v_log [B,C]  log1p(t_query − t_last[v]) candidate recency.
           -> logits [B, C].
        """
        eu = F.normalize(E_u, dim=-1)                              # [B, d]
        ev = F.normalize(E_v, dim=-1)                              # [B, C, d]
        ew = F.normalize(tok_emb, dim=-1)                          # [B, n, d]
        B, n, d = ew.shape
        m = tok_mask.float()                                       # [B, n]

        # --- map neighbours + candidates into T_{E[u]} -------------------------
        g = self._logmap(eu.unsqueeze(1), ew)                     # [B, n, d]
        nu = self._logmap(eu.unsqueeze(1), ev)                    # [B, C, d]

        # --- recency-weighted Gaussian over the neighbour tangents -------------
        lam = F.softplus(self.log_lambda)
        wlog = (-lam * tok_age).masked_fill(~tok_mask, float("-inf"))   # [B, n]
        w = torch.softmax(wlog, dim=-1)                           # [B, n] (Σ=1; cold→nan)
        w = torch.nan_to_num(w, nan=0.0)                          # cold source -> all 0
        mu = (w.unsqueeze(-1) * g).sum(dim=1)                     # [B, d]  predicted position

        # weighted-centered neighbour matrix A_i = √w_i (g_i − μ);  C = Aᵀ A.
        # clamp_min before sqrt: softmax weights underflow to EXACTLY 0 for old /
        # masked tokens, and √(·) has an infinite (NaN) backward at 0 — the clamp
        # keeps the gradient finite without changing the math (those weights ≈ 0).
        A = torch.sqrt(w.clamp_min(self.eps)).unsqueeze(-1) * (g - mu.unsqueeze(1))  # [B,n,d]
        A = A * m.unsqueeze(-1)                                   # zero padded rows

        # --- Mahalanobis via Woodbury (only an [n,n] inverse) ------------------
        # m² = (1/τ)[ ‖δ‖² − (Aδ)ᵀ (τI_n + AAᵀ)⁻¹ (Aδ) ],  δ = ν − μ
        tau = F.softplus(self.log_tau) + self.eps                # scalar > 0
        AAt = torch.einsum("bnd,bmd->bnm", A, A)                  # [B, n, n]  (rank ≤ n)
        G = torch.linalg.inv(AAt + tau * torch.eye(n, device=A.device).unsqueeze(0))  # [B,n,n]

        delta = nu - mu.unsqueeze(1)                              # [B, C, d]
        q = torch.einsum("bnd,bcd->bcn", A, delta)               # [B, C, n] = Aδ
        quad = torch.einsum("bcn,bnm,bcm->bc", q, G, q)          # [B, C] = (Aδ)ᵀ G (Aδ)
        sq = (delta * delta).sum(-1)                             # [B, C] = ‖δ‖²
        m2 = (sq - quad) / tau                                   # [B, C] = Mahalanobis²  (≥0)
        maha = torch.sqrt(m2.clamp_min(self.eps))                # [B, C]

        logit = -self.scale.clamp_min(1e-3) * maha               # [B, C]

        # --- proven additive terms (optional) ----------------------------------
        logit = logit + self.rec_head(self.t2v_rec(rec_v_log)).squeeze(-1)
        if self.use_pair_features:
            feat = torch.cat(
                [self.t2v_pair(pair_rec_log),
                 pair_ever.unsqueeze(-1), pair_count_log.unsqueeze(-1)],
                dim=-1)
            logit = logit + self.pair_head(feat).squeeze(-1)
        return logit
