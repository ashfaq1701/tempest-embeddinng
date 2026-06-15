"""Geometric link head — point version (recency-weighted mean + anisotropic ellipse).

Base point E[u]; u's temporal-walk neighbours log-mapped into the flat tangent
space T_{E[u]}; a recency-weighted MEAN μ predicts "where v should belong"; the
candidate is scored by an anisotropic (ellipse) distance from that predicted point,
oriented along the source's heading — so being off in the wrong DIRECTION costs
more than being off in DISTANCE. The channels are mixed with learnable coefficients.

  base point   p = E[u]
  neighbours   g_i = Log_{E[u]}(E[node_i])          tangent vectors at E[u]
  candidate    ν   = Log_{E[u]}(E[v])
  recency      w_i = softmax_i(−λ·age_i)            (λ ≥ 0 learnable; Σ w_i = 1)
  prediction   μ   = Σ_i w_i g_i                    the predicted position
  heading      r   = μ/‖μ‖ ;  δ = ν − μ = δ∥ (along r) + δ⊥ (⊥ r)
  geo channel  d   = √( a·‖δ∥‖² + b·‖δ⊥‖² )         anisotropic ellipse, a,b ≥ 0
  logit        = coef_geo·(−α·d) + coef_rec·rec(v) [+ coef_pair·pair(u,v)]

The geometric distance is an ELLIPSE oriented along each source's heading r (not an
isotropic circle): the model learns a,b so being off ALONG the heading is weighted
differently from being off SIDEWAYS — on tgbl-wiki it learns a/b ≈ 1/100 ("direction
matters, distance-along-heading is ~free"). a=b recovers the plain circle ‖ν−μ‖.
The channels (geometric / recency / pair) are mixed with learnable per-channel
coefficients (init 1 = plain sum) so the model can rebalance them. Few scalars
(α, λ, a, b, coef_*) — almost nothing to overfit; the anisotropy is adaptive
(≈isotropic where no direction signal exists).

Lineage: an explicit angle term −β·θ was tried and dropped (redundant with ‖ν−μ‖ by
the law of cosines, needed a ‖μ‖-floor guard). The Gaussian head's per-source
covariance was falsified — unestimable from a few recency-weighted neighbours, it
either shrinks away → this head, or hurts on cold-start. A global AMBIENT metric also
lost (49× anisotropy but in a frame that rotates per source); the per-source
intrinsic frame above is the correct, estimable basis (2 global scalars).

E stays the single sphere parameter (link-trained, no detach).
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


class GeometricPointHead(nn.Module):
    def __init__(self, d_emb: int, d_time: int = 16,
                 use_pair_features: bool = False):
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6

        # --- geometric channel -------------------------------------------------
        # u enters as the BASE POINT (everything relative to E[u]); no separate
        # u-vs-v term needed.
        self.log_lambda = nn.Parameter(torch.zeros(1))    # λ = softplus(·) ≥ 0  (recency)
        self.alpha = nn.Parameter(torch.tensor(10.0))     # distance weight
        # Intrinsic-frame anisotropy (a,b ≥ 0, global): the candidate distance is
        # an ELLIPSE oriented along each source's heading r=μ/‖μ‖, not a circle —
        # d² = a‖δ∥‖² + b‖δ⊥‖² (δ=ν−μ split into along-heading δ∥ and sideways δ⊥).
        # The model learns that being off ALONG the heading is ~free while being
        # off SIDEWAYS is costly: direction matters more than exact distance.
        # a=b ⇒ the isotropic ‖ν−μ‖. +0.0025 test wiki / +0.0095 test review
        # (2-seed) over the isotropic head. (The angle term −β·θ this replaces was
        # redundant with ‖ν−μ‖ by the law of cosines; dropped.)
        self.log_a = nn.Parameter(torch.zeros(1))         # radial (along-heading)
        self.log_b = nn.Parameter(torch.zeros(1))         # tangential (off-heading)

        # --- proven additive terms (optional) ----------------------------------
        self.t2v_rec = Time2Vec(d_time)
        self.rec_head = nn.Linear(d_time, 1)
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.t2v_pair = Time2Vec(d_time)
            self.pair_head = nn.Linear(d_time + 2, 1)

        # --- learnable per-channel mix coefficients (init 1 = plain sum) --------
        # One learnable gain per channel (on the raw channels) so the model can
        # rebalance the geometric vs recency vs pair terms rather than a fixed
        # sum. +0.005 val / +0.009 test on tgbl-wiki (with pair features on).
        self.coef_geo = nn.Parameter(torch.ones(1))
        self.coef_rec = nn.Parameter(torch.ones(1))
        if use_pair_features:
            self.coef_pair = nn.Parameter(torch.ones(1))

    def _logmap(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Sphere log-map at base point p (closed form = geoopt.Sphere().logmap)."""
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def forward(self, tok_emb: torch.Tensor, tok_age: torch.Tensor,
                tok_mask: torch.Tensor, E_u: torch.Tensor, E_v: torch.Tensor,
                rec_v_log: torch.Tensor,
                pair_rec_log: torch.Tensor = None,
                pair_ever: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """tok_emb [B,n,d]  source walk-neighbour embeddings (context only).
           tok_age [B,n]    age = t_query − t_node per token (≥0; 0 where masked).
           tok_mask[B,n]    bool, True at real neighbour positions.
           E_u     [B,d]    source embeddings (tangent-space BASE POINT).
           E_v     [B,C,d]  candidate embeddings.
           rec_v_log [B,C]  log1p(t_query − t_last[v]) candidate recency.
           -> logits [B, C].
        """
        eu = F.normalize(E_u, dim=-1)                              # [B, d]
        ev = F.normalize(E_v, dim=-1)                              # [B, C, d]
        ew = F.normalize(tok_emb, dim=-1)                          # [B, n, d]

        # --- map neighbours + candidates into T_{E[u]} -------------------------
        g = self._logmap(eu.unsqueeze(1), ew)                     # [B, n, d]
        nu = self._logmap(eu.unsqueeze(1), ev)                    # [B, C, d]

        # --- recency-weighted predicted position μ ----------------------------
        lam = F.softplus(self.log_lambda)
        wlog = (-lam * tok_age).masked_fill(~tok_mask, float("-inf"))   # [B, n]
        w = torch.nan_to_num(torch.softmax(wlog, dim=-1), nan=0.0)      # cold src -> all 0
        mu = (w.unsqueeze(-1) * g).sum(dim=1)                      # [B, d]

        # --- distance: anisotropic ellipse in the per-source heading frame ------
        delta = nu - mu.unsqueeze(1)                              # [B, C, d]
        r = mu / mu.norm(dim=-1, keepdim=True).clamp_min(self.eps)   # [B, d] heading
        dpar2 = torch.einsum("bcd,bd->bc", delta, r).pow(2)         # [B, C] ‖δ∥‖²
        dperp2 = ((delta * delta).sum(-1) - dpar2).clamp_min(0.0)   # [B, C] ‖δ⊥‖²
        a = F.softplus(self.log_a)
        b = F.softplus(self.log_b)
        d = (a * dpar2 + b * dperp2).clamp_min(self.eps).sqrt()     # [B, C]
        # cold/degenerate μ≈0 ⇒ r→0 ⇒ d → √b·‖ν‖ = geodesic(E[u],E[v]).

        # --- channels combined with learnable per-channel coefficients ---------
        geo = -self.alpha.clamp_min(1e-3) * d                      # [B, C]
        rec = self.rec_head(self.t2v_rec(rec_v_log)).squeeze(-1)   # [B, C]
        logit = self.coef_geo * geo + self.coef_rec * rec
        if self.use_pair_features:
            feat = torch.cat(
                [self.t2v_pair(pair_rec_log),
                 pair_ever.unsqueeze(-1), pair_count_log.unsqueeze(-1)],
                dim=-1)
            pair = self.pair_head(feat).squeeze(-1)                # [B, C]
            logit = logit + self.coef_pair * pair
        return logit
