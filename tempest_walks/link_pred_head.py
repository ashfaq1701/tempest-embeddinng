"""Geometric link head — point version (tangent-space recency-weighted mean + d/θ).

The cheap baseline of the family (the Gaussian head's τ→∞ limit, plus an explicit
angle term). Same crafted idea: base point E[u]; u's temporal-walk neighbours
log-mapped into the flat tangent space T_{E[u]}; a recency-weighted MEAN predicts
"where v should belong"; the candidate is scored by how far it is from that
predicted point — split into distance (wrong place) and angle (wrong heading).

  base point   p = E[u]
  neighbours   g_i = Log_{E[u]}(E[node_i])          tangent vectors at E[u]
  candidate    ν   = Log_{E[u]}(E[v])
  recency      w_i = softmax_i(−λ·age_i)            (λ ≥ 0 learnable; Σ w_i = 1)
  prediction   μ   = Σ_i w_i g_i                    the predicted position
  distance     d   = ‖ν − μ‖                        how far off (place)
  angle        θ   = ∠(ν, μ)                        direction mismatch (heading)
  logit        = −α·d − β·θ

d already fuses place + heading; exposing θ separately lets the model weight
"wrong place" vs "wrong heading" independently. Three scalars (α, β, λ) — almost
nothing to overfit.

Relation to the Gaussian head: that one replaces (d, θ) with a single Mahalanobis
distance to a fitted ellipse N(μ, C); as its shrinkage τ→∞ it collapses to ‖ν−μ‖,
i.e. this head's d term. Build this first; climb to the Gaussian only if the single
point underfits (μ is one location, so it assumes u heads toward ONE region).

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
    def __init__(self, d_emb: int, d_time: int = 16, mu_floor: float = 1e-3,
                 use_pair_features: bool = False):
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6
        self.mu_floor = mu_floor      # below this ‖μ‖, the angle term is switched off

        # --- geometric channel: three scalars ----------------------------------
        # u enters as the BASE POINT (everything relative to E[u]); no separate
        # u-vs-v term needed.
        self.log_lambda = nn.Parameter(torch.zeros(1))    # λ = softplus(·) ≥ 0  (recency)
        self.alpha = nn.Parameter(torch.tensor(10.0))     # distance weight
        self.beta = nn.Parameter(torch.tensor(1.0))       # angle weight

        # --- proven additive terms (optional) ----------------------------------
        self.t2v_rec = Time2Vec(d_time)
        self.rec_head = nn.Linear(d_time, 1)
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.t2v_pair = Time2Vec(d_time)
            self.pair_head = nn.Linear(d_time + 2, 1)

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

        # --- distance (place) --------------------------------------------------
        delta = nu - mu.unsqueeze(1)                              # [B, C, d]
        d = delta.norm(dim=-1)                                    # [B, C] = ‖ν−μ‖
        # cold/degenerate μ≈0 ⇒ d → ‖ν‖ = geodesic(E[u],E[v]) automatically.

        # --- angle (heading), guarded against ‖μ‖→0 cancellation ---------------
        mu_norm = mu.norm(dim=-1)                                 # [B]
        nu_norm = nu.norm(dim=-1).clamp_min(self.eps)            # [B, C]
        dot = torch.einsum("bcd,bd->bc", nu, mu)                 # [B, C]
        cos = (dot / (nu_norm * mu_norm.unsqueeze(1).clamp_min(self.eps))
               ).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(cos)                                # [B, C] ∈ [0, π]
        # switch the angle off where μ is too small to define a direction.
        theta = torch.where((mu_norm > self.mu_floor).unsqueeze(1),
                            theta, torch.zeros_like(theta))

        logit = (-self.alpha.clamp_min(1e-3) * d
                 - self.beta.clamp_min(0.0) * theta)             # [B, C]

        # --- proven additive terms (optional) ----------------------------------
        logit = logit + self.rec_head(self.t2v_rec(rec_v_log)).squeeze(-1)
        if self.use_pair_features:
            feat = torch.cat(
                [self.t2v_pair(pair_rec_log),
                 pair_ever.unsqueeze(-1), pair_count_log.unsqueeze(-1)],
                dim=-1)
            logit = logit + self.pair_head(feat).squeeze(-1)
        return logit
