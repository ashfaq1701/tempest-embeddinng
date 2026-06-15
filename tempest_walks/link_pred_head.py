"""Geometric link head — walk-velocity / trajectory extrapolation.

Instead of scoring a candidate by how near it sits to a static recency-weighted
centroid of u's walk-neighbours, this head asks where u's activity is HEADING: each of
u's K temporal walks is fit as its own short trajectory through the tangent space at
E[u], the per-walk predictions are averaged, and the candidate is scored by an
anisotropic distance from that forward-projected point.

Input is walk-structured [B, K, L, d] (K walks × L steps), kept separate per walk.

  base point   p = E[u]                            (everything in the tangent space at u)
  per walk k:  g_i = Log_{E[u]}(E[node_i])         walk-k nodes as tangent vectors
               fit g(age) ≈ ḡ_k + V_k·(age−ā_k)    LS over walk k's L steps, weighted by
                                                   softmax(−λ·age + edge_proj(edge_feat))
                                                   — recency × learned edge-type weight
               μ_k = ḡ_k − V_k                      walk-k prediction at query time (age=0)
  aggregate :  μ = mean_k μ_k ,  V = mean_k V_k     over VALID walks (≥1 real token)
  candidate :  ν = Log_{E[u]}(E[v]) ;  δ = ν − μ ;  V̂ = V/‖V‖
  score     :  d = √( a·⟨δ,V̂⟩² + b·‖δ⊥‖² )         ellipse along the mean motion (a,b ≥ 0)
               logit = coef_geo·(−α·d) + coef_rec·rec(v) [+ coef_pair·pair(u,v)]

The distance is an ellipse oriented along the mean direction of motion V̂: a weights
"off ALONG the line" (right heading, wrong distance), b weights "off the line" (wrong
heading); a=b recovers the isotropic ‖ν−μ‖. Channels are mixed with learnable
per-channel gains (init 1). Few scalars (λ, α, a, b, coef_*) — little to overfit.

No fallback branches; thin/cold sources degrade by the arithmetic itself: an empty
walk is excluded from the average; a walk with no time-spread has V_k=0 ⇒ μ_k=ḡ_k (its
own centroid); a fully-cold source has V=0, μ=0 ⇒ d=√b·‖ν‖ (distance to u). The only
non-modelling guards are the ε's that keep denominators finite.

E stays the single sphere parameter (link-trained, no detach).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(dim - 1))
        self.b = nn.Parameter(torch.rand(dim - 1) * 2 * math.pi)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.unsqueeze(-1)
        return torch.cat([self.w0 * tau + self.b0, torch.sin(tau * self.w + self.b)], dim=-1)


class GeometricVelocityPerWalkAvgHead(nn.Module):
    def __init__(self, d_emb: int, d_time: int = 16, use_pair_features: bool = False,
                 edge_dim: int = 0):
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6
        self.log_lambda = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.log_a = nn.Parameter(torch.zeros(1))
        self.log_b = nn.Parameter(torch.zeros(1))
        # Edge-feature re-weighting of the per-walk LS fit: a learnable scalar per
        # walk STEP, added to that step's recency log-weight (which steps matter).
        # ZERO-INIT ⇒ e≡0 ⇒ exact recency-only fit at start; the projection grows
        # only if edge-type re-weighting lowers the loss (its magnitude = the learned
        # amount). Single Linear, no MLP; only enters the fit weights (no candidate
        # channel — the (u,v) edge does not exist at query time).
        self.edge_proj = nn.Linear(edge_dim, 1)
        nn.init.zeros_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)
        self.t2v_rec = Time2Vec(d_time)
        self.rec_head = nn.Linear(d_time, 1)
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.t2v_pair = Time2Vec(d_time)
            self.pair_head = nn.Linear(d_time + 2, 1)
        self.coef_geo = nn.Parameter(torch.ones(1))
        self.coef_rec = nn.Parameter(torch.ones(1))
        if use_pair_features:
            self.coef_pair = nn.Parameter(torch.ones(1))

    def _logmap(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _per_walk_fit(self, g, age, mask, edge_feat):
        """g [B,K,L,d], age [B,K,L], mask [B,K,L] (bool),
        edge_feat [B,K,L,edge_dim]
        -> mu_k [B,K,d], V_k [B,K,d], walk_valid [B,K]."""
        lam = F.softplus(self.log_lambda)
        e = self.edge_proj(edge_feat).squeeze(-1)                  # [B,K,L]  0 at init
        # e added BEFORE the masked_fill so padded steps stay −inf ⇒ w=0.
        wlog = (-lam * age + e).masked_fill(~mask, float("-inf"))  # over L, per walk
        w = torch.nan_to_num(torch.softmax(wlog, dim=-1), nan=0.0)  # [B,K,L]
        gbar = (w.unsqueeze(-1) * g).sum(dim=2)                    # [B,K,d]
        abar = (w * age).sum(dim=2)                               # [B,K]
        an = age / (abar.unsqueeze(-1) + self.eps)                # [B,K,L]
        anc = an - 1.0                                            # centred regressor
        Saa = (w * anc * anc).sum(dim=2)                         # [B,K]
        Sag = (w.unsqueeze(-1) * anc.unsqueeze(-1)
               * (g - gbar.unsqueeze(2))).sum(dim=2)              # [B,K,d]
        V = Sag / (Saa.unsqueeze(-1) + self.eps)                  # [B,K,d]
        mu = gbar - V                                            # [B,K,d]
        return mu, V, mask.any(dim=2)                            # walk_valid [B,K]

    def forward(self, tok_emb, tok_age, tok_mask, tok_edge_feat, E_u, E_v, rec_v_log,
                pair_rec_log=None, pair_ever=None, pair_count_log=None):
        """tok_emb [B,K,L,d], tok_age [B,K,L], tok_mask [B,K,L],
           tok_edge_feat [B,K,L,edge_dim] (per-step edge features, aligned with
           tok_age/tok_mask), E_u [B,d], E_v [B,C,d], rec_v_log [B,C] -> logits [B,C]."""
        eu = F.normalize(E_u, dim=-1)                             # [B,d]
        ev = F.normalize(E_v, dim=-1)                             # [B,C,d]
        ew = F.normalize(tok_emb, dim=-1)                         # [B,K,L,d]

        g = self._logmap(eu.unsqueeze(1).unsqueeze(1), ew)        # [B,K,L,d]
        nu = self._logmap(eu.unsqueeze(1), ev)                   # [B,C,d]

        mu_k, V_k, walk_valid = self._per_walk_fit(g, tok_age, tok_mask, tok_edge_feat)
        wv = walk_valid.float().unsqueeze(-1)                     # [B,K,1]
        denom = wv.sum(dim=1).clamp_min(self.eps)                # [B,1]
        mu = (wv * mu_k).sum(dim=1) / denom                      # [B,d]  averaged prediction
        V = (wv * V_k).sum(dim=1) / denom                       # [B,d]  averaged motion

        delta = nu - mu.unsqueeze(1)                             # [B,C,d]
        vhat = V / V.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        dpar2 = torch.einsum("bcd,bd->bc", delta, vhat).pow(2)    # [B,C]
        dperp2 = ((delta * delta).sum(-1) - dpar2).clamp_min(0.0)
        a, b = F.softplus(self.log_a), F.softplus(self.log_b)
        d = (a * dpar2 + b * dperp2).clamp_min(self.eps).sqrt()   # [B,C]

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
