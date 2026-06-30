"""Velocity head — extrapolate u's drifting neighbourhood to the query time.

The previous head built a recency-weighted CENTROID of u's past neighbours in the tangent space
at E[u] — an AVERAGING read-out that can only land *among* nodes u has already touched, never
*ahead* of them. Where u's neighbourhood DRIFTS through embedding space, the next partner is a
point no past neighbour occupies and the centroid structurally cannot reach it.

The velocity head fits a LINE (intercept + slope) to u's neighbour trajectory in tangent space
and evaluates it at the query time — an EXTRAPOLATION, not an average. Per query (seed u, cutoff
t), base point p = normalize(E[u]):

  context tokens: every real, non-seed walk position; v = Log_p(E[node]) ∈ T_p, node-aligned time
  signed normalized time:  s = (time − t) / T_train          (≤ 0; query → s = 0)
  recency weight:          w = exp(λ·s) · ctx                 (λ = softplus(log_lambda))

  Weighted free-line fit (per query), evaluated at s = 0 (i.e. at t_query):
      W   = Σ w ; s̄ = Σ w·s / W ; v̄ = Σ w·v / W
      b   = Σ w·(s−s̄)·v / Σ w·(s−s̄)²            (velocity / slope; v̄ is the weighted centroid)
      μ   = v̄ − b·s̄                             (the line at s = 0)

Two channels (point-head style), blended by learnable coefficients:
      score(u,v) = γ·( a·⟨exp_p(v̄), Ê_v⟩  +  c·⟨exp_p(μ), Ê_v⟩ )
  * IDENTITY  a·⟨exp_p(v̄), Ê_v⟩  — score v against the CENTROID point: "is v in u's recent
    neighbourhood?" (the recurrence detector). a init 1 ⇒ at init the head IS the centroid baseline.
  * VELOCITY  c·⟨exp_p(μ), Ê_v⟩  — score v against the EXTRAPOLATED point: "is v where u is
    heading?" c init 0 ⇒ velocity earns weight (on the drift slice).

Fallbacks fall out of the algebra (never worse than the centroid by construction):
  * COLD (no context, W = 0): v̄ = μ = 0 ⇒ both points = p ⇒ score = γ·a·cosine(E[u], E[v]).
  * DEGENERATE time (Σ w·(s−s̄)² ≈ 0, one distinct timestamp / single neighbour): b = 0 ⇒ μ = v̄ —
    velocity collapses onto identity; it only DIVERGES where there is genuine temporal spread.

The whole head is four learnable scalars {log_lambda, logit_scale, coef_identity, coef_velocity}
over the sphere embedding table. No ellipse/heading gate, no forward Δτ horizon. T_train is the
frozen train-split span (NOT a per-batch t_max).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


# ──────────────────────────────────────────────────────────────────────────
# Sphere geometry primitives (shape-agnostic over leading axes)
# ──────────────────────────────────────────────────────────────────────────

def sphere_log(p: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Log_p(x) ∈ T_p S^{d-1}: the tangent vector (⊥ p) of length = geodesic angle(p, x).
    p, x unit and broadcastable; -> [..., d]. ⟨p,x⟩ clamped to the injectivity radius."""
    c = (p * x).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    orth = x - c * p
    return torch.arccos(c) * orth / orth.norm(dim=-1, keepdim=True).clamp_min(eps)


def sphere_exp(p: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """exp_p(v) = cos‖v‖·p + sin‖v‖·v/‖v‖ ∈ S^{d-1}. v ⊥ p (a tangent vector); ‖v‖ capped to the
    injectivity radius (<π); v→0 ⇒ exp_p(v)=p (cold drifts nowhere). Final normalize is a belt."""
    norm = v.norm(dim=-1, keepdim=True)
    theta = norm.clamp(max=math.pi - eps)
    coef = torch.sin(theta) / norm.clamp_min(eps)              # → 0 as norm → 0 (q̂ → p)
    return F.normalize(torch.cos(theta) * p + coef * v, dim=-1)


class VelocityHead(nn.Module):
    """Velocity (free-WLS-line) link head — see module docstring. forward -> logits [Q, C]."""

    def __init__(self, d_emb: int, t_train: float = 1.0):
        super().__init__()
        self.d_emb = int(d_emb)
        self.T_train = max(float(t_train), 1.0)         # time normalizer for s = (time − t)/T_train
        self.eps = 1e-6
        # recency λ; init ≈ 3 so λ·s spans ~O(1) over the normalized window s ∈ [−1, 0].
        self.log_lambda = nn.Parameter(torch.tensor(math.log(math.expm1(3.0)), dtype=torch.float32))
        # logit temperature on cosine ∈ [−1, 1]; init ≈ 10 to sharpen the softmax-CE.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(math.expm1(10.0)), dtype=torch.float32))

        # Two channels (point-head style): IDENTITY scores v against the CENTROID point exp_p(v̄)
        # — "is v in u's recent neighbourhood?" (the recurrence detector, init 1 = proven baseline);
        # VELOCITY scores v against the EXTRAPOLATED point exp_p(μ) — "is v where u is heading?"
        # (init 0, earns weight on the drift slice). At init the head IS the centroid baseline.
        self.coef_identity = nn.Parameter(torch.ones(1))
        self.coef_velocity = nn.Parameter(torch.zeros(1))

    # ──────────────────────────────────────────────────────────────────
    # Pieces (each does one job; `forward` orchestrates)
    # ──────────────────────────────────────────────────────────────────

    def _base_point(self, e_weight: torch.Tensor, tokens: WalkTokens) -> torch.Tensor:
        """p = normalize(E[u]) per query -> [Q, d]. From tokens.seeds (robust to cold/empty walks
        where the seed is not placed in `nodes`)."""
        return F.normalize(F.embedding(tokens.seeds, e_weight), dim=-1)

    def _context(self, e_weight: torch.Tensor, tokens: WalkTokens, p: torch.Tensor):
        """Per context token (real, non-seed): tangent vector v = Log_p(E[node]), signed time s,
        recency weight w. p [Q, d] -> v [Q, K, L, d], s [Q, K, L], w [Q, K, L] (0 off-context)."""
        cut = tokens.cutoffs.view(-1, 1, 1)
        is_seed = (tokens.timestamps == cut) & tokens.nodes_mask         # seed slot = the t==cutoff one
        ctx = (tokens.nodes_mask & ~is_seed).to(p.dtype)                 # [Q, K, L] 1 on context

        s = (tokens.timestamps - cut).to(p.dtype) / self.T_train         # ≤ 0; 0 at the (excluded) seed
        s = s * ctx                                                      # 0 off-context (w kills it too)
        w = torch.exp(F.softplus(self.log_lambda) * s) * ctx             # [Q, K, L]

        ev = F.normalize(F.embedding(tokens.nodes.clamp_min(0), e_weight), dim=-1)   # [Q, K, L, d]
        v = sphere_log(p[:, None, None, :], ev, self.eps)                # [Q, K, L, d]
        return v, s, w

    def _fit_at_query(self, v: torch.Tensor, s: torch.Tensor, w: torch.Tensor):
        """Weighted free-line fit over (K, L) -> (v̄, μ), both [Q, d]: v̄ = weighted CENTROID (the
        identity prediction), μ = the line evaluated at s = 0 (the velocity prediction). Cold
        (W ≤ ε) → v̄ = μ = 0 (exact). Degenerate time (Σw·(s−s̄)² ≈ 0) → b = 0 → μ = v̄."""
        wsum = w.sum((1, 2))                                             # [Q] raw (for the cold mask)
        W = wsum.clamp_min(self.eps)
        sbar = (w * s).sum((1, 2)) / W                                   # [Q]
        vbar = (w.unsqueeze(-1) * v).sum((1, 2)) / W.unsqueeze(-1)       # [Q, d] centroid
        ds = s - sbar[:, None, None]                                     # [Q, K, L]
        Sss = (w * ds * ds).sum((1, 2))                                  # [Q]
        Sdv = (w.unsqueeze(-1) * ds.unsqueeze(-1) * v).sum((1, 2))       # [Q, d]
        b = Sdv / Sss.clamp_min(self.eps).unsqueeze(-1)                  # [Q, d] velocity (0 if degenerate)
        mu = vbar - b * sbar.unsqueeze(-1)                               # [Q, d] line at s = 0
        cold = (wsum <= self.eps).unsqueeze(-1)
        z = torch.zeros_like(mu)
        return torch.where(cold, z, vbar), torch.where(cold, z, mu)

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(self, e_weight: torch.Tensor, tokens: WalkTokens,
                cand_ids: torch.Tensor) -> torch.Tensor:
        """e_weight [N, d] the whole sphere table; tokens the source walks (self-contained:
        seeds + cutoffs); cand_ids [Q, C]. -> logits [Q, C]."""
        p = self._base_point(e_weight, tokens)                          # [Q, d]
        v, s, w = self._context(e_weight, tokens, p)
        vbar, mu = self._fit_at_query(v, s, w)                          # [Q, d] centroid, velocity
        q_id = sphere_exp(p, vbar, self.eps)                            # [Q, d] centroid point
        q_vel = sphere_exp(p, mu, self.eps)                            # [Q, d] extrapolated point

        evc = F.normalize(F.embedding(cand_ids, e_weight), dim=-1)      # [Q, C, d]
        identity = (q_id.unsqueeze(1) * evc).sum(-1)                   # [Q, C] ⟨centroid, E_v⟩
        velocity = (q_vel.unsqueeze(1) * evc).sum(-1)                  # [Q, C] ⟨extrapolation, E_v⟩
        gamma = F.softplus(self.logit_scale)
        return gamma * (self.coef_identity * identity + self.coef_velocity * velocity)   # [Q, C]
