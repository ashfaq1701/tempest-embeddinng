"""Geometric link head — REACH (one-sided drift), COUNT-FREE.

Builds a displacement μ_u for the source ONLY, from its walk tokens, and scores each candidate
by how close its STATIC embedding E[v] is to u's drifted position:

  μ_u = drift in T_{E[u]}                             (recency centroid of u's Log vectors)
  q_u = exp_{E[u]}(μ_u)                               u's drifted position, back ON the sphere
  reach(u,v) = ⟨q_u, E[v]⟩                            does u's drift reach v?

The candidate side samples NO walks — there is no μ_v. q_u is u's neighbourhood pushed off
E[u]; the inner product with the unit E[v] asks "how close is v to where u's neighbourhood is
heading?". reach rises when v sits in the region u drifts toward, even if E[u], E[v] are far
apart. coef_reach init 0 ⇒ the head starts as the proven query-anchored baseline and reach earns
its weight. Age stays t_query − t_edge throughout (the trainer forms it; the head never
re-defines it). The score is identity + reach:

  TOKEN BASIS — the source seed carries a COUNT-FREE bag of walk-reached positions p (one token
  per reached position; a node recurring k times is k tokens). μ sums the softmax over those
  tokens directly — multiplicity is implicit in token repetition, no explicit count. age_p =
  t_query − t_edge(p).

  μ_u = Σ_p softmax_p(−λ·age_p)·Log_{E[u]}(E[node_p ∈ u-tokens]) ; r_u = ĝate(μ_u)

  identity = −α·ellipse( Log_{E[u]}(E_v) − μ_u ; r_u )          is v in u's region?     [B,C]
  reach    = ⟨ exp_{E[u]}(μ_u), E_v ⟩                          does u's drift reach v?  [B,C]
  logit = coef_identity·identity + coef_reach·reach

Only the source builds μ — the candidate enters solely through its static embedding E[v]
(identity + reach). This is the MINIMAL one-sided geometric head: no staleness, no time-encoding
(ExpDecayBasis/Time2Vec), no pair channel — a clean base for an end-to-end redesign.

BASELINE AT INIT — coef_identity=1, coef_reach=0 ⇒ at init the head IS the proven identity term
and nothing else; reach earns its weight from zero.

COUNT-FREE — a node reached k times appears as k tokens, so its μ softmax-mass is summed across
them; the old γμ·log(1+k) emphasis (learned ≈ −0.4, suppressed) is gone.

TIME UNITS — raw ages; μ softmax scale-invariant (λ self-scales). E stays the single sphere
parameter (link-trained, no detach).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class GeometricPointHead(nn.Module):
    def __init__(self, d_emb: int, t_train: float = 1.0):
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

        # --- geometric mix coefficients ----------------------------------------
        # identity is the proven baseline (init 1); reach earns weight (init 0).
        self.coef_identity = nn.Parameter(torch.ones(1))
        # REACH — ⟨exp_{E[u]}(μ_u), E_v⟩: does u's one-sided drift reach v? coef init 0.
        self.coef_reach = nn.Parameter(torch.zeros(1))

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
                cand_ids: torch.Tensor,        # [B, C]  candidate node ids
                ) -> torch.Tensor:
        """-> logits [B, C]. The head owns all embedding lookups and timing: E_u, E_v and the
        token embeddings all come from `e_weight` (E_u = e_weight[src_tokens.seeds],
        E_v = e_weight[cand_ids]); token ages come from src_tokens.cutoffs − src_tokens.pos_ts.
        The trainer only hands over the table, the (self-contained) source walk tokens, and the
        candidate ids. Score = identity + reach (no staleness / time-encoding / pair channels)."""
        B, C = cand_ids.shape[0], cand_ids.shape[1]
        d = self.d_emb
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

        # --- REACH: push u's drift off E[u] back onto the sphere and inner-product it with the
        # candidate's STATIC embedding. No μ_v — the candidate side samples no walks. q_u is u's
        # neighbourhood centroid drifted off E[u]; ⟨q_u, E_v⟩ asks whether v sits where u's
        # neighbourhood is heading. Rises when v is in u's drift region even if E[u], E[v] are
        # structurally far apart. coef_reach init 0 ⇒ no-op at init.
        q_u = self._expmap(eu, mu_u)                              # [B,d]   drifted source pos
        reach = (q_u.unsqueeze(1) * ev).sum(-1)                  # [B,C]   ⟨q_u, E_v⟩

        logit = (self.coef_identity * q_ident
                 + self.coef_reach * reach)

        return logit
