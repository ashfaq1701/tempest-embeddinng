"""Link head — DualMu (symmetric walk-mean), CROSS-IDENTITY score.

Both endpoints build a recency-weighted predicted point from their OWN temporal walks
(μ_u in T_{E[u]}, μ_v in T_{E[v]}; Exp-mapped to sphere points P_u, P_v). The geometric
score pairs each side's PREDICTION against the OTHER side's bare IDENTITY, symmetrically:

  μ_u = Σ softmax(−λ·age)·Log_{E[u]}(E[w])   ;  P_u = Exp_{E[u]}(μ_u)
  μ_v = Σ softmax(−λ·age)·Log_{E[v]}(E[w])   ;  P_v = Exp_{E[v]}(μ_v)
  geo = α · ½( ⟨P_u, E_v⟩ + ⟨P_v, E_u⟩ )

Why cross-identity instead of ⟨P_u, P_v⟩ (prediction↔prediction): the bare embeddings
E_v and E_u each appear RAW in the score, so each receives a DIRECT, full-strength,
candidate-specific gradient (∂⟨P_u,E_v⟩/∂E_v = P_u). The proven point head's edge is
exactly this direct supervision of candidate IDENTITY — which is what a recurrence-heavy
workload rewards. ⟨P_u,P_v⟩ buries BOTH identities behind softmax-means, so neither gets
the direct signal; this head restores it for both endpoints while keeping both μ's in play.

Gradient facts (why this is the right fix, and why detaching μ-weights is NOT needed):
  - The μ softmax weights depend only on age (timestamps) and λ — NOT on embeddings — so
    ∂w/∂E = 0: there is no softmax-Jacobian dispersion. Each walk-neighbour E[w_i] ALREADY
    gets a direct gradient w_i·∂g_i (scaled by its recency weight, which is correct). So
    neighbours are well-supervised in both heads; only the candidate IDENTITY was buried,
    and the cross-identity score is what surfaces it.
  - λ keeps its gradient (weights NOT detached); init λ≈C/t_train so the μ softmax is a
    focused-but-not-argmax mean (the symmetric-head sweet spot — argmax one-hots BOTH sides
    and sparsifies gradient; near-flat gives diffuse μ).

μ_v is computed ONCE PER UNIQUE candidate (query-independent: the recency softmax is shift-
invariant so t_query cancels) and scattered via v_inv — no [B,C,M,d] blow-up. Cold node
(no walk-neighbours) ⇒ μ=0 ⇒ P=E[node], so geo → α·½(⟨E_u,E_v⟩+⟨E_v,E_u⟩) = α·⟨E_u,E_v⟩,
the plain embedding similarity (graceful).

Additive channels (each its own learnable coef): rec (v's staleness, ExpDecayBasis) and a
flagged pair channel ((u,v) recurrence recency + log1p count, count-coef init 0).
E stays the single sphere parameter (link-trained, no detach).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpDecayBasis(nn.Module):
    def __init__(self, dim: int, t_train: float):
        super().__init__()
        r_lo = -math.log(max(float(t_train), 1.0)); r_hi = 0.0
        self.log_rates = nn.Parameter(torch.linspace(r_lo, r_hi, dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.exp(self.log_rates) * dt.unsqueeze(-1))


class DualMuHead(nn.Module):
    def __init__(self, d_emb: int, d_time: int = 16,
                 use_pair_features: bool = False, t_train: float = 1.0):
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6
        # focused-but-not-argmax μ: λ ≈ C/t_train (C=10) — see docstring.
        lam0 = 10.0 / max(float(t_train), 1.0)
        self.log_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.basis_rec = ExpDecayBasis(d_time, t_train)
        self.rec_head = nn.Linear(d_time, 1)
        self.use_pair_features = use_pair_features
        if use_pair_features:
            self.basis_pair = ExpDecayBasis(d_time, t_train)
            self.pair_head = nn.Linear(d_time, 1)
        self.coef_geo = nn.Parameter(torch.ones(1))
        self.coef_rec = nn.Parameter(torch.ones(1))
        if use_pair_features:
            self.coef_pair = nn.Parameter(torch.ones(1))
            self.coef_pair_count = nn.Parameter(torch.zeros(1))

    def _logmap(self, p, x):
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c); orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _expmap(self, p, v):
        vn = v.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.cos(vn) * p + torch.sin(vn) * (v / vn)

    def _mu(self, base, ew, age, mask):
        """μ in T_base (weights depend on age+λ only, NOT embeddings → neighbours get
           direct weight-scaled gradients; λ keeps its gradient)."""
        g = self._logmap(base.unsqueeze(-2), ew)
        lam = F.softplus(self.log_lambda)
        wlog = (-lam * age).masked_fill(~mask, float("-inf"))
        w = torch.nan_to_num(torch.softmax(wlog, dim=-1), nan=0.0)
        return (w.unsqueeze(-1) * g).sum(dim=-2)

    def forward(self, tok_emb, tok_age, tok_mask, E_u,
                rec_v_dt, uniq_v_ids, v_inv, cand_ids, cand_age, cand_mask, e_weight,
                pair_dt=None, pair_count_log=None):
        B, C = rec_v_dt.shape
        eu = F.normalize(E_u, dim=-1)                                   # [B,d] source identity

        # source prediction P_u
        ew_u = F.normalize(tok_emb, dim=-1)                            # [B,n,d]
        P_u = self._expmap(eu, self._mu(eu, ew_u, tok_age, tok_mask))  # [B,d]

        # candidate identity + prediction, per UNIQUE node, scattered
        ev_u = F.normalize(F.embedding(uniq_v_ids, e_weight), dim=-1)              # [Mv,d]
        ew_v = F.normalize(F.embedding(cand_ids.clamp_min(0), e_weight), dim=-1)   # [Mv,M,d]
        P_v_u = self._expmap(ev_u, self._mu(ev_u, ew_v, cand_age, cand_mask))      # [Mv,d]
        ev = ev_u[v_inv].view(B, C, -1)                              # [B,C,d] bare identity (unit)
        P_v = P_v_u[v_inv].view(B, C, -1)                             # [B,C,d] prediction

        # CROSS-IDENTITY score: each prediction vs the other's RAW identity (direct grad to both)
        pu_ev = (P_u.unsqueeze(1) * ev).sum(-1)                       # [B,C]  ⟨P_u, ev⟩
        pv_eu = (P_v * eu.unsqueeze(1)).sum(-1)                       # [B,C]  ⟨P_v, eu⟩
        geo = self.alpha.clamp_min(1e-3) * 0.5 * (pu_ev + pv_eu)      # [B,C]

        rec = self.rec_head(self.basis_rec(rec_v_dt)).squeeze(-1)
        logit = self.coef_geo * geo + self.coef_rec * rec
        if self.use_pair_features:
            pair = self.pair_head(self.basis_pair(pair_dt)).squeeze(-1)
            logit = logit + self.coef_pair * pair + self.coef_pair_count * pair_count_log
        return logit
