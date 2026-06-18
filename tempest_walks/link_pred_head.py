"""Link head — DualMu version (symmetric walk-mean, predicted-point cosine on the hypersphere).

Both endpoints build a recency-weighted PREDICTED NEIGHBOUR POSITION from their own
temporal walks, then the geometric channel scores the link by the cosine between the
two predicted points:

  μ_u = Σ softmax(−λ·age)·Log_{E[u]}(E[w])   over u's walk-neighbours w   (tangent at E[u])
  P_u = normalize((1−g)·E[u] + g·Exp_{E[u]}(μ_u))   gated identity↔prediction blend
  μ_v = Σ softmax(−λ·age)·Log_{E[v]}(E[w])   over v's walk-neighbours w   (tangent at E[v])
  P_v = normalize((1−g)·E[v] + g·Exp_{E[v]}(μ_v))
  geo = α · ⟨P_u, P_v⟩                         cosine (≡ neg-sq-Euclidean on the sphere;
                                               avoids the arccos gradient singularity)

  g = σ(gate_logit) shared by both sides: g=1 ⇒ pure walk-prediction (plain DualMu);
  g→0 ⇒ pure identity E[node] (re-injects the candidate identity the point head relies on).

  logit = coef_geo·geo + coef_rec·rec(v) [+ coef_pair·pair(u,v) + coef_pair_count·log1p(count)]

The geometry is a few scalars (decoupled recency rates λ_u, λ_v — one per side, since u and v
play asymmetric roles — one distance scale α, and the identity gate g): no ellipse, no heading frame,
no co-reachability channel. μ_u and μ_v ARE the geometric model. μ_u, μ_v live in DIFFERENT
tangent spaces (T_{E[u]} vs T_{E[v]}), so Exp-mapping each to the sphere first is what makes
the cosine well-defined. A cold node with no walk-neighbours has μ ≈ 0 ⇒ P = E[node], so
geo degrades gracefully to α·⟨E[u], E[v]⟩ (the plain embedding similarity).

μ_v is computed ONCE PER UNIQUE candidate node and scattered to the [B,C] grid via v_inv —
P_v depends only on v's walks + E[v] (the recency softmax is shift-invariant, so the query's
t_query cancels), so per-cell recomputation is pure redundancy. This keeps the candidate
activation at [Mv,M,d] instead of [B,C,M,d] (Mv = #unique candidates) — ~8-20× less memory
and compute, exactly equal to the per-cell form (verified bit-identical). rec/pair stay
per-[B,C] (they depend on t_query outside any softmax).

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

        # --- geometric channel: ⟨P_u, P_v⟩ (predicted-point similarity) ------------
        # Each side builds a recency-weighted predicted point P = Exp_E[node](μ); the score
        # is their cosine ⟨P_u, P_v⟩ (≡ neg-squared-Euclidean on the unit sphere; cosine
        # avoids the arccos gradient singularity at coincidence). Ablations showed the two
        # identity↔prediction cross signals and the anisotropic ellipse add nothing over this
        # symmetric pred↔pred cosine on wiki — so the head keeps only the P↔P term.
        #
        # λ = softplus(log_lambda) on RAW age in the μ softmax. Init λ ≈ C/t_train so μ is a
        # FOCUSED recency mean over the most-recent ~N/C neighbours: C=30 (~8 eff of ~80) — a
        # sharp μ leaning toward the most-recent neighbours, with the identity gate giving the
        # model a stable fallback (g→0 ⇒ rely on E[node]) that lets it tolerate sharper λ than
        # the pure head could. NOT full argmax (point-head's log_lambda=0): for the SYMMETRIC
        # head both μ_u and μ_v near-one-hot → sparse gradients on BOTH sides → collapse risk
        # (the point head survives argmax only because its candidate is the dense stable E[v]).
        # NOT 1/t_train either (near-flat → diffuse weak μ). softplus⁻¹(x).
        #
        # DECOUPLED per side: λ_u (source/query μ_u) and λ_v (candidate μ_v) are SEPARATE
        # learnable params — u and v play asymmetric roles (e.g. bipartite users vs pages have
        # different neighbour-time distributions), so each side can pick its own sharpness.
        # Both init C=10; if decoupling helps they diverge from there.
        lam0 = 10.0 / max(float(t_train), 1.0)
        log_lam0 = math.log(math.expm1(lam0))
        self.log_lambda_u = nn.Parameter(torch.tensor([log_lam0], dtype=torch.float32))
        self.log_lambda_v = nn.Parameter(torch.tensor([log_lam0], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(10.0))     # cosine → logit scale
        # Identity gate g = σ(gate_logit): P = normalize((1−g)·E[node] + g·Exp_E[node](μ)).
        # Blends the node's own embedding (IDENTITY) with its walk-predicted point. g=1 ⇒ pure
        # prediction (the plain DualMu head); g→0 ⇒ pure identity E[node] — re-injecting the
        # candidate identity that the point head leans on. Shared by both u and v (symmetric).
        # Init logit 0 (g=0.5). A cold node (μ≈0 ⇒ Exp=E[node]) gives P=E[node] for any g.
        self.gate_logit = nn.Parameter(torch.zeros(1))

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
            age: torch.Tensor, mask: torch.Tensor,
            log_lambda: torch.Tensor, T: int = 3) -> torch.Tensor:
        """Variant A — IRLS CONSENSUS CENTER (outlier-robust recency mean of Log_base(nbr)).
           Starts from the plain recency mean, then T fixed unrolled IRLS steps down-weight
           neighbours far from the EMERGING center μ (rejection is emergent, no k). c is the
           recency-weighted RMS residual = emergent scale; ψ = (r²+c²)^−½ the robust weight.
           c and μ are .detach()'d inside the weights (IRLS quasi-constants); the final
           μ = Σ ω·g keeps gradient through g → E. base/ew/age/mask as before; uses the SPLIT
           log_lambda arg (λ_u for source, λ_v for candidate). Masked / cold ⇒ μ = 0."""
        g = self._logmap(base.unsqueeze(-2), ew)                       # [..., M, d]
        lam = F.softplus(log_lambda)
        wlog = (-lam * age).masked_fill(~mask, float("-inf"))          # [..., M]
        w = torch.nan_to_num(torch.softmax(wlog, dim=-1), nan=0.0)     # [..., M] recency (cold ⇒ 0)
        mu = (w.unsqueeze(-1) * g).sum(dim=-2)                         # [..., d] recency mean (init)
        for _ in range(T):                                            # fixed-T unrolled IRLS
            r = (g - mu.detach().unsqueeze(-2)).norm(dim=-1)          # [..., M] residual to center
            c = torch.sqrt((w * r * r).sum(-1) / w.sum(-1).clamp_min(self.eps)
                           ).detach().unsqueeze(-1)                    # [..., 1] emergent scale (wtd RMS)
            psi = torch.rsqrt(r * r + c * c + self.eps)               # [..., M] robust weight
            omega = (w * psi).masked_fill(~mask, 0.0)                 # recency × robustness
            omega = omega / omega.sum(-1, keepdim=True).clamp_min(self.eps)
            mu = (omega.unsqueeze(-1) * g).sum(dim=-2)                # [..., d] reweighted center
        return mu

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(self, tok_emb: torch.Tensor, tok_age: torch.Tensor,
                tok_mask: torch.Tensor, E_u: torch.Tensor,
                rec_v_dt: torch.Tensor,
                uniq_v_ids: torch.Tensor, v_inv: torch.Tensor,
                cand_ids: torch.Tensor, cand_age: torch.Tensor,
                cand_mask: torch.Tensor, e_weight: torch.Tensor,
                pair_dt: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """Source side (μ_u): tok_emb [B,n,d], tok_age [B,n], tok_mask [B,n], E_u [B,d].
           Candidate side (μ_v) — PER UNIQUE NODE (dedup): uniq_v_ids [Mv] the unique
           candidate node ids, v_inv [B*C] scatters P_v back to the [B,C] grid, and
           cand_ids/cand_age/cand_mask [Mv,M] are v's walk-neighbour node ids / per-batch-
           reference ages / mask. P_v is computed once per unique v then indexed — the
           [B,C,M,d] tensor never materialises (only [Mv,M,d]). Exact: μ_v is query-
           independent (see _candidate_walk_tokens), so the dedup changes nothing.
           rec_v_dt [B,C] RAW Δt_v candidate staleness (→ basis_rec); pair_dt/pair_count_log
           [B,C] when pair on. -> logits [B, C]."""
        B, C = rec_v_dt.shape
        eu = F.normalize(E_u, dim=-1)                                  # [B, d]
        g = torch.sigmoid(self.gate_logit)                            # identity↔prediction gate

        # --- geometric channel: P_u (u's walks) vs P_v (v's walks) ----------------
        # P = normalize((1−g)·E[node] + g·Exp_E[node](μ)): gated blend of identity and the
        # walk-predicted point. Cold μ≈0 ⇒ Exp=E[node] ⇒ P=E[node] for any g.
        ew_u = F.normalize(tok_emb, dim=-1)                            # [B, n, d]
        mu_u = self._mu(eu, ew_u, tok_age, tok_mask, self.log_lambda_u)   # [B, d]
        P_u = F.normalize((1.0 - g) * eu + g * self._expmap(eu, mu_u), dim=-1)    # [B, d]

        # μ_v once per UNIQUE candidate node, then scatter via v_inv (no [B,C,M,d] blow-up).
        ev_u = F.normalize(F.embedding(uniq_v_ids, e_weight), dim=-1)             # [Mv, d]
        ew_v = F.normalize(F.embedding(cand_ids.clamp_min(0), e_weight), dim=-1)  # [Mv, M, d]
        mu_v = self._mu(ev_u, ew_v, cand_age, cand_mask, self.log_lambda_v)  # [Mv, d]
        P_v_u = F.normalize((1.0 - g) * ev_u + g * self._expmap(ev_u, mu_v), dim=-1)  # [Mv, d]
        P_v = P_v_u[v_inv].view(B, C, -1)                           # [B, C, d] cheap index

        # --- predicted-point similarity: α·⟨P_u, P_v⟩ -----------------------------
        # Both endpoints are unit-sphere points, so cosine ≡ neg-squared-Euclidean for
        # ranking; α scales it into a logit. Cold node ⇒ μ=0 ⇒ P=E[node] (graceful).
        geo = self.alpha.clamp_min(1e-3) * (P_u.unsqueeze(1) * P_v).sum(-1)  # [B, C]

        # --- channels combined with learnable per-channel coefficients ------------
        rec = self.rec_head(self.basis_rec(rec_v_dt)).squeeze(-1)     # [B, C]
        logit = self.coef_geo * geo + self.coef_rec * rec
        if self.use_pair_features:
            pair = self.pair_head(self.basis_pair(pair_dt)).squeeze(-1)  # [B, C]
            logit = (logit + self.coef_pair * pair
                     + self.coef_pair_count * pair_count_log)
        return logit
