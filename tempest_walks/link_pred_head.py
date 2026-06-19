"""Geometric link head — point version (recency-weighted mean + anisotropic ellipse)
+ co-reachability (∃-witness) channel.

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
                 + coef_coreach·coreach(u,v)

The geometric distance is an ELLIPSE oriented along each source's heading r (not an
isotropic circle): the model learns a,b so being off ALONG the heading is weighted
differently from being off SIDEWAYS — on tgbl-wiki it learns a/b ≈ 1/100 ("direction
matters, distance-along-heading is ~free"). a=b recovers the plain circle ‖ν−μ‖.
The channels (geometric / recency / pair / co-reachability) are mixed with learnable per-channel
coefficients (init 1 for the proven channels = plain sum) so the model can rebalance
them. Few scalars (α, λ, a, b, coef_*) — almost nothing to overfit; the anisotropy is
adaptive (≈isotropic where no direction signal exists).

TIME UNITS — ages are RAW (t_query − t_edge), never normalized at runtime.
  - The source-side recency μ = Σ softmax(−λ·age)·g is a SOFTMAX, which is
    scale-invariant, so raw age is exactly the proven baseline (λ self-scales).
  - The CO-REACH-side recency lives inside a LOGSUMEXP, which is NOT scale-invariant:
    with a normal rate on raw wiki ages (~1e6), −rate·age ≈ −7e5, the co-reach logit
    swings over ~1e6 and even an init-0 coef_coreach diverges on step 1. So co-reach
    gets its OWN decay ρ = exp(log_rate_coreach) on RAW age, init log_rate_coreach =
    −log(t_train) ⇒ ρ ≈ 1/t_train ⇒ ρ·age ~ O(1) at init (bounded). The
    exp-log-rate parameterization is scale-free & well-conditioned (∂z/∂log_rate = z);
    t_train sets the init, it is NOT a per-step scaler. (μ keeps softplus(log_lambda)
    on raw age — the proven argmax-init baseline.)

CO-REACHABILITY (∃-witness) channel — a soft, temporal, geometric COMMON-NEIGHBOUR
signal. For the new×both-seen cell, where the candidate v is seen but never linked from
u, the candidate term d(ν) is blind. Instead of asking "is v in u's region", it asks "is
any recent CONNECTOR of v (a source that recently reached v) in u's region":

  connectors   c_j = Log_{E[u]}(E[w_j])             v's recent direct neighbours, SAME tangent space
  conn dist    d(c_j) = √( a·‖δ∥‖² + b·‖δ⊥‖² ),  δ = c_j − μ     SAME μ, r, a, b, α as candidates
  coreach      x(u,v) = logsumexp_j(−α·d(c_j) − ρ·age_j)         ∃-witness (soft-min): the single
                                                                  best recent, in-region connector
  logit       += coef_coreach·x(u,v)                init 0 = channel off, earns its weight

This is the classic common-neighbour heuristic SOFTENED into geometry: proximity-in-
prediction-space instead of an exact shared-node count, and an EXISTENTIAL soft-min (one
witness suffices) instead of a sum/count. Design choices that make it consistent rather
than a bolted-on second model:
  - Connectors are scored against the SAME μ and the SAME ellipse (a, b, heading r, α)
    as the candidates — they are just another set of tangent vectors pushed through the
    existing geometry. No second scale/temperature (that compounding broke earlier heads).
  - Connector recency uses its OWN ρ = exp(log_rate_coreach) (separate from μ's λ),
    on RAW age, init −log(t_train) (see TIME UNITS). coef_coreach init 0 keeps the channel
    off while it earns its weight.
  - Soft-min (logsumexp) WITHIN connectors (existential: one in-region witness fires),
    additive coef_coreach BETWEEN channels. A sum over connectors would mean-pool and wash
    out the single relevant witness.
  - A candidate with no recent connectors contributes a neutral 0 (no −∞ leak).
  - Co-reach is a plain additive channel. Two conditioning mechanisms were tried and
    REJECTED on tgbl-wiki (2×2 {gate}×{hop} grid, all cells within ~0.002 test of
    baseline = inside the 0.015 noise band): a learnable source-DEGREE gate
    g(deg_u)·coreach (to silence co-reach on warm/repeat sources) and a connector
    HOP-distance penalty −κ·(hop−1) in the logsumexp (a soft candidate-walk-length
    sweep; longer reach was net-neutral-to-negative). The gate can't move wiki — its
    target cold/new-pair slice is only ~13% of the data — so both are deferred to a
    cold-start workload (review), where the co-reach channel is designed to matter.

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


class ExpDecayBasis(nn.Module):
    """Multi-rate exponential-decay staleness encoder: φ(Δt) = [exp(−ρ_k·Δt)]_{k=1..K}.
    ρ_k = exp(log_rates_k) > 0 learnable, log-spaced init from 1/t_train (train-span
    scale) to 1 (most-recent scale). The Hawkes/TPP recency feature — scale-free on RAW
    Δt, bounded [0,1], monotone, multi-timescale — feeding the rec/pair channels. A
    never-seen event (Δt → +inf, encoded as a huge value) maps to φ = 0 for free."""

    def __init__(self, dim: int, t_train: float):
        super().__init__()
        r_lo = -math.log(max(float(t_train), 1.0))    # ρ ≈ 1/t_train
        r_hi = 0.0                                    # ρ ≈ 1
        self.log_rates = nn.Parameter(torch.linspace(r_lo, r_hi, dim))   # [K]

    def forward(self, dt: torch.Tensor) -> torch.Tensor:   # dt [...]  raw Δt ≥ 0
        return torch.exp(-torch.exp(self.log_rates) * dt.unsqueeze(-1))  # [..., K]


class GeometricPointHead(nn.Module):
    def __init__(self, d_emb: int, d_time: int = 16,
                 use_pair_features: bool = False, t_train: float = 1.0):
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6

        # --- geometric channel -------------------------------------------------
        # u enters as the BASE POINT (everything relative to E[u]); no separate
        # u-vs-v term needed.
        # λ = softplus(log_lambda) ≥ 0, on RAW age inside the μ softmax. Init λ ≈ C/t_train
        # (C=10) — NOT log_lambda=0: at zero-init λ = softplus(0) = 0.693 on raw ages makes
        # λ·age enormous, so the softmax saturates to a hard argmax and ∂loss/∂λ → 0 (λ pinned,
        # μ frozen as the single most-recent neighbour). Scaling the init by 1/t_train puts
        # λ·age ~ O(1) so λ actually TRAINS. (Same fix already baked into log_rate_coreach below,
        # which inits at −log(t_train) ⇒ ρ≈1/t_train ⇒ ρ·age O(1); only log_lambda was unscaled.)
        lam0 = 10.0 / max(float(t_train), 1.0)
        self.log_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))
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
        # rec channel: candidate v's own recency, exp-decay basis on RAW Δt_v.
        self.basis_rec = ExpDecayBasis(d_time, t_train)
        self.rec_head = nn.Linear(d_time, 1)
        self.use_pair_features = use_pair_features
        if use_pair_features:
            # pair channel (FLAGGED): (u,v) last-interaction recency, exp-decay basis on
            # RAW Δt_uv. Never-seen pair → Δt huge → basis 0 (no ever-bit/count needed).
            self.basis_pair = ExpDecayBasis(d_time, t_train)
            self.pair_head = nn.Linear(d_time, 1)

        # --- learnable per-channel mix coefficients (init 1 = plain sum) --------
        # One learnable gain per channel (on the raw channels) so the model can
        # rebalance the geometric vs recency vs pair terms rather than a fixed
        # sum. +0.005 val / +0.009 test on tgbl-wiki (with pair features on).
        self.coef_geo = nn.Parameter(torch.ones(1))
        self.coef_rec = nn.Parameter(torch.ones(1))
        if use_pair_features:
            self.coef_pair = nn.Parameter(torch.ones(1))
            # (u,v) interaction-COUNT term: coef_pair_count · log1p(count). A separate
            # learnable scalar gain (init 0 ⇒ off, earns its weight). Never-seen pair →
            # count=0 → log1p=0 → no contribution (same clean baseline as the recency).
            self.coef_pair_count = nn.Parameter(torch.zeros(1))

        # --- co-reachability (∃-witness) channel -------------------------------
        # Reuses μ and the ellipse (a, b, heading r, α) from the geometric channel.
        # Two new params:
        #   coef_coreach : channel gain, init 0 ⇒ starts as the proven baseline.
        #   log_rate_coreach : SEPARATE connector-recency decay ρ = exp(log_rate)
        #     on RAW age. init −log(t_train) ⇒ ρ ≈ 1/t_train ⇒ ρ·age ~ O(1) at init,
        #     so the (non-scale-invariant) co-reach logsumexp stays bounded — same
        #     exp-log-rate idiom as ExpDecayBasis, replacing the softplus(λ)+0.1/t_train
        #     hack. Same recency family, just a cleaner well-conditioned parameterization.
        self.coef_coreach = nn.Parameter(torch.zeros(1))
        self.log_rate_coreach = nn.Parameter(
            torch.tensor([-math.log(max(float(t_train), 1.0))], dtype=torch.float32))

    def _logmap(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Sphere log-map at base point p (closed form = geoopt.Sphere().logmap)."""
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def _ellipse_dist_sq(self, delta: torch.Tensor, r: torch.Tensor,
                         a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Anisotropic ellipse distance² in the per-source heading frame.
           delta [B, ..., d]  displacement ν−μ (or c−μ);  r [B, d] heading.
           Broadcasts r over delta's leading axes after B.  -> [B, ...] (no d axis)."""
        extra = delta.dim() - 2                      # axes between B and d
        r_b = r.view(r.shape[0], *([1] * extra), r.shape[-1])
        dpar2 = (delta * r_b).sum(-1).pow(2)         # [B, ...]
        dperp2 = ((delta * delta).sum(-1) - dpar2).clamp_min(0.0)
        return a * dpar2 + b * dperp2                # [B, ...]

    def _coreach(self, eu: torch.Tensor, mu: torch.Tensor, r: torch.Tensor,
                 a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor,
                 conn_ids: torch.Tensor, conn_age: torch.Tensor,
                 conn_mask: torch.Tensor, e_weight: torch.Tensor) -> torch.Tensor:
        """Co-reachability (∃-witness) logit per candidate — a soft, temporal, geometric
        common-neighbour signal: does v have ONE recent connector that falls in u's
        predicted region?

        Connectors are v's recent walk-neighbours (sources that reached v), scored
        against u's SAME μ / ellipse as the candidates, with a SEPARATE recency decay
        ρ = exp(log_rate_coreach) (on RAW age; init −log(t_train)). The soft-min
        (logsumexp) is EXISTENTIAL — it returns the single best recent, in-region
        connector (the "witness"), not an average; candidates with no valid connector
        contribute a neutral 0.

           eu        [B, d]        unit source embedding (tangent base point)
           mu, r     [B, d]        predicted position and heading (from neighbours)
           conn_ids  [B, C, M]     connector NODE IDS (M per candidate; −1 at padded slots)
           conn_age  [B, C, M]     connector age = t_query − t_edge ≥ 0 (RAW units)
           conn_mask [B, C, M]     bool, True at a real connector
           e_weight  [N, d]        embedding matrix (connectors gathered from it)
           -> [B, C]

        SINGLE PASS — no chunking, no checkpointing: the full [B,C,M,d] connector
        activation is built in one shot, so peak memory scales with B·C·M·d and never
        the speed. This is deliberately MEMORY-DEPENDENT: a larger GPU supports longer
        candidate walks (larger M); a smaller GPU OOMs at high lengths. To go longer,
        use a bigger GPU or fewer/shorter candidate walks (K_cand·L_cand)."""
        ec = F.normalize(F.embedding(conn_ids.clamp_min(0), e_weight), dim=-1)  # [B,C,M,d]
        gc = self._logmap(eu[:, None, None, :], ec)               # [B, C, M, d]
        delta = gc - mu[:, None, None, :]
        d_conn = self._ellipse_dist_sq(delta, r, a, b).clamp_min(self.eps).sqrt()  # [B,C,M]
        rho = torch.exp(self.log_rate_coreach)
        lw = (-alpha * d_conn - rho * conn_age).masked_fill(~conn_mask, -1e9)
        coreach = torch.logsumexp(lw, dim=-1)                     # [B, C]
        return torch.where(conn_mask.any(dim=-1), coreach, torch.zeros_like(coreach))

    def forward(self, tok_emb: torch.Tensor, tok_age: torch.Tensor,
                tok_mask: torch.Tensor, E_u: torch.Tensor, E_v: torch.Tensor,
                rec_v_dt: torch.Tensor,
                conn_ids: torch.Tensor, conn_age: torch.Tensor,
                conn_mask: torch.Tensor, e_weight: torch.Tensor,
                pair_dt: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """tok_emb [B,n,d]  source walk-neighbour embeddings (context only).
           tok_age [B,n]    age = t_query − t_node per token (≥0, RAW units).
           tok_mask[B,n]    bool, True at real neighbour positions.
           E_u     [B,d]    source embeddings (tangent-space BASE POINT).
           E_v     [B,C,d]  candidate embeddings.
           rec_v_dt [B,C]   RAW Δt = t_query − t_last[v] candidate recency (→ basis_rec).
           conn_ids [B,C,M]    v's connector NODE IDS (gathered from e_weight).
           conn_age [B,C,M]    connector edge age = t_query − t_edge (≥0, RAW units).
           conn_mask[B,C,M]    bool, True at a real connector.
           e_weight [N,d]      embedding matrix (for the connector gather).
           -> logits [B, C].
        """
        eu = F.normalize(E_u, dim=-1)                              # [B, d]
        ev = F.normalize(E_v, dim=-1)                              # [B, C, d]
        ew = F.normalize(tok_emb, dim=-1)                          # [B, n, d]

        # --- map neighbours + candidates into T_{E[u]} -------------------------
        g = self._logmap(eu.unsqueeze(1), ew)                     # [B, n, d]
        nu = self._logmap(eu.unsqueeze(1), ev)                    # [B, C, d]

        # --- recency-weighted predicted position μ (RAW age; softmax is scale-
        # invariant so λ self-scales — exactly the proven baseline) -------------
        lam = F.softplus(self.log_lambda)
        wlog = (-lam * tok_age).masked_fill(~tok_mask, float("-inf"))   # [B, n]
        w = torch.nan_to_num(torch.softmax(wlog, dim=-1), nan=0.0)      # cold src -> all 0
        mu = (w.unsqueeze(-1) * g).sum(dim=1)                      # [B, d]

        # --- distance: anisotropic ellipse in the per-source heading frame ------
        r = mu / mu.norm(dim=-1, keepdim=True).clamp_min(self.eps)   # [B, d] heading
        a = F.softplus(self.log_a)
        b = F.softplus(self.log_b)
        alpha = self.alpha.clamp_min(1e-3)

        delta = nu - mu.unsqueeze(1)                              # [B, C, d]
        d = self._ellipse_dist_sq(delta, r, a, b).clamp_min(self.eps).sqrt()   # [B, C]
        # cold/degenerate μ≈0 ⇒ r→0 ⇒ d → √b·‖ν‖ = geodesic(E[u],E[v]).

        # --- channels combined with learnable per-channel coefficients ---------
        geo = -alpha * d                                          # [B, C]
        rec = self.rec_head(self.basis_rec(rec_v_dt)).squeeze(-1)  # [B, C]
        logit = self.coef_geo * geo + self.coef_rec * rec

        # --- co-reachability (∃-witness) channel (same μ / ellipse; own ρ, raw age) ---
        coreach = self._coreach(eu, mu, r, a, b, alpha,
                                conn_ids, conn_age, conn_mask, e_weight)  # [B, C]
        logit = logit + self.coef_coreach * coreach

        if self.use_pair_features:
            pair = self.pair_head(self.basis_pair(pair_dt)).squeeze(-1)  # [B, C]
            logit = (logit + self.coef_pair * pair
                     + self.coef_pair_count * pair_count_log)

        return logit
