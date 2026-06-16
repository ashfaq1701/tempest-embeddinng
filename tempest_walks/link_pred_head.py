"""Geometric link head — point version (recency-weighted mean + anisotropic ellipse)
+ co-reachability (cross) channel.

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
                 + coef_cross·cross(u,v)

The geometric distance is an ELLIPSE oriented along each source's heading r (not an
isotropic circle): the model learns a,b so being off ALONG the heading is weighted
differently from being off SIDEWAYS — on tgbl-wiki it learns a/b ≈ 1/100 ("direction
matters, distance-along-heading is ~free"). a=b recovers the plain circle ‖ν−μ‖.
The channels (geometric / recency / pair / cross) are mixed with learnable per-channel
coefficients (init 1 for the proven channels = plain sum) so the model can rebalance
them. Few scalars (α, λ, a, b, coef_*) — almost nothing to overfit; the anisotropy is
adaptive (≈isotropic where no direction signal exists).

TIME UNITS — ages are RAW (t_query − t_edge), never normalized at runtime.
  - The source-side recency μ = Σ softmax(−λ·age)·g is a SOFTMAX, which is
    scale-invariant, so raw age is exactly the proven baseline (λ self-scales).
  - The CROSS-side recency lives inside a LOGSUMEXP, which is NOT scale-invariant:
    with the source λ (init 0 ⇒ λ≈0.69) on raw wiki ages (~1e6), −λ·age ≈ −7e5, the
    cross logit swings over ~1e6, and even an init-0 coef_cross diverges on step 1.
    So the cross gets its OWN decay λ_cross, with a DATASET-DERIVED low init:
    λ_cross ≈ 0.1/t_train (train-split span) ⇒ λ_cross·age ~ O(0.1) at init on any
    dataset (no runtime normalization, no magic constant). It then learns freely.
    t_train is used ONCE at construction to set this init, not as a per-step scaler.

CO-REACHABILITY (cross) channel — for the new×both-seen cell, where the candidate v is
seen but never linked from u, so the candidate term d(ν) is blind. Instead of asking
"is v in u's region", it asks "is any recent CONNECTOR of v (a source that recently
reached v) in u's region":

  connectors   c_j = Log_{E[u]}(E[w_j])             v's recent direct neighbours, SAME tangent space
  conn dist    d(c_j) = √( a·‖δ∥‖² + b·‖δ⊥‖² ),  δ = c_j − μ     SAME μ, r, a, b, α as candidates
  cross        x(u,v) = logsumexp_j(−α·d(c_j) − λ_cross·age_j)   soft-min: the single best
                                                                  recent, in-character connector
  logit       += coef_cross·x(u,v)                  init 0 = channel off, earns its weight

Design choices that make this consistent rather than a bolted-on second model:
  - Connectors are scored against the SAME μ and the SAME ellipse (a, b, heading r, α)
    as the candidates — they are just another set of tangent vectors pushed through the
    existing geometry. No second scale/temperature (that compounding broke earlier heads).
  - Connector recency uses its OWN λ_cross (separate from μ's neighbour λ), on RAW age,
    init dataset-derived low (see TIME UNITS) and learned freely. coef_cross init 0
    keeps the channel off while it earns its weight.
  - Soft-min (logsumexp) WITHIN connectors (existential: one in-character witness fires),
    additive coef_cross BETWEEN channels. A sum over connectors would mean-pool and wash
    out the single relevant witness.
  - A candidate with no recent connectors contributes a neutral 0 (no −∞ leak).
  - No degree gate yet (deferred): cross is a plain additive channel. The gate, if added
    later, is a one-line g(deg_u)·cross in front of the cross term.

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
                 use_pair_features: bool = False, t_train: float = 1.0,
                 use_hop_weight: bool = False):
        super().__init__()
        self.d_emb = d_emb
        self.eps = 1e-6
        self.use_hop_weight = use_hop_weight

        # --- geometric channel -------------------------------------------------
        # u enters as the BASE POINT (everything relative to E[u]); no separate
        # u-vs-v term needed.
        # λ = softplus(log_lambda) ≥ 0, on RAW age inside the μ softmax (scale-
        # invariant, so λ self-scales — exactly the proven baseline; see TIME UNITS).
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

        # --- co-reachability (cross) channel -----------------------------------
        # Reuses μ and the ellipse (a, b, heading r, α) from the geometric channel.
        # Two new params:
        #   coef_cross : channel gain, init 0 ⇒ starts as the proven baseline.
        #   log_lambda_cross : SEPARATE connector-recency decay (not the source λ),
        #     on RAW age. Its init is DATASET-DERIVED — λ_cross ≈ 0.1/T_train, so
        #     λ_cross·age ~ O(0.1) at init on any dataset (low recency, distance-led;
        #     no runtime age-normalization, no magic constant). It then learns freely.
        self.coef_cross = nn.Parameter(torch.zeros(1))
        lam0 = 0.1 / max(float(t_train), 1.0)
        self.log_lambda_cross = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))

        # --- hop-distance (κ) weighting on the CONNECTOR side (optional) --------
        # Down-weight a connector by its HOP distance from the candidate v inside the
        # cross logsumexp: penalty −κ·(hop−1), so a 1-hop (direct) connector pays 0, a
        # 2-hop pays −κ, a 3-hop −2κ. κ = softplus(log_kappa) ≥ 0, learnable. This is
        # a SOFT, continuous length sweep on the (free-length) connector set: κ→∞ ⇒
        # only 1-hop counts (≡ candidate len-2); κ→0 ⇒ all hops equal (≡ full len-L).
        # The gradient down-weighting is automatic — a deeper connector's larger
        # penalty lowers its logsumexp softmax share, so it gets a smaller gradient.
        # Init log_kappa=0 ⇒ κ≈0.69 (moderate shallow-bias, hop-2 penalty O(0.69)
        # ~ the O(1) distance term), then learns freely. CONNECTOR-SIDE ONLY: the
        # query-side μ pool is already a recency-weighted mean and recency≈hop in
        # backward temporal walks, so a query-side hop term was inert on wiki (A/B:
        # within ~0.0015). Gated by use_hop_weight so off ⇒ baseline byte-identical.
        if use_hop_weight:
            self.log_kappa = nn.Parameter(torch.zeros(1))

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

    def _cross(self, eu: torch.Tensor, mu: torch.Tensor, r: torch.Tensor,
               a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor,
               conn_ids: torch.Tensor, conn_age: torch.Tensor,
               conn_hop: torch.Tensor, conn_mask: torch.Tensor,
               e_weight: torch.Tensor) -> torch.Tensor:
        """Soft-min co-reachability logit per candidate.

        Connectors are v's recent walk-neighbours (sources that reached v), scored
        against u's SAME μ / ellipse as the candidates, with a SEPARATE recency decay
        λ_cross (on RAW age; init dataset-derived low, then learned). The soft-min
        picks the single best recent, in-character connector; candidates with no
        valid connector contribute a neutral 0.

           eu        [B, d]        unit source embedding (tangent base point)
           mu, r     [B, d]        predicted position and heading (from neighbours)
           conn_ids  [B, C, M]     connector NODE IDS (M per candidate; −1 at padded slots)
           conn_age  [B, C, M]     connector age = t_query − t_edge ≥ 0 (RAW units)
           conn_hop  [B, C, M]     hop distance from v (1 = direct neighbour, 2, 3, …);
                                   used only when use_hop_weight (penalty −κ·(hop−1))
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
        lam_cross = F.softplus(self.log_lambda_cross)
        lw = -alpha * d_conn - lam_cross * conn_age
        if self.use_hop_weight:
            # −κ·(hop−1): 1-hop pays 0, 2-hop −κ, 3-hop −2κ … (deeper = down-weighted)
            lw = lw - F.softplus(self.log_kappa) * (conn_hop - 1).clamp_min(0.0)
        lw = lw.masked_fill(~conn_mask, -1e9)
        cross = torch.logsumexp(lw, dim=-1)                       # [B, C]
        return torch.where(conn_mask.any(dim=-1), cross, torch.zeros_like(cross))

    def forward(self, tok_emb: torch.Tensor, tok_age: torch.Tensor,
                tok_mask: torch.Tensor, E_u: torch.Tensor, E_v: torch.Tensor,
                rec_v_log: torch.Tensor,
                conn_ids: torch.Tensor, conn_age: torch.Tensor,
                conn_hop: torch.Tensor, conn_mask: torch.Tensor,
                e_weight: torch.Tensor,
                pair_rec_log: torch.Tensor = None,
                pair_ever: torch.Tensor = None,
                pair_count_log: torch.Tensor = None) -> torch.Tensor:
        """tok_emb [B,n,d]  source walk-neighbour embeddings (context only).
           tok_age [B,n]    age = t_query − t_node per token (≥0, RAW units).
           tok_mask[B,n]    bool, True at real neighbour positions.
           E_u     [B,d]    source embeddings (tangent-space BASE POINT).
           E_v     [B,C,d]  candidate embeddings.
           rec_v_log [B,C]  log1p(t_query − t_last[v]) candidate recency.
           conn_ids [B,C,M]    v's connector NODE IDS (gathered from e_weight).
           conn_age [B,C,M]    connector edge age = t_query − t_edge (≥0, RAW units).
           conn_hop [B,C,M]    connector hop distance from v (1 = direct); penalty
                               −κ·(hop−1) in the cross logsumexp when use_hop_weight.
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
        rec = self.rec_head(self.t2v_rec(rec_v_log)).squeeze(-1)  # [B, C]
        logit = self.coef_geo * geo + self.coef_rec * rec
        if self.use_pair_features:
            feat = torch.cat(
                [self.t2v_pair(pair_rec_log),
                 pair_ever.unsqueeze(-1), pair_count_log.unsqueeze(-1)],
                dim=-1)
            pair = self.pair_head(feat).squeeze(-1)               # [B, C]
            logit = logit + self.coef_pair * pair

        # --- co-reachability (cross) channel (same μ / ellipse; own λ_cross, raw age) -
        cross = self._cross(eu, mu, r, a, b, alpha,
                            conn_ids, conn_age, conn_hop, conn_mask, e_weight)  # [B, C]
        logit = logit + self.coef_cross * cross
        return logit
