"""Velocity link head — one module owning the node embeddings AND the scoring head.

E rows live on the unit sphere as a ``geoopt.ManifoldParameter`` on ``geoopt.Sphere()``, kept
unit-norm by RiemannianAdam (see trainer.py). E is link-trained (the link loss is the only
gradient path into it); the head weights are Euclidean. There is no separate embedding table — the
head owns ``self.E`` and every embedding lookup (E_u, E_v, token embeddings) reads ``self.E.weight``.

Scoring (COUNT-FREE, one-sided drift EXTRAPOLATION): builds, for the source ONLY, a weighted free
LINE through u's walk-token trajectory in T_{E[u]} and scores each candidate by how close its
STATIC embedding E[v] is to the line evaluated at the QUERY time — an extrapolation, not an average.
Per query, over the flat token bag (the K walks pooled; see flatten_and_exclude_seed):

  v̄ = Σ_p softmax_p(−λ·age_p)·Log_{E[u]}(E[node_p])    recency CENTROID (the identity prediction)
  μ  = v̄ − b·s̄  with b the weighted slope, s = −age   the LINE at the query time (s = 0)
  q_u = exp_{E[u]}(μ)                                  u's extrapolated position, back ON the sphere

The candidate side samples NO walks — there is no μ_v. q_u is u's trajectory pushed off E[u]; the
inner product with the unit E[v] asks "how close is v to where u's neighbourhood is HEADING?".
The score is identity + velocity:

  identity = −α·‖ Log_{E[u]}(E_v) − v̄ ‖                        is v in u's region?         [B,C]
  velocity = ⟨ exp_{E[u]}(μ), E_v ⟩                            does u's drift extrapolate to v?  [B,C]
  logit = coef_identity·identity + coef_velocity·velocity

Identity is anchored on the CENTROID v̄ (the proven recurrence baseline); only the velocity channel
uses the extrapolated μ. coef_velocity init 0 ⇒ at init the head IS the centroid-identity baseline
and velocity earns its weight from zero. Degenerate time (one distinct timestamp / single token)
⇒ slope b = 0 ⇒ μ = v̄ ⇒ velocity collapses onto the centroid; it only diverges where there is
genuine temporal spread.

TOKEN BASIS — COUNT-FREE: a node recurring k times is k tokens, summed automatically; μ is
scale-invariant in time (λ self-scales). E stays the single sphere parameter (link-trained, no
detach).
"""
import math

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens, flatten_and_exclude_seed


class SphereManifold:
    """Unit-sphere geometry, behind a 5-method contract so the head is manifold-agnostic.

    The head does ALL of its scoring in the tangent space (centroid, WLS line, isotropic distance);
    the only geometry it touches is this contract:

        manifold        the geoopt manifold for E's ManifoldParameter (RiemannianAdam retraction)
        proj(x)         map an arbitrary vector onto the manifold (feasible-init + pre-score projection)
        logmap(p, x)    Log_p(x): tangent vector at p pointing to x
        expmap(p, δ)    Exp_p(δ): move off p along tangent δ, back onto the manifold
        similarity(a,b) natural "closeness" of two on-manifold points, HIGHER = closer

    To swap geometry, provide another class with these five members. `similarity` is the one
    semantic hook: the sphere returns the inner product ⟨a,b⟩ (= cosine, since points are unit),
    whereas a distance manifold (Poincaré/Lorentz/Euclidean) would return −dist(a,b). The bodies
    below are the exact primitives the head used inline — this refactor changes call sites only,
    not numerics.
    """
    eps = 1e-6

    def __init__(self):
        self.manifold = geoopt.Sphere()

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def logmap(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        c = (p * x).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        theta = torch.arccos(c)
        orth = x - c * p
        return theta * orth / orth.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def expmap(self, p: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """exp_p(δ) = cos‖δ‖·p + sin‖δ‖·δ/‖δ‖ ∈ S^{d-1}. δ = μ is a sum of Log_p's (all ⊥ p),
        so the result is unit-norm; ‖δ‖ capped to the injectivity radius (<π); δ→0 ⇒ exp_p(δ)=p
        (a cold node drifts nowhere). Final proj is a numeric belt."""
        norm = delta.norm(dim=-1, keepdim=True)                     # ‖μ‖ = drift angle
        theta = norm.clamp(max=math.pi - self.eps)
        coef = torch.sin(theta) / norm.clamp_min(self.eps)         # → 0 as norm→0 (q→p)
        return self.proj(torch.cos(theta) * p + coef * delta)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(-1)


class VelocityHead(nn.Module):
    """Sphere node embeddings + one-sided centroid/velocity drift head, in a single module."""

    def __init__(self, num_nodes: int, d_emb: int, t_train: float = 1.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.eps = 1e-6

        # --- geometry: all manifold ops go through this (swap the class to swap the space) --------
        self.geom = SphereManifold()

        # --- node embeddings: the head OWNS the table (unit sphere, link-trained) ----------------
        self.E = nn.Embedding(num_nodes, d_emb)
        nn.init.normal_(self.E.weight, mean=0.0, std=1.0 / math.sqrt(d_emb))
        # Feasible init: project every row onto the manifold (RiemannianAdam assumes the parameter
        # starts on the manifold).
        with torch.no_grad():
            w = self.E.weight.data
            w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.E.weight = geoopt.ManifoldParameter(w, manifold=self.geom.manifold)

        # --- head parameters (Euclidean) ---------------------------------------------------------
        # Shared μ recency λ (softmax, scale-invariant), init λ ≈ C/t_train so λ·age~O(1).
        lam0 = 10.0 / max(float(t_train), 1.0)
        self.log_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(lam0))], dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(10.0))     # shared distance weight

        # --- geometric mix coefficients ----------------------------------------
        # identity is the proven baseline (init 1); velocity earns weight (init 0).
        self.coef_identity = nn.Parameter(torch.ones(1))
        # VELOCITY — ⟨exp_{E[u]}(μ_line), E_v⟩: does u's drift EXTRAPOLATION reach v? coef init 0.
        self.coef_velocity = nn.Parameter(torch.zeros(1))

    # ──────────────────────────────────────────────────────────────────
    # μ — recency center of mass over a token bag (frame-agnostic)
    # ──────────────────────────────────────────────────────────────────

    def _centroid_and_line(self, base_emb: torch.Tensor, ids: torch.Tensor, nmask: torch.Tensor,
                           ages: torch.Tensor):
        """Recency-softmax CENTROID v̄ (identity prediction) AND the weighted free-LINE μ
        extrapolated to the query time s = 0 (velocity prediction), over one flat token bag.
        base [...,d] ; ids/nmask/ages [...,U]  ->  v̄ [...,d], μ [...,d]. Token embeddings come
        from the head's own table ``self.E.weight``.

          w_p = softmax_p(−λ·age_p)                              recency weights (Σ = 1, or 0 if cold)
          v̄   = Σ_p w_p · g_p          (g_p = Log_base(E[node_p]))   the centroid
          b   = Σ w·(s−s̄)·g / Σ w·(s−s̄)²   (s = −age, query at s = 0)  the WLS slope
          μ   = v̄ − b·s̄                                          the line at s = 0

        μ is SCALE-INVARIANT in s (b·s̄ is unit-free), so raw ages give the same line as any time
        unit. Count-free: a node recurring k times is k tokens, summed automatically. Cold (all
        masked) → w = 0 → v̄ = μ = 0 (exact)."""
        base = self.geom.proj(base_emb)
        ew = self.geom.proj(F.embedding(ids.clamp_min(0), self.E.weight))       # [...,U,d]
        g = self.geom.logmap(base.unsqueeze(-2), ew)                       # [...,U,d]
        lam = F.softplus(self.log_lambda)
        ell = (-lam * ages).masked_fill(~nmask, float("-inf"))            # [...,U]
        w = torch.nan_to_num(torch.softmax(ell, dim=-1), nan=0.0)         # [...,U]  Σ = 1 (0 if cold)

        vbar = (w.unsqueeze(-1) * g).sum(dim=-2)                          # [...,d]  centroid (Σw = 1)
        s = -ages.to(g.dtype)                                             # signed time; query at s = 0
        sbar = (w * s).sum(dim=-1)                                        # [...]    weighted mean (Σw = 1)
        ds = s - sbar.unsqueeze(-1)                                       # [...,U]
        Sss = (w * ds * ds).sum(dim=-1)                                   # [...]
        Sgv = (w.unsqueeze(-1) * ds.unsqueeze(-1) * g).sum(dim=-2)        # [...,d]
        b = Sgv / Sss.clamp_min(self.eps).unsqueeze(-1)                   # [...,d]  slope (0 if degenerate)
        mu = vbar - b * sbar.unsqueeze(-1)                               # [...,d]  line at s = 0
        return vbar, mu

    # ──────────────────────────────────────────────────────────────────
    # identity — probe an embedding against a prediction (frame-agnostic)
    # ──────────────────────────────────────────────────────────────────

    def _identity(self, frame: torch.Tensor, probe: torch.Tensor, mu: torch.Tensor,
                  alpha: torch.Tensor) -> torch.Tensor:
        """−α·‖ Log_frame(probe) − μ ‖ — ISOTROPIC tangent distance to the centroid prediction.
        All [...,d] (caller broadcasts) -> [...]."""
        nu = self.geom.logmap(frame, probe)                              # [...,d]
        delta = nu - mu
        dist = (delta * delta).sum(-1).clamp_min(self.eps).sqrt()
        return -alpha * dist

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(self,
                src_tokens: WalkTokens,        # source walk tokens — self-contained: `seeds`
                                               # ARE the sources, `cutoffs` ARE the query times
                cand_ids: torch.Tensor,        # [B, C]  candidate node ids
                ) -> torch.Tensor:
        """-> logits [B, C]. The head owns E and all embedding lookups/timing: E_u, E_v and the
        token embeddings all read ``self.E.weight`` (E_u = E[src_tokens.seeds], E_v = E[cand_ids]);
        token ages come from src_tokens.cutoffs − token times. The trainer only hands over the
        (self-contained) source walk tokens and the candidate ids. Score = identity + velocity."""
        e_weight = self.E.weight
        B, C = cand_ids.shape[0], cand_ids.shape[1]
        d = self.d_emb
        E_u = F.embedding(src_tokens.seeds, e_weight)            # [B, d]
        E_v = F.embedding(cand_ids, e_weight)                    # [B, C, d]
        eu = self.geom.proj(E_u)                                  # [B, d]
        ev = self.geom.proj(E_v)                                  # [B, C, d]
        alpha = self.alpha.clamp_min(1e-3)

        # --- predictions: centroid v̄ (identity) + line extrapolation μ (velocity) ---
        # Flatten the raw [B, K, L] walks to one [B, T] token bag, masking padding + the seed node
        # u (the K/per-walk structure is unused by this head — see flatten_and_exclude_seed). One
        # weighted free-line fit per query over the pooled tokens gives BOTH the centroid v̄ and the
        # line at the query time μ.
        ids, nmask, ages = flatten_and_exclude_seed(src_tokens)       # [B, T] each
        vbar, mu_line = self._centroid_and_line(
            E_u, ids, nmask, ages.to(eu.dtype))                      # [B,d], [B,d]
        eu_bc = eu.unsqueeze(1).expand(B, C, d)                   # [B,C,d]
        vbar_bc = vbar.unsqueeze(1).expand(B, C, d)

        # --- identity: is v in u's region? ISOTROPIC distance to the CENTROID v̄ ---
        q_ident = self._identity(eu_bc, ev, vbar_bc, alpha)      # [B,C]

        # --- VELOCITY: push u's drift to the EXTRAPOLATED point exp_{E[u]}(μ) (the free line at the
        # query time, not the centroid) back onto the sphere and inner-product it with the
        # candidate's STATIC embedding. No μ_v — the candidate side samples no walks. ⟨q_u, E_v⟩
        # asks whether v sits where u's neighbourhood is HEADING (extrapolation, not average).
        # coef_velocity init 0 ⇒ no-op at init; the head starts as the centroid-identity baseline.
        q_u = self.geom.expmap(eu, mu_line)                      # [B,d]   extrapolated source pos
        velocity = self.geom.similarity(q_u.unsqueeze(1), ev)   # [B,C]   ⟨q_u, E_v⟩

        logit = (self.coef_identity * q_ident
                 + self.coef_velocity * velocity)

        return logit
