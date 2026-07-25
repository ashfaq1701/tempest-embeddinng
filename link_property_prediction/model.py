"""Link head — sphere node embeddings + a deep neighbourhood-projection channel, in one module.

    logit = <P[u], E[v]>                               is v near u's neighbourhood?

where P[u] = exp_{E[u]}(mu_u) pushes the source off E[u] toward mu_u — the deep pooling of u's
walk-token tangents (in the tangent space at E[u]) + a TPNet Time2Vec of each token's age, produced
by NeighborhoodProjection. One-sided: only u is walked/projected; each candidate v enters through
its static embedding E[v]. The head owns self.E (a ManifoldParameter on a geoopt.Sphere, link-
trained on it); geometry goes through self.geom (SphereManifold): logmap proxies to geoopt, expmap and
similarity are ours (geoopt's expmap NaNs the gradient at the cold-row zero tangent).

(The dual-sided variant — walk every candidate too and score <P[u], P[v]> — was falsified on wiki:
at matched walks it lost to one-sided at ~8x the cost. It lives one `git revert` away; see the
"Important: revert this commit to bring back dual side walks" commit.)
"""
import math

import geoopt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens, flatten_tokens


class SphereManifold:
    """Unit-sphere geometry over geoopt.Sphere. dist and logmap PROXY straight to geoopt; expmap is
    OURS — geoopt's Sphere.expmap uses sin(‖u‖)/‖u‖ under a torch.where and leaks a NaN gradient at
    ‖u‖ = 0 (the cold-row tangent mu = 0, a node with no walk tokens — common on wiki). E is kept
    on-sphere by RiemannianAdam, so no read-time re-projection is needed."""

    def __init__(self):
        self.manifold = geoopt.Sphere()

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Sphere similarity: inner product ⟨a, b⟩ (= cosine for unit vectors). HIGHER = closer."""
        return (a * b).sum(-1)

    def logmap(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """Log map at `base` of `point` — geoopt.Sphere.logmap (its gradient is finite at coincidence)."""
        return self.manifold.logmap(base, point)

    def expmap(self, base: torch.Tensor, tangent: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Exponential map: move from `base` along `tangent`.  exp(base, u) = cos(‖u‖)·base +
        sinc(‖u‖/π)·u.  torch.sinc is a smooth primitive (no explicit /‖u‖), so ‖u‖ = 0 is an exact
        differentiable no-op — returns `base` with a FINITE gradient. (geoopt's expmap NaNs the
        gradient there.) The squared norm is clamped off 0 before the sqrt for extra safety."""
        angle = (tangent ** 2).sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()   # ‖u‖, finite grad at 0
        return torch.cos(angle) * base + torch.sinc(angle / math.pi) * tangent


class TimeEncoder(nn.Module):
    """Time2Vec time encoding, ported verbatim from TPNet (TGB_TPNet/models/modules.py)."""

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class NeighborhoodProjection(nn.Module):
    """DISPLACED-tangent attention pooling of a source's walk-token tangents into one tangent vector
    mu_u at E[u]. Single-head.

    Generalises the old "rescale-only" centroid (mu_u = Σ_p w_p·g_p, value = the RAW tangent
    g_p = Log_{E[u]}(E[token_p]), so the only learnable per token was the scalar attention weight —
    mu_u was trapped in the conic hull of the FIXED neighbour tangents) by giving the VALUE a learned
    displacement:

        v_p = g_p + γ ⊙ ( W_v · g_p  +  Enc([Time2Vec(age_p), log1p(hop_p), edge_p]) )

    a content-LINEAR reshape of the tangent (W_v) plus a DEEP displacement driven only by the STABLE
    features (Enc). Depth lives on the stable path on purpose — a 2-layer projection on the
    embedding/code path overfit on wiki; width/linear is the safe lever there. γ is a per-channel
    LayerScale init 0, so at start v == g (displacement OFF) and the module is a plain tangent
    centroid, earning capacity channel-by-channel. A seed-conditioned query (decoupled from the keys)
    attends over the tokens; mu_u = Σ_p w_p · v_p.

    Sphere validity: each value v_p (and mu_u) is a FREE R^d vector (the displacement leaves the
    tangent subspace); mu_u is projected ONCE onto the tangent space at E[u] (mu -= ⟨mu, E[u]⟩·E[u]),
    which is all exp_{E[u]} needs — the standard retraction / extrinsic-mean pattern (compute freely
    off the tangent space, project back once). Project-once retains strictly MORE information than
    per-step projection (see researches/off-tangent-intermediate-values-and-project-once.txt). mu is
    then radially clamped below π so exp_{E[u]}(mu) stays injective. Candidate-independent (never sees
    E[v]); cold rows (no token) -> weights 0 -> mu_u = 0 -> P[u] = E[u].

    (The multi-head variant — H decoupled heads + a W_o head-combine — is at commit 889c59f. On coin
    it bought only ~+0.004 val/test over this single-head displacement at 3x the head params and
    ~1.5x the compute; recover it from there if the multi-head lever is wanted again.)
    """

    def __init__(self, d_emb: int, t2v_dim: int = 16, d_ef: int = 0):
        super().__init__()
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.d_a = d_emb // 2                           # attention (query/key) dim
        self.scale = 1.0 / math.sqrt(self.d_a)
        self.max_radius = math.pi - 1e-2               # keep ‖mu‖ < π so exp_{E[u]} stays injective

        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        s_in = t2v_dim + 1 + d_ef                      # stable-feature dim: [Time2Vec, log-hop, edge]
        desc_in = d_emb + s_in                         # full token descriptor: [tangent, stable]

        # Attention: DECOUPLED query (from the seed) and key (from the token descriptor).
        self.w_q = nn.Linear(desc_in, self.d_a)
        self.w_k = nn.Linear(desc_in, self.d_a)

        # Value displacement:
        #   content path — LINEAR reshape of the tangent (depth here overfits; keep it linear);
        #   stable  path — DEEP (safe: driven only by time / hop / edge).
        self.w_v = nn.Linear(d_emb, d_emb, bias=False)
        self.disp_enc = nn.Sequential(
            nn.Linear(s_in, d_emb), nn.GELU(), nn.Linear(d_emb, d_emb))
        self.gamma = nn.Parameter(torch.zeros(d_emb))    # LayerScale, init 0 -> displacement off

    def forward(self, source: torch.Tensor, token_tangents: torch.Tensor,
                ages: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """source [B,d_emb] (E[u], unit-norm); token_tangents [B,T,d_emb] (Log_{E[u]}(E[token]), ⊥ E[u]);
        ages [B,T]; mask [B,T] bool (True = real token); positions [B,T] int hop-from-seed (1=seed,
        0=pad); edge_features [B,T,d_ef]. Returns mu_u [B,d_emb], a tangent at E[u] (⊥ E[u], ‖·‖<π);
        cold rows (no token) -> 0."""
        b = source.shape[0]

        # TPNet scales delta-times by log(Δt + 1) before the time encoder.
        t2v = self.time_encoder(torch.log1p(ages.clamp_min(0.0)))                  # [B,T,t2v_dim]
        log_hop = torch.log1p(positions.clamp_min(0).to(t2v.dtype)).unsqueeze(-1)  # [B,T,1]
        stable = torch.cat([t2v, log_hop, edge_features], dim=-1)                  # [B,T,s_in]

        # ── attention (decoupled Q/K) ──────────────────────────────────────────────────────────
        keys = self.w_k(torch.cat([token_tangents, stable], dim=-1))              # [B,T,d_a]
        # Query = the seed: content = E[u], stable features (age/hop/edge) zero.
        query = self.w_q(torch.cat([source, stable.new_zeros(b, stable.shape[-1])], dim=-1))  # [B,d_a]
        scores = (query.unsqueeze(1) * keys).sum(-1) * self.scale                 # [B,T]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)        # [B,T]; cold row -> 0

        # ── displaced values ───────────────────────────────────────────────────────────────────
        disp = self.w_v(token_tangents) + self.disp_enc(stable)                   # [B,T,d]
        values = token_tangents + self.gamma * disp                               # [B,T,d]; γ init 0

        # ── pool -> project onto T_{E[u]} -> radial clamp ──────────────────────────────────────
        mu = (weights.unsqueeze(-1) * values).sum(dim=-2)                         # [B,d]
        mu = mu - (mu * source).sum(-1, keepdim=True) * source                    # Π_{E[u]}: ⊥ E[u]
        norm = (mu * mu).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()            # finite grad at 0
        return mu * (self.max_radius / norm).clamp(max=1.0)                       # ‖mu‖ ≤ max_radius


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int,
                 t2v_dim: int = 16, d_ef: int = 0):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.geom = SphereManifold()

        # E lives on the unit sphere: init uniformly at random on it (geoopt.Sphere.random_uniform),
        # then wrap as a ManifoldParameter so RiemannianAdam keeps it there.
        self.E = nn.Embedding(num_nodes, d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random_uniform(num_nodes, d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        self.neighbourhood = NeighborhoodProjection(
            d_emb=d_emb, t2v_dim=t2v_dim, d_ef=d_ef)

        # Combiner MLP over the 4 pairwise SIMILARITIES (inner products = cosines) on the sphere
        # between both sides' identity (E[x]) and neighbourhood (P[x]) points. Rotation-invariant, so
        # the scorer stays sphere-faithful (no raw coordinates); HIGHER similarity = closer.
        self.scorer = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))

    def _token_edge_features(self, tokens: WalkTokens, q: int) -> torch.Tensor:
        """Per-token edge features [Q, T, d_ef] aligned with the flattened token bag. tokens holds
        edge_features as [Q, K, L*d_ef]; reshape to [Q, K*L, d_ef]. When the dataset has no edge
        features (d_ef == 0) this is an empty [Q, T, 0] tensor (a no-op in the key concat)."""
        _, k, length = tokens.nodes.shape
        if tokens.edge_features is not None:
            return tokens.edge_features.reshape(q, k, length, self.d_ef).reshape(q, k * length, self.d_ef)
        return tokens.nodes.new_zeros((q, k * length, self.d_ef), dtype=torch.float32)

    def _project(self, tokens: WalkTokens):
        """Project one bag of N queries. Returns (e_seed, p): e_seed [N, d] = E[seed] (the identity on
        the sphere), p [N, d] = exp_{E[seed]}(mu) (seed pushed toward its walk-token centroid). Both
        on-sphere."""
        e_weight = self.E.weight
        e_seed = F.embedding(tokens.seeds, e_weight)                                  # E[x]  [N, d] (E is on-sphere)
        n = e_seed.shape[0]

        token_ids, token_mask, token_pos = flatten_tokens(
            tokens, exclude_seed_positions=True)
        token_ages = tokens.ages.reshape(n, -1).clamp_min(0)                          # ages read from the instance
        token_ef = self._token_edge_features(tokens, n)                              # [N, T, d_ef]
        token_emb = F.embedding(token_ids.clamp_min(0), e_weight)                     # [N, T, d]
        token_tangent = self.geom.logmap(e_seed.unsqueeze(-2), token_emb)             # [N, T, d] tangent
        mu = self.neighbourhood(
            e_seed, token_tangent, token_ages.to(e_seed.dtype), token_mask, token_pos, token_ef)  # [N, d]
        return e_seed, self.geom.expmap(e_seed, mu)                                   # (E[x], P[x])  [N, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in query-major order, each walked with its query's cutoff. Score = MLP over
        the FOUR pairwise similarities (inner products = cosines) on the sphere between both sides'
        identity (seed = E[x]) and neighbourhood (nbhd = P[x]) points (rotation-invariant, so the
        scorer is sphere-faithful — no raw coordinates; higher = closer). Returns logits [B, C]."""
        seed_u, nbhd_u = self._project(src_tokens)                            # E[u], P[u]  [B, d]
        seed_v, nbhd_v = self._project(cand_tokens)                           # E[v], P[v]  [B*C, d]
        b, d = seed_u.shape
        c = seed_v.shape[0] // b
        seed_v = seed_v.reshape(b, c, d)                                      # [B, C, d]
        nbhd_v = nbhd_v.reshape(b, c, d)                                      # [B, C, d]
        seed_u = seed_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]
        nbhd_u = nbhd_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]

        # Four sphere similarities (inner products = cosines) between u's and v's identity /
        # neighbourhood points.
        similarities = torch.stack([
            self.geom.similarity(seed_u, seed_v),                           # ⟨E[u], E[v]⟩  identity affinity
            self.geom.similarity(seed_u, nbhd_v),                           # ⟨E[u], P[v]⟩  is u in v's neighbourhood
            self.geom.similarity(nbhd_u, seed_v),                           # ⟨P[u], E[v]⟩  is v in u's neighbourhood
            self.geom.similarity(nbhd_u, nbhd_v),                           # ⟨P[u], P[v]⟩  neighbourhood overlap
        ], dim=-1)                                                            # [B, C, 4]
        return self.scorer(similarities).squeeze(-1)                         # [B, C]
