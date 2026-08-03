"""Link head — Poincaré-ball node embeddings + a walk-neighbourhood scoring head, in one module.

Per (u, candidate v) the logit is an MLP (self.scorer) over:
  - 6 pairwise GEODESIC DISTANCES between {E[·], P[·]} of u and v:
        d(E_u,E_v), d(E_u,P_v), d(P_u,E_v), d(P_u,P_v)  (4 cross)  +  d(E_u,P_u), d(E_v,P_v)  (2 self-disp)
  - each node's per-dim-standardised STATIC node features   nf[u], nf[v]
  - each node's pooled WALK-NEIGHBOUR feature encoding       nbhd_feat[u], nbhd_feat[v]
        (NeighborFeatureProjection over the walk tokens' node/edge features, masked-mean pooled)

P[x] = exp_{E[x]}(mu_x) pushes x off E[x] toward mu_x — the ROTATION-INVARIANT weighted mean of x's
walk-token tangents (NeighborhoodProjection: pure geometry, no learned mixing and no feature gate).

TWO-SIDED: both the source u AND every candidate v are walked and projected (P[u], P[v] both used).
The head owns self.E (a ManifoldParameter on a geoopt.PoincareBall, link-trained on it); geometry goes
through self.geom (PoincareManifold): dist/logmap/expmap all proxy to geoopt (the ball's expmap has a
finite gradient at the cold-row zero tangent).
"""
from typing import Optional

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class PoincareManifold:
    """Poincaré-ball geometry over geoopt.PoincareBall(c=1). dist / logmap / expmap all PROXY straight to
    geoopt — the ball's expmap already has a FINITE gradient at the zero tangent (the cold-row mu = 0, a
    node with no walk tokens), so no custom expmap is needed (unlike the sphere). The tangent space at any
    point is all of R^d — no orthogonal projection. E is kept in the ball by RiemannianAdam."""

    def __init__(self, c: float = 1.0):
        self.manifold = geoopt.PoincareBall(c=c)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic (hyperbolic) distance on the ball — geoopt.PoincareBall.dist. LOWER = closer."""
        return self.manifold.dist(x, y)

    def logmap(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """Log map at `base` of `point` — geoopt.PoincareBall.logmap (finite gradient at coincidence)."""
        return self.manifold.logmap(base, point)

    def expmap(self, base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """Exp map from `base` along `tangent` — geoopt.PoincareBall.expmap. Finite gradient at u = 0, so
        the cold-row zero tangent is a safe differentiable no-op returning `base`."""
        return self.manifold.expmap(base, tangent)


class NeighborhoodProjection(nn.Module):
    """Weighted-MEAN pooling of a source's walk-token TANGENTS, mapped to the ball point P[u] = exp_{E[u]}(mu_u):

        mu_u = Σ_p softmax(a_p) · g_p,   g_p = Log_{E[u]}(E[token_p])   (raw geometric tangent)
        P[u] = exp_{E[u]}(mu_u)                                          (back to the ball)

    The per-token weight logits a_p are computed ONCE by LinkPredHead (its shared recency/hop weight net)
    and passed in; here they are softmaxed over the GEOMETRY context (mask & ~seed_mask & ~seed_node_mask —
    seed-node revisits are dropped, since Log_{E[u]}(E[u]) = 0 contributes nothing but would dilute). PURE
    GEOMETRY and PARAMETER-FREE: raw tangents pooled by rotation-invariant scalar weights, so mu_u — and
    hence P[u] — is ROTATION-INVARIANT. The exp map is applied HERE so forward returns a valid ball POINT
    directly (not a tangent). Cold rows (no context) -> mu_u = 0 -> exp_{E[u]}(0) = E[u]."""

    def __init__(self):
        super().__init__()
        self.geom = PoincareManifold()                 # stateless; used for the token log-map

    def forward(self, walk_bag: WalkTokens, emb: torch.Tensor, w_logit: torch.Tensor) -> torch.Tensor:
        """walk_bag: flattened WalkTokens. emb: full [num_nodes,d_emb] table in the ball. w_logit: [B,T]
        shared per-token weight logits. Returns P[u] = exp_{E[u]}(mu_u) [B,d_emb], a POINT in the ball;
        cold rows (no context) -> mu_u = 0 -> P[u] = E[u]."""
        node_ids = walk_bag.nodes.clamp_min(0)                                    # [B,T]  (padding → row 0)
        source = F.embedding(walk_bag.seeds, emb)                                 # E[u]  [B,d_emb]
        token_tangents = self.geom.logmap(source.unsqueeze(-2), F.embedding(node_ids, emb))  # [B,T,d_emb]

        mask = walk_bag.mask & ~walk_bag.seed_mask & ~walk_bag.seed_node_mask    # [B,T]  geometry context
        weights = torch.nan_to_num(
            torch.softmax(w_logit.masked_fill(~mask, float("-inf")), dim=-1), nan=0.0)   # [B,T]; cold row -> 0

        # mu = weighted mean of the RAW token tangents (rotation-invariant); exp_{E[u]}(mu) maps it back to
        # the ball as P[u]. (Cold rows: mu = 0 -> exp_{E[u]}(0) = E[u], a differentiable no-op.)
        mu = (weights.unsqueeze(-1) * token_tangents).sum(dim=-2)                 # [B,d]  tangent at E[u]
        return self.geom.expmap(source, mu)                                      # P[u]  [B,d]  ball point


class NeighborFeatureProjection(nn.Module):
    """WEIGHTED-mean-pooled encoder for the walk tokens' STATIC node features + per-token edge features —
    the non-geometric channel, kept separate from the (rotation-invariant) NeighborhoodProjection. Per
    token: concat [node_feat, edge_feat] -> Linear-GELU-Linear -> LayerNorm. Pooled to one [N, feature_dim]
    vector by the SAME shared recency/hop weights (passed in as w_logit), softmaxed over (mask & ~seed_mask)
    — only non-seed tokens carry an edge feature. Returns [N, 0] when d_nf == d_ef == 0."""

    def __init__(self, d_nf: int, d_ef: int, feature_dim: int = 16):
        super().__init__()
        self.d_nf = d_nf
        self.d_ef = d_ef
        in_dim = d_nf + d_ef
        self.feature_dim = feature_dim if in_dim > 0 else 0        # overridden to 0 when no features present
        if self.feature_dim > 0:
            self.encode = nn.Sequential(
                nn.Linear(in_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim))
            self.out_norm = nn.LayerNorm(feature_dim)

    def forward(self, walk_bag: WalkTokens, node_features: Optional[torch.Tensor],
                w_logit: torch.Tensor) -> torch.Tensor:
        """Returns [N, feature_dim] the recency/hop-weighted mean of the per-token feature encodings, or
        [N, 0] if the dataset has no node/edge features. w_logit: [N, T] shared per-token weight logits."""
        node_ids = walk_bag.nodes.clamp_min(0)                                    # [N,T]  (padding → row 0)
        n, t = node_ids.shape
        dev = node_ids.device
        if self.feature_dim == 0:
            return torch.zeros(n, 0, device=dev)

        if node_features is None or self.d_nf == 0:
            nf_token = torch.zeros(n, t, self.d_nf, device=dev)
        else:
            nf_token = F.embedding(node_ids, node_features)                       # [N,T,d_nf]

        if walk_bag.edge_features is None or self.d_ef == 0:
            edge_features = torch.zeros(n, t, self.d_ef, device=dev)
        else:
            edge_features = walk_bag.edge_features                                # [N,T,d_ef]

        ft = self.out_norm(self.encode(torch.cat([nf_token, edge_features], dim=-1)))    # [N,T,F]  per-token

        # Weighted mean via the shared weights, softmaxed over the non-seed real tokens (seed slot / padding
        # get weight 0 -> excluded; cold rows -> all-0 weights -> 0 vector).
        mask = walk_bag.mask & ~walk_bag.seed_mask                                # [N,T]
        weights = torch.nan_to_num(
            torch.softmax(w_logit.masked_fill(~mask, float("-inf")), dim=-1), nan=0.0)   # [N,T]
        return (weights.unsqueeze(-1) * ft).sum(dim=-2)                           # [N,F]  weighted mean


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int,
                 t2v_dim: int = 16, d_ef: int = 0, feature_dim: int = 16,
                 node_features: Optional[torch.Tensor] = None, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.d_nf = 0 if node_features is None else int(node_features.shape[1])
        # Static per-node feature table [num_nodes, d_nf] (dataset-derived, NOT learned). Per-dimension
        # STANDARDISED (z-score across nodes) so nf[u], nf[v] enter the scorer at ~unit scale — consistent
        # with the LayerNorm-ed nbhd feats, and well-conditioned for the MLP (raw features can be arbitrary
        # scale). A buffer so it rides model.to(device) and stays out of the optimizer; non-persistent.
        if node_features is not None and self.d_nf > 0:
            mu = node_features.mean(dim=0, keepdim=True)
            sd = node_features.std(dim=0, keepdim=True).clamp_min(1e-6)
            node_features = (node_features - mu) / sd                    # [num_nodes, d_nf]  per-dim z-score
        self.register_buffer("node_features", node_features, persistent=False)
        self.geom = PoincareManifold()

        # E lives in the Poincaré ball: init near the origin (small-std wrapped normal) so the conformal
        # metric — which blows up near the boundary — stays well-conditioned early; then wrap as a
        # ManifoldParameter so RiemannianAdam keeps it in the ball.
        self.E = nn.Embedding(num_nodes, d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random_normal(num_nodes, d_emb, std=1e-2)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        self.neighbourhood = NeighborhoodProjection()   # parameter-free tangent pooling (weights passed in)
        # Non-geometric feature channel (static node + per-token edge features). feature_dim from config;
        # 0-width (no-op) when the dataset has no node/edge features. Weighted-mean pooled per node -> scorer.
        self.neighbour_feats = NeighborFeatureProjection(d_nf=self.d_nf, d_ef=d_ef, feature_dim=feature_dim)
        # Per-token pooling weights are now a FIXED (non-learned) recency/hop prior — see
        # _token_weight_logits. The old learned weight net (Time2Vec + MLP) is removed; t2v_dim
        # is retained in the signature only for CLI compatibility and is unused.

        # Scorer: MLP over the 4 pairwise GEODESIC DISTANCES between {E[x], P[x]} of u and v, plus each node's
        # static node features nf[·] and its pooled walk-neighbour feature encoding (NeighborFeatureProjection).
        # Distances are isometry-invariant; nf / nbhd-feats are external, frame-free channels.
        # 6 dists = 4 cross + 2 self-displacements d(E,P); + nf[u,v] + nbhd_feat[u,v].
        score_in = 6 + 2 * self.d_nf + 2 * self.neighbour_feats.feature_dim
        # Dropout on the scorer hidden layer breaks the head's co-adaptation/memorisation — the lever that
        # converts the post-peak overfit CLIFF into a gentle valley (wiki: dropout 0.1 held a stable
        # ~0.795-0.805 valley vs a hard -0.066 crash with no dropout). Load-bearing for stability.
        self.scorer = nn.Sequential(
            nn.Linear(score_in, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        # Single learned temperature on the whole distance block: distances / tau.clamp(...). The raw
        # geodesic distances inflate ~77x over training (median ~0.03 early -> ~2.1 late) while nf / nbhd_feat
        # sit at O(1); one clamped scalar absorbs that global scale so the scorer's first Linear sees
        # comparable channels. Pairs with boundary_penalty (which caps inflation on the E side). Init 1 = no-op.
        self.dist_tau = nn.Parameter(torch.tensor(1.0))

    def _token_weight_logits(self, tokens: WalkTokens) -> torch.Tensor:
        """FIXED (non-learned) shared per-token pooling weights [N, T], mirroring the alignment
        loss's recency/hop prior:

            log_weight = -( log1p(age) + log1p(hop - 1) )

        These are LOG-weights: each pooling channel masked_fills non-context slots to -inf and
        softmaxes, so the token with the smallest (age, hop) — most RECENT and CLOSEST to the seed —
        gets the highest weight. The leading minus is what does that: lower age → smaller log1p(age),
        lower hop → smaller log1p(hop-1) → larger (less negative) log_weight → larger softmax weight.
        Convention (WalkTokens): age = 0 at the seed, ≥1 for context; hop = 1 at the seed, 2 for the
        closest context, so hop-1 = 0 at the seed (masked out) and ≥1 for context."""
        dtype = self.E.weight.dtype
        age = tokens.ages.clamp_min(0).to(dtype)                                       # [N, T]  seed=0, ctx≥1
        hop = tokens.positions.clamp_min(1).to(dtype)                                  # [N, T]  seed=1, ctx≥2
        return -(torch.log1p(age) + torch.log1p(hop - 1.0))                            # [N, T]  ≤ 0

    def _project(self, tokens: WalkTokens, w_logit: torch.Tensor) -> torch.Tensor:
        """Neighbourhood projection P[x] = exp_{E[seed]}(mu) [N, d] — the ball point returned by
        NeighborhoodProjection (which applies the exp map itself). w_logit: the bag's shared per-token
        weight logits. E is read LIVE — the link loss trains E end-to-end through the head (no detach)."""
        return self.neighbourhood(tokens, self.E.weight, w_logit)                    # P[x]  [N, d]

    def _node_feats(self, seeds: torch.Tensor) -> torch.Tensor:
        """Static node features [N, d_nf] for the given seed ids; a zero-width [N, 0] tensor when the
        dataset has no node features (so it concatenates as a no-op — no branching at the call site)."""
        if self.node_features is None:
            return self.E.weight.new_zeros(seeds.shape[0], 0)
        return F.embedding(seeds, self.node_features)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in query-major order, each walked with its query's cutoff. Logit = MLP over
        [ d(E_u,E_v), d(E_u,P_v), d(P_u,E_v), d(P_u,P_v), nf[u], nf[v], nbhd_feat[u], nbhd_feat[v] ]
        per (u, candidate v). Returns logits [B, C]."""
        emb = self.E.weight                                                   # E trained end-to-end by the link loss
        # Shared per-token weight logits, computed once per bag and used to pool BOTH channels.
        w_src = self._token_weight_logits(src_tokens)                         # [B, T]
        w_cand = self._token_weight_logits(cand_tokens)                       # [B*C, T]
        seed_u = F.embedding(src_tokens.seeds, emb)                           # E[u]   [B, d]
        nbhd_u = self._project(src_tokens, w_src)                             # P[u]   [B, d]
        nf_u = self._node_feats(src_tokens.seeds)                             # nf[u]  [B, d_nf]
        nbhd_feat_u = self.neighbour_feats(src_tokens, self.node_features, w_src)   # nbhd_feat[u]  [B, F]
        seed_v = F.embedding(cand_tokens.seeds, emb)                          # E[v]   [B*C, d]
        nbhd_v = self._project(cand_tokens, w_cand)                           # P[v]   [B*C, d]
        nf_v = self._node_feats(cand_tokens.seeds)                            # nf[v]  [B*C, d_nf]
        nbhd_feat_v = self.neighbour_feats(cand_tokens, self.node_features, w_cand)  # nbhd_feat[v]  [B*C, F]

        b, d = seed_u.shape
        c = seed_v.shape[0] // b
        # candidate (v) side -> [B, C, ·]; source (u) side broadcast over the C candidates -> [B, C, ·].
        seed_v = seed_v.reshape(b, c, d)                                      # [B, C, d]
        nbhd_v = nbhd_v.reshape(b, c, d)                                      # [B, C, d]
        nf_v = nf_v.reshape(b, c, self.d_nf)                                  # [B, C, d_nf]
        seed_u = seed_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]
        nbhd_u = nbhd_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]
        nf_u = nf_u.unsqueeze(1).expand(b, c, self.d_nf)                      # [B, C, d_nf]
        fd = self.neighbour_feats.feature_dim
        nbhd_feat_v = nbhd_feat_v.reshape(b, c, fd)                                   # [B, C, F]
        nbhd_feat_u = nbhd_feat_u.unsqueeze(1).expand(b, c, fd)                       # [B, C, F]

        # Six pairwise geodesic distances (self.geom.dist): the 4 cross distances + the 2 self-displacements
        # d(E[·], P[·]) that complete the frame-free set (0 exactly for a cold row, where P = E).
        distances = torch.stack([
            self.geom.dist(seed_u, seed_v),                                  # d(E[u], E[v])  identity affinity
            self.geom.dist(seed_u, nbhd_v),                                  # d(E[u], P[v])  is u in v's nbhd
            self.geom.dist(nbhd_u, seed_v),                                  # d(P[u], E[v])  is v in u's nbhd
            self.geom.dist(nbhd_u, nbhd_v),                                  # d(P[u], P[v])  nbhd overlap
            self.geom.dist(seed_u, nbhd_u),                                  # d(E[u], P[u])  u's self-displacement
            self.geom.dist(seed_v, nbhd_v),                                  # d(E[v], P[v])  v's self-displacement
        ], dim=-1)                                                            # [B, C, 6]
        distances = distances / self.dist_tau.clamp(0.05, 20.0)              # learned global scale (conditioning)
        feats = torch.cat([distances, nf_u, nf_v, nbhd_feat_u, nbhd_feat_v], dim=-1)  # [B, C, 6 + 2*d_nf + 2*F]
        return self.scorer(feats).squeeze(-1)                                 # [B, C]
