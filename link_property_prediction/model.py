"""Link head — Poincaré-ball node embeddings + a walk-neighbourhood scoring head, in one module.

Per (u, candidate v) the logit is an MLP (self.scorer) over:
  - 6 pairwise GEODESIC DISTANCES between {E[·], M[·]} of u and v:
        d(E_u,E_v), d(E_u,M_v), d(M_u,E_v), d(M_u,M_v)  (4 cross)  +  d(E_u,M_u), d(E_v,M_v)  (2 self-disp)
  - each node's per-dim-standardised STATIC node features   nf[u], nf[v]
  - each node's pooled WALK-NEIGHBOUR feature encoding       nbhd_feat[u], nbhd_feat[v]
        (NeighborFeatureProjection over the walk tokens' node/edge features, masked-mean pooled)

M[x] is the intrinsic weighted GYROMIDPOINT of x's walk-token points (NeighborhoodProjection via
geoopt.weighted_midpoint) — base-point-free, rotation-EQUIVARIANT, manifold-preserving (no tangent-space
approximation at a learned base point). Cold rows (no context) -> M[x] = E[x], so d(E_x,M_x) = 0 is the
cold-row indicator.

TWO-SIDED: both the source u AND every candidate v are walked and aggregated (M[u], M[v] both used).
The head owns self.E (a ManifoldParameter on a geoopt.PoincareBall, link-trained on it); geometry goes
through self.geom (PoincareManifold): dist/logmap/expmap all proxy to geoopt (the ball's expmap has a
finite gradient at the cold-row zero tangent).
"""
from typing import Optional

import geoopt
import numpy as np
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

    def pairwise_dist(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """All-pairs geodesic distances between rows of X [n,d] and Y [m,d] (both in the c=1 ball), as
        [n, m] — WITHOUT materialising an [n,m,d] tensor. Everything factors through the Gram matrix
        ⟨x,y⟩ = X @ Yᵀ:  d(x,y) = arccosh(1 + 2‖x-y‖²/((1-‖x‖²)(1-‖y‖²))),  ‖x-y‖² = ‖x‖²+‖y‖²-2⟨x,y⟩.
        Differentiable in E (grad flows through the matmul). Matches geoopt.PoincareBall(c=1).dist to ~1e-7."""
        xn = (X * X).sum(-1)                                              # ‖x‖²  [n]
        yn = (Y * Y).sum(-1)                                              # ‖y‖²  [m]
        sq = xn[:, None] + yn[None, :] - 2.0 * (X @ Y.t())               # ‖x-y‖²  [n,m]
        arg = 1.0 + 2.0 * sq / ((1.0 - xn)[:, None] * (1.0 - yn)[None, :])
        return torch.arccosh(arg.clamp_min(1.0 + 1e-7))                  # [n,m]


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
    """Weighted GYROMIDPOINT of a source's walk-token points into one ball point M[u] (Ungar's gyromidpoint
    via geoopt.PoincareBall.weighted_midpoint):

        M[u] = ½ ⊗ ( Σ_p w_p·γ_p·E[token_p] / Σ_p w_p·(γ_p − 1) ),   γ_p = 2/(1 − ‖E[token_p]‖²)

    INTRINSIC and base-point-free: unlike the previous tangent-pool (Log_{E[u]} → mean → Exp_{E[u]}, a
    first-order approximation at the learned base point E[u]), the gyromidpoint combines the token POINTS
    directly in the ball, so M[u] is the exact manifold-preserving weighted mean and is rotation-EQUIVARIANT.

    The per-token weight logits a_p are computed ONCE by LinkPredHead (shared recency/hop weight net) and
    passed in; here they are softmaxed over the GEOMETRY context (mask & ~seed_mask). Because there is NO
    base point, seed-node occurrences are KEPT — only the trivial walk-origin slot (seed_mask) is dropped;
    u's mid-walk recurrences are legitimate points of the intrinsic mean (contrast the tangent pool, where
    Log_{E[u]}(E[u]) = 0 made them inert and they were excluded via seed_node_mask). Softmax is retained
    for POSITIVITY (keeps the gyromidpoint denominator Σw(γ−1) bounded away from 0 / inside the gyroconvex
    hull); its normalisation is redundant since weighted_midpoint(lincomb=False) is scale-invariant. Cold
    rows (no context tokens) -> M[u] = E[u] (a differentiable no-op, so d(E[u],M[u]) = 0 stays the cold-row
    indicator the scorer reads)."""

    def __init__(self):
        super().__init__()
        self.geom = PoincareManifold()                 # stateless; used for weighted_midpoint

    def forward(self, walk_bag: WalkTokens, emb: torch.Tensor, w_logit: torch.Tensor) -> torch.Tensor:
        """walk_bag: flattened WalkTokens. emb: full [num_nodes,d_emb] table in the ball. w_logit: [B,T]
        shared per-token weight logits. Returns M[u] [B,d_emb], a POINT in the ball; cold rows -> E[u]."""
        node_ids = walk_bag.nodes.clamp_min(0)                                    # [B,T]  (padding → row 0)
        source = F.embedding(walk_bag.seeds, emb)                                 # E[u]  [B,d_emb]
        tok_pts = F.embedding(node_ids, emb)                                      # [B,T,d_emb]  token POINTS

        # Geometry context: valid, non-origin-slot tokens. seed_node_mask is NOT applied — with no base
        # point, u's own mid-walk recurrences are real points of the mean (only the walk-origin slot drops).
        mask = walk_bag.mask & ~walk_bag.seed_mask                                # [B,T]
        # Zero weight == token removed for the midpoint, so masked slots (weight 0) drop out cleanly; softmax
        # guarantees the surviving weights are strictly positive (denominator stays bounded away from 0).
        weights = torch.nan_to_num(
            torch.softmax(w_logit.masked_fill(~mask, float("-inf")), dim=-1), nan=0.0)   # [B,T]; cold row -> all 0

        # Intrinsic weighted mean of the token points (reduce over the T token axis). No base point, no
        # Log/Exp round-trip. weighted_midpoint clamps its denominator, so an all-zero (cold) row is finite
        # (lands at the origin) — masked back to E[u] below rather than relied upon.
        mid = self.geom.manifold.weighted_midpoint(tok_pts, weights, reducedim=[1])      # [B,d]
        has_ctx = mask.any(dim=-1, keepdim=True)                                          # [B,1]
        return torch.where(has_ctx, mid, source)                                          # cold row -> E[u]


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
        # SHARED recency/hop weight net: [Time2Vec(age), log1p(hop)] -> one per-token weight logit, used to
        # pool BOTH the geometry tangents (NeighborhoodProjection) and the feature encodings
        # (NeighborFeatureProjection). One "token relevance" function; each channel softmaxes over its own mask.
        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        w_in = t2v_dim + 1
        self.weight_net = nn.Sequential(
            nn.Linear(w_in, w_in), nn.GELU(), nn.Linear(w_in, 1))

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
        """Shared per-token recency/hop weight logits [N, T] from [Time2Vec(log1p(age)), log1p(hop)].
        Computed once per bag; each channel softmaxes over its own context mask."""
        ages = tokens.ages.clamp_min(0).to(self.E.weight.dtype)                       # [N, T]
        t2v = self.time_encoder(torch.log1p(ages))                                   # [N, T, t2v_dim]
        log_hop = torch.log1p(tokens.positions.clamp_min(0).to(t2v.dtype)).unsqueeze(-1)  # [N, T, 1]
        return self.weight_net(torch.cat([t2v, log_hop], dim=-1)).squeeze(-1)         # [N, T]

    def _project(self, tokens: WalkTokens, w_logit: torch.Tensor) -> torch.Tensor:
        """Neighbourhood gyromidpoint M[x] [N, d] — the intrinsic weighted mean of x's walk-token points
        (base-point-free; cold rows -> E[x]). w_logit: the bag's shared per-token weight logits. E is read
        LIVE — the link loss trains E end-to-end through the head (no detach)."""
        return self.neighbourhood(tokens, self.E.weight, w_logit)                    # M[x]  [N, d]

    def _node_feats(self, seeds: torch.Tensor) -> torch.Tensor:
        """Static node features [N, d_nf] for the given seed ids; a zero-width [N, 0] tensor when the
        dataset has no node features (so it concatenates as a no-op — no branching at the call site)."""
        if self.node_features is None:
            return self.E.weight.new_zeros(seeds.shape[0], 0)
        return F.embedding(seeds, self.node_features)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in query-major order, each walked with its query's cutoff. M[x] is the walk
        neighbourhood's intrinsic GYROMIDPOINT (base-point-free). Logit = MLP over
        [ d(E_u,E_v), d(E_u,M_v), d(M_u,E_v), d(M_u,M_v), nf[u], nf[v], nbhd_feat[u], nbhd_feat[v] ]
        per (u, candidate v). Returns logits [B, C]."""
        emb = self.E.weight                                                   # E trained end-to-end by the link loss
        # Shared per-token weight logits, computed once per bag and used to pool BOTH channels.
        w_src = self._token_weight_logits(src_tokens)                         # [B, T]
        w_cand = self._token_weight_logits(cand_tokens)                       # [B*C, T]
        seed_u = F.embedding(src_tokens.seeds, emb)                           # E[u]   [B, d]
        nbhd_u = self._project(src_tokens, w_src)                             # M[u]   [B, d]  gyromidpoint
        nf_u = self._node_feats(src_tokens.seeds)                             # nf[u]  [B, d_nf]
        nbhd_feat_u = self.neighbour_feats(src_tokens, self.node_features, w_src)   # nbhd_feat[u]  [B, F]
        seed_v = F.embedding(cand_tokens.seeds, emb)                          # E[v]   [B*C, d]
        nbhd_v = self._project(cand_tokens, w_cand)                           # M[v]   [B*C, d]  gyromidpoint
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
        # d(E[·], M[·]) that complete the frame-free set (0 exactly for a cold row, where M = E).
        distances = torch.stack([
            self.geom.dist(seed_u, seed_v),                                  # d(E[u], E[v])  identity affinity
            self.geom.dist(seed_u, nbhd_v),                                  # d(E[u], M[v])  is u in v's nbhd
            self.geom.dist(nbhd_u, seed_v),                                  # d(M[u], E[v])  is v in u's nbhd
            self.geom.dist(nbhd_u, nbhd_v),                                  # d(M[u], M[v])  nbhd overlap
            self.geom.dist(seed_u, nbhd_u),                                  # d(E[u], M[u])  u's self-displacement
            self.geom.dist(seed_v, nbhd_v),                                  # d(E[v], M[v])  v's self-displacement
        ], dim=-1)                                                            # [B, C, 6]
        distances = distances / self.dist_tau.clamp(0.05, 20.0)              # learned global scale (conditioning)
        feats = torch.cat([distances, nf_u, nf_v, nbhd_feat_u, nbhd_feat_v], dim=-1)  # [B, C, 6 + 2*d_nf + 2*F]
        return self.scorer(feats).squeeze(-1)                                 # [B, C]
