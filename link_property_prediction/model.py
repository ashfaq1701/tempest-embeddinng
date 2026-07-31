"""Link head — Poincaré-ball node embeddings + a walk-neighbourhood scoring head, in one module.

Per (u, candidate v) the logit is an MLP (self.scorer) over:
  - 4 pairwise GEODESIC DISTANCES between {E[·], P[·]} of u and v:
        d(E_u,E_v), d(E_u,P_v), d(P_u,E_v), d(P_u,P_v)
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
    """Weighted-MEAN pooling of a source's walk-token tangents into one tangent vector mu_u at E[u].

        mu_u = Σ_p softmax(a_p) · g_p,
        g_p  = Log_{E[u]}(E[token_p])                                  (raw geometric tangent)
        a_p  = weight_net([Time2Vec(age_p), log1p(hop_p)])            (scalar from time/position ONLY)

    PURE GEOMETRY — no learned mixing (no W_v) and no feature gate. The value is the raw tangent; the weight
    is a rotation-invariant scalar from time/position, softmax-normalised over the context tokens. Because
    every value is a raw tangent and every weight a rotation-invariant scalar, mu_u is ROTATION-INVARIANT
    (it co-rotates with E). Fed to exp_{E[u]} downstream -> P[u]. Candidate-independent; cold rows (no
    context token) -> mu_u = 0 -> P[u] = E[u]. (Static node / edge features are handled separately by
    NeighborFeatureProjection.)"""

    def __init__(self, d_emb: int, t2v_dim: int = 16):
        super().__init__()
        self.d_emb = d_emb
        self.geom = PoincareManifold()                 # stateless; used for the token log-map
        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        # WEIGHT net: [Time2Vec(age), log1p(hop)] -> scalar per token (softmax over context tokens).
        w_in = t2v_dim + 1
        self.weight_net = nn.Sequential(
            nn.Linear(w_in, w_in), nn.GELU(), nn.Linear(w_in, 1))

    def forward(self, walk_bag: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """walk_bag: the flattened WalkTokens bag. emb: the FULL node-embedding table [num_nodes,d_emb] in
        the Poincaré ball. Returns mu_u [B,d_emb], a tangent vector at E[u]; cold rows -> 0."""
        node_ids = walk_bag.nodes.clamp_min(0)                                    # [B,T]  (padding → row 0)
        source = F.embedding(walk_bag.seeds, emb)                                 # E[u]  [B,d_emb]
        token_tangents = self.geom.logmap(source.unsqueeze(-2), F.embedding(node_ids, emb))  # [B,T,d_emb]

        mask = walk_bag.mask & ~walk_bag.seed_mask & ~walk_bag.seed_node_mask    # [B,T]  context (non-seed-node real)
        ages = walk_bag.ages.clamp_min(0).to(token_tangents.dtype)               # [B,T]

        # TPNet scales delta-times by log(Δt + 1) before the time encoder.
        t2v = self.time_encoder(torch.log1p(ages))                               # [B,T,t2v_dim]
        log_hop = torch.log1p(walk_bag.positions.clamp_min(0).to(t2v.dtype)).unsqueeze(-1)  # [B,T,1]

        # WEIGHT: scalar per token from [Time2Vec(age), log-hop]; softmax over the context tokens.
        w_logit = self.weight_net(torch.cat([t2v, log_hop], dim=-1)).squeeze(-1)  # [B,T]
        w_logit = w_logit.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(w_logit, dim=-1), nan=0.0)       # [B,T]; cold row -> 0

        # mu = weighted mean of the RAW token tangents (rotation-invariant). No ⊥-projection on the ball;
        # mu is fed to exp_{E[u]} downstream -> P[u].  (cold rows: mu = 0 -> P[u] = E[u].)
        return (weights.unsqueeze(-1) * token_tangents).sum(dim=-2)               # [B,d]


class NeighborFeatureProjection(nn.Module):
    """Per-token encoder for the walk tokens' STATIC node features + per-token edge features — the
    NON-geometric channel, kept separate from the (rotation-invariant) NeighborhoodProjection. Concatenates
    [node_feat, edge_feat] per token, encodes to a feature_dim vector, and LayerNorms it. Returns a 0-width
    [B, T, 0] tensor when the dataset has neither node nor edge features (d_nf == d_ef == 0)."""

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

    def forward(self, walk_bag: WalkTokens,
                node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns [B, T, feature_dim] per-token LayerNorm-ed feature encodings, or [B, T, 0] if the dataset
        has no node/edge features. node_features: the FULL static node-feature table [num_nodes, d_nf]."""
        node_ids = walk_bag.nodes.clamp_min(0)                                    # [B,T]  (padding → row 0)
        b, t = node_ids.shape
        dev = node_ids.device
        if self.feature_dim == 0:
            return torch.zeros(b, t, 0, device=dev)

        # Only non-seed real tokens carry a (non-empty) edge feature — the seed slot has no outgoing edge.
        mask = (walk_bag.mask & ~walk_bag.seed_mask).unsqueeze(-1)                # [B,T,1]

        if node_features is None or self.d_nf == 0:
            nf_token = torch.zeros(b, t, self.d_nf, device=dev)
        else:
            nf_token = F.embedding(node_ids, node_features)                       # [B,T,d_nf]

        if walk_bag.edge_features is None or self.d_ef == 0:
            edge_features = torch.zeros(b, t, self.d_ef, device=dev)
        else:
            edge_features = walk_bag.edge_features                                # [B,T,d_ef]

        feats = torch.cat([nf_token, edge_features], dim=-1) * mask                # zero seed slot + padding
        return self.out_norm(self.encode(feats)) * mask                           # [B,T,feature_dim], masked


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int,
                 t2v_dim: int = 16, d_ef: int = 0,
                 node_features: Optional[torch.Tensor] = None):
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

        self.neighbourhood = NeighborhoodProjection(d_emb=d_emb, t2v_dim=t2v_dim)
        # Non-geometric per-token feature channel (static node + per-token edge features). feature_dim 16;
        # 0-width (no-op) when the dataset has no node/edge features. Pooled per-node and fed to the scorer.
        self.neighbour_feats = NeighborFeatureProjection(d_nf=self.d_nf, d_ef=d_ef, feature_dim=16)

        # Scorer: MLP over the 4 pairwise GEODESIC DISTANCES between {E[x], P[x]} of u and v, plus each node's
        # static node features nf[·] and its pooled walk-neighbour feature encoding (NeighborFeatureProjection).
        # Distances are isometry-invariant; nf / nbhd-feats are external, frame-free channels.
        score_in = 4 + 2 * self.d_nf + 2 * self.neighbour_feats.feature_dim   # 4 dists + nf[u,v] + nbhd_feat[u,v]
        self.scorer = nn.Sequential(
            nn.Linear(score_in, 32), nn.GELU(), nn.Linear(32, 1))

    def _project(self, tokens: WalkTokens) -> torch.Tensor:
        """Neighbourhood projection P[x] = exp_{E[seed]}(mu) [N, d] for a bag of N queries. The
        neighbourhood takes the bag + the full E table and derives E[u] / token tangents / stable
        feats / masks itself; E[seed] is looked up separately by the caller."""
        emb = self.E.weight.detach()                                                 # link head reads E DETACHED
        e_seed = F.embedding(tokens.seeds, emb)                                      # E[seed]  [N, d]
        mu = self.neighbourhood(tokens, emb)                                         # [N, d]
        return self.geom.expmap(e_seed, mu)                                          # P[x]  [N, d]

    def _node_feats(self, seeds: torch.Tensor) -> torch.Tensor:
        """Static node features [N, d_nf] for the given seed ids; a zero-width [N, 0] tensor when the
        dataset has no node features (so it concatenates as a no-op — no branching at the call site)."""
        if self.node_features is None:
            return self.E.weight.new_zeros(seeds.shape[0], 0)
        return F.embedding(seeds, self.node_features)

    def _neighbour_feats(self, tokens: WalkTokens) -> torch.Tensor:
        """Pooled walk-neighbour feature encoding for a bag of N queries -> [N, feature_dim]: a masked MEAN
        over the seed's non-seed real tokens of NeighborFeatureProjection's per-token encodings. Zero-width
        [N, 0] when the dataset has no node/edge features."""
        ft = self.neighbour_feats(tokens, self.node_features)                        # [N, T, F] (0 at seed/pad)
        n = (tokens.mask & ~tokens.seed_mask).unsqueeze(-1).to(ft.dtype)             # [N, T, 1]
        return ft.sum(dim=-2) / n.sum(dim=-2).clamp_min(1.0)                          # [N, F]  masked mean

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in query-major order, each walked with its query's cutoff. Logit = MLP over
        [ d(E_u,E_v), d(E_u,P_v), d(P_u,E_v), d(P_u,P_v), nf[u], nf[v], nbhd_feat[u], nbhd_feat[v] ]
        per (u, candidate v). Returns logits [B, C]."""
        emb = self.E.weight.detach()                                          # link head reads E DETACHED
        seed_u = F.embedding(src_tokens.seeds, emb)                           # E[u]   [B, d]
        nbhd_u = self._project(src_tokens)                                    # P[u]   [B, d]
        nf_u = self._node_feats(src_tokens.seeds)                             # nf[u]  [B, d_nf]
        seed_v = F.embedding(cand_tokens.seeds, emb)                          # E[v]   [B*C, d]
        nbhd_v = self._project(cand_tokens)                                   # P[v]   [B*C, d]
        nf_v = self._node_feats(cand_tokens.seeds)                            # nf[v]  [B*C, d_nf]
        nbhd_feat_u = self._neighbour_feats(src_tokens)                           # nbhd_feat[u]  [B, F]
        nbhd_feat_v = self._neighbour_feats(cand_tokens)                          # nbhd_feat[v]  [B*C, F]

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

        # Four pairwise geodesic distances (self.geom.dist) between u's and v's identity / nbhd points.
        distances = torch.stack([
            self.geom.dist(seed_u, seed_v),                                  # d(E[u], E[v])  identity affinity
            self.geom.dist(seed_u, nbhd_v),                                  # d(E[u], P[v])  is u in v's nbhd
            self.geom.dist(nbhd_u, seed_v),                                  # d(P[u], E[v])  is v in u's nbhd
            self.geom.dist(nbhd_u, nbhd_v),                                  # d(P[u], P[v])  nbhd overlap
        ], dim=-1)                                                            # [B, C, 4]
        feats = torch.cat([distances, nf_u, nf_v, nbhd_feat_u, nbhd_feat_v], dim=-1)  # [B, C, 4 + 2*d_nf + 2*F]
        return self.scorer(feats).squeeze(-1)                                 # [B, C]
