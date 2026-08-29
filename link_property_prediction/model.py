"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = geo_temp * (-d_H(P_u, P_v)) [+ pop_bias[v]].

The distance term is scaled by a learned geo_temp (init 1.0). When the popularity channel is on, a
learned per-node scalar pop_bias[v] (zero-init, so it contributes exactly 0 at step 0) is added at
FIXED unit weight -- the model sharpens the geometry via geo_temp while popularity rides alongside at
a constant scale.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over a
selectable-feature MLP over [rec, pos, rad, dev] at a FIXED hidden width of 32. Learned head params:
geo_temp, the MLP pooler, and num_nodes popularity scalars when the channel is on."""

from typing import Sequence

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class PoincareManifold:
    """Poincaré-ball geometry (c=1). `manifold` is kept for E's init + RiemannianAdam."""

    def __init__(self, c: float = 1.0):
        self.manifold = geoopt.PoincareBall(c=c)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Elementwise geodesic distance, broadcasting over leading dims. LOWER = closer."""
        return self.manifold.dist(x, y)

    def dist0(self, x: torch.Tensor) -> torch.Tensor:
        """Hyperbolic radius: geodesic distance from the origin (0 at center, large near boundary)."""
        return self.manifold.dist0(x)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted gyro-midpoint: x [Q,T,d], w [Q,T] -> [Q,d]."""
        return self.manifold.weighted_midpoint(x, weights=w, reducedim=[-2], dim=-1, keepdim=False)


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP(features)); hidden = HIDDEN, randomly initialised, with mnia as
    a fixed age scale. The softmax over tokens dampens the random init.

    The feature set is selectable so that adding a feature is a single-variable change. HIDDEN is a
    FIXED 32 regardless of how many features are on -- under the old `8 * N_FEAT` rule the hidden
    width moved with the feature count, so a feature-count A/B also changed pooler capacity and the
    two effects could not be separated.

    Features (both geometric ones are detached, so the pooler reads geometry but does not backprop
    through it):
      rec -- -(age / mnia), token recency at a fixed age scale
      pos -- -(position - 1), depth along the walk
      rad -- geodesic distance from the origin: the token's hyperbolic radius
      dev -- geodesic distance to the bag's UNWEIGHTED centroid: a per-token spread signal
    """

    ALL_FEATURES = ("rec", "pos", "rad", "dev")
    HIDDEN = 32

    def __init__(self, mnia: float, features: Sequence[str] = ALL_FEATURES):
        super().__init__()
        feats = tuple(features)
        unknown = [f for f in feats if f not in self.ALL_FEATURES]
        if unknown:
            raise ValueError(f"unknown pooler feature(s) {unknown}; known: {list(self.ALL_FEATURES)}")
        if not feats:
            raise ValueError("pooler needs at least one feature")
        self.features = feats
        self.mnia = float(mnia)                 # fixed age scale
        self.net = nn.Sequential(nn.Linear(len(feats), self.HIDDEN), nn.GELU(),
                                 nn.Linear(self.HIDDEN, 1))

    def forward(self, geom: "PoincareManifold", tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        cols = []
        for f in self.features:                 # built in self.features order
            if f == "rec":
                cols.append(-(tokens.ages.clamp_min(0).float() / self.mnia))
            elif f == "pos":
                cols.append(-(tokens.positions.clamp_min(1).float() - 1.0))
            elif f == "rad":
                cols.append(geom.dist0(x.detach()))                             # [Q, T] hyperbolic radius
            elif f == "dev":                                                    # token -> unweighted centroid
                m0 = geom.midpoint(x, valid.float() / valid.sum(-1, keepdim=True).clamp_min(1))
                cols.append(geom.dist(x, m0.unsqueeze(1)).detach())
        feat = torch.stack(cols, dim=-1).to(x.dtype)                            # [Q, T, n_feat]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; the pooling weights carry no learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float,
                 init_irange: float = 1e-3, use_pop_bias: bool = False,
                 pooler_features: Sequence[str] = BagWeights.ALL_FEATURES):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.use_pop_bias = bool(use_pop_bias)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(mean_node_inter_arrival, pooler_features)

        # Near-origin init: uniform(-irange, irange) per coord -> r ~ 2*irange*sqrt(d/3).
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.projx(
                (torch.rand(self.num_nodes, self.d_emb) * 2 - 1) * float(init_irange))
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        # Learned per-node popularity scalar, zero-init: the channel contributes exactly 0 at step 0,
        # so turning it on cannot perturb the starting point.
        if self.use_pop_bias:
            self.pop_bias = nn.Embedding(self.num_nodes, 1)
            nn.init.zeros_(self.pop_bias.weight)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> P [Q,d], the pooling-weighted gyro-midpoint of the walk-token cloud."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)                                               # all-padding walk -> use seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(self.geom, tokens, x, valid)                       # [Q, T] sums to 1, 0 on padding
        return self.geom.midpoint(x, w)                                         # [Q, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        score = geo_temp * (-geo) [+ pop_bias[v]]  (distance scaled by geo_temp; the learned per-node
        popularity scalar, when on, added at fixed unit weight)."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        score = self.geo_temp * (-geo)                                         # [B, C] scaled distance term
        if self.use_pop_bias:
            v_nodes = cand_tokens.seeds.view(b, c)                              # [B, C] candidate node ids
            score = score + self.pop_bias(v_nodes).squeeze(-1)
        return score
