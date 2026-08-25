"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = w . feats.

feats is [-d_H(P_u, P_v)] and, when the popularity channel is on, [-d_H(P_u,P_v), pop_bias[v]] -- a
learned per-node scalar, zero-init, so the channel contributes exactly 0 at step 0. w is the mixing
vector over those features, init ones, so the distance enters at unit weight from the start and the
model learns how much popularity to mix in.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over
pooling_temp * (-age/mnia - (pos-1)), both priors RAW (no log1p). Learned head params: w (1 or 2),
the pooling temperature, and num_nodes popularity scalars when the channel is on."""

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

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted gyro-midpoint: x [Q,T,d], w [Q,T] -> [Q,d]."""
        return self.manifold.weighted_midpoint(x, weights=w, reducedim=[-2], dim=-1, keepdim=False)


class BagWeights(nn.Module):

    def __init__(self, mnia: float):
        super().__init__()
        self.mnia = float(mnia)                                              # fixed age scale
        self._raw = nn.Parameter(torch.zeros(()))               # sigmoid -> temp in (0, 1)

    @property
    def temp(self) -> torch.Tensor:
        """Pooling temperature, constrained to (0, 1) by a sigmoid on the raw Parameter."""
        return torch.sigmoid(self._raw)

    def forward(self, tokens: WalkTokens, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> pooling weights [Q,T] summing to 1, 0 on padding. (x sets dtype.)"""
        age = tokens.ages.clamp_min(0).float()                               # seed = 0, ctx >= 1, pad -> 0
        rec = -age / self.mnia                                               # LINEAR, unbounded  [Q, T]
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)                 # LINEAR, in [-(L-1), 0]
        logits = (self.temp * (rec + pos)).to(x.dtype)                       # [Q, T]
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; BagWeights is the only other trained module."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float,
                 init_irange: float = 1e-3, use_pop_bias: bool = False):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.use_pop_bias = bool(use_pop_bias)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(mean_node_inter_arrival)

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

        # Mixing weights over the score features, init ones. Length tracks the feature count.
        self.w = nn.Parameter(torch.ones(2 if self.use_pop_bias else 1))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> P [Q,d], the pooling-weighted gyro-midpoint of the walk-token cloud."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)                                               # all-padding walk -> use seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(tokens, x, valid)                                  # [Q, T] sums to 1, 0 on padding
        return self.geom.midpoint(x, w)                                         # [Q, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        score = w . feats, over [-geo] or [-geo, pop_bias[v]]."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        feats = [-geo]                                                          # closer -> higher
        if self.use_pop_bias:
            v_nodes = cand_tokens.seeds.view(b, c)                              # [B, C] candidate node ids
            feats.append(self.pop_bias(v_nodes).squeeze(-1))                    # [B, C] learned per-node scalar
        return (self.w * torch.stack(feats, dim=-1)).sum(dim=-1)                # [B, C]
