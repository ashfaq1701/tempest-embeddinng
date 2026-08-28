"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = w . [-d_v, d_u + d_v - d_uv].

A learned 2-vector w (init [1,1]) mixes two isometry-invariant channels: the candidate depth (-d_v,
d_v = dist0(P_v)) and the Gromov overlap d_u + d_v - d_uv = 2*(u|v)_o (d_u = dist0(P_u)). At w=[1,1]
the score is -d_uv up to a per-query constant (pure geodesic), so any mix is a learned departure; the
ratio w[1]/w[0] reads how far the model moves from metric scoring. No popularity term.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over
-age/mnia - (pos-1), both priors RAW (no log1p) and at fixed unit scale -- there is no learned pooling
temperature. Only learned head param: geo_temp."""

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

    def __init__(self, mnia: float):
        super().__init__()
        self.mnia = float(mnia)                                              # fixed age scale

    def forward(self, tokens: WalkTokens, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> pooling weights [Q,T] summing to 1, 0 on padding. (x sets dtype.)"""
        age = tokens.ages.clamp_min(0).float()                               # seed = 0, ctx >= 1, pad -> 0
        rec = -age / self.mnia                                               # LINEAR, unbounded  [Q, T]
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)                 # LINEAR, in [-(L-1), 0]
        logits = (rec + pos).to(x.dtype)                                     # [Q, T]
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; the pooling weights carry no learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float,
                 init_irange: float = 1e-3):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(mean_node_inter_arrival)

        # Near-origin init: uniform(-irange, irange) per coord -> r ~ 2*irange*sqrt(d/3).
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.projx(
                (torch.rand(self.num_nodes, self.d_emb) * 2 - 1) * float(init_irange))
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        # Learned 2-vector mix over [depth, overlap], init [1,1] -- at init the score equals -d_uv
        # (pure geodesic) up to a per-query constant; departures are learned. w[0] scales candidate
        # depth (-d_v), w[1] scales the Gromov overlap (d_u + d_v - d_uv = 2*(u|v)_o).
        self.w = nn.Parameter(torch.tensor([1.0, 1.0]))

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
        score = w . [-d_v, d_u + d_v - d_uv] = w . [candidate depth, 2*Gromov overlap]. At w=[1,1] this
        equals -d_uv up to a per-query constant (pure geodesic); the mix is a learned departure."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        d_u = self.geom.dist0(p_u).unsqueeze(1)                                 # [B, 1] source radius
        d_v = self.geom.dist0(p_v)                                              # [B, C] candidate radius
        radial = -d_v                                                          # candidate depth
        angular = d_u + d_v - geo                                              # 2 * Gromov product (u|v)_o
        return (self.w * torch.stack([radial, angular], dim=-1)).sum(dim=-1)   # [B, C] learned mix
