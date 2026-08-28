"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = -temp * (w . [d_uv, d_u * d_v]).

A learned 2-vector w (init [1,1]) mixes the geodesic distance with the radius product d_u * d_v
(d_u = dist0(P_u), d_v = dist0(P_v)); the outer minus makes w[0] > 0 a closer-is-better term and
w[1] > 0 a penalty on peripheral-peripheral pairs (favouring hubs) -- both weights stay positive.
An outer temperature temp (plain scalar, init 1) scales the logits. No popularity term.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over
-age/mnia - (pos-1), both priors RAW (no log1p) and at fixed unit scale -- there is no learned pooling
temperature. Learned head params: the 2 mix weights w and the outer temp."""

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

        # Learned 2-vector mix over [-d_uv, d_u*d_v], init [1,1]: w[0] scales the geodesic-proximity
        # term, w[1] scales the radius product (sign learned: negative => penalty on peripheral pairs).
        self.w = nn.Parameter(torch.tensor([1.0, 1.0]))
        # Outer temperature, plain linear parameter (init 1): scales the logits alongside the mix.
        self.temp = nn.Parameter(torch.tensor(1.0))

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
        score = -temp * (w . [d_uv, d_u*d_v]), temp a plain learned scalar (init 1). Both features
        are positive; the outer minus makes w[0]>0 = closer-better and w[1]>0 = penalty on peripheral
        pairs, so both weights stay positive (init [1,1])."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        d_uv = self.geom.dist(p_u.unsqueeze(1), p_v)                           # [B, C] geodesic distance
        d_u = self.geom.dist0(p_u).unsqueeze(1)                                 # [B, 1] source radius
        d_v = self.geom.dist0(p_v)                                              # [B, C] candidate radius
        mix = (self.w * torch.stack([d_uv, d_u * d_v], dim=-1)).sum(dim=-1)    # [B, C] channel mix
        return -self.temp * mix                                               # [B, C] outer minus
