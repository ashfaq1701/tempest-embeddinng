"""Dead-simple centroid-to-centroid head on the Poincaré ball: s(u,v) = temperature * (-d(P_u,P_v)).
No spread, cosine, or pop_bias channels — the score is purely the radius-sensitive geodesic distance,
which forces the model to use the radial/hierarchy dimension. P_x is the weighted gyro-midpoint of x's
walk-token bag; the pooling weights are a PARAMETERLESS softmax over two RAW fixed priors,
-age/mnia and -(pos-1) (no log1p). ONE head param (temperature) — a minimal base to scale up from."""

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
    """Parameterless token pooling: softmax over -age/mnia - (pos-1), a recency prior plus a hop
    prior, both raw (no log compression). mnia (mean node inter-arrival) is a fixed age scale, so
    -age/mnia reads as "how many typical inter-arrival gaps old is this token".

    Sign structure worth knowing: both priors are <= 0, and both are exactly 0 for the seed (age 0
    at position 1). The seed is therefore always the argmax and every other token sits below it --
    steeper priors mean a sharper softmax and more pooling mass on the seed. Since the seed IS the
    query node, that mass is what keeps P_u near E[u] for the geodesic score to work from."""

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
    """E is a ManifoldParameter; BagWeights is the only other trained module."""

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

        # Learned scalar on the geodesic logit (init 1). Scales -geo ahead of the per-query
        # softmax CE, letting the model sharpen or soften the ranking without a second channel.
        self.temperature = nn.Parameter(torch.tensor(1.0))

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
        score = temperature * (-geo)  (spread, cosine channels and pop_bias all removed -> pure geodesic)."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        return self.temperature * (-geo)                                       # [B, C] temperature-scaled score
