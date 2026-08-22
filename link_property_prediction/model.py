"""Dead-simple centroid-to-centroid head on the Poincaré ball: s(u,v) = temperature * (-d(P_u,P_v)).
No spread, cosine, or pop_bias channels — the score is purely the radius-sensitive geodesic distance,
which forces the model to use the radial/hierarchy dimension. P_x is the weighted gyro-midpoint of x's
walk-token bag; the pooling weights are a softmax over each walk's tokens of a learnable linear combo of
two per-token features, -(age/age_temp) and -pos (age_temp and the feature weights both learnable). A
minimal base (4 head params: temperature, age_temp, 2 feature weights) to scale up from."""

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
    """Dead-simple token pooling: softmax over a walk's tokens of a learnable linear combination of two
    per-token features, recency -log1p(age/s) and -pos. No MLP, no geometry features (a base to scale
    up from).

    The age scale is LOG-parameterised: s = exp(age_temp), age_temp init log(mnia) (mean node
    inter-arrival). Learning the log matters — Adam's step is ~lr in ABSOLUTE units, so a parameter
    living at ~1e6 moves ~1e-9 of itself per step and never leaves its init; on the log the same step
    is a relative one, so s can traverse decades within an epoch. exp() also keeps s > 0 with no clamp.
    Larger s flattens the recency signal (bag -> more uniform); smaller s sharpens it onto the newest
    tokens."""

    def __init__(self, mnia: float):
        super().__init__()
        self.age_temp = nn.Parameter(torch.log1p(torch.tensor(float(mnia))))  # log age scale, init log1p(mnia)
        self.w = nn.Parameter(torch.ones(2))                      # learnable per-feature weights, init 1

    def forward(self, tokens: WalkTokens, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> pooling weights [Q,T] summing to 1, 0 on padding. (x unused here.)"""
        age = tokens.ages.clamp_min(0).float()                               # seed = 0, ctx >= 1, pad -> 0
        rec = -torch.log1p(age / self.age_temp.exp())                        # -log1p(age/s)  [Q, T]
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)                 # -pos         [Q, T]
        feats = torch.stack([rec, pos], dim=-1).to(x.dtype)                  # [Q, T, 2]
        logits = (feats * self.w).sum(dim=-1)                               # [Q, T]  linear over features
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

        # Learned scalar temperature on the geodesic logit (init 1). Scales -geo before the per-query
        # softmax CE so the model can sharpen/soften the ranking without any extra score channel.
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
