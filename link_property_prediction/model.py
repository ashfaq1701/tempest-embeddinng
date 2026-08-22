"""Dead-simple centroid-to-centroid head on the Poincaré ball: s(u,v) = temperature * (-d(P_u,P_v)).
No spread, cosine, or pop_bias channels — the score is purely the radius-sensitive geodesic distance,
which forces the model to use the radial/hierarchy dimension. P_x is the weighted gyro-midpoint of x's
walk-token bag; the pooling weights are a PARAMETERLESS softmax over d(token, centroid) *
(-log1p(age/mnia) - log1p(pos-1)). ONE head param (temperature) — a minimal base to scale up from."""

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
    """Parameterless token pooling: softmax over d(token, bag_centroid) * (rec + pos), with
    rec = -log1p(age/mnia) and pos = -log1p(pos-1). Both priors are fixed (mnia = mean node
    inter-arrival) and nothing here is learned; the centroid is the UNWEIGHTED gyro-midpoint.

    Structural consequences of the `dist *` factor, all sign-driven (rec <= 0, pos <= 0, dist >= 0):
      - The SEED still has logit exactly 0 (age 0 and position 1 -> rec = pos = 0) and every other
        token is <= 0, so the seed remains the argmax whatever the geometry does. This scales the
        suppression of the others; it does not contest the top slot.
      - Among non-seed tokens, those NEAR the centroid are suppressed LESS (dist ~ 0 pulls their
        logit toward 0), so the pooler now favours the bag's geometric core over its outliers.
      - Logit SCALE is now tied to |E|: early on E is near the origin, distances are ~0.1, so the
        logits shrink and pooling goes near-UNIFORM; as E expands the same priors sharpen. The
        pooling temperature is no longer fixed -- it is coupled to embedding radius.
      - Pooling weights now depend on E, so gradients flow from the softmax into E. Previously the
        weights came only from token metadata and E saw no gradient through this path.

    Measured seed share of the pooling mass on YouTube (all training bags, cold excluded; uniform
    would be 0.244):

        recency only                      0.485
        -log1p(pos-1) only                0.494
        BOTH, this pooler                 0.693
        -(pos-1) linear only              0.657
        recency + -(pos-1), w=[1,1]       0.800     <- the previous feature pair, untrained
        same, trained to epoch 6          0.960

    Log-compressing the position channel is what pulls the seed down from 0.800 to 0.693: it halves
    that channel's standalone pull (0.657 -> 0.494). The residual 0.693 is structural and no reshaping
    reaches it -- the seed is the argmax of BOTH channels at once and occupies 5 of ~20.5 slots."""

    def __init__(self, mnia: float, geom: "PoincareManifold"):
        super().__init__()
        self.mnia = float(mnia)                                              # fixed age scale
        self.geom = geom                                                     # plain object -> NOT registered
                                                                             # as a submodule, state_dict unchanged

    @staticmethod
    def centroid(geom: "PoincareManifold", x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """UNWEIGHTED gyro-midpoint of the valid tokens -> [Q, 1, d], broadcastable against x [Q,T,d]."""
        vf = valid.to(x.dtype)
        n = vf.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return geom.midpoint(x, vf / n).unsqueeze(-2)

    def forward(self, tokens: WalkTokens, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> pooling weights [Q,T] summing to 1, 0 on padding. (x sets dtype.)"""
        age = tokens.ages.clamp_min(0).float()                               # seed = 0, ctx >= 1, pad -> 0
        rec = -torch.log1p(age / self.mnia)                                  # -log1p(age/mnia)  [Q, T]
        pos = -torch.log1p(tokens.positions.clamp_min(1).float() - 1.0)      # -log1p(pos-1)     [Q, T]
        c = self.centroid(self.geom, x, valid)                               # [Q, 1, d] bag centroid
        dist = self.geom.dist(x, c)                                          # [Q, T] >= 0, geodesic to centroid
        logits = (dist * (rec + pos)).to(x.dtype)                            # [Q, T] prior SCALED by spread
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; BagWeights is the only other trained module."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float,
                 init_irange: float = 1e-3):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(mean_node_inter_arrival, self.geom)

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
