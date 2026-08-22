"""Dead-simple centroid-to-centroid head on the Poincaré ball:

    s(u,v) = temperature * (-d(P_u, P_v) + rho_v),   rho_v = dist0(P_v)

Two FIXED unit-weight priors — the pair's geodesic distance and the candidate's own radius — with no
learnable balance between them, so the temperature is the ONLY head parameter. rho_v carries no
dependence on u and so acts as a popularity / prominence prior. P_x is the weighted gyro-midpoint of
x's walk-token bag; the pooling weights are likewise a PARAMETERLESS softmax over two fixed priors,
-log1p(age/mnia) and -log1p(pos-1)."""

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
        """Hyperbolic radius from the origin, 2*artanh(||x||)."""
        return self.manifold.dist0(x)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted gyro-midpoint: x [Q,T,d], w [Q,T] -> [Q,d]."""
        return self.manifold.weighted_midpoint(x, weights=w, reducedim=[-2], dim=-1, keepdim=False)


class BagWeights(nn.Module):
    """Parameterless token pooling: softmax over -log1p(age/mnia) - log1p(pos-1). Both channels are
    fixed priors on a FIXED scale (mnia = mean node inter-arrival); nothing here is learned.

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

    def __init__(self, mnia: float):
        super().__init__()
        self.mnia = float(mnia)                                              # fixed age scale

    def forward(self, tokens: WalkTokens, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> pooling weights [Q,T] summing to 1, 0 on padding. (x sets dtype.)"""
        age = tokens.ages.clamp_min(0).float()                               # seed = 0, ctx >= 1, pad -> 0
        rec = -torch.log1p(age / self.mnia)                                  # -log1p(age/mnia)  [Q, T]
        pos = -torch.log1p(tokens.positions.clamp_min(1).float() - 1.0)      # -log1p(pos-1)     [Q, T]
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

        score = temperature * (-d(P_u,P_v) + rho_v),   rho_v = dist0(P_v)

        Both channels enter as FIXED unit-weight priors -- there is no learnable balance between
        them, so the only head parameter is the temperature. rho_v depends on the candidate alone,
        not on u, so it acts as a popularity / prominence prior; the source's own radius is omitted
        because it is constant within a query and cancels from the loss and the gradient under the
        per-query softmax CE (sum_c dL/ds_c = 0)."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        rho_v = self.geom.dist0(p_v)                                            # [B, C] candidate radius
        return self.temperature * (-geo + rho_v)                               # [B, C]
