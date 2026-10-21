import math

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .product import ProductManifold
from .walk_tokens import WalkTokens


class BagWeights(nn.Module):

    def __init__(self, geom: geoopt.Manifold, E: nn.Embedding, T_train: int,
                 max_walk_len: int, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.geom = geom
        self.E = E
        self.log_T_train = math.log1p(T_train)
        self.max_walk_len = int(max_walk_len)
        self.hidden = int(hidden_dim)
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_feat = 4
        layers = [nn.Linear(self.n_feat, self.hidden), nn.GELU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)

        self.skip = nn.Linear(self.n_feat, 1, bias=False)
        with torch.no_grad():
            self.skip.weight.copy_(torch.tensor([[-1.0, -1.0, 0.0, 0.0]]))

    def forward(self, tokens: WalkTokens, r_mean: torch.Tensor):
        """Token points [Q, T, d], log-weights [Q, T] (-inf on padding) and weighted midpoint [Q, d]."""
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()                 # [Q, T]
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        points = F.embedding(nodes, self.E.weight)                           # [Q, T, d]
        # Features see positions only; E's gradient flows through `points`.
        fixed = points.detach()                                              # [Q, T, d]

        valid_f = valid.to(fixed.dtype)                                      # [Q, T]
        mid = self.geom.weighted_midpoint(fixed, valid_f / valid_f.sum(-1, keepdim=True))  # [Q, d]

        age = tokens.ages.flatten(1).clamp_min(0).to(fixed.dtype)            # [Q, T]
        pos = tokens.positions.flatten(1).to(fixed.dtype)                    # [Q, T]
        age = torch.log1p(age) / self.log_T_train                            # [Q, T]
        pos = pos / self.max_walk_len                                        # [Q, T]

        tiny = torch.finfo(fixed.dtype).tiny

        d_tok_mid = self.geom.dist(fixed, mid.unsqueeze(-2))                 # [Q, T]
        d_mid_mean = (d_tok_mid * valid_f).sum(-1, keepdim=True) / valid_f.sum(-1, keepdim=True)  # [Q, 1]
        d_mid = d_tok_mid * valid_f / d_mid_mean.clamp_min(tiny)             # [Q, T]

        d_0 = self.geom.dist0(fixed) * valid_f / r_mean                      # [Q, T]

        feats = torch.stack([age, pos, d_mid, d_0], dim=-1)                  # [Q, T, 4]
        logits = (self.skip(feats) + self.net(feats)).squeeze(-1)            # [Q, T]
        log_weights = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)  # [Q, T]
        pooled = self.geom.weighted_midpoint(points, log_weights.exp())      # [Q, d]
        return points, log_weights, pooled


class LinkPredHead(nn.Module):

    INIT_RADIUS = 1e-3

    def __init__(self, num_nodes: int, T_train: int, max_walk_len: int, d_emb: int,
                 hidden_dim: int = 32, n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.T_train = int(T_train)
        self.max_walk_len = int(max_walk_len)
        self.d_emb = int(d_emb)
        self.geom = ProductManifold(self.d_emb // 2, self.d_emb - self.d_emb // 2)   # [Euclidean | Lorentz]
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, radius=self.INIT_RADIUS)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, self.T_train, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler)

        self.w = nn.Parameter(torch.tensor([1.0]))
        self.beta = nn.Parameter(torch.tensor([0.0]))                       # >= 0; 0 is the weighted mean

    @torch.no_grad()
    def project_(self) -> None:
        """Clamp beta to >= 0 after an optimiser step."""
        self.beta.clamp_(min=0.0)

    def mean_radius(self) -> torch.Tensor:
        """Mean geodesic radius of the table, detached."""
        r_mean = self.geom.dist0(self.E.weight.detach()).mean()              # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    def gap_to_bag(self, x: torch.Tensor, points: torch.Tensor,
                   log_weights: torch.Tensor) -> torch.Tensor:
        """Distance from x to a bag under softmax(log w_j - beta * d_j): mean at 0, nearest as beta grows."""
        dist_to_tokens = self.geom.dist(x.unsqueeze(-2), points)             # [..., T]
        attn = torch.softmax(log_weights - self.beta * dist_to_tokens, dim=-1)  # [..., T]
        return (attn * dist_to_tokens).sum(-1)                               # [...]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Score candidates by the pooled-point-to-bag gaps in both directions."""
        r_mean = self.mean_radius()                                          # []
        points_u, log_weights_u, pooled_u = self.bag_weights(src_tokens, r_mean)   # [b, T, d], [b, T], [b, d]
        points_v, log_weights_v, pooled_v = self.bag_weights(cand_tokens, r_mean)  # [b*c, T, d], [b*c, T], [b*c, d]
        b, t, d = points_u.shape
        points_v = points_v.view(b, -1, t, d)                                # [b, c, T, d]
        log_weights_v = log_weights_v.view(b, -1, t)                         # [b, c, T]
        pooled_v = pooled_v.view(b, -1, d)                                   # [b, c, d]

        gap_uv = (self.gap_to_bag(pooled_u.unsqueeze(1), points_v, log_weights_v)
                  + self.gap_to_bag(pooled_v, points_u.unsqueeze(1), log_weights_u.unsqueeze(1)))  # [b, c]
        return self.w * -gap_uv
