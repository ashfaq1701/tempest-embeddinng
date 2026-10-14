import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_SQ_DIST_FLOOR = 1e-12


def dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean distance over the last axis; the floor keeps the gradient finite at x == y."""
    return ((x - y) ** 2).sum(-1).clamp_min(_SQ_DIST_FLOOR).sqrt()


def weighted_midpoint(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean of points x [..., T, d] under weights w [..., T] summing to 1."""
    return (w.unsqueeze(-1) * x).sum(-2)                                     # [..., d]


class BagWeights(nn.Module):

    def __init__(self, E: nn.Embedding, T_train: int, max_walk_len: int,
                 hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
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

    def forward(self, tokens: WalkTokens):
        return self.pool(tokens)

    def pool(self, tokens: WalkTokens):
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()                 # [Q, T]
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x_tokens = F.embedding(nodes, self.E.weight)                         # [Q, T, d]
        xt = x_tokens.detach()                                               # [Q, T, d]

        u = valid.to(xt.dtype)                                               # [Q, T]
        mid = weighted_midpoint(xt, u / u.sum(-1, keepdim=True))             # [Q, d]

        age = tokens.ages.flatten(1).clamp_min(0).to(xt.dtype)               # [Q, T]
        pos = tokens.positions.flatten(1).to(xt.dtype)                       # [Q, T]
        age = torch.log1p(age) / self.log_T_train                            # [Q, T]
        pos = pos / self.max_walk_len                                        # [Q, T]

        tiny = torch.finfo(xt.dtype).tiny

        d_tok_mid = dist(xt, mid.unsqueeze(-2))                              # [Q, T]
        d_mid_mean = (d_tok_mid * u).sum(-1, keepdim=True) / u.sum(-1, keepdim=True)  # [Q, 1]
        d_mid = d_tok_mid * u / d_mid_mean.clamp_min(tiny)                   # [Q, T]

        r_mean = self.E.weight.detach().norm(dim=-1).mean()                  # []
        d_0 = xt.norm(dim=-1) * u / r_mean.clamp_min(tiny)                   # [Q, T]

        feats = torch.stack([age, pos, d_mid, d_0], dim=-1)                  # [Q, T, 4]
        logits = (self.skip(feats) + self.net(feats)).squeeze(-1)            # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        p = weighted_midpoint(x_tokens, w)                                   # [Q, d]

        d_tok_p = dist(xt, p.unsqueeze(-2))                                  # [Q, T]
        spread = (d_tok_p * u).sum(-1) / u.sum(-1)                           # [Q]
        return p, spread


class LinkPredHead(nn.Module):

    INIT_RADIUS = 1e-3

    def __init__(self, num_nodes: int, T_train: int, max_walk_len: int, d_emb: int,
                 hidden_dim: int = 32, n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.T_train = int(T_train)
        self.max_walk_len = int(max_walk_len)
        self.d_emb = int(d_emb)
        torch.manual_seed(seed)

        # Isotropic normal with |x| concentrating at INIT_RADIUS: master's wrapped normal at the
        # origin, whose exponential map is the identity in Euclidean space.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            self.E.weight.copy_(torch.randn(self.num_nodes, self.d_emb)
                                * (self.INIT_RADIUS / self.d_emb ** 0.5))

        self.bag_weights = BagWeights(self.E, self.T_train, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        p_u, _ = self.bag_weights(src_tokens)                                # [b, d]
        p_v, _ = self.bag_weights(cand_tokens)                               # [b*c, d]
        b, d = p_u.shape
        p_v = p_v.view(b, p_v.shape[0] // b, d)                              # [b, c, d]

        geo = dist(p_u.unsqueeze(1), p_v)                                    # [b, c]
        return self.geo_temp * (-geo)
