import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_SQ_DIST_FLOOR = 1e-12


def dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean distance over the last axis; the floor keeps the gradient finite at x == y."""
    return ((x - y) ** 2).sum(-1).clamp_min(_SQ_DIST_FLOOR).sqrt()


def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean distances between x [..., N, d] and y [..., M, d] -> [..., N, M], with no [N, M, d] tensor."""
    sq = (x * x).sum(-1, keepdim=True) + (y * y).sum(-1).unsqueeze(-2) - 2 * x @ y.transpose(-1, -2)
    return sq.clamp_min(_SQ_DIST_FLOOR).sqrt()


def weighted_midpoint(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean of points x [..., T, d] under weights w [..., T] summing to 1."""
    return (w.unsqueeze(-1) * x).sum(-2)                                     # [..., d]


class BagWeights(nn.Module):
    """Pooling weights from token features [age, pos, d_mid, d_0, d_other]: the last is each token's
    distance to the pair's other side, so a bag is pooled once per node it is seen from."""

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
        self.n_feat = 5                                                      # age, pos, d_mid, d_0, d_other
        layers = [nn.Linear(self.n_feat, self.hidden), nn.GELU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens: WalkTokens, other_side: torch.Tensor, r_mean: torch.Tensor) -> torch.Tensor:
        """The bag's pooled point seen from each of its K other-side nodes: other_side [Q, K] node ids
        -> [Q, K, d], the weighted midpoint of its tokens under per-view weights."""
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
        mid = weighted_midpoint(fixed, valid_f / valid_f.sum(-1, keepdim=True))  # [Q, d]

        age = tokens.ages.flatten(1).clamp_min(0).to(fixed.dtype)            # [Q, T]
        pos = tokens.positions.flatten(1).to(fixed.dtype)                    # [Q, T]
        age = torch.log1p(age) / self.log_T_train                            # [Q, T]
        pos = pos / self.max_walk_len                                        # [Q, T]

        tiny = torch.finfo(fixed.dtype).tiny

        d_tok_mid = dist(fixed, mid.unsqueeze(-2))                           # [Q, T]
        d_mid_mean = (d_tok_mid * valid_f).sum(-1, keepdim=True) / valid_f.sum(-1, keepdim=True)  # [Q, 1]
        d_mid = d_tok_mid * valid_f / d_mid_mean.clamp_min(tiny)             # [Q, T]

        d_0 = fixed.norm(dim=-1) * valid_f / r_mean                          # [Q, T]

        others = F.embedding(other_side, self.E.weight).detach()             # [Q, K, d]
        d_other = pairwise_dist(others, fixed) * valid_f.unsqueeze(1) / r_mean  # [Q, K, T]

        feats = torch.stack([age, pos, d_mid, d_0], dim=-1)                  # [Q, T, 4]
        feats = torch.cat([feats.unsqueeze(1).expand(-1, d_other.shape[1], -1, -1),
                           d_other.unsqueeze(-1)], dim=-1)                   # [Q, K, T, 5]
        logits = self.net(feats).squeeze(-1)                                 # [Q, K, T]
        log_weights = torch.log_softmax(logits.masked_fill(~valid.unsqueeze(1), float("-inf")), dim=-1)  # [Q, K, T]
        return weighted_midpoint(points.unsqueeze(1), log_weights.exp())     # [Q, K, d]


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

        # Isotropic normal with |x| concentrating at INIT_RADIUS.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            self.E.weight.copy_(torch.randn(self.num_nodes, self.d_emb)
                                * (self.INIT_RADIUS / self.d_emb ** 0.5))

        self.bag_weights = BagWeights(self.E, self.T_train, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler)

        self.w = nn.Parameter(torch.tensor([1.0]))

    def mean_radius(self) -> torch.Tensor:
        """Mean norm of the table (its distance from the origin), detached."""
        r_mean = self.E.weight.detach().norm(dim=-1).mean()                  # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Score candidates by the raw distance between u's bag pooled as seen from v and v's bag
        pooled as seen from u."""
        r_mean = self.mean_radius()                                          # [] (the pooler's own units)
        u = src_tokens.seeds                                                 # [b]
        v = cand_tokens.seeds.view(u.shape[0], -1)                           # [b, c]
        pooled_u = self.bag_weights(src_tokens, v, r_mean)                   # [b, c, d]  one view per candidate
        u_per_cand = u.repeat_interleave(v.shape[1]).unsqueeze(1)            # [b*c, 1]
        pooled_v = self.bag_weights(cand_tokens, u_per_cand, r_mean)         # [b*c, 1, d]
        pooled_v = pooled_v.view(pooled_u.shape)                             # [b, c, d]
        return self.w * -dist(pooled_u, pooled_v)                            # [b, c]
