import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_SQ_DIST_FLOOR = 1e-12
_NORM_FLOOR = 1e-12


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

        feats = torch.stack([age, pos, d_mid, d_0], dim=-1)                  # [Q, T, 4]
        logits = (self.skip(feats) + self.net(feats)).squeeze(-1)            # [Q, T]
        log_weights = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)  # [Q, T]
        pooled = weighted_midpoint(points, log_weights.exp())                # [Q, d]
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
        torch.manual_seed(seed)

        # Isotropic normal with |x| concentrating at INIT_RADIUS: master's wrapped normal at the
        # origin, whose exponential map is the identity in Euclidean space.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            self.E.weight.copy_(torch.randn(self.num_nodes, self.d_emb)
                                * (self.INIT_RADIUS / self.d_emb ** 0.5))

        self.bag_weights = BagWeights(self.E, self.T_train, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler)

        self.w = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))            # [distance, dot, cosine]; equal at init
        self.beta = nn.Parameter(torch.tensor([0.0]))                       # >= 0; 0 is the weighted mean

    @torch.no_grad()
    def project_(self) -> None:
        """Clamp beta to >= 0 after an optimiser step."""
        self.beta.clamp_(min=0.0)

    def mean_radius(self) -> torch.Tensor:
        """Mean norm of the table (its distance from the origin), detached."""
        r_mean = self.E.weight.detach().norm(dim=-1).mean()                  # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    @staticmethod
    def pair_metrics(x: torch.Tensor, points: torch.Tensor, r_mean: torch.Tensor) -> torch.Tensor:
        """[-distance, dot / r_mean, cos * r_mean] from x to every token, all in units of the mean radius."""
        x = x.unsqueeze(-2)                                                  # [..., 1, d]
        dist_to_tokens = dist(x, points)                                     # [..., T]
        dot = (x * points).sum(-1)                                           # [..., T]
        cos = dot / (x.norm(dim=-1) * points.norm(dim=-1)).clamp_min(_NORM_FLOOR)  # [..., T]
        return torch.stack([-dist_to_tokens, dot / r_mean, cos * r_mean], dim=-1)  # [..., T, 3]

    def bag_metrics(self, x: torch.Tensor, points: torch.Tensor, log_weights: torch.Tensor,
                    r_mean: torch.Tensor) -> torch.Tensor:
        """The pair metrics averaged over a bag under softmax(log w_j - beta * d_j): mean at 0, nearest as beta grows."""
        metrics = self.pair_metrics(x, points, r_mean)                       # [..., T, 3]
        attn = torch.softmax(log_weights + self.beta * metrics[..., 0], dim=-1)  # [..., T]; metrics[..., 0] = -d
        return (attn.unsqueeze(-1) * metrics).sum(-2)                        # [..., 3]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Score candidates by distance, dot and cosine from each side's pooled point to the other's bag."""
        r_mean = self.mean_radius()                                          # []
        points_u, log_weights_u, pooled_u = self.bag_weights(src_tokens, r_mean)   # [b, T, d], [b, T], [b, d]
        points_v, log_weights_v, pooled_v = self.bag_weights(cand_tokens, r_mean)  # [b*c, T, d], [b*c, T], [b*c, d]
        b, t, d = points_u.shape
        points_v = points_v.view(b, -1, t, d)                                # [b, c, T, d]
        log_weights_v = log_weights_v.view(b, -1, t)                         # [b, c, T]
        pooled_v = pooled_v.view(b, -1, d)                                   # [b, c, d]

        features = (self.bag_metrics(pooled_u.unsqueeze(1), points_v, log_weights_v, r_mean)
                    + self.bag_metrics(pooled_v, points_u.unsqueeze(1), log_weights_u.unsqueeze(1),
                                       r_mean))                              # [b, c, 3]
        return features @ self.w                                             # [b, c]
