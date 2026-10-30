import math

import torch
import torch.nn as nn

from .walk_tokens import WalkTokens

_SQ_DIST_FLOOR = 1e-12


def dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean distance over the last axis; the floor keeps the gradient finite at x == y."""
    return ((x - y) ** 2).sum(-1).clamp_min(_SQ_DIST_FLOOR).sqrt()


def weighted_midpoint(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean of points x [..., T, d] under weights w [..., T] summing to 1."""
    return (w.unsqueeze(-1) * x).sum(-2)                                     # [..., d]


class RunningNorm(nn.Module):
    """Standardises x with an EMA of the mean/var of past training batches.
    A batch is never scored with its own stats: standardise first, then fold the batch in.
    Eval reads the frozen buffers. Stats are buffers: no gradient, saved in state_dict."""

    def __init__(self, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(()))
        self.register_buffer("running_var", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        if self.training:
            with torch.no_grad():
                self.running_mean.lerp_(x.mean(), self.momentum)
                self.running_var.lerp_(x.var(unbiased=False), self.momentum)
        return out


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

    def forward(self, tokens: WalkTokens, r_mean: torch.Tensor) -> torch.Tensor:
        """The bag's pooled point: the weighted midpoint of its tokens [Q, d]."""
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()                 # [Q, T]
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        points = self.E(nodes)                                               # [Q, T, d]
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
        logits = self.net(feats).squeeze(-1)                                 # [Q, T]
        log_weights = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)  # [Q, T]
        return weighted_midpoint(points, log_weights.exp())                  # [Q, d]


class LinkPredHead(nn.Module):

    def __init__(self, num_nodes: int, T_train: int, max_walk_len: int, d_emb: int,
                 hidden_dim: int = 32, n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.T_train = int(T_train)
        self.max_walk_len = int(max_walk_len)
        self.d_emb = int(d_emb)
        torch.manual_seed(seed)

        # N(0, 1/d): row norm ~1 at any width (Flax / fairseq embedding init).
        self.E = nn.Embedding(self.num_nodes, self.d_emb, sparse=True)      # sparse grad: SparseAdam
        nn.init.normal_(self.E.weight, std=self.d_emb ** -0.5)

        self.bag_weights = BagWeights(self.E, self.T_train, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler)

        self.dist_norm = RunningNorm()

    def mean_radius(self) -> torch.Tensor:
        """Mean norm of the table (its distance from the origin), detached."""
        r_mean = self.E.weight.detach().norm(dim=-1).mean()                  # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Score candidates by the distance between the two pooled points, standardised by a
        running estimate of its own mean and variance over past training batches."""
        r_mean = self.mean_radius()                                          # [] (the pooler's own units)
        pooled_u = self.bag_weights(src_tokens, r_mean)                      # [b, d]
        pooled_v = self.bag_weights(cand_tokens, r_mean)                     # [b*c, d]
        pooled_v = pooled_v.view(pooled_u.shape[0], -1, pooled_u.shape[-1])  # [b, c, d]
        return self.dist_norm(-dist(pooled_u.unsqueeze(1), pooled_v))        # [b, c]
