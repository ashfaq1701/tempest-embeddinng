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

    def __init__(self, E: nn.Embedding, median_inter_event_time: float, max_walk_len: int,
                 hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.E = E
        self.age_scale = float(median_inter_event_time)
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

        points = F.embedding(nodes, self.E.weight)                           # [Q, T, d]
        # Features see positions only; E's gradient flows through `points`.
        fixed = points.detach()                                              # [Q, T, d]

        valid_f = valid.to(fixed.dtype)                                      # [Q, T]
        mid = weighted_midpoint(fixed, valid_f / valid_f.sum(-1, keepdim=True))  # [Q, d]

        age = tokens.ages.flatten(1).clamp_min(0).to(fixed.dtype)            # [Q, T]
        pos = tokens.positions.flatten(1).to(fixed.dtype)                    # [Q, T]
        age = torch.log1p(age / self.age_scale)                              # [Q, T]  unit-free
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

    def __init__(self, num_nodes: int, median_inter_event_time: float, max_walk_len: int, d_emb: int,
                 hidden_dim: int = 32, n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.median_inter_event_time = float(median_inter_event_time)
        self.max_walk_len = int(max_walk_len)
        self.d_emb = int(d_emb)
        torch.manual_seed(seed)

        # Uniform inside the unit ball: isotropic direction, radius U^(1/d) (uniform by volume).
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            direction = F.normalize(torch.randn(self.num_nodes, self.d_emb), dim=-1)
            radius = torch.rand(self.num_nodes, 1).pow(1.0 / self.d_emb)
            self.E.weight.copy_(direction * radius)

        self.bag_weights = BagWeights(self.E, self.median_inter_event_time, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler)

        self.temp = nn.Parameter(torch.tensor([1.0]))                       # score temperature

    def mean_radius(self) -> torch.Tensor:
        """Mean norm of the table (its distance from the origin), detached."""
        r_mean = self.E.weight.detach().norm(dim=-1).mean()                  # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Score candidates by the raw distance between the two pooled points."""
        r_mean = self.mean_radius()                                          # [] (the pooler's own units)
        pooled_u = self.bag_weights(src_tokens, r_mean)                      # [b, d]
        pooled_v = self.bag_weights(cand_tokens, r_mean)                     # [b*c, d]
        pooled_v = pooled_v.view(pooled_u.shape[0], -1, pooled_u.shape[-1])  # [b, c, d]
        return self.temp * -dist(pooled_u.unsqueeze(1), pooled_v)            # [b, c]
