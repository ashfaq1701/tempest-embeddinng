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


class BagSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention over a query's walk-token bag, replacing a learned pooler.

    Tokens are content (detached embedding / r_mean) + time encoding + hop embedding. Values are the
    raw embeddings and the heads' attention maps are averaged, so each token's contextual point
    sum_k A_jk E_k is a convex point in E-space; the bag's pooled point is their mean, i.e. each token
    is weighted by the attention it receives."""

    def __init__(self, E: nn.Embedding, max_walk_len: int, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.E = E
        self.n_heads = int(n_heads)
        self.d_head = int(d_model) // self.n_heads

        self.content = nn.Linear(E.weight.shape[1], d_model, bias=False)
        self.time_freq = nn.Parameter(torch.randn(d_model))                # time encoding cos(w * t + phi)
        self.time_phase = nn.Parameter(torch.zeros(d_model))
        self.hop = nn.Embedding(int(max_walk_len) + 1, d_model)            # 0 = padding, 1 = seed, ... = older
        self.norm = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)

    def forward(self, tokens: WalkTokens, r_mean: torch.Tensor):
        """Token points [Q, T, d], log pooling weights [Q, T] (-inf on padding) and pooled point [Q, d]."""
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()                 # [Q, T]
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        hop = tokens.positions.flatten(1).clone()                            # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True
            hop[cold, 0] = 1

        points = F.embedding(nodes, self.E.weight)                           # [Q, T, d]
        # Attention sees positions only; E's gradient flows through `points`.
        fixed = points.detach()                                              # [Q, T, d]
        log_age = torch.log1p(tokens.ages.flatten(1).clamp_min(0).to(fixed.dtype))  # [Q, T]

        x = (self.content(fixed / r_mean)
             + torch.cos(log_age.unsqueeze(-1) * self.time_freq + self.time_phase)
             + self.hop(hop))                                                # [Q, T, d_model]
        x = self.norm(x)                                                     # [Q, T, d_model]

        q_len, t_len, _ = x.shape
        q = self.W_q(x).view(q_len, t_len, self.n_heads, self.d_head).transpose(1, 2)   # [Q, H, T, d_head]
        k = self.W_k(x).view(q_len, t_len, self.n_heads, self.d_head).transpose(1, 2)   # [Q, H, T, d_head]
        scores = q @ k.transpose(-1, -2) / self.d_head ** 0.5                # [Q, H, T, T]
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1).mean(1)                         # [Q, T, T], heads averaged

        # Pooling weight of token k: the attention it receives, averaged over the bag's real tokens.
        valid_f = valid.to(fixed.dtype)                                      # [Q, T]
        weights = (valid_f.unsqueeze(-1) * attn).sum(1) / valid_f.sum(-1, keepdim=True)  # [Q, T]
        log_weights = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        log_weights = log_weights.masked_fill(~valid, float("-inf"))         # [Q, T]
        pooled = weighted_midpoint(points, weights)                          # [Q, d] = mean_j sum_k A_jk E_k
        return points, log_weights, pooled


class LinkPredHead(nn.Module):

    INIT_RADIUS = 1e-3

    def __init__(self, num_nodes: int, T_train: int, max_walk_len: int, d_emb: int,
                 d_model: int = 64, n_heads: int = 4, seed: int = 42):
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

        self.bag_attention = BagSelfAttention(self.E, self.max_walk_len, d_model=d_model, n_heads=n_heads)

        self.w = nn.Parameter(torch.tensor([1.0]))
        self.beta = nn.Parameter(torch.tensor([0.0]))                       # >= 0; 0 is the weighted mean

    @torch.no_grad()
    def project_(self) -> None:
        """Clamp beta to >= 0 after an optimiser step."""
        self.beta.clamp_(min=0.0)

    def mean_radius(self) -> torch.Tensor:
        """Mean norm of the table (its distance from the origin), detached."""
        r_mean = self.E.weight.detach().norm(dim=-1).mean()                  # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    def gap_to_bag(self, x: torch.Tensor, points: torch.Tensor, log_weights: torch.Tensor,
                   r_mean: torch.Tensor) -> torch.Tensor:
        """Distance from x to a bag in units of the table's mean radius, under softmax(log w_j - beta * d_j):
        mean at 0, nearest as beta grows. Invariant to rescaling the whole table."""
        dist_to_tokens = dist(x.unsqueeze(-2), points) / r_mean              # [..., T]
        attn = torch.softmax(log_weights - self.beta * dist_to_tokens, dim=-1)  # [..., T]
        return (attn * dist_to_tokens).sum(-1)                               # [...]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Score candidates by the pooled-point-to-bag gaps in both directions."""
        r_mean = self.mean_radius()                                          # []
        points_u, log_weights_u, pooled_u = self.bag_attention(src_tokens, r_mean)   # [b, T, d], [b, T], [b, d]
        points_v, log_weights_v, pooled_v = self.bag_attention(cand_tokens, r_mean)  # [b*c, T, d], [b*c, T], [b*c, d]
        b, t, d = points_u.shape
        points_v = points_v.view(b, -1, t, d)                                # [b, c, T, d]
        log_weights_v = log_weights_v.view(b, -1, t)                         # [b, c, T]
        pooled_v = pooled_v.view(b, -1, d)                                   # [b, c, d]

        gap_uv = (self.gap_to_bag(pooled_u.unsqueeze(1), points_v, log_weights_v, r_mean)
                  + self.gap_to_bag(pooled_v, points_u.unsqueeze(1), log_weights_u.unsqueeze(1),
                                    r_mean))                                 # [b, c]
        return self.w * -gap_uv
