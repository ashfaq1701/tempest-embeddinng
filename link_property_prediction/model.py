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

    @staticmethod
    def bag_slots(tokens: WalkTokens):
        """Flattened bag: node ids, real-slot mask and hop index [Q, T]. A cold bag keeps its seed."""
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()                 # [Q, T]
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        hop = tokens.positions.flatten(1).clone()                            # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True
            hop[cold, 0] = 1
        return nodes, valid, hop

    def time_encode(self, ages: torch.Tensor) -> torch.Tensor:
        """cos(w * log1p(age) + phi) per channel: [Q, T] ages -> [Q, T, d_model]."""
        log_age = torch.log1p(ages.clamp_min(0).to(self.time_freq.dtype))    # [Q, T]
        return torch.cos(log_age.unsqueeze(-1) * self.time_freq + self.time_phase)

    def encode(self, fixed: torch.Tensor, ages: torch.Tensor, hop: torch.Tensor,
               r_mean: torch.Tensor) -> torch.Tensor:
        """Token vectors: content + time + hop, layer-normalised. [Q, T, d] -> [Q, T, d_model]."""
        x = self.content(fixed / r_mean) + self.time_encode(ages) + self.hop(hop)
        return self.norm(x)

    def attention(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Scaled dot-product self-attention over real tokens, heads averaged: [Q, T, T], rows sum to 1."""
        q_len, t_len, _ = x.shape
        q = self.W_q(x).view(q_len, t_len, self.n_heads, self.d_head).transpose(1, 2)   # [Q, H, T, d_head]
        k = self.W_k(x).view(q_len, t_len, self.n_heads, self.d_head).transpose(1, 2)   # [Q, H, T, d_head]
        scores = q @ k.transpose(-1, -2) / self.d_head ** 0.5                # [Q, H, T, T]
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        return torch.softmax(scores, dim=-1).mean(1)                         # [Q, T, T]

    @staticmethod
    def pooling_weights(attn: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Each token's weight in the bag: the attention it receives from the real tokens, averaged. [Q, T], sums to 1."""
        valid_f = valid.to(attn.dtype)                                       # [Q, T]
        return (valid_f.unsqueeze(-1) * attn).sum(1) / valid_f.sum(-1, keepdim=True)

    def forward(self, tokens: WalkTokens, r_mean: torch.Tensor) -> torch.Tensor:
        """The bag's attention-pooled point [Q, d]."""
        nodes, valid, hop = self.bag_slots(tokens)                           # [Q, T] each
        points = F.embedding(nodes, self.E.weight)                           # [Q, T, d]
        # Attention sees positions only; E's gradient flows through `points`.
        x = self.encode(points.detach(), tokens.ages.flatten(1), hop, r_mean)  # [Q, T, d_model]

        weights = self.pooling_weights(self.attention(x, valid), valid)      # [Q, T]
        return weighted_midpoint(points, weights)                            # [Q, d] = mean_j sum_k A_jk E_k


class LinkPredHead(nn.Module):

    INIT_RADIUS = 1e-3

    def __init__(self, num_nodes: int, max_walk_len: int, d_emb: int,
                 d_model: int = 64, n_heads: int = 4, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)

        # Isotropic normal with |x| concentrating at INIT_RADIUS.
        self.E = nn.Embedding(num_nodes, d_emb)
        with torch.no_grad():
            self.E.weight.copy_(torch.randn(num_nodes, d_emb) * (self.INIT_RADIUS / d_emb ** 0.5))

        self.bag_attention = BagSelfAttention(self.E, max_walk_len, d_model=d_model, n_heads=n_heads)

        self.w = nn.Parameter(torch.tensor([1.0]))

    def mean_radius(self) -> torch.Tensor:
        """Mean norm of the table (its distance from the origin), detached."""
        r_mean = self.E.weight.detach().norm(dim=-1).mean()                  # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Score candidates by the distance between the two attention-pooled points, in units of the
        table's mean radius."""
        r_mean = self.mean_radius()                                          # []
        pooled_u = self.bag_attention(src_tokens, r_mean)                    # [b, d]
        pooled_v = self.bag_attention(cand_tokens, r_mean)                   # [b*c, d]
        pooled_v = pooled_v.view(pooled_u.shape[0], -1, pooled_u.shape[-1])  # [b, c, d]
        return self.w * -dist(pooled_u.unsqueeze(1), pooled_v) / r_mean      # [b, c]
