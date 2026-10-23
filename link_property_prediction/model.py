import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_SQ_DIST_FLOOR = 1e-12
_SDPA_MAX_BATCH = 65_535        # CUDA's grid y/z limit: the most bags one memory-efficient SDPA call can launch


def dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean distance over the last axis; the floor keeps the gradient finite at x == y."""
    return ((x - y) ** 2).sum(-1).clamp_min(_SQ_DIST_FLOOR).sqrt()


def weighted_midpoint(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean of points x [..., T, d] under weights w [..., T] summing to 1."""
    return (w.unsqueeze(-1) * x).sum(-2)                                     # [..., d]


class SelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention over the real tokens."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)                          # query, key, value, fused
        self.out = nn.Linear(d_model, d_model)

    @staticmethod
    def chunked_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """SDPA over chunks of at most _SDPA_MAX_BATCH bags; bags are independent, so this is exact."""
        return torch.cat([F.scaled_dot_product_attention(qc, kc, vc, attn_mask=mc)
                          for qc, kc, vc, mc in zip(q.split(_SDPA_MAX_BATCH), k.split(_SDPA_MAX_BATCH),
                                                    v.split(_SDPA_MAX_BATCH), mask.split(_SDPA_MAX_BATCH))])

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """h [Q, T, d_model], valid [Q, T] (keys that may be attended to) -> [Q, T, d_model]."""
        q_len, t_len, d_model = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)                               # [Q, T, d_model] each
        q, k, v = (x.view(q_len, t_len, self.n_heads, self.d_head).transpose(1, 2)
                   for x in (q, k, v))                                       # [Q, H, T, d_head] each
        attended = self.chunked_sdpa(q, k, v, valid[:, None, None, :])      # [Q, H, T, d_head]
        return self.out(attended.transpose(1, 2).reshape(q_len, t_len, d_model))


class BagWeights(nn.Module):
    """Pooling weights from self-attention over the bag's token features.

    Each token is its feature vector [age, pos, d_mid, d_0] -- no embedding content. Tokens see each
    other through one post-LN self-attention with a residual, then each gets one logit; softmax over the bag.
    The pooled point is the weighted midpoint of the raw embeddings."""

    N_FEAT = 4                                                               # age, pos, d_mid, d_0

    def __init__(self, E: nn.Embedding, T_train: int, max_walk_len: int,
                 d_model: int = 32, n_heads: int = 4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.E = E
        self.log_T_train = math.log1p(T_train)
        self.max_walk_len = int(max_walk_len)

        self.embed = nn.Linear(self.N_FEAT, d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.attn_norm = nn.LayerNorm(d_model)                              # post-LN: after the residual
        self.logit = nn.Linear(d_model, 1)

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
        h = F.gelu(self.embed(feats))                                        # [Q, T, d_model]
        h = self.attn_norm(h + self.attn(h, valid))                          # [Q, T, d_model]
        logits = self.logit(h).squeeze(-1)                                   # [Q, T]
        log_weights = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)  # [Q, T]
        pooled = weighted_midpoint(points, log_weights.exp())                # [Q, d]
        return points, log_weights, pooled


class LinkPredHead(nn.Module):

    INIT_RADIUS = 1e-3

    def __init__(self, num_nodes: int, T_train: int, max_walk_len: int, d_emb: int,
                 d_model: int = 32, n_heads: int = 4, seed: int = 42):
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
                                      d_model=d_model, n_heads=n_heads)

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
        points_u, log_weights_u, pooled_u = self.bag_weights(src_tokens, r_mean)   # [b, T, d], [b, T], [b, d]
        points_v, log_weights_v, pooled_v = self.bag_weights(cand_tokens, r_mean)  # [b*c, T, d], [b*c, T], [b*c, d]
        b, t, d = points_u.shape
        points_v = points_v.view(b, -1, t, d)                                # [b, c, T, d]
        log_weights_v = log_weights_v.view(b, -1, t)                         # [b, c, T]
        pooled_v = pooled_v.view(b, -1, d)                                   # [b, c, d]

        gap_uv = (self.gap_to_bag(pooled_u.unsqueeze(1), points_v, log_weights_v, r_mean)
                  + self.gap_to_bag(pooled_v, points_u.unsqueeze(1), log_weights_u.unsqueeze(1),
                                    r_mean))                                 # [b, c]
        return self.w * -gap_uv
