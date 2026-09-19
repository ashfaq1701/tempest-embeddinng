import math

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold.

    Features per token are [log1p(age)/log1p(T_train), hop/max_walk_len, d(token, mid)],
    softmax over the bag, then the Lorentzian midpoint: [Q, T, d] -> [Q, d].

    `mid` is the bag's own UNWEIGHTED midpoint -- the Lorentzian centroid under uniform
    weights over the valid tokens -- so the third feature is a per-token spread signal:
    how far this token sits from the middle of its own bag.

    The pooling does NOT depend on who is being scored -- each query is summarised once,
    independently of its candidates.

    Geometric features are detached; the midpoint is NOT. That split is load-bearing:
    detaching the points would leave E with no gradient path at all.
    """

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, T_train: int,
                 max_walk_len: int, hidden_dim: int = 32):
        super().__init__()
        self.geom = geom
        self.E = E
        self.hidden = int(hidden_dim)
        # Divisors that put age and hop on [0, 1]-ish scales: the train-split time span
        # and the walk-length cap. Required, because a 0 here is a NaN in the weights.
        self.T_train = int(T_train)
        self.max_walk_len = int(max_walk_len)
        self.n_feat = 3
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, tokens: WalkTokens) -> torch.Tensor:
        """`tokens` has Q rows of T walk tokens -> [Q, d]."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        # One lookup. Live for the pooled midpoint, detached for the features.
        x_tokens = F.embedding(nodes, self.E.weight)                         # [Q, T, d]
        xt = x_tokens.detach()                                               # [Q, T, d]

        # The bag's unweighted midpoint: uniform weights over the valid tokens. The
        # cold-start fix above guarantees at least one valid token per row, so the
        # denominator is never zero.
        u = valid.to(xt.dtype)
        mid = self.geom.midpoint(xt, u / u.sum(-1, keepdim=True))            # [Q, d]

        # Every feature is [Q, T]: two from the walk, one pair distance, no radii.
        age = torch.log1p(tokens.ages.clamp_min(0).to(xt.dtype)) / math.log1p(self.T_train)
        pos = tokens.positions.to(xt.dtype) / self.max_walk_len
        d_tok_mid = self.geom.dist(xt, mid.unsqueeze(-2))

        feat = torch.stack([age, pos, d_tok_mid], dim=-1).to(xt.dtype)       # [Q, T, 3]
        logits = self.net(feat).squeeze(-1)                                  # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
        return self.geom.midpoint(x_tokens, w)                               # [Q, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, T_train: int, max_walk_len: int,
                 hidden_dim: int = 32, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, irange=self.INIT_IRANGE)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.T_train = int(T_train)
        self.max_walk_len = int(max_walk_len)

        self.bag_weights = BagWeights(self.geom, self.E, T_train=self.T_train,
                                      max_walk_len=self.max_walk_len, hidden_dim=hidden_dim)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        # Each side is pooled once, independently of the other. The candidate rows arrive
        # flattened as b*c, so the only reshaping left is folding c back out.
        p_u = self.bag_weights(src_tokens)                                   # [b, d]
        p_v = self.bag_weights(cand_tokens)                                  # [b*c, d]
        b, d = p_u.shape
        p_v = p_v.view(b, p_v.shape[0] // b, d)                              # [b, c, d]

        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                          # [b, c]
        return self.geo_temp * (-geo)
