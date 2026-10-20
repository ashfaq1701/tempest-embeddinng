import math

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens


class BagWeights(nn.Module):

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, T_train: int,
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

        # Focus sharpness: beta -> 0 pools with the pooler's own weights (master); 0.1 ~ master.
        self.log_beta = nn.Parameter(torch.tensor([math.log(0.1)]))

    def forward(self, tokens: WalkTokens, others: torch.Tensor, r_mean: torch.Tensor) -> torch.Tensor:
        """Pool each bag once per point it is seen from. others [Q, K, d]: K query points per bag
        (the other side's E). Returns pooled [Q, K, d], the midpoint under the pooler's weights
        tilted toward tokens near each query: softmax(log w_j - beta * d(query, y_j) / r_mean).
        A bag with no walks holds its seed alone. `r_mean` is the embedding table's mean radius,
        computed once per forward by the caller."""
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()                 # [Q, T]
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x_tokens = F.embedding(nodes, self.E.weight)                         # [Q, T, d]
        xt = x_tokens.detach()                                               # [Q, T, d]

        u = valid.to(xt.dtype)                                               # [Q, T]
        mid = self.geom.weighted_midpoint(xt, u / u.sum(-1, keepdim=True))            # [Q, d]

        age = tokens.ages.flatten(1).clamp_min(0).to(xt.dtype)               # [Q, T]
        pos = tokens.positions.flatten(1).to(xt.dtype)                       # [Q, T]
        age = torch.log1p(age) / self.log_T_train                            # [Q, T]
        pos = pos / self.max_walk_len                                        # [Q, T]

        tiny = torch.finfo(xt.dtype).tiny

        d_tok_mid = self.geom.dist(xt, mid.unsqueeze(-2))                    # [Q, T]
        d_mid_mean = (d_tok_mid * u).sum(-1, keepdim=True) / u.sum(-1, keepdim=True)  # [Q, 1]
        d_mid = d_tok_mid * u / d_mid_mean.clamp_min(tiny)                   # [Q, T]

        d_0 = self.geom.dist0(xt) * u / r_mean.clamp_min(tiny)               # [Q, T]

        feats = torch.stack([age, pos, d_mid, d_0], dim=-1)                  # [Q, T, 4]
        logits = (self.skip(feats) + self.net(feats)).squeeze(-1)            # [Q, T]
        log_w = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)  # [Q, T]

        # The query only selects tokens; detached so E is not pulled through the focus path.
        d_query = self.geom.dist(others.detach().unsqueeze(-2), xt.unsqueeze(1)) / r_mean.clamp_min(tiny)  # [Q, K, T]
        focus = torch.softmax(log_w.unsqueeze(1) - self.log_beta.exp() * d_query, dim=-1)  # [Q, K, T]
        return self.geom.weighted_midpoint(x_tokens.unsqueeze(1), focus)     # [Q, K, d]


class LinkPredHead(nn.Module):

    INIT_RADIUS = 1e-3

    def __init__(self, num_nodes: int, T_train: int, max_walk_len: int, d_emb: int,
                 hidden_dim: int = 32, n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.T_train = int(T_train)
        self.max_walk_len = int(max_walk_len)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, radius=self.INIT_RADIUS)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, self.T_train, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler)

        self.w = nn.Parameter(torch.tensor([1.0]))

    def mean_radius(self) -> torch.Tensor:
        """Mean distance of the embedding table from the origin, detached; the d_0 scale."""
        return self.geom.dist0(self.E.weight.detach()).mean()                # []

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Score each candidate v of source u by the distance between u's bag seen from v and v's
        bag seen from u. At beta -> 0 both are the plain pooled points: master."""
        r_mean = self.mean_radius()                                          # []
        b = src_tokens.seeds.shape[0]
        e_u = self.E.weight[src_tokens.seeds]                                # [b, d]
        e_v = self.E.weight[cand_tokens.seeds].view(b, -1, e_u.shape[-1])    # [b, c, d]
        c = e_v.shape[1]

        p_u_from_v = self.bag_weights(src_tokens, e_v, r_mean)               # [b, c, d]
        e_u_per_cand = e_u.repeat_interleave(c, dim=0).unsqueeze(1)          # [b*c, 1, d]
        p_v_from_u = self.bag_weights(cand_tokens, e_u_per_cand, r_mean).view(b, c, -1)  # [b, c, d]

        d_uv = self.geom.dist(p_u_from_v, p_v_from_u)                        # [b, c]
        return self.w * -d_uv
