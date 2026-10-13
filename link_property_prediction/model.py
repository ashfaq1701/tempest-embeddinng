import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens, group_by_query

_VAR_FLOOR = 1e-12


def standardise(feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """feat [G, N, F]: standardise each group separately over its N valid entries."""
    m = valid.unsqueeze(-1).to(feat.dtype)                                   # [G, N, 1]
    n = m.sum(dim=1, keepdim=True).clamp_min(1.0)                            # [G, 1, 1]
    mu = (feat * m).sum(dim=1, keepdim=True) / n                             # [G, 1, F]
    var = (((feat - mu) ** 2) * m).sum(dim=1, keepdim=True) / n              # [G, 1, F]
    return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m                # [G, N, F]


class BagWeights(nn.Module):

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, hidden_dim: int = 32,
                 n_layers: int = 2):
        super().__init__()
        self.geom = geom
        self.E = E
        self.hidden = int(hidden_dim)
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_feat = 3
        layers = [nn.Linear(self.n_feat, self.hidden), nn.GELU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)

        self.skip = nn.Linear(self.n_feat, 1, bias=False)
        with torch.no_grad():
            self.skip.weight.copy_(torch.tensor([[-1.0, -1.0, 0.0]]))

    def forward(self, tokens: WalkTokens, group_size: int = None):
        return self.pool(tokens, group_size)

    def pool(self, tokens: WalkTokens, group_size: int = None):
        """`group_size` consecutive bags are standardised together; None = the whole call."""
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

        age = torch.log1p(tokens.ages.flatten(1).clamp_min(0).to(xt.dtype))  # [Q, T]
        pos = tokens.positions.flatten(1).to(xt.dtype)                       # [Q, T]

        d_tok_mid = self.geom.dist(xt, mid.unsqueeze(-2))                    # [Q, T]

        feats = torch.stack([age, pos, d_tok_mid], dim=-1)                   # [Q, T, 3]
        q, t, f = feats.shape
        g = group_size or q
        feats = standardise(feats.view(q // g, g * t, f),
                            valid.view(q // g, g * t)).view(q, t, f).to(xt.dtype)  # [Q, T, 3]
        logits = (self.skip(feats) + self.net(feats)).squeeze(-1)            # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        p = self.geom.weighted_midpoint(x_tokens, w)                         # [Q, d]

        d_tok_p = self.geom.dist(xt, p.unsqueeze(-2))                        # [Q, T]
        spread = (d_tok_p * u).sum(-1) / u.sum(-1)                           # [Q]
        return p, spread


class LinkPredHead(nn.Module):

    INIT_RADIUS = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, radius=self.INIT_RADIUS)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, hidden_dim=hidden_dim,
                                      n_layers=n_layers_pooler)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        b = src_tokens.seeds.shape[0]
        c = cand_tokens.seeds.shape[0] // b
        tokens = group_by_query(src_tokens, cand_tokens)                     # [b*(1+c)] bags
        p, _ = self.bag_weights(tokens, group_size=1 + c)                    # [b*(1+c), d]
        p = p.view(b, 1 + c, -1)                                             # [b, 1+c, d]
        p_u, p_v = p[:, 0], p[:, 1:]                                         # [b, d], [b, c, d]

        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                          # [b, c]
        return self.geo_temp * (-geo)
