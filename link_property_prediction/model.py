import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12


def standardise(feat: torch.Tensor, valid: torch.Tensor = None) -> torch.Tensor:
    if valid is None:
        valid = torch.ones(feat.shape[:-1], dtype=torch.bool, device=feat.device)
    dims = tuple(range(feat.dim() - 1))
    m = valid.unsqueeze(-1).to(feat.dtype)                                   # [..., 1]
    n = m.sum(dim=dims).clamp_min(1.0)                                       # [F]
    mu = (feat * m).sum(dim=dims) / n                                        # [F]
    var = (((feat - mu) ** 2) * m).sum(dim=dims) / n                         # [F]
    return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m                # [..., F]


class BagWeights(nn.Module):

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding):
        super().__init__()
        self.geom = geom
        self.E = E
        self.n_feat = 3
        self.w = nn.Linear(self.n_feat, 1, bias=False)
        with torch.no_grad():
            self.w.weight.copy_(torch.tensor([[-1.0, -1.0, 0.0]]))

    def forward(self, tokens: WalkTokens):
        return self.pool(tokens)

    def pool(self, tokens: WalkTokens):
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

        feats = standardise(torch.stack([age, pos, d_tok_mid], dim=-1),
                            valid).to(xt.dtype)                        # [Q, T, 3]
        logits = self.w(feats).squeeze(-1)                                   # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        p = self.geom.weighted_midpoint(x_tokens, w)                         # [Q, d]

        d_tok_p = self.geom.dist(xt, p.unsqueeze(-2))                        # [Q, T]
        spread = (d_tok_p * u).sum(-1) / u.sum(-1)                           # [Q]
        return p, spread


class LinkPredHead(nn.Module):

    INIT_STD = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, std=self.INIT_STD)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        p_u, _ = self.bag_weights(src_tokens)                                # [b, d]
        p_v, _ = self.bag_weights(cand_tokens)                               # [b*c, d]
        b, d = p_u.shape
        p_v = p_v.view(b, p_v.shape[0] // b, d)                              # [b, c, d]

        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                          # [b, c]
        return self.geo_temp * (-geo)
