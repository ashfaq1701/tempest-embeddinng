import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12


def standardise(feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    m = valid.unsqueeze(-1).to(feat.dtype)
    n = m.sum(dim=(0, 1)).clamp_min(1.0)
    mu = (feat * m).sum(dim=(0, 1)) / n
    var = (((feat - mu) ** 2) * m).sum(dim=(0, 1)) / n
    return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m


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

    def forward(self, tokens: WalkTokens) -> torch.Tensor:
        return self.pool(tokens)[0]

    def pool(self, tokens: WalkTokens):
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()
        valid = tokens.mask.flatten(1).clone()
        cold = ~valid.any(dim=-1)
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x_tokens = F.embedding(nodes, self.E.weight)
        xt = x_tokens.detach()

        u = valid.to(xt.dtype)
        mid = self.geom.weighted_midpoint(xt, u / u.sum(-1, keepdim=True))

        age = torch.log1p(tokens.ages.flatten(1).clamp_min(0).to(xt.dtype))
        pos = tokens.positions.flatten(1).to(xt.dtype)

        d_tok_mid = self.geom.dist(xt, mid.unsqueeze(-2))

        feats = standardise(torch.stack([age, pos, d_tok_mid], dim=-1),
                            valid).to(xt.dtype)
        logits = (self.skip(feats) + self.net(feats)).squeeze(-1)
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
        return self.geom.weighted_midpoint(x_tokens, w), w, x_tokens


class LinkPredHead(nn.Module):

    INIT_STD = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, std=self.INIT_STD)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, hidden_dim=hidden_dim,
                                      n_layers=n_layers_pooler)

        self.w = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        p_u = self.bag_weights(src_tokens)
        p_v, w_v, x_v = self.bag_weights.pool(cand_tokens)
        d_tok = self.geom.dist(x_v.detach(), p_v.unsqueeze(-2))
        spread_v = (w_v * d_tok).sum(-1)

        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)

        geo = self.geom.dist(p_u.unsqueeze(1), p_v)
        feats = torch.stack([-geo, spread_v.view(b, c)], dim=-1)
        return (self.w * feats).sum(-1)
