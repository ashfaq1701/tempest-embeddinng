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


def group_by_node(nodes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[Q, T] -> [Q, T]: each token's index among its row's distinct nodes. Invalid tokens
    share one group of their own."""
    keyed = torch.where(valid, nodes, torch.full_like(nodes, -1))
    ordered, order = keyed.sort(dim=-1)
    new_run = torch.ones_like(valid)
    new_run[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    run_id = new_run.cumsum(dim=-1) - 1
    return torch.empty_like(run_id).scatter_(1, order, run_id)


def group_sum(values: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(values).scatter_add_(1, group, values)


def group_min(values: torch.Tensor, group: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    inf = torch.full_like(values, float("inf"))
    out = inf.scatter_reduce(1, group, values.masked_fill(~valid, float("inf")), "amin")
    return out.masked_fill(out.isinf(), 0.0)


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
        self.n_feat = 6
        layers = [nn.Linear(self.n_feat, self.hidden), nn.GELU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)

        self.skip = nn.Linear(self.n_feat, 1, bias=False)
        with torch.no_grad():
            self.skip.weight.copy_(torch.tensor([[-1.0, -1.0, -1.0, -1.0, 0.0, 0.0]]))

    def forward(self, tokens: WalkTokens):
        return self.pool(tokens)

    def pool(self, tokens: WalkTokens):
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()                 # [Q, T]
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        group = group_by_node(nodes, valid)                                  # [Q, T]
        tok = valid.to(self.E.weight.dtype)                                  # [Q, T]
        count = group_sum(tok, group)                                        # [Q, U]
        node_valid = count > 0                                               # [Q, U]
        node_ids = torch.zeros_like(nodes).scatter_(1, group, nodes)         # [Q, U]
        n = count.clamp_min(1.0)

        age = tokens.ages.flatten(1).clamp_min(0).to(tok.dtype)              # [Q, T]
        pos = tokens.positions.flatten(1).to(tok.dtype)                      # [Q, T]
        min_age = torch.log1p(group_min(age, group, valid))                  # [Q, U]
        mean_age = torch.log1p(group_sum(age * tok, group) / n)              # [Q, U]
        min_pos = group_min(pos, group, valid)                               # [Q, U]
        mean_pos = group_sum(pos * tok, group) / n                           # [Q, U]

        x_nodes = F.embedding(node_ids, self.E.weight)                       # [Q, U, d]
        xn = x_nodes.detach()                                                # [Q, U, d]

        u = node_valid.to(xn.dtype)                                          # [Q, U]
        mid = self.geom.weighted_midpoint(xn, u / u.sum(-1, keepdim=True))   # [Q, d]
        d_node_mid = self.geom.dist(xn, mid.unsqueeze(-2))                   # [Q, U]

        feats = torch.stack([min_age, mean_age, min_pos, mean_pos,
                             torch.log1p(count), d_node_mid], dim=-1)        # [Q, U, 6]
        feats = standardise(feats, node_valid).to(xn.dtype)                  # [Q, U, 6]
        logits = (self.skip(feats) + self.net(feats)).squeeze(-1)            # [Q, U]
        w = torch.softmax(logits.masked_fill(~node_valid, float("-inf")), dim=-1)  # [Q, U]
        p = self.geom.weighted_midpoint(x_nodes, w)                          # [Q, d]

        d_node_p = self.geom.dist(xn, p.unsqueeze(-2))                       # [Q, U]
        spread = (d_node_p * u).sum(-1) / u.sum(-1)                          # [Q]
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
        p_u, _ = self.bag_weights(src_tokens)                                # [b, d]
        p_v, _ = self.bag_weights(cand_tokens)                               # [b*c, d]
        b, d = p_u.shape
        p_v = p_v.view(b, p_v.shape[0] // b, d)                              # [b, c, d]

        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                          # [b, c]
        return self.geo_temp * (-geo)
