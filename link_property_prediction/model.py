"""Multi-query pooler on the flat [Q, T] bag: learned query points relative to the bag centre.

Master scores each token from [log1p(age), pos, d(token, mid)], where mid is the bag's uniform
Lorentz midpoint: attention pooling with one fixed query, the centre. This pooler keeps the
centre as the anchor and adds n_queries learned offsets, q_off [H, d], stored as tangent
vectors at the origin and initialised to zero. For every bag each offset is parallel-
transported to the bag's midpoint and applied with the exponential map, giving H query points
per bag that sit at the same learned displacement from each bag's centre. Each token is scored
from [log1p(age), pos, d(token, q_1), ..., d(token, q_H)]. At q_off = 0 every query is the
midpoint and the pooler is master with d_mid repeated; the offsets are the only parameters
added. The geometry (xt, mid) is detached, so the queries are learned but the embeddings are
not reshaped by the pooler.
"""
import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold, scored against H learned queries."""

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, d_emb: int,
                 hidden_dim: int = 32, n_layers: int = 2, n_queries: int = 4):
        super().__init__()
        self.geom = geom
        self.E = E
        self.d_emb = int(d_emb)
        self.hidden = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_queries = int(n_queries)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if self.n_queries < 1:
            raise ValueError(f"n_queries must be >= 1, got {n_queries}")
        self.n_feat = 2 + self.n_queries                                     # age, pos, H distances

        # H learned offsets, tangent vectors at the origin; zero -> every query is the midpoint.
        self.q_off = nn.Parameter(torch.zeros(self.n_queries, self.d_emb))  # [H, d]

        layers = [nn.Linear(self.n_feat, self.hidden), nn.GELU()]           # master's net, wider input
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _standardise(feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Per-feature standardisation over the valid tokens of the whole batch."""
        m = valid.unsqueeze(-1).to(feat.dtype)                               # [Q, T, 1]
        n = m.sum(dim=(0, 1)).clamp_min(1.0)                                 # [F]
        mu = (feat * m).sum(dim=(0, 1)) / n                                  # [F]
        var = (((feat - mu) ** 2) * m).sum(dim=(0, 1)) / n                   # [F]
        return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m            # [Q, T, F]

    def _queries(self, mid: torch.Tensor) -> torch.Tensor:
        """Bag midpoints [Q, d] -> H query points per bag [Q, H, d]: mid shifted by each offset."""
        q_count = mid.shape[0]
        origin = torch.zeros_like(mid)                                       # [Q, d]  chart origin
        base = mid.unsqueeze(1).expand(-1, self.n_queries, -1)               # [Q, H, d]
        off = self.q_off.unsqueeze(0).expand(q_count, -1, -1)                # [Q, H, d]  same offsets, every bag
        v = self.geom.transp(origin.unsqueeze(1), base, off)                 # [Q, H, d]  offsets carried to each centre
        return self.geom.expmap(base, v)                                     # [Q, H, d]  step from the centre

    def forward(self, tokens: WalkTokens) -> torch.Tensor:
        nodes = tokens.nodes.clamp_min(0).clone()                            # [Q, T]
        valid = tokens.mask.clone()                                          # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x_tokens = F.embedding(nodes, self.E.weight)                         # [Q, T, d]
        xt = x_tokens.detach()                                               # [Q, T, d]

        m = valid.to(xt.dtype)                                               # [Q, T]
        n = m.sum(-1, keepdim=True).clamp_min(1.0)                           # [Q, 1]
        mid = self.geom.midpoint(xt, m / n)                                  # [Q, d]   bag centre, detached geometry

        age = torch.log1p(tokens.ages.clamp_min(0).to(xt.dtype))             # [Q, T]
        pos = tokens.positions.to(xt.dtype)                                  # [Q, T]

        q = self._queries(mid)                                               # [Q, H, d]
        d_q = self.geom.dist(xt.unsqueeze(2), q.unsqueeze(1))                # [Q, T, H]  each token to each query

        feats = self._standardise(torch.cat([age.unsqueeze(-1), pos.unsqueeze(-1), d_q], dim=-1),
                                  valid).to(xt.dtype)                        # [Q, T, 2 + H]
        logits = self.net(feats).squeeze(-1)                                 # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        return self.geom.midpoint(x_tokens, w)                               # [Q, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 n_layers_pooler: int = 2, n_queries: int = 4, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, irange=self.INIT_IRANGE)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, self.d_emb, hidden_dim=hidden_dim,
                                      n_layers=n_layers_pooler, n_queries=n_queries)

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
