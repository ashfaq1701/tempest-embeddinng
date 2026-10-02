"""Master plus one column: cos_o, the angle at the origin between the bag centre and the token.

Origin O, bag centre M, token X form a hyperbolic triangle with sides a = d(M, X) = d_mid,
b = d(O, X) = r_tok, c = d(O, M) = r_mid. The angle at O, by the hyperbolic law of cosines,
    cos_o = (cosh b cosh c - cosh a) / (sinh b sinh c),
is +1 when the token and the centre lie on the same ray from the origin (same branch of the
hierarchy), 0 when in unrelated directions, -1 on opposite sides. Computed from distances
only, so the chart never enters. Features: [log1p(age), pos, d_mid, cos_o].
"""
import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12
_SIN_FLOOR = 1e-6


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold."""

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, hidden_dim: int = 32,
                 n_layers: int = 2):
        super().__init__()
        self.geom = geom
        self.E = E
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

    @staticmethod
    def _standardise(feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Per-feature standardisation over the valid tokens of the whole batch."""
        m = valid.unsqueeze(-1).to(feat.dtype)                               # [Q, T, 1]
        n = m.sum(dim=(0, 1)).clamp_min(1.0)                                 # [F]
        mu = (feat * m).sum(dim=(0, 1)) / n                                  # [F]
        var = (((feat - mu) ** 2) * m).sum(dim=(0, 1)) / n                   # [F]
        return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m            # [Q, T, F]

    def forward(self, tokens: WalkTokens) -> torch.Tensor:
        nodes = tokens.nodes.clamp_min(0).clone()                            # [Q, T]
        valid = tokens.mask.clone()                                          # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x_tokens = F.embedding(nodes, self.E.weight)                         # [Q, T, d]
        xt = x_tokens.detach()                                               # [Q, T, d]

        u = valid.to(xt.dtype)                                               # [Q, T]
        mid = self.geom.midpoint(xt, u / u.sum(-1, keepdim=True))            # [Q, d]

        age = torch.log1p(tokens.ages.clamp_min(0).to(xt.dtype))             # [Q, T]
        pos = tokens.positions.to(xt.dtype)                                  # [Q, T]

        a = self.geom.dist(xt, mid.unsqueeze(-2))                            # [Q, T]  d_mid
        b = self.geom.dist0(xt)                                              # [Q, T]  r_tok
        c = self.geom.dist0(mid).unsqueeze(-1)                               # [Q, 1]  r_mid
        cos_o = (torch.cosh(b) * torch.cosh(c) - torch.cosh(a)) \
            / (torch.sinh(b) * torch.sinh(c)).clamp_min(_SIN_FLOOR)          # [Q, T]  angle at O
        cos_o = cos_o.clamp(-1.0, 1.0) * u                                   # degenerate -> ~0

        feats = self._standardise(torch.stack([age, pos, a, cos_o], dim=-1),
                                  valid).to(xt.dtype)                        # [Q, T, 4]
        logits = self.net(feats).squeeze(-1)                                 # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        return self.geom.midpoint(x_tokens, w)                               # [Q, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, irange=self.INIT_IRANGE)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, hidden_dim=hidden_dim,
                                      n_layers=n_layers_pooler)

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
