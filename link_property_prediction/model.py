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
_COS_EPS = 1e-12


def standardise(feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-feature standardisation over the valid tokens of the whole batch."""
    m = valid.unsqueeze(-1).to(feat.dtype)                                   # [Q, T, 1]
    n = m.sum(dim=(0, 1)).clamp_min(1.0)                                     # [F]
    mu = (feat * m).sum(dim=(0, 1)) / n                                      # [F]
    var = (((feat - mu) ** 2) * m).sum(dim=(0, 1)) / n                       # [F]
    return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m                # [Q, T, F]


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

    def forward(self, tokens: WalkTokens) -> torch.Tensor:
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

        a = self.geom.dist(xt, mid.unsqueeze(-2))                            # [Q, T]  d_mid
        # AMBIENT COORDS: a point at radius r on ray n is (cosh r, sinh(r) n), so the angle at
        # the origin is the Euclidean cosine of the SPATIAL parts. Taking it on the full
        # (x0, x') vector is wrong -- x0*y0 contaminates it by up to 1.84 (measured). On the
        # spatial parts it matches the hyperbolic law of cosines to ~1e-15 at every radius,
        # so triangle_cos is not needed here at all.
        xsp = self.geom._sp(xt)                                              # [Q, T, n]
        msp = self.geom._sp(mid).unsqueeze(-2)                               # [Q, 1, n]
        # No origin guard: cosine_similarity already returns exactly 0 for a zero vector
        # (its denominator is clamped at eps), and the feature block is detached, so the only
        # thing a guard did was suppress a ~1e12 gradient that cannot flow. See the commit.
        cos_o = F.cosine_similarity(xsp, msp, dim=-1, eps=_COS_EPS) * u      # [Q, T]  angle at O

        feats = standardise(torch.stack([age, pos, a, cos_o], dim=-1),
                            valid).to(xt.dtype)                        # [Q, T, 4]
        logits = self.net(feats).squeeze(-1)                                 # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        return self.geom.weighted_midpoint(x_tokens, w)                               # [Q, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        # d_emb is the HYPERBOLIC dimension, as in every other branch. The ambient Lorentz
        # vector stores one derived coordinate on top of it: H^d lives in R^{d+1}, with x0
        # fixed by the other d. So Euclidean R^64, Poincare B^64, the intrinsic chart and
        # this branch all have 64 degrees of freedom per node; only the tensor is 65 wide.
        # Allocating d_emb here instead would silently give H^{d_emb-1}.
        self.E = nn.Embedding(self.num_nodes, self.d_emb + 1)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb + 1, irange=self.INIT_IRANGE)
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
