"""Poincare ball pooler, WITHOUT the `cos_o` column -- the fourth cell of the 2x2.

`geoopt.PoincareBall(c=1.0)` in place of the Lorentz chart, features
`[log1p(age), pos, d_mid]`, n_feat 3, pooler 1,217 params at nl2 (161 at nl1).

THE 2x2 THIS COMPLETES, chart x feature:

                     with cos_o                 without cos_o
    Lorentz          feature/with-cos-o         (default)
    Poincare ball    feature/poincare           THIS BRANCH

Both axes were measured one at a time and both came out small. The chart was worth
nothing on five of six TGB-Seq datasets (ball ahead only on YouTube, by 0.0142) and the
pooler is provably the SAME operation in either chart -- geoopt's Einstein midpoint IS
the Lorentzian centroid, agreeing to 1e-10 across the chart map. Removing `cos_o` was
worth +0.0159 YouTube / +0.0121 GoogleLocal against -0.0070 Flickr / -0.0033 ML-20M.
Neither axis has been tested in combination with the other.

Note the ball clamps at `r_max = 2 artanh(0.99599993) = 6.2126` in float32, and removing
`cos_o` LENGTHENS the radial tail (r_max 1.3-1.5x at matched epoch, r_mean unchanged), so
this arm is the most likely of the four to hit that ceiling. Watch for `r_max = 6.213`
pinned in the log -- it means clamped, not converged.
"""
import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12


def standardise(feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-feature standardisation over the valid tokens of the whole batch."""
    m = valid.unsqueeze(-1).to(feat.dtype)                                   # [Q, T, 1]
    n = m.sum(dim=(0, 1)).clamp_min(1.0)                                     # [F]
    mu = (feat * m).sum(dim=(0, 1)) / n                                      # [F]
    var = (((feat - mu) ** 2) * m).sum(dim=(0, 1)) / n                       # [F]
    return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m                # [Q, T, F]


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold."""

    def __init__(self, geom: geoopt.PoincareBall, E: nn.Embedding, hidden_dim: int = 32,
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
        mid = self.geom.weighted_midpoint(
            xt, weights=u / u.sum(-1, keepdim=True), reducedim=[-2])          # [Q, d]

        age = torch.log1p(tokens.ages.flatten(1).clamp_min(0).to(xt.dtype))  # [Q, T]
        pos = tokens.positions.flatten(1).to(xt.dtype)                       # [Q, T]

        d_mid = self.geom.dist(xt, mid.unsqueeze(-2))                        # [Q, T]

        feats = standardise(torch.stack([age, pos, d_mid], dim=-1),
                            valid).to(xt.dtype)                              # [Q, T, 4]
        logits = self.net(feats).squeeze(-1)                                 # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        return self.geom.weighted_midpoint(x_tokens, weights=w, reducedim=[-2])   # [Q, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = geoopt.PoincareBall(c=1.0)
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, std=self.INIT_IRANGE)
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
