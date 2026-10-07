"""Walk-bag pooler and the link-prediction head.

Pooling weights are `softmax(skip(f) + MLP(f))` over the walk-token bag with
`f = standardise(features)`, features `[log1p(age), pos, d_mid]`, n_feat 3, pooler 1,220
params at nl2 (164 at nl1). `skip` is a `Linear(3, 1, bias=False)` initialised at
`[-1, -1, 0]`: the residual's identity branch, carrying whatever part of the weighting is
linear in the standardised features, with the MLP left to carry the curved part. That init
makes step 0 the sigma-unit recency prior rather than master, and the weights stay free, so
`skip.weight` after training reads out the pooler's linear law in standardised units.
The pooled point is a weighted Lorentz midpoint; the scorer is `geo_temp * (-d(p_u, p_v))`.

`cos_o` -- the angle at the origin between token and bag centre -- was removed at
791360a. It lowered the TRAINING loss and widened the val->test gap: on YouTube the loss
ratio reached 2.15 at ep18, both runs reached the same best val (0.6823 vs 0.6819) and the
gap to test was 0.0890 with it against 0.0714 without, worth +0.0159 on test to remove.
Recover it from d36ce26^ if you want to re-run those arms; WikiLink is the one dataset
that preferred it.
"""
import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
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
        self.net = nn.Sequential(*layers)                 # correction (residual) branch

        # Identity-class branch: a linear read of the standardised features onto the logit.
        # Declared AFTER self.net so net's draws are untouched. The output is a scalar and the
        # input is n_feat-dim, so the ResNet "identity" is a Linear(n_feat, 1), not the
        # literal identity.
        #
        # Initialised at [-1, -1, 0] rather than zero, so at step 0 the skip contributes
        # exactly -(z_age + z_pos): the sigma-unit recency prior of e509656, which scores
        # 0.7572 on tgbl-wiki against master's 0.7282. The weights stay free, so the arm can
        # keep that law, rescale it, or unlearn it -- unlike the fixed prior, the sign is not
        # pinned. Zero init instead reproduces master at step 0 (see 471f3f1), where the skip
        # plateaued near [-0.07, -0.03, +0.03] and was far too small to matter.
        self.skip = nn.Linear(self.n_feat, 1, bias=False)
        if self.n_feat != 3:
            raise ValueError(f"skip init [-1,-1,0] assumes n_feat 3, got {self.n_feat}")
        with torch.no_grad():
            self.skip.weight.copy_(torch.tensor([[-1.0, -1.0, 0.0]]))

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

        feats = standardise(torch.stack([age, pos, a], dim=-1),
                            valid).to(xt.dtype)                        # [Q, T, 3]
        logits = (self.skip(feats) + self.net(feats)).squeeze(-1)            # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        return self.geom.weighted_midpoint(x_tokens, w)                               # [Q, d]


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
