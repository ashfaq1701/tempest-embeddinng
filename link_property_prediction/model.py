"""Walk-bag pooler and the link-prediction head.

Pooling weights are `softmax(MLP(standardise(features)))` over the walk-token bag,
features `[log1p(age), pos, d_mid]`, n_feat 3, pooler 1,217 params at nl2 (161 at nl1)
-- identical to master. The weights are INITIALISED so the step-0 logit is the linear
prior `-z_age - z_pos` up to O(eps); see `_init_prior`.
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
    """One query's walk bag -> one point on the manifold.

    Master's pooler with its weights arranged so that at step 0 the logit is the linear
    prior `-z_age - z_pos` up to O(eps), for any n_layers. Same architecture and parameter
    count as master; only the starting values differ.
    """

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
        self._init_prior(prior=(-1.0, -1.0, 0.0), eps=0.05, jitter=0.01)

    def _init_prior(self, prior, eps: float, jitter: float) -> None:
        """Set the MLP so that MLP(z) ~= prior . z at step 0.

        First Linear: every hidden unit reads eps*(prior . z), plus a small random
        perturbation so the rows are not exact copies (equal rows would receive equal
        gradients and never separate). Middle Linears: identity. Last Linear: one equal
        gain per unit, 2^n_layers / (eps*H), which undoes the eps, one GELU halving per
        layer (GELU(x) ~= x/2 for |x| << 1), and the H-fold sum. All biases zero.
        """
        linears = [m for m in self.net if isinstance(m, nn.Linear)]          # n_layers + 1
        first, mids, last = linears[0], linears[1:-1], linears[-1]
        H = self.hidden
        pr = torch.tensor(prior, dtype=first.weight.dtype)
        if pr.numel() != self.n_feat:
            raise ValueError(f"prior has {pr.numel()} entries, n_feat is {self.n_feat}")
        with torch.no_grad():
            first.weight.copy_(eps * pr.expand(H, -1)
                               + jitter * eps * torch.randn(H, self.n_feat))
            first.bias.zero_()
            for m in mids:
                m.weight.copy_(torch.eye(H))
                m.bias.zero_()
            last.weight.fill_((2.0 ** self.n_layers) / (eps * H))
            last.bias.zero_()

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
        logits = self.net(feats).squeeze(-1)                                 # [Q, T]
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
