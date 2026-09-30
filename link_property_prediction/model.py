import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold. Memory: oldest -> seed."""

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, hidden_dim: int = 32,
                 n_layers: int = 2, alpha_init: float = 0.5):
        super().__init__()
        self.geom = geom
        self.E = E
        self.hidden = int(hidden_dim)
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_feat = 3
        a0 = torch.full((self.n_feat,), float(alpha_init))
        self.log_alpha = nn.Parameter(torch.logit(a0))  # alpha = sigmoid(log_alpha), one per column
        layers = [nn.Linear(2 * self.n_feat, self.hidden), nn.GELU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _standardise(feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Per-feature standardisation over the valid tokens of the whole batch."""
        m = valid.unsqueeze(-1).to(feat.dtype)                               # [..., 1]
        dims = tuple(range(feat.dim() - 1))
        n = m.sum(dim=dims).clamp_min(1.0)                                   # [F]
        mu = (feat * m).sum(dim=dims) / n                                    # [F]
        var = (((feat - mu) ** 2) * m).sum(dim=dims) / n                     # [F]
        return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m            # [..., F]

    def _walk_memory(self, feats: torch.Tensor, pos: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
        """h_l = (1-alpha) * sum_{j: hop_j >= hop_l} alpha^(hop_j - hop_l) f_j, same walk only.
        feats [Q, K, L, F] (0 on padding), pos [Q, K, L] hops, valid [Q, K, L] -> h [Q, K, L, F]."""
        alpha = torch.sigmoid(self.log_alpha)                                # [F]
        lag = (pos.unsqueeze(-2) - pos.unsqueeze(-1)).to(feats.dtype)        # [Q, K, L, L]  hop_j - hop_l
        keep = (lag >= 0) & valid.unsqueeze(-1) & valid.unsqueeze(-2)        # j on the far side of l
        decay = alpha ** lag.clamp_min(0).unsqueeze(-1)                      # [Q, K, L, L, F]
        W = torch.where(keep.unsqueeze(-1), decay, torch.zeros_like(decay))
        return (1.0 - alpha) * torch.einsum("qkljc,qkjc->qklc", W, feats)    # [Q, K, L, F]

    def forward(self, tokens: WalkTokens) -> torch.Tensor:
        nodes = tokens.nodes.clamp_min(0).clone()                            # [Q, K, L]
        valid = tokens.mask.clone()                                          # [Q, K, L]
        q, k, l = nodes.shape

        cold = ~valid.reshape(q, -1).any(dim=-1)                             # [Q]
        if bool(cold.any()):
            nodes[cold, 0, 0] = tokens.seeds[cold]
            valid[cold, 0, 0] = True

        x_tokens = F.embedding(nodes, self.E.weight)                         # [Q, K, L, d]
        xt = x_tokens.detach()                                               # [Q, K, L, d]
        d = xt.shape[-1]

        m = valid.to(xt.dtype)                                               # [Q, K, L]
        m_flat = m.reshape(q, k * l)                                         # [Q, T]
        n_bag = m_flat.sum(-1, keepdim=True).clamp_min(1.0)                  # [Q, 1]
        mid = self.geom.midpoint(xt.reshape(q, k * l, d), m_flat / n_bag)    # [Q, d]

        age = torch.log1p(tokens.ages.clamp_min(0).to(xt.dtype))             # [Q, K, L]
        pos = tokens.positions.to(xt.dtype)                                  # [Q, K, L]
        d_mid = self.geom.dist(xt, mid.view(q, 1, 1, d))                     # [Q, K, L]
        feats = self._standardise(torch.stack([age, pos, d_mid], dim=-1),
                                  valid).to(xt.dtype)                        # [Q, K, L, 3]

        h = self._walk_memory(feats, pos, valid)                             # [Q, K, L, 3]
        z = torch.cat([feats, h], dim=-1).reshape(q, k * l, 2 * self.n_feat)  # [Q, T, 6]

        logits = self.net(z).squeeze(-1)                                     # [Q, T]
        valid_flat = valid.reshape(q, k * l)                                 # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid_flat, float("-inf")), dim=-1)
        return self.geom.midpoint(x_tokens.reshape(q, k * l, d), w)         # [Q, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, irange=self.INIT_IRANGE)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, hidden_dim=hidden_dim)

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
