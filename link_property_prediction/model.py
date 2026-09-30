import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12
_TAP_FLOOR = 1e-12


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold. Memory over when + where."""

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, hidden_dim: int = 32,
                 n_layers: int = 2, alpha_init: float = 0.5, floor: float = 0.2):
        super().__init__()
        self.geom = geom
        self.E = E
        self.hidden = int(hidden_dim)
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_feat = 3
        self.stem = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU())  # master's stem
        self.log_alpha = nn.Parameter(torch.logit(torch.tensor(float(alpha_init))))  # one shared alpha
        layers = []                                                          # master's layers after the stem
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)
        # Built LAST so stem/net consume the same RNG draws as the content-free arm; zero-init makes
        # the content branch contribute exactly nothing at step 0, so this starts as that arm.
        self.proj_e = nn.Linear(int(E.weight.shape[1]), self.hidden)
        nn.init.zeros_(self.proj_e.weight); nn.init.zeros_(self.proj_e.bias)
        self.norm_e = nn.LayerNorm(self.hidden)
        self.floor = float(floor)                                        # lam; 0 = off

    @staticmethod
    def _standardise(feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Per-feature standardisation over the valid tokens of the whole batch."""
        m = valid.unsqueeze(-1).to(feat.dtype)                               # [..., 1]
        dims = tuple(range(feat.dim() - 1))
        n = m.sum(dim=dims).clamp_min(1.0)                                   # [F]
        mu = (feat * m).sum(dim=dims) / n                                    # [F]
        var = (((feat - mu) ** 2) * m).sum(dim=dims) / n                     # [F]
        return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m            # [..., F]

    def _walk_memory(self, u: torch.Tensor, pos: torch.Tensor,
                     valid: torch.Tensor) -> torch.Tensor:
        """h[l] = sum_{j older-or-equal} alpha^(hop_j - hop_l) u[j] / sum of those taps.
        u [Q, K, L, H] (0 on padding), pos [Q, K, L] hops, valid [Q, K, L] -> h [Q, K, L, H]."""
        alpha = torch.sigmoid(self.log_alpha)                                # scalar
        lag = (pos.unsqueeze(-2) - pos.unsqueeze(-1)).to(u.dtype)            # [Q, K, L, L]  hop_j - hop_l
        keep = (lag >= 0) & valid.unsqueeze(-1) & valid.unsqueeze(-2)        # j at or beyond l, both real
        W = torch.where(keep, alpha ** lag.clamp_min(0), torch.zeros_like(lag))  # [Q, K, L, L]
        num = torch.einsum("qklj,qkjh->qklh", W, u)                          # [Q, K, L, H]
        den = W.sum(dim=-1, keepdim=True).clamp_min(_TAP_FLOOR)              # [Q, K, L, 1]
        return num / den

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

        # Content: the arrow from the bag midpoint to the token, carried to the ORIGIN's frame by
        # parallel transport so every bag's arrows live in one shared frame. Intrinsic chart, so the
        # transported vector is a full d-vector -- there is no ambient time coordinate to drop.
        mid_e = mid.detach().view(q, 1, 1, d).expand_as(xt)                   # [Q, K, L, d]
        v_mid = self.geom.logmap(mid_e, xt)                                   # [Q, K, L, d]  at mid
        v = self.geom.transp(mid_e, torch.zeros_like(xt), v_mid)              # [Q, K, L, d]  at origin
        content = self.norm_e(self.proj_e(v))                                 # [Q, K, L, H]  where
        u = (self.stem(feats) + content) * m.unsqueeze(-1)                    # [Q, K, L, H]  when + where
        h = self._walk_memory(u, pos, valid)                                 # [Q, K, L, H]

        logits = self.net(h).squeeze(-1).reshape(q, k * l)                   # [Q, T]
        valid_flat = valid.reshape(q, k * l)                                 # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid_flat, float("-inf")), dim=-1)
        if self.floor > 0.0:
            w = (1.0 - self.floor) * w + self.floor * (m_flat / n_bag)    # every real token keeps floor/n
        return self.geom.midpoint(x_tokens.reshape(q, k * l, d), w)         # [Q, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 n_layers_pooler: int = 2, floor: float = 0.2, seed: int = 42):
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
                                      n_layers=n_layers_pooler, floor=floor)

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
