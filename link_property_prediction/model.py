import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold, tokens scored after reading the bag."""

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, d_emb: int,
                 hidden_dim: int = 32, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.geom = geom
        self.E = E
        self.d_emb = int(d_emb)
        self.hidden = int(hidden_dim)
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_feat = 3

        self.stem = nn.Sequential(nn.Linear(self.n_feat + self.d_emb, self.hidden), nn.GELU())
        self.attn = nn.MultiheadAttention(self.hidden, num_heads=int(n_heads), batch_first=True)
        self.norm = nn.LayerNorm(self.hidden)
        layers = []                                                          # master's remaining layers
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.head = nn.Sequential(*layers)

    @staticmethod
    def _standardise(feat: torch.Tensor, valid: torch.Tensor, joint: bool = False) -> torch.Tensor:
        """Standardisation over the valid tokens of the whole batch, 0 on padding.

        joint=False: each column gets its own mean and scale (named scalar features).
        joint=True:  each column gets its own mean, but all columns share ONE scale, the RMS of
                     the centred values over tokens and columns. Every token is shifted by the
                     same vector and divided by the same number, so relative lengths and all
                     angles between tokens' vectors are preserved (tangent vectors).
        """
        m = valid.unsqueeze(-1).to(feat.dtype)                               # [..., 1]
        dims = tuple(range(feat.dim() - 1))
        n = m.sum(dim=dims).clamp_min(1.0)                                   # [F]
        mu = (feat * m).sum(dim=dims) / n                                    # [F]
        var = (((feat - mu) ** 2) * m).sum(dim=dims) / n                     # [F]
        if joint:
            var = var.mean()                                                 # scalar, shared by all columns
        return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m            # [..., F]

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
        mid = self.geom.midpoint(xt, m / n)                                  # [Q, d]

        age = torch.log1p(tokens.ages.clamp_min(0).to(xt.dtype))             # [Q, T]
        pos = tokens.positions.to(xt.dtype)                                  # [Q, T]
        d_mid = self.geom.dist(xt, mid.unsqueeze(-2))                        # [Q, T]
        feats = self._standardise(torch.stack([age, pos, d_mid], dim=-1),
                                  valid).to(xt.dtype)                        # [Q, T, 3]   when, how far

        v = self.geom.logmap0(xt)                                            # [Q, T, d]   where (tangent at origin)
        c = self._standardise(v, valid, joint=True).to(xt.dtype)             # [Q, T, d]   same frame, unit RMS

        h = self.stem(torch.cat([feats, c], dim=-1))                         # [Q, T, H]   one vector per token
        a, _ = self.attn(h, h, h, key_padding_mask=~valid, need_weights=False)  # [Q, T, H]  each token reads the bag
        h = self.norm(h + a)                                                 # [Q, T, H]   residual + LayerNorm

        logits = self.head(h).squeeze(-1)                                    # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        return self.geom.midpoint(x_tokens, w)                               # [Q, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 n_layers_pooler: int = 2, n_heads: int = 4, seed: int = 42):
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
                                      n_layers=n_layers_pooler, n_heads=n_heads)

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
