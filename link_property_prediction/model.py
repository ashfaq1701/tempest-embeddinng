import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold."""

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, hidden_dim: int = 32,
                 n_layers: int = 1, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.geom = geom
        self.E = E
        self.hidden = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if self.hidden % self.n_heads != 0:
            raise ValueError(f"hidden_dim {self.hidden} must divide by n_heads {self.n_heads}")
        self.n_feat = 3
        # A SET ENCODER over the walk-token features: every token's hidden vector is updated by
        # attending over the other VALID tokens of its own bag, then scored. Padded tokens are
        # invisible via src_key_padding_mask. Pre-norm, because with attention in the stack the
        # LayerNorms are load-bearing rather than decoration.
        self.stem = nn.Linear(self.n_feat, self.hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=self.n_heads, dim_feedforward=4 * self.hidden,
            dropout=self.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.n_layers)
        self.norm = nn.LayerNorm(self.hidden)
        self.head = nn.Linear(self.hidden, 1)

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

        d_mid = self.geom.dist(xt, mid.unsqueeze(-2))                        # [Q, T]

        feats = self._standardise(torch.stack([age, pos, d_mid], dim=-1),
                                  valid).to(xt.dtype)                        # [Q, T, 3]

        h = self.stem(feats)                                                 # [Q, T, H]
        h = self.encoder(h, src_key_padding_mask=~valid)                     # [Q, T, H]
        logits = self.head(self.norm(h)).squeeze(-1)                         # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1) # [Q, T]
        return self.geom.midpoint(x_tokens, w)                               # [Q, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32,
                 n_layers: int = 1, n_heads: int = 4, dropout: float = 0.1,
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

        self.bag_weights = BagWeights(self.geom, self.E, hidden_dim=hidden_dim,
                                      n_layers=n_layers, n_heads=n_heads, dropout=dropout)

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
