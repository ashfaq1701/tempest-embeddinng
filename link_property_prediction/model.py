import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_VAR_FLOOR = 1e-12
_DENOM_FLOOR = 1e-12


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold, walk by walk.

    Stage 1 within each walk: a bidirectional GRU reads the walk's token features, a per-token
    head gives logits, softmax over that walk's L tokens, Lorentz midpoint -> one point per walk.
    Stage 2 across walks: the GRU's final state summarises each walk, a per-walk head gives
    logits, softmax over the K walks, Lorentz midpoint of those points -> the bag's point.
    """

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, hidden_dim: int = 32):
        super().__init__()
        self.geom = geom
        self.E = E
        self.hidden = int(hidden_dim)
        self.n_feat = 3
        self.gru = nn.GRU(self.n_feat, self.hidden, batch_first=True, bidirectional=True)
        self.head_tok = nn.Linear(2 * self.hidden, 1)       # per-token logit within its walk
        self.head_walk = nn.Linear(2 * self.hidden, 1)      # per-walk logit within its bag

    @staticmethod
    def _standardise(feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Per-feature standardisation over the valid tokens of the whole batch."""
        m = valid.unsqueeze(-1).to(feat.dtype)                               # [..., 1]
        dims = tuple(range(feat.dim() - 1))
        n = m.sum(dim=dims).clamp_min(1.0)                                   # [F]
        mu = (feat * m).sum(dim=dims) / n                                    # [F]
        var = (((feat - mu) ** 2) * m).sum(dim=dims) / n                     # [F]
        return (feat - mu) / var.clamp_min(_VAR_FLOOR).sqrt() * m            # [..., F]

    def forward(self, tokens: WalkTokens) -> torch.Tensor:
        nodes = tokens.nodes.clamp_min(0).clone()                            # [Q, K, L]
        valid = tokens.mask.clone()                                          # [Q, K, L]
        q, k, l = nodes.shape

        cold = ~valid.reshape(q, -1).any(dim=-1)                             # [Q]
        if bool(cold.any()):
            nodes[cold, 0, 0] = tokens.seeds[cold]
            valid[cold, 0, 0] = True
        walk_valid = valid.any(dim=-1)                                       # [Q, K]

        x_tokens = F.embedding(nodes, self.E.weight)                         # [Q, K, L, d]
        xt = x_tokens.detach()                                               # [Q, K, L, d]
        d = xt.shape[-1]

        m = valid.to(xt.dtype)                                               # [Q, K, L]
        n_bag = m.reshape(q, -1).sum(-1, keepdim=True).clamp_min(1.0)        # [Q, 1]
        mid = self.geom.midpoint(xt.reshape(q, k * l, d),
                                 (m.reshape(q, k * l) / n_bag))              # [Q, d]

        age = torch.log1p(tokens.ages.clamp_min(0).to(xt.dtype))             # [Q, K, L]
        pos = tokens.positions.to(xt.dtype)                                  # [Q, K, L]
        d_mid = self.geom.dist(xt, mid.view(q, 1, 1, d))                     # [Q, K, L]
        bag_mean = ((d_mid * m).reshape(q, -1).sum(-1) / n_bag.squeeze(-1)) \
            .clamp_min(_DENOM_FLOOR).view(q, 1, 1)                           # [Q, 1, 1]
        r_mid = d_mid / bag_mean * m                                         # [Q, K, L]

        feats = self._standardise(torch.stack([age, pos, r_mid], dim=-1),
                                  valid).to(xt.dtype)                        # [Q, K, L, 3]

        seq = feats.reshape(q * k, l, self.n_feat)                           # [Q*K, L, 3]
        lens = valid.reshape(q * k, l).sum(-1)                               # [Q*K]  padding is at the end
        lens_c = lens.clamp_min(1).cpu()                                     # empty walks fed length 1, masked below
        packed = nn.utils.rnn.pack_padded_sequence(seq, lens_c, batch_first=True,
                                                   enforce_sorted=False)
        out, h_n = self.gru(packed)                                          # h_n [2, Q*K, H]
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=l)
        out = out.reshape(q, k, l, 2 * self.hidden)                          # [Q, K, L, 2H]
        h_walk = torch.cat([h_n[0], h_n[1]], dim=-1).reshape(q, k, 2 * self.hidden)  # [Q, K, 2H]

        tok_logits = self.head_tok(out).squeeze(-1)                          # [Q, K, L]
        tok_logits = tok_logits.masked_fill(~valid, float("-inf"))
        w_tok = torch.softmax(tok_logits, dim=-1)                            # [Q, K, L]  sums to 1 per walk
        w_tok = torch.where(walk_valid.unsqueeze(-1), w_tok, torch.zeros_like(w_tok))

        m_k = self.geom.midpoint(x_tokens.reshape(q * k, l, d),
                                 w_tok.reshape(q * k, l)).reshape(q, k, d)   # [Q, K, d]

        walk_logits = self.head_walk(h_walk).squeeze(-1)                     # [Q, K]
        walk_logits = walk_logits.masked_fill(~walk_valid, float("-inf"))
        w_walk = torch.softmax(walk_logits, dim=-1)                          # [Q, K]  sums to 1 per bag

        return self.geom.midpoint(m_k, w_walk)                               # [Q, d]


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
