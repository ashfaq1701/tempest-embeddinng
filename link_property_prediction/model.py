import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens


class BagWeights(nn.Module):
    """One query's walk bag -> one point per counterpart, conditioned on that counterpart.

    The pooling weights depend on WHO is being scored: token t gets a different weight
    against counterpart m than against counterpart m'. Features per (m, t) pair are
    [log1p(age), hop, d0(token), d0(other), d(token, other)], softmax over the bag, then
    the Lorentzian midpoint -- so the bag collapses to [Q, M, d], not [Q, d].

    Geometric features are detached; the midpoint is NOT. That split is load-bearing:
    detaching the points would leave E with no gradient path at all.
    """

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, hidden_dim: int = 32):
        super().__init__()
        self.geom = geom
        self.E = E
        self.hidden = int(hidden_dim)
        self.n_feat = 5
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, tokens: WalkTokens, other_ids: torch.Tensor) -> torch.Tensor:
        """`tokens` has Q rows, `other_ids` is [Q, M] counterpart nodes -> [Q, M, d]."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        # Live for the midpoint, detached for the features.
        x_tokens = F.embedding(nodes, self.E.weight)                         # [Q, T, d]
        xt = x_tokens.detach()
        x_other = F.embedding(other_ids, self.E.weight).detach()             # [Q, M, d]

        d_tok_oth = self.geom.dist(xt.unsqueeze(-3), x_other.unsqueeze(-2))  # [Q, M, T]
        shape = d_tok_oth.shape

        age = torch.log1p(tokens.ages.clamp_min(0).to(xt.dtype)).unsqueeze(-2).expand(shape)
        pos = tokens.positions.to(xt.dtype).unsqueeze(-2).expand(shape)
        d0_tok = self.geom.dist0(xt).unsqueeze(-2).expand(shape)             # [Q, 1, T]
        d0_oth = self.geom.dist0(x_other).unsqueeze(-1).expand(shape)        # [Q, M, 1]

        feat = torch.stack([age, pos, d0_tok, d0_oth, d_tok_oth], dim=-1).to(xt.dtype)
        logits = self.net(feat).squeeze(-1)                                  # [Q, M, T]
        keep = valid.unsqueeze(-2).expand(shape)
        w = torch.softmax(logits.masked_fill(~keep, float("-inf")), dim=-1)

        m = shape[-2]
        xe = x_tokens.unsqueeze(-3).expand(x_tokens.shape[:-2] + (m,) + x_tokens.shape[-2:])
        return self.geom.midpoint(xe, w)                                     # [Q, M, d]


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, irange=self.INIT_IRANGE)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, hidden_dim)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        b = src_tokens.nodes.shape[0]
        c = cand_tokens.nodes.shape[0] // b

        # Source side: Q = b, M = c. Candidate side: Q = b*c, M = 1 -- each candidate has
        # exactly one counterpart, its own query's source. The factor c is absorbed by the
        # COLUMNS on one side and by the ROWS on the other, so one code path serves both.
        p_uv = self.bag_weights(src_tokens, cand_tokens.seeds.view(b, c))    # [b, c, d]
        src_ids = src_tokens.seeds.repeat_interleave(c).unsqueeze(-1)        # [b*c, 1]
        p_vu = self.bag_weights(cand_tokens, src_ids).view(b, c, -1)         # [b, c, d]

        d0v = self.geom.dist0(p_vu)
        duv = self.geom.dist(p_uv, p_vu)
        return self.geo_temp * (-duv + d0v)
