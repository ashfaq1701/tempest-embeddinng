import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold, CONDITIONED ON THE OTHER SIDE.

    softmax over the bag from [log1p(age), hop, radius, d(token, other)], then the
    Lorentzian midpoint. Identical to the unconditioned version except for the fourth
    feature, which makes the weights -- and so the pooled point -- differ per counterpart.

    That is cross-attention in the manifold: -d(., .) is the score kernel, the softmax is
    the attention distribution, and midpoint() is the manifold's weighted sum, since a
    convex combination of hyperboloid points is not itself a hyperboloid point.

    `rad` is detached: geometric features describe the bag, they are not a second gradient
    path into E. `cross` is NOT detached -- it is the query-dependent term and the only
    route by which the other side reaches the weights.
    """

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, hidden_dim: int = 32):
        super().__init__()
        self.geom = geom
        self.E = E
        self.hidden = int(hidden_dim)
        self.n_feat = 4
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

        x = F.embedding(nodes, self.E.weight)
        e = F.embedding(other_ids, self.E.weight)
        cross = self.geom.dist(x.unsqueeze(-3), e.unsqueeze(-2))

        age = torch.log1p(tokens.ages.clamp_min(0).to(x.dtype)).unsqueeze(-2).expand(cross.shape)
        pos = tokens.positions.to(x.dtype).unsqueeze(-2).expand(cross.shape)
        rad = self.geom.dist0(x.detach()).unsqueeze(-2).expand(cross.shape)
        feat = torch.stack([age, pos, rad, cross], dim=-1).to(x.dtype)

        logits = self.net(feat).squeeze(-1)
        keep = valid.unsqueeze(-2).expand(cross.shape)
        w = torch.softmax(logits.masked_fill(~keep, float("-inf")), dim=-1)

        m = cross.shape[-2]
        xe = x.unsqueeze(-3).expand(x.shape[:-2] + (m,) + x.shape[-2:])
        return self.geom.midpoint(xe, w)


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

        p_uv = self.bag_weights(src_tokens, cand_tokens.seeds.view(b, c))
        src_ids = src_tokens.seeds.repeat_interleave(c).unsqueeze(-1)
        p_vu = self.bag_weights(cand_tokens, src_ids).view(b, c, -1)

        geo = self.geom.dist(p_uv, p_vu)
        return self.geo_temp * (-geo)
