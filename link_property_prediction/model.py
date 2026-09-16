import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens


class BagWeights(nn.Module):
    """One query's walk bag -> one point on the manifold.

    softmax over the bag from [log1p(age), hop, radius], then the Lorentzian midpoint.
    `rad` is detached: geometric features describe the bag, they are not a second
    gradient path into E.
    """

    def __init__(self, geom: "LorentzManifold", E: nn.Embedding, hidden_dim: int = 32):
        super().__init__()
        self.geom = geom
        self.E = E
        self.hidden = int(hidden_dim)
        self.n_feat = 3
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, tokens: WalkTokens) -> torch.Tensor:
        # A seed whose walk found no history gets itself in slot 0, so every row has at
        # least one valid token and the midpoint never sees an empty bag.
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, self.E.weight)
        age = torch.log1p(tokens.ages.clamp_min(0).to(x.dtype)).unsqueeze(-1)
        pos = tokens.positions.unsqueeze(-1).to(x.dtype)
        rad = self.geom.dist0(x.detach()).unsqueeze(-1)
        feat = torch.cat([age, pos, rad], dim=-1).to(x.dtype)
        logits = self.net(feat).squeeze(-1)
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
        return self.geom.midpoint(x, w)


class LinkPredHead(nn.Module):

    INIT_IRANGE = 1e-3

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        # Intrinsic coordinates: a point IS x' in R^d. The time coordinate is
        # derived in float64 inside the manifold and never stored, so the
        # embedding table is d wide, not d+1.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, irange=self.INIT_IRANGE)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, hidden_dim)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        p_u = self.bag_weights(src_tokens)
        p_v = self.bag_weights(cand_tokens)
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)
        return self.geo_temp * (-geo)
