"""Centroid-to-centroid head on the LORENTZ hyperboloid: s(u,v) = geo_temp * (-d_H(P_u, P_v)).

Lorentz model (k=1), isometric to the Poincaré ball but with no finite coordinate boundary. The
distance term is scaled by a learned geo_temp (init 1.0). The score is the scaled geodesic distance
and nothing else.

P_x is the weighted Lorentzian centroid of x's walk-token bag; the pooling weights are a softmax over an
MLP of [log1p(age) | raw position | rad] at a fixed hidden width. Nothing is standardised: log1p
is a fixed function of the age alone, so no batch-dependent or dataset-derived quantity enters
the pooler. Learned head params: geo_temp and the MLP pooler."""


import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class LorentzManifold:

    def __init__(self, k: float = 1.0):
        self.manifold = geoopt.Lorentz(k=k)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.manifold.dist(x, y)

    def dist0(self, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.dist0(x)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        s = (w.unsqueeze(-1) * x).sum(dim=-2)
        mink = -s[..., :1] ** 2 + (s[..., 1:] ** 2).sum(-1, keepdim=True)
        return s / (-mink).clamp_min(1e-9).sqrt()


class BagWeights(nn.Module):

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden = int(hidden_dim)
        self.n_feat = 3
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, geom: "LorentzManifold", tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        age = torch.log1p(tokens.ages.clamp_min(0).to(x.dtype)).unsqueeze(-1)
        pos = tokens.positions.unsqueeze(-1).to(x.dtype)
        rad = geom.dist0(x.detach()).unsqueeze(-1)
        feat = torch.cat([age, pos, rad], dim=-1).to(x.dtype)
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):

    def __init__(self, num_nodes: int, d_emb: int,
                 init_irange: float = 1e-3, hidden_dim: int = 32):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        self.bag_weights = BagWeights(hidden_dim)

        self.E = nn.Embedding(self.num_nodes, self.d_emb + 1)
        with torch.no_grad():
            init = self.geom.manifold.projx(
                (torch.rand(self.num_nodes, self.d_emb + 1) * 2 - 1) * float(init_irange))
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)
        w = self.bag_weights(self.geom, tokens, x, valid)
        return self.geom.midpoint(x, w)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)
        p_v = self.pool(cand_tokens, emb)
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)
        return self.geo_temp * (-geo)
