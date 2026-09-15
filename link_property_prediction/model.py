import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens


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

    INIT_IRANGE = 1e-3
    NUM_FEATURES = 3
    SCORER_HIDDEN = 32

    def __init__(self, num_nodes: int, d_emb: int, hidden_dim: int = 32, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)
        self.bag_weights = BagWeights(hidden_dim)

        # Intrinsic coordinates: a point IS x' in R^d. The time coordinate is
        # derived in float64 inside the manifold and never stored, so the
        # embedding table is d wide, not d+1.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb, irange=self.INIT_IRANGE)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        # Random init, left random: the residual below supplies the prior, so the MLP
        # does not need an engineered starting point.
        self.mix = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, self.SCORER_HIDDEN, bias=False),
            nn.GELU(),
            nn.Linear(self.SCORER_HIDDEN, 1, bias=False),
        )

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
        d0_u = self.geom.dist0(p_u).unsqueeze(1).expand_as(geo)
        d0_v = self.geom.dist0(p_v)
        feats = torch.stack([-geo, d0_u, d0_v], dim=-1)
        # RESIDUAL. -geo is unparameterised, so the head starts as the baseline and the
        # MLP learns a correction to distance instead of having to rediscover distance.
        # At INIT_IRANGE=1e-3 the features are ~1e-3 and the MLP's output is ~1e-4
        # against geo's ~1.6e-3, so it is negligible at init by construction.
        return -geo + self.mix(feats).squeeze(-1)
