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

        # Two Linears, NO nonlinearity between them, so this is mathematically a single
        # Linear(3,1): W2 @ W1 is 1x3. Depth changes the optimisation dynamics, not the
        # expressive power. Both layers are bias-free.
        #
        # Init. Every W1 row is [1,0,0] and W2 sums to 1, so the effective weight
        # sum_j W2_j * W1_j is EXACTLY [1,0,0]: the score starts as the plain baseline
        # -d_H with temperature 1, and the d0_u / d0_v channels start exactly inert.
        # W2 carries small jitter purely to break symmetry -- dL/dW1[j,:] is proportional
        # to W2_j, so unequal W2_j makes the hidden rows diverge. With W2 uniform the rows
        # would receive identical gradients forever and the layer would stay rank-1, i.e.
        # SCORER_HIDDEN identical copies of one unit.
        self.mix_in = nn.Linear(self.NUM_FEATURES, self.SCORER_HIDDEN, bias=False)
        self.mix_out = nn.Linear(self.SCORER_HIDDEN, 1, bias=False)
        with torch.no_grad():
            row = torch.zeros(self.NUM_FEATURES)
            row[0] = 1.0
            self.mix_in.weight.copy_(row.expand(self.SCORER_HIDDEN, -1))
            w2 = 1.0 + 0.01 * torch.randn(self.SCORER_HIDDEN)
            self.mix_out.weight.copy_((w2 / w2.sum()).view(1, -1))

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
        # NOTE d0_u is a per-query CONSTANT and this head is linear in it, so under the
        # softmax CE loss its gradient is identically zero (measured 2.05e-08 on this
        # suite: softmax row-sums of dL/ds vanish, so any per-row-constant channel gets
        # nothing). The d0_u channel is inert under CE by construction, not by init, and
        # only becomes live under a pointwise loss such as BCE.
        feats = torch.stack([-geo, d0_u, d0_v], dim=-1)
        return self.mix_out(self.mix_in(feats)).squeeze(-1)
