"""Gated identity<->context head on the Poincaré ball. Each side is a geodesic blend of IDENTITY (the
seed embedding E_s) and CONTEXT (C_s = gyro-midpoint of the walk bag with the seed EXCLUDED via seed_mask
-> inductive neighbourhood centroid): R_s = interp(E_s, C_s; alpha_s), alpha_s = gate(n_ctx) in (0,1) a
per-query blend learned from the neighbour count (cold/untrained-identity -> context). Score:
s(u,v) = w . [geo, cos, geo_spread_v, cos_spread_v, pop_bias], geo=-d(R_u,R_v), cos=cos(R_u,R_v), spreads
on the candidate context cloud C_v, w init [1,1,0,0,1]. Empty-neighbour bags fall back to C=E (=> R=E, no
NaN). C_x is the weighted gyro-midpoint of x's walk-token bag. Pooling weights come from
four per-token scalars (recency, position, and geodesic distance and cosine alignment to the bag's
unweighted midpoint, both centre features with the per-bag level removed) via w·[rec, pos, spr, ang] plus a
zero-init MLP correction, w init (1, 1, 0, 0)."""
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class PoincareManifold:
    """Poincaré-ball geometry (c=1). `manifold` is kept for E's init + RiemannianAdam."""

    def __init__(self, c: float = 1.0):
        self.manifold = geoopt.PoincareBall(c=c)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Elementwise geodesic distance, broadcasting over leading dims. LOWER = closer."""
        return self.manifold.dist(x, y)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted gyro-midpoint: x [Q,T,d], w [Q,T] -> [Q,d]."""
        return self.manifold.weighted_midpoint(x, weights=w, reducedim=[-2], dim=-1, keepdim=False)

    def interp(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Geodesic interpolation: R = exp_x(t . log_x(y)). t [Q] -> R = x at t=0, y at t=1; x==y -> x."""
        return self.manifold.expmap(x, t.unsqueeze(-1) * self.manifold.logmap(x, y))


class BagWeights(nn.Module):
    """logit = w·[rec, pos, spr, ang] + MLP([rec, pos, spr, ang]); w init (1, 1, 0, 0), MLP zero-init."""

    def __init__(self, mnia: float, hidden: int = 32):
        super().__init__()
        self.mnia = float(mnia)
        self.w = nn.Parameter(torch.tensor([1.0, 1.0, 0.0, 0.0]))
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def centre_feats(geom: PoincareManifold, x: torch.Tensor,
                     valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token geometry relative to the bag's unweighted midpoint C, per-bag level removed from both:
        spr = d(x_p, C) / mean_q d(x_q, C),  ang = cos(x_p, C) - mean_q cos(x_q, C).  Each -> [Q, T]."""
        vf = valid.to(x.dtype)
        n = vf.sum(dim=-1, keepdim=True).clamp_min(1.0)
        c = geom.midpoint(x, vf / n).unsqueeze(-2)                              # [Q, 1, d]
        dc = geom.dist(c, x) * vf
        spr = dc / (dc.sum(dim=-1, keepdim=True) / n).clamp_min(1e-6)
        cs = F.cosine_similarity(c, x, dim=-1, eps=1e-6) * vf
        return spr, (cs - cs.sum(dim=-1, keepdim=True) / n) * vf

    def forward(self, geom: PoincareManifold, tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        rec = -torch.log1p(tokens.ages.clamp_min(0).float() / self.mnia)
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)
        spr, ang = self.centre_feats(geom, x.detach(), valid)
        feat = torch.stack([rec, pos, spr, ang], dim=-1).to(x.dtype)            # [Q, T, 4]
        logits = (feat * self.w).sum(dim=-1) + self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class ContextGate(nn.Module):
    """alpha = sigmoid(a . log1p(n_ctx) + b) in (0,1): the weight on CONTEXT in the identity<->context
    blend. a<0 so more neighbours -> smaller alpha -> more identity; per-dataset training also lets b set
    a global level (Patent -> alpha high / context, since every test source's identity is untrained)."""

    def __init__(self, a_init: float = -1.0, b_init: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))

    def forward(self, n_ctx: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * torch.log1p(n_ctx.float()) + self.b)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; BagWeights, ContextGate and score_w are the other trained params."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float,
                 init_irange: float = 1e-3):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(mean_node_inter_arrival)

        # Near-origin init: uniform(-irange, irange) per coord -> r ~ 2*irange*sqrt(d/3).
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.projx(
                (torch.rand(self.num_nodes, self.d_emb) * 2 - 1) * float(init_irange))
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        self.pop_bias = nn.Embedding(self.num_nodes, 1)
        nn.init.zeros_(self.pop_bias.weight)
        self.gate = ContextGate()   # per-query identity<->context blend for R_s = interp(E_s, C_s; alpha)
        # Learnable channel weights [geo, cos, geo_spread_v, cos_spread_v, pop]; init 1,1,0,0,1.
        self.score_w = nn.Parameter(torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0]))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor):
        """Neighbour-only bag -> (C [Q,d], geo_spread [Q], cos_spread [Q], n_ctx [Q]): the seed's own slot
        is EXCLUDED (seed_mask) so C is the inductive context centroid; empty-neighbour bags fall back to
        the seed (=> later R=E). Spreads are the pooling-weighted mean geo/cos of the cloud to C; n_ctx is
        the neighbour count (coldness proxy for the gate)."""
        nodes = tokens.nodes.clamp_min(0).clone()
        nbr = tokens.mask & ~tokens.seed_mask                                   # neighbours only
        n_ctx = nbr.sum(dim=-1)                                                 # [Q] neighbour count
        valid = nbr.clone()
        cold = ~valid.any(dim=-1)                                               # no neighbours -> use the seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(self.geom, tokens, x, valid)                       # [Q, T] sums to 1, 0 on padding
        c = self.geom.midpoint(x, w)                                            # [Q, d] context centroid
        ce = c.unsqueeze(-2)                                                    # [Q, 1, d]
        geo_sp = (w * self.geom.dist(ce, x)).sum(dim=-1)                        # [Q] weighted mean geo to C
        cos_sp = (w * F.cosine_similarity(ce, x, dim=-1, eps=1e-6)).sum(dim=-1) # [Q] weighted mean cos to C
        return c, geo_sp, cos_sp, n_ctx

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        Per side R = interp(E_seed, C_context; alpha=gate(n_ctx)); score = w . [geo(R_u,R_v), cos(R_u,R_v),
        geo_spread_v, cos_spread_v, pop_bias] (spreads on the candidate CONTEXT cloud C_v)."""
        emb = self.E.weight
        c_u, _, _, n_u = self.pool(src_tokens, emb)                             # context [B,d], n_ctx [B]
        c_v, gsp_v, csp_v, n_v = self.pool(cand_tokens, emb)                    # context [B*C,d], spreads, n_ctx
        e_u = F.embedding(src_tokens.seeds, emb)                                # identity [B, d]
        e_v = F.embedding(cand_tokens.seeds, emb)                              # identity [B*C, d]
        r_u = self.geom.interp(e_u, c_u, self.gate(n_u))                        # [B, d]  blended source rep
        r_v = self.geom.interp(e_v, c_v, self.gate(n_v))                        # [B*C, d] blended cand rep

        b, d = r_u.shape
        c = r_v.shape[0] // b
        r_v = r_v.view(b, c, d)                                                 # [B, C, d]
        gsp_v = gsp_v.view(b, c); csp_v = csp_v.view(b, c)
        v_nodes = cand_tokens.seeds.view(b, c)                                  # [B, C] candidate node ids
        geo = -self.geom.dist(r_u.unsqueeze(1), r_v)                            # [B, C] geodesic
        cos = F.cosine_similarity(r_u.unsqueeze(1), r_v, dim=-1, eps=1e-6)      # [B, C] direction agreement
        feats = torch.stack([
            geo, cos, gsp_v, csp_v,                                             # geo, cos, geo_spread_v, cos_spread_v
            self.pop_bias(v_nodes).squeeze(-1),                                 # pop-bias
        ], dim=-1)                                                             # [B, C, 5]
        return (feats * self.score_w).sum(dim=-1)                              # [B, C]  s = w . f
