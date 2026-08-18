"""Split-neighbourhood head on the Poincaré ball. Each side is factored into IDENTITY (the seed's own
embedding E_s) and CONTEXT (P_s = gyro-midpoint of the walk bag with the seed EXCLUDED via seed_mask, so
P_s is an inductive neighbourhood centroid). The scorer sees 11 features per (u,v):
  the 4 pairs <P_u,E_v>, <E_u,P_v>, <E_u,E_v>, <P_u,P_v> each as geodesic (-d_H) and cosine (=8),
  the four cloud spreads (geodesic and cosine, for P_u and P_v; pooling-weighted mean dist/cos to the
  centroid), and the candidate popularity bias b_v.
The P_u spreads are source-constant, so in a LINEAR head they cancel under the row-softmax; the MLP is not
additively separable, so it can use them to modulate the candidate-varying terms (non-cancelling here).
These 13 are reduced by a 2-layer MLP (13->5->1, random init). P_x is the weighted gyro-midpoint of x's
walk-token bag (seed excluded; a bag with no neighbours falls back to the seed). Pooling weights come from
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


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; BagWeights and the scorer MLP are the other trained modules."""

    NUM_FEATS = 13   # 4 pairs x {geo, cos} + {geo,cos}-spread for P_u and P_v + pop-bias

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

        # Popularity bias: zero-init so an UNSEEN (test-only) node defaults to a neutral 0 — any other
        # init would inject a spurious bias for nodes the training never updates.
        self.pop_bias = nn.Embedding(self.num_nodes, 1)
        nn.init.zeros_(self.pop_bias.weight)
        # Scorer: reduce the 11 pair/spread/bias features -> scalar. Randomly initialised (default),
        # so unlike the additive head there is no privileged channel at step 0.
        self.scorer = nn.Sequential(nn.Linear(self.NUM_FEATS, 5), nn.GELU(), nn.Linear(5, 1))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Neighbour-only bag -> (P [Q,d], geo_spread [Q], cos_spread [Q]): the seed's own walk-origin slot
        is EXCLUDED (seed_mask), so P is an inductive context centroid; a bag with no neighbours falls back
        to the seed. The spreads are the pooling-weighted mean geodesic distance / cosine of the cloud to
        its centroid P (same definition as the previous additive head)."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask & ~tokens.seed_mask                                 # neighbours only (drop seed)
        cold = ~valid.any(dim=-1)                                               # no neighbours -> use the seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(self.geom, tokens, x, valid)                       # [Q, T] sums to 1, 0 on padding
        p = self.geom.midpoint(x, w)                                            # [Q, d]
        pe = p.unsqueeze(-2)                                                    # [Q, 1, d]
        geo_sp = (w * self.geom.dist(pe, x)).sum(dim=-1)                        # [Q] weighted mean geo to P
        cos_sp = (w * F.cosine_similarity(pe, x, dim=-1, eps=1e-6)).sum(dim=-1) # [Q] weighted mean cos to P
        return p, geo_sp, cos_sp

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        11 features (identity/context pairs, spreads, pop-bias) reduced by the scorer MLP."""
        emb = self.E.weight
        p_u, gsp_u, csp_u = self.pool(src_tokens, emb)                          # context [B,d], spreads [B]
        p_v, gsp_v, csp_v = self.pool(cand_tokens, emb)                         # context [B*C,d], spreads [B*C]
        e_u = F.embedding(src_tokens.seeds, emb)                                # identity [B, d]
        e_v = F.embedding(cand_tokens.seeds, emb)                              # identity [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d); e_v = e_v.view(b, c, d)
        gsp_v = gsp_v.view(b, c); csp_v = csp_v.view(b, c)
        v_nodes = cand_tokens.seeds.view(b, c)                                  # [B, C] candidate node ids
        pu = p_u.unsqueeze(1); eu = e_u.unsqueeze(1)                            # [B, 1, d] broadcast over C
        gsp_u = gsp_u.unsqueeze(1).expand(b, c); csp_u = csp_u.unsqueeze(1).expand(b, c)  # source-constant

        def gd(a, x): return -self.geom.dist(a, x)                             # geodesic, higher = closer [B,C]
        def cs(a, x): return F.cosine_similarity(a, x, dim=-1, eps=1e-6)       # cosine [B, C]

        feats = torch.stack([
            gd(pu, e_v), cs(pu, e_v),                                          # <P_u, E_v>  context->identity
            gd(eu, p_v), cs(eu, p_v),                                          # <E_u, P_v>  identity->context
            gd(eu, e_v), cs(eu, e_v),                                          # <E_u, E_v>  identity->identity
            gd(pu, p_v), cs(pu, p_v),                                          # <P_u, P_v>  context->context
            gsp_u, csp_u,                                                       # P_u geo + cos spread (source-const)
            gsp_v, csp_v,                                                       # P_v geo + cos spread
            self.pop_bias(v_nodes).squeeze(-1),                                # popularity bias
        ], dim=-1)                                                             # [B, C, 13]
        return self.scorer(feats).squeeze(-1)                                  # [B, C]
