"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = w_g*(-d(P_u,P_v)) + w_c*cos(P_u,P_v) + b_v,
learnable [w_g, w_c] init (1, geo/cos spread); P_x is the weighted gyro-midpoint of x's walk-token bag. Pooling weights come from
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
    """E is a ManifoldParameter; BagWeights is the only other trained module."""

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
        # Learnable [geo, cos] channel weights, init to the calibrated linear head (geo 1, cos = geo/cos spread).
        self.score_w = nn.Parameter(torch.stack([
            torch.ones((), dtype=init.dtype), self.channel_scale(self.geom, init)]))

    @staticmethod
    @torch.no_grad()
    def channel_scale(geom: PoincareManifold, points: torch.Tensor, n_pairs: int = 4096) -> torch.Tensor:
        """Scalar putting scale*cos on the same per-pair spread as the geodesic, measured on `points`."""
        i, j = torch.randint(0, points.shape[0], (2, n_pairs), device=points.device)
        return (geom.dist(points[i], points[j]).std()
                / F.cosine_similarity(points[i], points[j], dim=-1).std().clamp_min(1e-8)).clamp_min(1e-8)

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> one manifold point [Q, d]."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)                                               # all-padding walk -> use seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(self.geom, tokens, x, valid)                       # [Q, T]
        return self.geom.midpoint(x, w)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        score: w_g*geo + w_c*cos + b_v."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        v_nodes = cand_tokens.seeds.view(b, c)                                  # [B, C] candidate node ids
        geo = -self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic
        cos = F.cosine_similarity(p_u.unsqueeze(1), p_v, dim=-1, eps=1e-6)      # [B, C] direction agreement
        return self.score_w[0] * geo + self.score_w[1] * cos + self.pop_bias(v_nodes).squeeze(-1)
