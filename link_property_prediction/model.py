"""Centroid-to-centroid head on the unit sphere: s(u,v) = w . f + pop_bias_v, f = [geo, cos,
geo_spread_v, cos_spread_v] (geo=-d(P_u,P_v), cos=cos(P_u,P_v); the spreads are the candidate cloud's
pooling-weighted mean geo/cos to its centroid; pop_bias_v is a learned per-node scalar added at its own
scale), learnable w init [1,1,0,0]; P_x is the normalized weighted resultant (spherical mean) of x's
walk-token bag. (Phase 1: geometry swapped to the sphere; channel surgery to follow.) Pooling
weights come from a small MLP (hidden = 8*N_FEAT) over four per-token scalars: -(age/mnia), -pos, and the
geodesic distance / cosine alignment to the bag's unweighted midpoint (the last two are centre features
with the per-bag level removed). mnia (mean node inter-arrival) is a fixed age scale."""
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class SphereGeometry:
    """Unit-hypersphere geometry. `manifold` is kept for E's random init + RiemannianAdam. The
    sphere's weighted midpoint is the normalized weighted resultant (the vMF MLE mean direction),
    and distance is the great-circle angle."""

    def __init__(self):
        self.manifold = geoopt.Sphere()

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Great-circle (geodesic) distance = arccos(<x,y>), broadcasting over leading dims.
        LOWER = closer."""
        return self.manifold.dist(x, y)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted spherical mean = normalize(sum_i w_i x_i): x [Q,T,d], w [Q,T] (sums to 1 over
        valid) -> [Q,d]. Minimizes weighted squared chordal distance == maximizes weighted mean
        cosine; the pre-normalization norm is the resultant length R (cloud concentration)."""
        resultant = (x * w.unsqueeze(-1)).sum(dim=-2)                           # [Q, d]
        return F.normalize(resultant, dim=-1, eps=1e-8)


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([-(age/mnia), -pos, ang])); hidden = 8*N_FEAT, randomly
    initialised, with mnia as a fixed age scale. The softmax over tokens dampens the random init.
    On the sphere there is no meaningful geodesic 'spread' (distance == arccos(cos)), so the only
    geometric per-token feature is the cosine similarity to the bag's unweighted center."""

    N_FEAT = 3

    def __init__(self, mnia: float):
        super().__init__()
        self.mnia = float(mnia)                 # fixed age scale
        hidden = 8 * self.N_FEAT                # 8x expansion, random init
        self.net = nn.Sequential(nn.Linear(self.N_FEAT, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @staticmethod
    def centre_feats(geom: SphereGeometry, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of each token to the bag's unweighted center C, per-bag level removed:
        ang = cos(x_p, C) - mean_q cos(x_q, C).  -> [Q, T]."""
        vf = valid.to(x.dtype)
        n = vf.sum(dim=-1, keepdim=True).clamp_min(1.0)
        c = geom.midpoint(x, vf / n).unsqueeze(-2)                              # [Q, 1, d] unweighted center
        cs = F.cosine_similarity(c, x, dim=-1, eps=1e-6) * vf
        return (cs - cs.sum(dim=-1, keepdim=True) / n) * vf

    def forward(self, geom: SphereGeometry, tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        rec = -(tokens.ages.clamp_min(0).float() / self.mnia)                   # -(age/mnia)
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)                    # -pos
        ang = self.centre_feats(geom, x.detach(), valid)                        # cos to unweighted center
        feat = torch.stack([rec, pos, ang], dim=-1).to(x.dtype)                 # [Q, T, 3]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; BagWeights is the only other trained module."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float,
                 init_irange: float = 1e-3):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = SphereGeometry()
        self.bag_weights = BagWeights(mean_node_inter_arrival)

        # Uniform-random init on the sphere (Gaussian -> normalize). No origin to dilute toward, so
        # geodesic (great-circle) distances have full scale from step 0; init_irange is unused on the
        # sphere. RiemannianAdam keeps E on the sphere via the ManifoldParameter.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random_uniform(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        self.pop_bias = nn.Embedding(self.num_nodes, 1)
        nn.init.zeros_(self.pop_bias.weight)
        # Learnable channel weights [geo, cos, geo_spread_v, cos_spread_v], init 1,1,0,0 (the candidate
        # cloud-spread channels are off at step 0). pop_bias is added separately at weight 1.
        self.score_w = nn.Parameter(torch.tensor([1.0, 1.0, 0.0, 0.0]))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bag -> (P [Q,d], geo_spread [Q], cos_spread [Q]): the spreads are the pooling-weighted mean
        geodesic distance / cosine of the cloud tokens to their centroid P."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)                                               # all-padding walk -> use seed
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
        score = w . [-geo, cos, geo_spread_v, cos_spread_v] + pop_bias_v."""
        emb = self.E.weight
        p_u, _, _ = self.pool(src_tokens, emb)                                  # [B, d]
        p_v, sg_v, sc_v = self.pool(cand_tokens, emb)                           # [B*C,d] + 2x [B*C]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        v_nodes = cand_tokens.seeds.view(b, c)                                  # [B, C] candidate node ids
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        cos = F.cosine_similarity(p_u.unsqueeze(1), p_v, dim=-1, eps=1e-6)      # [B, C] direction agreement
        feats = torch.stack([-geo, cos,                                        # -geo: lower distance = higher score
                             sg_v.view(b, c), sc_v.view(b, c)], dim=-1)         # [B, C, 4] candidate cloud spreads
        # Per-node bias at its own learned scale, so absolute popularity carries across queries;
        # zero-init means it contributes exactly 0 at step 0.
        pop = self.pop_bias(v_nodes).squeeze(-1)                                # [B, C]
        return (feats * self.score_w).sum(dim=-1) + pop
