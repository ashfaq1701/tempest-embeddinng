"""Centroid-to-centroid head on the unit sphere: s(u,v) = w . f + pop_bias_v, f = [cos, cos_spread_v]
(cos = cos(P_u,P_v), the centroid cosine similarity; cos_spread_v = the candidate cloud's
pooling-weighted mean cosine to its centroid, i.e. the resultant length; pop_bias_v is a learned
per-node scalar added at its own scale), learnable w init [1,0]; P_x is the normalized weighted
resultant (spherical mean) of x's walk-token bag. No geodesic distance anywhere — on the sphere it is
just arccos(cosine), so the head is expressed purely via cosine. Pooling weights come from a small MLP
(hidden = 8*N_FEAT) over three per-token scalars: -(age/mnia), -pos, and the cosine similarity to the
bag's unweighted midpoint (per-bag level removed). mnia (mean node inter-arrival) is a fixed age scale."""
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class SphereGeometry:
    """Unit-hypersphere geometry. `manifold` is kept for E's random init + RiemannianAdam. The
    sphere's weighted midpoint is the normalized weighted resultant (the vMF MLE mean direction);
    similarity is the cosine. No geodesic distance — on the sphere it is just arccos(cosine), so
    everything the head needs is expressed via cosine."""

    def __init__(self):
        self.manifold = geoopt.Sphere()

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cosine similarity <x,y>/(|x||y|), broadcasting over leading dims. HIGHER = closer."""
        return F.cosine_similarity(x, y, dim=-1, eps=1e-6)

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

        # Uniform-random init on the sphere (Gaussian -> normalize). Cosine is scale-invariant, so
        # there is no near-origin dilution to worry about; init_irange is unused on the sphere.
        # RiemannianAdam keeps E on the sphere via the ManifoldParameter.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random_uniform(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        self.pop_bias = nn.Embedding(self.num_nodes, 1)
        nn.init.zeros_(self.pop_bias.weight)
        # Learnable channel weights [cos, cos_spread_v], init 1,0 (the cloud-spread channel is off
        # at step 0). pop_bias is added separately at weight 1.
        self.score_w = nn.Parameter(torch.tensor([1.0, 0.0]))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bag -> (P [Q,d], cos_spread [Q]): cos_spread is the pooling-weighted mean cosine of the
        cloud tokens to their centroid P (== the resultant length R, a cloud-concentration scalar)."""
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
        cos_sp = (w * self.geom.similarity(pe, x)).sum(dim=-1)                  # [Q] weighted mean cos to P
        return p, cos_sp

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        score = w . [cos, cos_spread_v] + pop_bias_v."""
        emb = self.E.weight
        p_u, _ = self.pool(src_tokens, emb)                                     # [B, d]
        p_v, sc_v = self.pool(cand_tokens, emb)                                 # [B*C,d] + cos_spread [B*C]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        v_nodes = cand_tokens.seeds.view(b, c)                                  # [B, C] candidate node ids
        cos = self.geom.similarity(p_u.unsqueeze(1), p_v)                       # [B, C] cosine similarity (higher=closer)
        feats = torch.stack([cos, sc_v.view(b, c)], dim=-1)                     # [B, C, 2] cos + cos-cloud-spread
        # Per-node bias at its own learned scale, so absolute popularity carries across queries;
        # zero-init means it contributes exactly 0 at step 0.
        pop = self.pop_bias(v_nodes).squeeze(-1)                                # [B, C]
        return (feats * self.score_w).sum(dim=-1) + pop
