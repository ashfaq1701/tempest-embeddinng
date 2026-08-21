"""Centroid-to-centroid head in plain Euclidean R^d: s(u,v) = w . f, f = [-dist,
dist_spread_v] (dist = ||P_u - P_v||, the centroid L2 distance, negated so lower distance = higher
score; dist_spread_v = the candidate cloud's pooling-weighted mean L2 distance to its centroid).
pop_bias removed; learnable w init [1,0]; P_x is the
plain weighted arithmetic mean (Sum_i w_i x_i) of x's walk-token bag. E is a plain unconstrained
nn.Parameter (Adam moves it freely). Pooling weights come from a small MLP (hidden = 8*N_FEAT) over
three per-token scalars: -(age/mnia), -pos, and the L2 distance to the bag's unweighted midpoint
(per-bag level removed). mnia (mean node inter-arrival) is a fixed age scale."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class EuclideanGeometry:
    """Plain Euclidean R^d geometry (no manifold constraint). Distance is L2 and the weighted
    midpoint is the plain weighted arithmetic mean (Sum_i w_i x_i) -- the unconstrained sibling of
    the sphere's normalized resultant and the ball's gyro-midpoint. E is a plain nn.Parameter, so
    RiemannianAdam falls back to standard Adam for it."""

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Euclidean (L2) distance ||x - y||, broadcasting over leading dims. LOWER = closer."""
        return torch.linalg.vector_norm(x - y, dim=-1)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted arithmetic mean = sum_i w_i x_i (weights sum to 1 over valid): x [Q,T,d],
        w [Q,T] -> [Q,d]. Minimizes the weighted sum of squared Euclidean distances to the tokens."""
        return (x * w.unsqueeze(-1)).sum(dim=-2)


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([-(age/mnia), -pos, spr])); hidden = 8*N_FEAT, randomly
    initialised, with mnia as a fixed age scale. The softmax over tokens dampens the random init.
    The single geometric per-token feature is the L2 distance to the bag's unweighted center,
    per-bag level removed (divided by the mean distance)."""

    N_FEAT = 3

    def __init__(self, mnia: float):
        super().__init__()
        self.mnia = float(mnia)                 # fixed age scale
        hidden = 8 * self.N_FEAT                # 8x expansion, random init
        self.net = nn.Sequential(nn.Linear(self.N_FEAT, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @staticmethod
    def centre_feats(geom: EuclideanGeometry, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """L2 distance of each token to the bag's unweighted center C, per-bag level removed by
        dividing by the mean distance: spr = d(x_p, C) / mean_q d(x_q, C).  -> [Q, T]."""
        vf = valid.to(x.dtype)
        n = vf.sum(dim=-1, keepdim=True).clamp_min(1.0)
        c = geom.midpoint(x, vf / n).unsqueeze(-2)                              # [Q, 1, d] unweighted center
        dc = geom.dist(c, x) * vf                                               # [Q, T] L2 distance to center
        return dc / (dc.sum(dim=-1, keepdim=True) / n).clamp_min(1e-6)

    def forward(self, geom: EuclideanGeometry, tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        rec = -(tokens.ages.clamp_min(0).float() / self.mnia)                   # -(age/mnia)
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)                    # -pos
        spr = self.centre_feats(geom, x.detach(), valid)                        # L2 dist to unweighted center
        feat = torch.stack([rec, pos, spr], dim=-1).to(x.dtype)                 # [Q, T, 3]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a plain (unconstrained) nn.Parameter; BagWeights is the only other trained module."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float,
                 init_irange: float = 1e-3):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = EuclideanGeometry()
        self.bag_weights = BagWeights(mean_node_inter_arrival)

        # Euclidean init: unit-norm random directions (Gaussian -> normalize), held as a PLAIN
        # unconstrained nn.Parameter that Adam moves freely (no manifold). init_irange is unused.
        # Starting at norm 1 gives distances real scale from step 0 (typical pairwise ||x-y|| ~ sqrt 2).
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = F.normalize(torch.randn(self.num_nodes, self.d_emb), dim=-1)
        self.E.weight = nn.Parameter(init)

        # Learnable channel weights [-dist, dist_spread_v], init 1,0 (the cloud-spread channel is off
        # at step 0; -dist so lower distance = higher score).
        self.score_w = nn.Parameter(torch.tensor([1.0, 0.0]))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bag -> (P [Q,d], dist_spread [Q]): dist_spread is the pooling-weighted mean L2 distance of
        the cloud tokens to their centroid P (a cloud-spread scalar; lower = tighter cloud)."""
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
        dist_sp = (w * self.geom.dist(pe, x)).sum(dim=-1)                       # [Q] weighted mean L2 dist to P
        return p, dist_sp

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        score = w . [-dist, dist_spread_v]  (pop_bias removed)."""
        emb = self.E.weight
        p_u, _ = self.pool(src_tokens, emb)                                     # [B, d]
        p_v, sd_v = self.pool(cand_tokens, emb)                                 # [B*C,d] + dist_spread [B*C]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        dist = self.geom.dist(p_u.unsqueeze(1), p_v)                           # [B, C] L2 distance (lower=closer)
        feats = torch.stack([-dist, sd_v.view(b, c)], dim=-1)                   # [B, C, 2] -dist + dist-cloud-spread
        return (feats * self.score_w).sum(dim=-1)
