"""Centroid-to-centroid head on the Poincaré ball. Each side's walk-token bag (seeds included) is pooled to a
single point and the score is the geodesic between them:
    s(u,v) = -d(P_u, P_v)
Two changes from master. (1) The pooling weights are LEARNED: instead of the fixed
-(log1p(age/mnia)+log1p(hop-1)) prior, a small MLP maps (fixed cos/sin age encoding, learned hop embedding)
-> one logit per token, softmaxed over the bag. Weights depend only on (age, hop), not on E, so there is no
feedback loop between an embedding and its own pooling weight. (2) Pooling is the TANGENT MEAN at the origin,
expmap0(sum_p w_p logmap0(x_p)), not the gyro-midpoint: weighted_midpoint reweights each token by its
conformal factor 2/(1-||x||^2), which with learned weights gives two coupled reweighting mechanisms chasing
each other as E moves. The tangent mean uses w exactly as computed."""
from typing import Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_NORM_EPS = 1e-5      # ||x||^2 clamp: stay strictly inside the ball
_ACOSH_EPS = 1e-7     # arcosh arg clamp: finite gradient at coincidence


class PoincareManifold:
    """Poincaré-ball geometry (c=1). `manifold` (geoopt ball) is kept for E's init + RiemannianAdam."""

    def __init__(self, c: float = 1.0):
        self.manifold = geoopt.PoincareBall(c=c)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Elementwise geodesic distance (geoopt), broadcasting over leading dims. LOWER = closer."""
        return self.manifold.dist(x, y)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Manifold -> tangent at the origin. One global chart, so all queries stay comparable."""
        return self.manifold.logmap0(x)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Tangent at the origin -> manifold. geoopt clamps ||out|| to 1-4e-3 (fp32); never hits the boundary."""
        return self.manifold.expmap0(v)

    @staticmethod
    def pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2))) for x [...,n,d], y [...,m,d] -> [...,n,m]."""
        x2 = (x * x).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., n]
        y2 = (y * y).sum(dim=-1).clamp(max=1.0 - _NORM_EPS)                     # [..., m]
        xy = torch.matmul(x, y.transpose(-1, -2))                               # [..., n, m]
        sq = (x2.unsqueeze(-1) + y2.unsqueeze(-2) - 2.0 * xy).clamp_min(0.0)     # [..., n, m]
        denom = (1.0 - x2).unsqueeze(-1) * (1.0 - y2).unsqueeze(-2)             # [..., n, m]
        arg = (1.0 + 2.0 * sq / denom).clamp_min(1.0 + _ACOSH_EPS)
        return torch.acosh(arg)


class BagWeights(nn.Module):
    """Learned pooling weights from (age, hop) only. Age -> fixed cos/sin frequencies (GraphMixer finds fixed
    beats learned) plus log1p for a monotone channel; hop -> learned embedding. LayerNorm before the MLP
    because log1p(age/mnia) reaches ~5 while the cos/sin channels are bounded by 1."""

    def __init__(self, mnia: float, max_hop: int, d_time: int = 32, d_hop: int = 16, hidden: int = 64):
        super().__init__()
        self.mnia = float(mnia)
        self.max_hop = int(max_hop)
        self.register_buffer("freqs", 1.0 / (10.0 ** torch.linspace(0.0, 3.0, d_time // 2)))
        self.hop_emb = nn.Embedding(self.max_hop, d_hop)
        d_feat = d_time + 1 + d_hop
        self.net = nn.Sequential(
            nn.Linear(d_feat, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: WalkTokens, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """(nodes [Q,T] with padding->0, w [Q,T] summing to 1). Cold-bag guard falls back to the seed so an
        all -inf softmax row can't go NaN."""
        nodes = tokens.nodes.clamp_min(0).clone()                               # [Q, T] padding(-1) -> 0
        valid = tokens.mask.clone()                                             # [Q, T] real slots (seed incl.)
        cold = ~valid.any(dim=-1)
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        a = tokens.ages.clamp_min(0).float() / self.mnia                        # [Q, T]  seed = 0
        hop_idx = tokens.positions.clamp_min(1).long().clamp_max(self.max_hop) - 1
        ang = a.unsqueeze(-1) * self.freqs                                      # [Q, T, d_time/2]
        feat = torch.cat([
            torch.cos(ang), torch.sin(ang),
            torch.log1p(a).unsqueeze(-1),
            self.hop_emb(hop_idx),
        ], dim=-1).to(dtype)                                                    # [Q, T, d_feat]

        logits = self.net(feat).squeeze(-1)                                     # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)    # [Q, T] sums to 1
        return nodes, w


class LinkPredHead(nn.Module):
    """Centroid-to-centroid head with tangent-mean pooling. E is a ManifoldParameter; BagWeights is the only
    other trained module."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float, max_walk_length: int,
                 d_time: int = 32, d_hop: int = 16):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        # walk length counts the seed, so positions run 1..max_walk_length -> that IS the hop cardinality.
        self.bag_weights = BagWeights(mean_node_inter_arrival, max_walk_length, d_time, d_hop)

        # Spread init: geoopt random (std=1), not the near-origin wrapped normal. ManifoldParameter so
        # RiemannianAdam keeps E in the ball.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    def bag_centroid(self, nodes: torch.Tensor, w: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """P_x = expmap0(sum_p w_p logmap0(x_p)) — tangent mean at the origin, no conformal reweighting."""
        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        z = self.geom.logmap0(x)                                                # [Q, T, d]
        return self.geom.expmap0((w.unsqueeze(-1) * z).sum(dim=-2))             # [Q, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries (seeds u); cand = B*C candidate queries (seeds v), query-major. -> [B, C]."""
        emb = self.E.weight

        nodes_u, w_u = self.bag_weights(src_tokens, emb.dtype)                  # [B, T]
        nodes_v, w_v = self.bag_weights(cand_tokens, emb.dtype)                 # [B*C, T]

        p_u = self.bag_centroid(nodes_u, w_u, emb)                              # [B, d]
        p_v = self.bag_centroid(nodes_v, w_v, emb)                              # [B*C, d]

        b, d = p_u.shape
        p_v = p_v.view(b, p_v.shape[0] // b, d)                                 # [B, C, d]
        return -self.geom.dist(p_u.unsqueeze(1), p_v)                           # [B, C] higher = closer
