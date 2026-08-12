"""Centroid-to-centroid head on the Poincaré ball.  s(u,v) = -d(P_u, P_v).

Each side's walk-token bag (seeds included) is lifted to the tangent space at the origin, passed through a
residual MLP conditioned on (age, hop), pooled by a softmax whose logits are the recency/hop prior plus a
learned gate, then mapped back to the manifold. The MLP and the gate are zero-init, so at step 0 the head is
exactly expmap0(sum_p w_p logmap0(x_p)) under the fixed prior."""
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


class TokenFeatures(nn.Module):
    """Per-token (age, hop) -> feature vector. Age is normalised by the dataset's mean-field per-node
    inter-event time, then encoded with FIXED cos/sin frequencies (GraphMixer finds fixed beats learned)
    plus its log. Hop is one-hot."""

    def __init__(self, mnia: float, max_hop: int, d_time: int = 32):
        super().__init__()
        self.mnia = float(mnia)
        self.max_hop = int(max_hop)
        self.register_buffer("freqs", 1.0 / (10.0 ** torch.linspace(0.0, 3.0, d_time // 2)))
        self.dim = d_time + 1 + self.max_hop

    def forward(self, tokens: WalkTokens) -> Tuple[torch.Tensor, torch.Tensor]:
        """(feat [Q,T,F], prior_logits [Q,T] = -(log1p(age/mnia) + log1p(hop-1)); 0 = max, at the seed)."""
        a = tokens.ages.clamp_min(0).float() / self.mnia                        # [Q, T]
        hop_idx = tokens.positions.clamp_min(1).long().clamp_max(self.max_hop) - 1
        ang = a.unsqueeze(-1) * self.freqs                                      # [Q, T, d_time/2]
        feat = torch.cat([
            torch.cos(ang), torch.sin(ang),
            torch.log1p(a).unsqueeze(-1),
            F.one_hot(hop_idx, self.max_hop).to(a.dtype),
        ], dim=-1)                                                              # [Q, T, F]
        prior = -(torch.log1p(a) + torch.log1p(hop_idx.to(a.dtype)))            # [Q, T]
        return feat, prior


class TokenEncoder(nn.Module):
    """Residual MLP in the tangent space at the origin, plus an additive gate on the pooling logits. One
    hidden layer of width 2*d_emb, shared by both output heads; the GELU between fc1 and fc2 is what makes
    per-token transformation more than a linear map on the pooled point. Both output heads are zero-init,
    so z' == z and gate == 0 at step 0."""

    def __init__(self, d_emb: int, d_feat: int):
        super().__init__()
        h = 2 * d_emb
        self.fc1 = nn.Linear(d_emb + d_feat, h)
        self.fc2 = nn.Linear(h, d_emb)
        self.gate = nn.Linear(h, 1)
        for layer in (self.fc2, self.gate):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, z: torch.Tensor, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """z [Q,T,d], feat [Q,T,F] -> (z' [Q,T,d], gate [Q,T])."""
        h = F.gelu(self.fc1(torch.cat([z, feat], dim=-1)))                      # [Q, T, hidden]
        return z + self.fc2(h), self.gate(h).squeeze(-1)


class LinkPredHead(nn.Module):
    """Centroid-to-centroid head. E is a ManifoldParameter; TokenEncoder is the only other trained module."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float, max_walk_length: int,
                 d_time: int = 32):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        # walk length counts the seed, so positions run 1..max_walk_length -> that IS the hop cardinality.
        self.feats = TokenFeatures(mean_node_inter_arrival, max_walk_length, d_time)
        self.encoder = TokenEncoder(self.d_emb, self.feats.dim)

        # Spread init: geoopt random (std=1), not the near-origin wrapped normal. ManifoldParameter so
        # RiemannianAdam keeps E in the ball.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random(self.num_nodes, self.d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> one manifold point [Q, d]. Cold-bag guard falls back to the seed so an all -inf softmax row
        can't go NaN."""
        nodes = tokens.nodes.clamp_min(0).clone()                               # [Q, T] padding(-1) -> 0
        valid = tokens.mask.clone()                                             # [Q, T] real slots (seed incl.)
        cold = ~valid.any(dim=-1)
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        feat, prior = self.feats(tokens)                                        # [Q,T,F], [Q,T]
        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        z, gate = self.encoder(self.geom.logmap0(x), feat.to(emb.dtype))        # [Q,T,d], [Q,T]

        logits = prior.to(emb.dtype) + gate                                     # [Q, T]
        w = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)    # [Q, T] sums to 1
        return self.geom.expmap0((w.unsqueeze(-1) * z).sum(dim=-2))             # [Q, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries (seeds u); cand = B*C candidate queries (seeds v), query-major. -> [B, C]."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        p_v = p_v.view(b, p_v.shape[0] // b, d)                                 # [B, C, d]
        return -self.geom.dist(p_u.unsqueeze(1), p_v)                           # [B, C] higher = closer
