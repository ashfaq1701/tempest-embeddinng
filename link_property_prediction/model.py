"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = geo_temp * (-d_H(P_u, P_v)) [+ pop_bias[v]].

The distance term is scaled by a learned geo_temp (init 1.0). When the popularity channel is on, a
learned per-node scalar pop_bias[v] (zero-init, so it contributes exactly 0 at step 0) is added at
FIXED unit weight -- the model sharpens the geometry via geo_temp while popularity rides alongside at
a constant scale.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over an
MLP of [time_enc(age) | hop_enc(hop) | rad] at a fixed hidden width. The time encoding is a fixed
cos/sin ladder over log-age normalised by a CONSTANT, so it is a pure function of age -- no batch
statistic and no dataset statistic enter the pooler at all. Learned head params: geo_temp, the MLP pooler, and num_nodes popularity scalars
when on."""


import math

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

    def dist0(self, x: torch.Tensor) -> torch.Tensor:
        """Hyperbolic radius: geodesic distance from the origin (0 at center, large near boundary)."""
        return self.manifold.dist0(x)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted gyro-midpoint: x [Q,T,d], w [Q,T] -> [Q,d]."""
        return self.manifold.weighted_midpoint(x, weights=w, reducedim=[-2], dim=-1, keepdim=False)


class TimeEncoder(nn.Module):
    """Fixed cos/sin ladder over log-age, plus a monotone linear term.

    Ages span seconds to years, so a ladder in raw seconds wastes most of its
    resolution: scaled to the largest age it cannot separate anything below
    roughly a day. Working in log-age gives uniform resolution per decade
    instead, and matches the heavy-tailed inter-event distribution.

    LMAX is a fixed constant rather than a dataset statistic. log1p of an
    elapsed time in seconds is bounded by arithmetic, not by measurement: one
    second is 0.69, one year is 17.3, three centuries is 23.0, so dividing by
    24 normalises every age without looking at the data. This keeps the
    encoder a pure function of age, which matters in a streaming setting where
    any adaptive divisor would make the same age encode differently over time
    and invalidate what the model has already learned.

    Frequencies are fixed: the gradient of cos(xw) wrt w scales with x, which
    destabilises training on large timestamps (Cong et al., ICLR 2023).
    """

    LMAX = 24.0     # log1p(1e10 s) = 23.03, about 317 years

    def __init__(self, time_dim: int, lam_min: float = 0.05,
                 lam_max: float = 2.0):
        super().__init__()
        time_dim = int(time_dim)
        if time_dim < 3:
            raise ValueError(
                f"time_dim must be >= 3 (1 linear + >= 2 periodic), got {time_dim}")
        if not 0.0 < lam_min <= lam_max:
            raise ValueError(
                f"need 0 < lam_min <= lam_max, got {lam_min} and {lam_max}")
        self.time_dim = time_dim
        self.n_per = time_dim - 1

        k = (self.n_per + 1) // 2                        # frequencies, ceil
        i = torch.arange(k, dtype=torch.float32) / max(k - 1, 1)
        lam = lam_min * (lam_max / lam_min) ** i         # geometric ladder
        self.register_buffer("w", 2.0 * math.pi / lam)   # [k], not a parameter

    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        """ages [...] (non-negative seconds) -> [..., time_dim] in [-1, 1]."""
        u = (torch.log1p(ages.clamp_min(0).float()) / self.LMAX).clamp_max(1.0)
        wu = u.unsqueeze(-1) * self.w                               # [..., k]
        # interleave cos_i, sin_i so an odd tail drops only the last sine
        pairs = torch.stack([torch.cos(wu), torch.sin(wu)], dim=-1).flatten(-2)
        lin = 1.0 - 2.0 * u                              # [+1, -1], monotone
        return torch.cat([lin.unsqueeze(-1), pairs[..., :self.n_per]], dim=-1)


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([time_enc(age) | hop_enc(hop) | rad])).

    Features (`rad` is detached, so the pooler reads geometry but does not backprop through it):
      time_enc -- TimeEncoder(age): d_time fixed features over log-age normalised by the
                  constant LMAX = 24. Absolutely anchored and dataset-independent -- a given age
                  maps to the same vector in every batch, on every dataset, forever.
      hop_enc  -- a learned embedding of the hop index. walk_tokens builds positions as
                  (lens - arange).clamp_min(0) with lens <= max_walk_len, so the values are
                  0..max_walk_len with 0 meaning padding -- hence max_walk_len + 1 rows and
                  padding_idx=0, which pins the pad row at zero and keeps it there.
      rad      -- geodesic distance from the origin: the token's hyperbolic radius.

    The hop embedding is initialised small (std 0.02): the time features are bounded in [-1, 1]
    and rad is O(1), so a default N(0, 1) table would dominate the input and the pooler would
    start on hop alone.

    `hidden` is the MLP width, pinned independently of the feature count: under an earlier rule
    where it scaled with the feature count, every feature change silently moved pooler capacity
    too and the two effects could not be separated (see the ablation in CLAUDE.md).
    """

    def __init__(self, max_walk_len: int, d_time: int = 16, d_pos: int = 4,
                 hidden_dim: int = 32):
        super().__init__()
        self.max_walk_len = int(max_walk_len)
        self.hidden = int(hidden_dim)
        self.time = TimeEncoder(d_time)
        # +1 row for the padding index 0; real hops are 1..max_walk_len.
        self.pos = nn.Embedding(self.max_walk_len + 1, int(d_pos), padding_idx=0)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.pos.weight[0].zero_()
        self.n_feat = int(d_time) + int(d_pos) + 1
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, geom: "PoincareManifold", tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        # TimeEncoder clamps the -1 padding sentinel to 0 internally, so padding is finite;
        # padded slots are masked out of the softmax below in any case.
        t_enc = self.time(tokens.ages)                                          # [Q, T, d_time]
        p_emb = self.pos(tokens.positions.clamp(0, self.max_walk_len))          # [Q, T, d_pos]
        rad = geom.dist0(x.detach()).unsqueeze(-1)                              # [Q, T, 1]
        feat = torch.cat([t_enc, p_emb, rad], dim=-1).to(x.dtype)               # [Q, T, n_feat]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; the pooling weights carry no learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int, max_walk_len: int,
                 init_irange: float = 1e-3, use_pop_bias: bool = False,
                 d_time: int = 16, d_pos: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.use_pop_bias = bool(use_pop_bias)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(max_walk_len, d_time, d_pos, hidden_dim)

        # Near-origin init: uniform(-irange, irange) per coord -> r ~ 2*irange*sqrt(d/3).
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.projx(
                (torch.rand(self.num_nodes, self.d_emb) * 2 - 1) * float(init_irange))
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        # Learned per-node popularity scalar, zero-init: the channel contributes exactly 0 at step 0,
        # so turning it on cannot perturb the starting point.
        if self.use_pop_bias:
            self.pop_bias = nn.Embedding(self.num_nodes, 1)
            nn.init.zeros_(self.pop_bias.weight)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> P [Q,d], the pooling-weighted gyro-midpoint of the walk-token cloud."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)                                               # all-padding walk -> use seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(self.geom, tokens, x, valid)                       # [Q, T] sums to 1, 0 on padding
        return self.geom.midpoint(x, w)                                         # [Q, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        score = geo_temp * (-geo) [+ pop_bias[v]]  (distance scaled by geo_temp; the learned per-node
        popularity scalar, when on, added at fixed unit weight)."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        score = self.geo_temp * (-geo)                                         # [B, C] scaled distance term
        if self.use_pop_bias:
            v_nodes = cand_tokens.seeds.view(b, c)                              # [B, C] candidate node ids
            score = score + self.pop_bias(v_nodes).squeeze(-1)
        return score
