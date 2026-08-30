"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = geo_temp * (-d_H(P_u, P_v)) [+ pop_bias[v]].

The distance term is scaled by a learned geo_temp (init 1.0). When the popularity channel is on, a
learned per-node scalar pop_bias[v] (zero-init, so it contributes exactly 0 at step 0) is added at
FIXED unit weight -- the model sharpens the geometry via geo_temp while popularity rides alongside at
a constant scale.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over an
MLP of [time encoding | position embedding | rad] at a fixed hidden width. Learned head params:
geo_temp, the MLP pooler (encoder included), and num_nodes popularity scalars when on."""

import math
from dataclasses import replace

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


class TimeEncoding(nn.Module):
    """Fixed cos/sin features over RELATIVE age u = age / T_train, plus the linear term -u.

    One cosine is a ruler whose ticks are `lam` apart: it resolves structure at that scale and
    cannot separate t from t + lam. Ages span decades, so a ladder of wavelengths is needed --
    geometric, so resolution is uniform in log-time.

    The ladder is fixed in units of T_train rather than in seconds. Measured over the suite, ages
    occupy ~2 decades of T_train on every dataset and the datasets overlap in relative units
    (p10 >= 7e-5, p90 <= 0.43), so one window serves all of them and T_train -- a statistic of the
    edge timestamps -- is the only per-dataset input. Nothing here depends on walk sampling.

    Frequencies are NOT learnable: with raw ages the gradient d/dw of sin(w*t) is ~t (1e7 here) and
    oscillating, so a badly placed frequency cannot be trained into a good one. GraphMixer reports
    the same, using a fixed encoding by choice.

    The linear term carries monotone recency; the cosines cannot express "fresher is better" on
    their own, which is why Time2Vec keeps one too.

    `time_dim` is the TOTAL output width, so the caller sizes the feature directly instead of
    reasoning about a frequency count that gets doubled: slot 0 is the linear term and the
    remaining time_dim - 1 slots are cos/sin in quadrature over ceil((time_dim - 1) / 2)
    geometric frequencies, truncated to fit. An even time_dim therefore leaves the highest
    frequency with its cosine only.
    """

    LAM_MIN, LAM_MAX = 1e-4, 1.0        # wavelengths in units of T_train

    def __init__(self, time_dim: int, t_train: float):
        super().__init__()
        if int(time_dim) < 3:
            raise ValueError(f"time_dim must be >= 3 (1 linear + >= 2 periodic), got {time_dim}")
        self.time_dim = int(time_dim)
        self.t_train = max(float(t_train), 1.0)
        n_per = self.time_dim - 1                                       # periodic slots
        k = (n_per + 1) // 2                                            # frequencies, ceil
        i = torch.arange(k, dtype=torch.float32) / max(k - 1, 1)
        wl = self.LAM_MIN * (self.LAM_MAX / self.LAM_MIN) ** i          # geometric, relative
        self.n_per = n_per
        self.register_buffer("w", 2.0 * math.pi / wl)                   # [k], fixed

    def forward(self, ages: torch.Tensor) -> torch.Tensor:
        """ages [Q,T] (>=0; padding pre-clamped) -> [Q,T,time_dim]"""
        u = (ages.clamp_min(0).float() / self.t_train).unsqueeze(-1)    # [Q,T,1] ~ [0,1]
        wu = u * self.w                                                 # [Q,T,k]
        # interleave cos_i, sin_i so truncating an odd tail drops only the last sine
        pairs = torch.stack([torch.cos(wu), torch.sin(wu)], dim=-1).flatten(-2)   # [Q,T,2k]
        return torch.cat([-u, pairs[..., :self.n_per]], dim=-1)


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([time encoding | position embedding | rad])).

    Features (`rad` is detached, so the pooler reads geometry but does not backprop through it):
      time -- TimeEncoding(age): 2k+1 fixed cos/sin/linear features over age / T_train
      pos  -- a learned embedding of the hop index, 1 = seed .. max_walk_len
      rad  -- geodesic distance from the origin: the token's hyperbolic radius

    Position is a LOOKUP, not a sinusoid: there are only max_walk_len distinct values (5 in the
    default config), so a table is exact, smaller and strictly more expressive than any smooth
    encoding -- it can represent the seed slot's specialness directly. Sinusoidal position codes
    earn their keep on long or unbounded sequences, which this is not.

    `hidden` is the MLP width, pinned independently of the feature count: under an earlier rule
    where it scaled with N_FEAT, every feature change silently moved pooler capacity too and the
    two effects could not be separated (see the pooler-feature ablation in CLAUDE.md).
    """

    def __init__(self, t_train: float, max_walk_len: int, time_dim: int = 16,
                 pos_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.hidden = int(hidden_dim)
        self.time = TimeEncoding(time_dim, t_train)
        # +1 row for the padding index 0; real positions are 1..max_walk_len.
        self.pos = nn.Embedding(int(max_walk_len) + 1, int(pos_dim), padding_idx=0)
        # Small init: the time features are bounded in [-1, 1] and rad is O(1), so a default
        # N(0,1) table would dominate the input and the pooler would start on position alone.
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.pos.weight[0].zero_()
        self.n_feat = int(time_dim) + int(pos_dim) + 1
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, geom: "PoincareManifold", tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        t_enc = self.time(tokens.ages)                                          # [Q, T, 2k+1]
        p_emb = self.pos(tokens.positions.clamp_min(0))                         # [Q, T, d_pos]
        rad = geom.dist0(x.detach()).unsqueeze(-1)                              # [Q, T, 1]
        feat = torch.cat([t_enc, p_emb, rad], dim=-1).to(x.dtype)               # [Q, T, n_feat]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; the pooling weights carry no learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int, t_train: float, max_walk_len: int,
                 init_irange: float = 1e-3, use_pop_bias: bool = False,
                 time_dim: int = 16, pos_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.use_pop_bias = bool(use_pop_bias)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(t_train, max_walk_len, time_dim, pos_dim, hidden_dim)

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
            # Slot 0 now IS the seed, so it must carry the seed's hop index. Its position is still
            # 0 from the padding fill, which indexes the pooler's frozen padding row instead of the
            # learned "position 1 = seed" row. Harmless today -- a cold bag has exactly one valid
            # token, so its softmax weight is 1.0 whatever the logit -- but it would quietly become
            # wrong the moment a cold bag carried more than one token.
            positions = tokens.positions.clone()
            positions[cold, 0] = 1
            tokens = replace(tokens, positions=positions)

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
