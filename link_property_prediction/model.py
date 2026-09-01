"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = geo_temp * (-d_H(P_u, P_v)).

The distance term is scaled by a learned geo_temp (init 1.0). The score is purely geometric: there
is no per-node popularity term.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over an
MLP of [TimeEncoder(age) | pos_emb(hop) | rad] at a fixed hidden width. Nothing is standardised:
the encoders are fixed functions of the age and the hop alone, so no batch-dependent or
dataset-derived quantity enters the pooler. Learned head params: geo_temp, the position embedding
and the MLP pooler -- the TimeEncoder is frozen."""


import numpy as np
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
    """GraphMixer/TGAT cosine encoding: cos(w * x + b), w a fixed geometric frequency ladder.

    The ladder is 1 / 10^linspace(0, 9, time_dim), i.e. angular frequencies from 1 down to 1e-9, so
    channel i has period 2*pi*10^(9i/(time_dim-1)). FROZEN -- the ladder is a fixed basis and there
    is no option to train it.

    Applied here to the token's AGE (cutoff - t_edge), not to a raw timestamp. That keeps the
    feature absolutely anchored: a given age maps to the same encoding in every batch and on every
    dataset, whereas a raw Unix timestamp would shift the whole input distribution per dataset.

    KNOWN HAZARD ON A COARSE TIMESTAMP GRID, measured on this suite. A channel whose period is
    shorter than twice the grid spacing is below Nyquist and cannot resolve anything: every age is
    a multiple of the grid, so the channel is a deterministic hash of the grid index rather than a
    time signal, and it still looks healthy by variance. Train-split grids, measured:
        Patent    604,800 s (WEEKLY)  -> ~19/32 channels sub-Nyquist at d_time=32
        YouTube    86,400 s (daily)   -> ~16/32
        Flickr     86,400 s (daily)   -> ~16/32
        WikiLink   86,400 s (daily)   -> ~16/32
        GoogleLocal 4 s / ML-20M 2 s / Taobao 1 s / Yelp 2 s  -> 0/32
    The ladder here is unfloored. Flooring it at the grid resolution is a separate change.
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = int(time_dim)
        self.w = nn.Linear(1, self.time_dim)
        self.w.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim, dtype=np.float32))
            .reshape(self.time_dim, -1), requires_grad=False)
        self.w.bias = nn.Parameter(torch.zeros(self.time_dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L] -> [B, L, time_dim]."""
        return torch.cos(self.w(x.unsqueeze(dim=2)))


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([TimeEncoder(age) | pos_emb(hop) | rad])).

    Features (`rad` is detached, so the pooler reads geometry but does not backprop through it):
      age  -- WalkTokens.ages, i.e. cutoff - t_edge: 0 at the seed, >= 1 for context edges, -1 on
              padding. clamp_min(0) sends the padding sentinel to 0; padded slots are masked out
              of the softmax below in any case. Encoded by the FROZEN TimeEncoder to `d_time`
              channels. Age rather than raw timestamp keeps the feature absolutely anchored.
      hop  -- the hop index from the seed, EMBEDDED to `d_pos` channels. Range is 1..max_walk_len
              for real slots (1 = seed, max_walk_len = oldest) with 0 for padding, so the table has
              max_walk_len + 1 rows and row 0 is the never-used padding slot.
      rad  -- geodesic distance from the origin: the token's hyperbolic radius.

    Both encoders replace raw scalars. The scalar hop forced the response to be affine in the hop
    index, so the pooler could only express a monotone depth preference and could not single out,
    say, the seed slot; a table can. log1p(age) gave the MLP one monotone channel; the cosine
    ladder gives it `d_time` channels at different timescales.

    `hidden` is the MLP width, pinned independently of the feature count: under an earlier rule
    where it scaled with the feature count, every feature change silently moved pooler capacity
    too and the two effects could not be separated (see the ablation in CLAUDE.md).
    """

    def __init__(self, max_walk_len: int, hidden_dim: int = 32,
                 d_time: int = 32, d_pos: int = 4):
        super().__init__()
        self.max_walk_len = int(max_walk_len)
        self.hidden = int(hidden_dim)
        self.d_time = int(d_time)
        self.d_pos = int(d_pos)
        self.time_enc = TimeEncoder(self.d_time)
        # +1 row: hop 0 is the padding slot (real hops are 1..max_walk_len).
        self.pos_emb = nn.Embedding(self.max_walk_len + 1, self.d_pos)
        self.n_feat = self.d_time + self.d_pos + 1
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, geom: "PoincareManifold", tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        age = tokens.ages.clamp_min(0).to(x.dtype)                              # [Q, T]
        t_e = self.time_enc(age)                                                # [Q, T, d_time]
        p_e = self.pos_emb(tokens.positions.clamp(0, self.max_walk_len))        # [Q, T, d_pos]
        rad = geom.dist0(x.detach()).unsqueeze(-1)                              # [Q, T, 1]
        feat = torch.cat([t_e, p_e, rad], dim=-1).to(x.dtype)                   # [Q, T, n_feat]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; the pooling weights carry no learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int, max_walk_len: int,
                 init_irange: float = 1e-3, hidden_dim: int = 32,
                 d_time: int = 32, d_pos: int = 4):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(max_walk_len, hidden_dim, d_time, d_pos)

        # Near-origin init: uniform(-irange, irange) per coord -> r ~ 2*irange*sqrt(d/3).
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.projx(
                (torch.rand(self.num_nodes, self.d_emb) * 2 - 1) * float(init_irange))
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

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
        score = geo_temp * (-geo), the distance scaled by geo_temp."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        return self.geo_temp * (-geo)                                           # [B, C] scaled distance
