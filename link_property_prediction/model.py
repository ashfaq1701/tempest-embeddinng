"""Centroid-to-centroid head on the Poincaré ball:

    s(u,v) = -geo_temp_net(TimeEncoder(t_query)) * d_H(P_u, P_v)

The distance is scaled by a TIME-DEPENDENT TEMPERATURE: `geo_temp_net`, a
Linear -> GELU -> Linear MLP over the cosine encoding of the query's own cutoff timestamp,
replacing the single learned scalar geo_temp. It reads the pooler's frozen TimeEncoder rather than
carrying a second copy of the same fixed basis.

UNCONSTRAINED SIGN, deliberately -- and at init this is NOT a corner case. The score is -temp * d,
so a NEGATIVE temperature INVERTS that query's ranking: farther candidates score higher. geo_temp
started at exactly +1.0 for every seed and could only drift from there; this starts wherever the
random MLP init lands, which straddles zero. Measured over 8192 real-scale YouTube cutoffs:

  seed 42   mean -0.0022  min -0.4562  max +0.3239   50.8% NEGATIVE
  seed  0   mean -0.0879  min -0.6701  max +0.3327   73.1%
  seed  1   mean -0.1414  min -0.4471  max +0.1715   96.9%
  seed  3   mean +0.1330  min -0.2724  max +0.4559    9.3%

At seed 42, the seed this project runs, HALF the queries begin with an inverted ranking, and the
fraction swings from 9% to 97% across seeds -- so this head's init is far more seed-sensitive than
the scalar it replaces. The sign is NOT constrained and the temperature is NOT logged, so a negative
branch is silent: nothing in the epoch line will show it. If a run underperforms, dumping the
temperature over the eval cutoffs is the first check, and constraining the output positive
(softplus, or 1 + net(...)) is the fix. The scale the
model applies is therefore a function of WHEN the query happens, not one global constant -- it can
sharpen in dense periods and flatten in sparse ones.

The temperature is constant across a query's candidate row, so like geo_temp it cannot reorder
candidates: it is a pure temperature, and its effect is on the cross-entropy and the gradients,
not the MRR of a fixed embedding. Unlike geo_temp it is SHARED ACROSS QUERIES only through the
network weights, so different query times get different scales.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over an
MLP of [TimeEncoder(age) | hop embedding | rad] at a fixed hidden width. Nothing is standardised:
the age is a fixed function of the timestamps alone, so no batch-dependent quantity enters
the pooler. Learned head params: geo_temp, the TimeEncoder, the hop embedding and the MLP pooler."""


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
    """GraphMixer/TGAT cosine time encoding: cos(w * t + b), w a fixed geometric frequency ladder.

    The ladder is 1 / 10^linspace(0, 9, time_dim), i.e. angular frequencies from 1 down to 1e-9, so
    channel i has period 2*pi*10^(9i/(time_dim-1)). w and b are FROZEN -- the ladder is a fixed
    basis and there is no option to train it.

    KNOWN HAZARD ON A COARSE TIMESTAMP GRID, measured previously on this project (commit 7c7c112a):
    a channel whose period is shorter than twice the grid spacing is below Nyquist and cannot
    resolve anything -- every age is a multiple of the grid, so the channel becomes a deterministic
    hash of the grid index rather than a time signal. It still looks healthy by variance, which is
    why no shape or parameter check catches it. YouTube is the worst case in the suite: 175
    distinct train timestamps, 24h apart. That earlier version floored the ladder at the grid
    resolution; this one is the plain TGAT ladder with no floor, so the low channels are expected
    to be sub-Nyquist here. See the run header for the measured count.
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = int(time_dim)
        self.w = nn.Linear(1, self.time_dim)
        # FROZEN. The ladder is a fixed basis, not something to fit: requires_grad=False on both
        # tensors, with no flag to turn it back on.
        self.w.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim, dtype=np.float32))
            .reshape(self.time_dim, -1), requires_grad=False)
        self.w.bias = nn.Parameter(torch.zeros(self.time_dim), requires_grad=False)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """timestamps [B, L] -> [B, L, time_dim]."""
        return torch.cos(self.w(timestamps.unsqueeze(dim=2)))


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([TimeEncoder(age) | hop_emb(hop) | rad])).

    Features (`rad` is detached, so the pooler reads geometry but does not backprop through it):
      age  -- cutoff - t_edge, computed here from WalkTokens.timestamps rather than stored. The
              seed slot sits at t_edge == cutoff so its age is 0; padding is clamped to 0 and is
              masked out of the softmax below in any case. Encoded by TimeEncoder to `d_time`
              channels. Age, not raw timestamp: a raw timestamp would make the feature
              dataset-dependent and unanchored across splits.
      hop  -- the hop index from the seed, EMBEDDED to `d_hop` channels. Range is 1..max_walk_len
              for real slots (1 = seed, max_walk_len = oldest) with 0 for padding, so the table has
              max_walk_len + 1 rows and row 0 is the never-used padding slot.
      rad  -- geodesic distance from the origin: the token's hyperbolic radius.

    A learned embedding per hop is strictly more expressive than the raw scalar it replaces: the
    scalar forced the response to be affine in the hop index, so the pooler could only express a
    monotone depth preference and could not single out, say, the seed slot. The table can.

    `hidden` is the MLP width, pinned independently of the feature count: under an earlier rule
    where it scaled with the feature count, every feature change silently moved pooler capacity
    too and the two effects could not be separated (see the ablation in CLAUDE.md).
    """

    def __init__(self, max_walk_len: int, hidden_dim: int = 32,
                 d_time: int = 32, d_hop: int = 4):
        super().__init__()
        self.max_walk_len = int(max_walk_len)
        self.hidden = int(hidden_dim)
        self.d_time = int(d_time)
        self.d_hop = int(d_hop)
        self.time_enc = TimeEncoder(self.d_time)
        # +1 row: hop 0 is the padding slot (real hops are 1..max_walk_len).
        self.hop_emb = nn.Embedding(self.max_walk_len + 1, self.d_hop)
        self.n_feat = self.d_time + self.d_hop + 1
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, geom: "PoincareManifold", tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        # age = cutoff - t_edge. Padding carries t_edge = -1, which would give a huge positive age,
        # so clamp after subtracting; those slots are masked out of the softmax regardless.
        age = (tokens.cutoffs.unsqueeze(-1) - tokens.timestamps).clamp_min(0).to(x.dtype)  # [Q, T]
        t_e = self.time_enc(age)                                                # [Q, T, d_time]
        h_e = self.hop_emb(tokens.positions.clamp(0, self.max_walk_len))        # [Q, T, d_hop]
        rad = geom.dist0(x.detach()).unsqueeze(-1)                              # [Q, T, 1]
        feat = torch.cat([t_e, h_e, rad], dim=-1).to(x.dtype)                   # [Q, T, n_feat]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; the pooling weights carry no learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int, max_walk_len: int,
                 init_irange: float = 1e-3, hidden_dim: int = 32,
                 d_time: int = 32, d_hop: int = 4):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(max_walk_len, hidden_dim, d_time, d_hop)

        # Near-origin init: uniform(-irange, irange) per coord -> r ~ 2*irange*sqrt(d/3).
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.projx(
                (torch.rand(self.num_nodes, self.d_emb) * 2 - 1) * float(init_irange))
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        # Time-dependent distance temperature, replacing the scalar geo_temp: an MLP over the
        # cosine encoding of the query's own cutoff. Reuses the pooler's TimeEncoder, which is
        # frozen and stateless, so there is one fixed basis in the head rather than two copies.
        # Shares hidden_dim with the pooler MLP; no separate flag, so --hidden-dim moves both.
        self.geo_temp_net = nn.Sequential(nn.Linear(d_time, hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, 1))

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
        score = -temp(t_query) * geo. The temperature comes from the SOURCE bag's cutoff, which
        every candidate in the row shares, so it scales the row without reordering it."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        t_enc = self.bag_weights.time_enc(src_tokens.cutoffs.to(p_u.dtype).unsqueeze(0))
        temp = self.geo_temp_net(t_enc.squeeze(0))                              # [B, 1] per-query
        return -temp * geo                                                      # [B, C]
