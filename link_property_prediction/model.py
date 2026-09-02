"""Centroid-to-centroid head in flat Euclidean space: s(u,v) = geo_temp * (-||P_u - P_v||) [+ pop_bias[v]].

Flat-space counterpart of the Poincare-ball head, and the control it exists to provide: it keeps
every other moving part identical and removes ONLY the curvature. A gap between this head and the
ball head is attributable to the geometry; anything the two share is not.

The distance is the ordinary L2 norm, scaled by a learned geo_temp (init 1.0). Distance is LOWER =
closer, as on the ball, so the score keeps the negation. When the popularity channel is on, a
learned per-node scalar pop_bias[v] (zero-init, so it contributes exactly 0 at step 0) is added at
FIXED unit weight.

P_x is the weighted arithmetic mean of x's walk-token bag -- the flat-space midpoint, and the exact
Frechet mean of the Euclidean metric, where the ball needs a gyro-midpoint and the sphere needs a
renormalisation. The pooling weights are a softmax over an MLP of [log1p(age) | raw position | norm]
at a fixed hidden width. Nothing is standardised: log1p is a fixed function of the age alone, so no
batch-dependent or dataset-derived quantity enters the pooler. Learned head params: geo_temp, the
MLP pooler, and num_nodes popularity scalars when on.

`norm` needs no special handling here, unlike on the sphere: in flat space the distance to the
origin IS the Euclidean norm, so the pooler's third feature is the exact analogue of the ball
head's `rad` (dist0) rather than a substitute for it.

E is unconstrained -- flat space has no constraint to impose. The init is uniform inside the unit
BALL, which only sets the starting scale; nothing keeps E there afterwards. geoopt's RiemannianAdam
falls back to its Euclidean default manifold for a plain nn.Parameter (identity egrad2rgrad, x+u
retraction), so E gets a standard Adam update and no optimizer change is needed."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class EuclideanManifold:
    """Flat-space geometry. The zero-curvature control for the ball and sphere heads."""

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Elementwise L2 distance, broadcasting over leading dims. LOWER = closer."""
        return (x - y).norm(dim=-1)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """Euclidean magnitude = the distance to the origin. The flat-space `dist0`."""
        return x.norm(dim=-1)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted arithmetic mean: x [Q,T,d], w [Q,T] summing to 1 over T -> [Q,d].

        The flat-space midpoint, and the exact Frechet mean of the Euclidean metric -- no
        projection or renormalisation, unlike the ball's gyro-midpoint or the sphere's."""
        return (w.unsqueeze(-1) * x).sum(dim=-2)


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([age_norm | raw position | norm])).

    Features (`norm` is detached, so the pooler reads geometry but does not backprop through it):
      age      -- log1p(age). NOT standardised: log1p is a fixed function of the age alone, so
                  the feature stays absolutely anchored and no batch-dependent quantity enters
                  the pooler.
      pos      -- the RAW hop index as one scalar, 1..max_walk_len, padding 0. Padded slots are
                  masked out of the softmax, so their value never matters.
      norm     -- the token's Euclidean magnitude |E[v]|, i.e. its distance from the origin.
                  The exact flat-space analogue of the ball head's `rad`.

    Position is a RAW SCALAR here, not one-hot. One-hot lets the first Linear learn an arbitrary
    per-hop weight; a scalar forces the response to be affine in the hop index, so the pooler can
    only express a monotone depth preference and cannot single out, say, the seed slot. That is
    the trade this arm measures: 3 features and 162 head params against 7 and 290.

    `hidden` is the MLP width, pinned independently of the feature count: under an earlier rule
    where it scaled with the feature count, every feature change silently moved pooler capacity
    too and the two effects could not be separated (see the ablation in CLAUDE.md).
    """

    def __init__(self, max_walk_len: int, hidden_dim: int = 32):
        super().__init__()
        self.max_walk_len = int(max_walk_len)
        self.hidden = int(hidden_dim)
        # 1 normalised age + 1 raw position scalar + 1 magnitude.
        self.n_feat = 3
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, geom: "EuclideanManifold", tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        # log1p compresses ages that span orders of magnitude and are heavily right-skewed
        # (measured on real YouTube tokens: skew +1.83 raw, -1.07 after; and the mean is 2-3x the
        # median on every dataset in the suite). It is a fixed function of the age alone, so the
        # feature stays absolutely anchored -- a given age maps to the same value in every batch
        # and on every dataset. clamp_min(0) sends the -1 padding sentinel to log1p(0) = 0 rather
        # than -inf; padded slots are masked out of the softmax below in any case.
        age = torch.log1p(tokens.ages.clamp_min(0).to(x.dtype)).unsqueeze(-1)   # [Q, T, 1]
        # RAW hop index: 1 = seed .. max_walk_len = oldest, padding 0. Masked out below.
        pos = tokens.positions.unsqueeze(-1).to(x.dtype)                        # [Q, T, 1]
        norm = geom.norm(x.detach()).unsqueeze(-1)                              # [Q, T, 1]
        feat = torch.cat([age, pos, norm], dim=-1).to(x.dtype)                  # [Q, T, 3]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is an ordinary nn.Parameter initialised inside the unit ball; the pooling weights carry
    no learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int, max_walk_len: int,
                 init_radius: float = 1.0, use_pop_bias: bool = False,
                 hidden_dim: int = 32):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.use_pop_bias = bool(use_pop_bias)
        self.geom = EuclideanManifold()
        self.bag_weights = BagWeights(max_walk_len, hidden_dim)

        # Uniform inside the ball of radius `init_radius`: a uniform DIRECTION (normalised
        # isotropic Gaussian) times a radius drawn as R * U^(1/d). The 1/d exponent is what makes
        # it uniform by VOLUME -- drawing the radius uniformly would pile mass near the origin,
        # and normalising a uniform cube would bias it towards the corners.
        #
        # In high d this concentrates near the surface (median radius R * 0.5^(1/d) = 0.989 R at
        # d=64), which is a property of the ball, not a bug: at d=64 a uniform-in-ball init is
        # nearly a uniform-on-sphere init. It only sets the starting scale -- flat space imposes
        # no constraint, and nothing keeps E inside the ball once training starts.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            direction = F.normalize(torch.randn(self.num_nodes, self.d_emb), dim=-1)
            radius = float(init_radius) * torch.rand(self.num_nodes, 1).pow(1.0 / self.d_emb)
            self.E.weight.copy_(direction * radius)

        # Learned per-node popularity scalar, zero-init: the channel contributes exactly 0 at step 0,
        # so turning it on cannot perturb the starting point.
        if self.use_pop_bias:
            self.pop_bias = nn.Embedding(self.num_nodes, 1)
            nn.init.zeros_(self.pop_bias.weight)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> P [Q,d], the pooling-weighted arithmetic mean of the walk-token cloud."""
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
        score = geo_temp * (-dist) [+ pop_bias[v]]  (L2 distance scaled by geo_temp; the learned
        per-node popularity scalar, when on, added at fixed unit weight). Distance is LOWER =
        closer, so the negation is kept -- as on the ball, unlike the sphere's similarity."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        dist = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] L2 distance
        score = self.geo_temp * (-dist)                                         # [B, C] scaled distance
        if self.use_pop_bias:
            v_nodes = cand_tokens.seeds.view(b, c)                              # [B, C] candidate node ids
            score = score + self.pop_bias(v_nodes).squeeze(-1)
        return score
