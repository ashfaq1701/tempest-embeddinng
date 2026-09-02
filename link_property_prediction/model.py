"""Centroid-to-centroid head on the unit hypersphere: s(u,v) = geo_temp * cos(P_u, P_v) [+ pop_bias[v]].

Spherical counterpart of the Poincare-ball head. Closeness is cosine SIMILARITY, so the sign
flips relative to the ball: there the score was geo_temp * (-d_H) because distance is lower =
closer, here it is geo_temp * cos because similarity is higher = closer. The similarity is
scaled by a learned geo_temp (init 1.0). When the popularity channel is on, a learned per-node
scalar pop_bias[v] (zero-init, so it contributes exactly 0 at step 0) is added at FIXED unit
weight.

P_x is the weighted spherical mean of x's walk-token bag -- the weighted Euclidean sum
renormalised back onto the sphere. The pooling weights are a softmax over an MLP of
[log1p(age) | raw position | norm] at a fixed hidden width. Nothing is standardised: log1p is a
fixed function of the age alone, so no batch-dependent or dataset-derived quantity enters the
pooler. Learned head params: geo_temp, the MLP pooler, and num_nodes popularity scalars when on.

E IS NOT CONSTRAINED TO THE SPHERE. It is initialised uniformly on it, but left an ordinary
nn.Parameter so its magnitude is free to move. That is deliberate and load-bearing for the
`norm` feature: on a constrained sphere every |E[v]| is exactly 1, so the pooler's third
feature would be a CONSTANT and the channel dead. Leaving the magnitude free makes |E[v]| a
learned per-node scalar the pooler can actually read. The geometry is unaffected -- cosine
similarity is scale-invariant, and the spherical mean renormalises -- so only the pooler
feature and the norm's own dynamics depend on this choice.

A consequence worth knowing: with E a plain nn.Parameter, geoopt's RiemannianAdam falls back
to its Euclidean default manifold for it (identity egrad2rgrad, x+u retraction), i.e. it is
plain Adam on E. No optimizer change is needed for this head."""


import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class SphereManifold:
    """Unit-hypersphere geometry. Every operation is plain torch -- the sphere needs no
    manifold library: its projection IS normalisation, and autograd through that normalisation
    reproduces the tangent projection exactly."""

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cosine similarity, broadcasting over leading dims. HIGHER = closer.

        Replaces the ball's geodesic `dist`. On the unit sphere this is the inner product, and
        it is a monotone function of the geodesic distance arccos(<x,y>), so ranking by it is
        equivalent to ranking by angle -- without arccos's vanishing gradient at the poles."""
        return F.cosine_similarity(x, y, dim=-1)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """Euclidean magnitude. Replaces the ball's `dist0` (hyperbolic radius).

        Meaningful only because E is left unconstrained -- see the module docstring."""
        return x.norm(dim=-1)

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted spherical mean: x [Q,T,d], w [Q,T] -> [Q,d] on the unit sphere.

        The weighted Euclidean sum renormalised back onto the sphere. This is the standard
        extrinsic (chordal) mean; it agrees with the intrinsic Karcher mean for concentrated
        bags and is closed-form, where the Karcher mean needs an inner iteration."""
        return F.normalize((w.unsqueeze(-1) * x).sum(dim=-2), dim=-1)


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([age_norm | raw position | norm])).

    Features (`norm` is detached, so the pooler reads geometry but does not backprop through it):
      age      -- log1p(age). NOT standardised: log1p is a fixed function of the age alone, so
                  the feature stays absolutely anchored and no batch-dependent quantity enters
                  the pooler.
      pos      -- the RAW hop index as one scalar, 1..max_walk_len, padding 0. Padded slots are
                  masked out of the softmax, so their value never matters.
      norm     -- the token's Euclidean magnitude |E[v]|. The spherical counterpart of the
                  ball head's `rad`. It is informative ONLY because E is left off the sphere
                  (module docstring): on a constrained sphere it would be identically 1.

    Position is a RAW SCALAR here, not one-hot. One-hot lets the first Linear learn an arbitrary
    per-hop weight; a scalar forces the response to be affine in the hop index, so the pooler can
    only express a monotone depth preference and cannot single out, say, the seed slot. That is
    the trade this arm measures: 3 features and 162 head params against 7 and 290.

    `hidden` is the MLP width, pinned independently of the feature count: under an earlier rule
    where it scaled with the feature count, every feature change silently moved pooler capacity
    too and the two effects could not be separated (see the ablation in CLAUDE.md).
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden = int(hidden_dim)
        # 1 normalised age + 1 raw position scalar + 1 magnitude.
        self.n_feat = 3
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, geom: "SphereManifold", tokens: WalkTokens, x: torch.Tensor,
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
    """E is an ordinary nn.Parameter initialised on the sphere; the pooling weights carry no
    learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int,
                 use_pop_bias: bool = False, hidden_dim: int = 32):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.use_pop_bias = bool(use_pop_bias)
        self.geom = SphereManifold()
        self.bag_weights = BagWeights(hidden_dim)

        # Spread init: directions UNIFORM on the unit sphere. Normalising an isotropic Gaussian
        # is the standard way to sample the sphere uniformly -- normalising a uniform CUBE would
        # bias mass towards the corners. The sphere is compact and homogeneous, so there is no
        # boundary to stay away from and the ball head's near-origin init has no analogue: a
        # near-origin cluster here would just be a near-degenerate direction set.
        #
        # E is a plain nn.Parameter, NOT constrained to the sphere. Magnitudes start at exactly 1
        # and are then free to move, which is what makes the pooler's `norm` feature live. See
        # the module docstring.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            self.E.weight.copy_(F.normalize(torch.randn(self.num_nodes, self.d_emb), dim=-1))

        # Learned per-node popularity scalar, zero-init: the channel contributes exactly 0 at step 0,
        # so turning it on cannot perturb the starting point.
        if self.use_pop_bias:
            self.pop_bias = nn.Embedding(self.num_nodes, 1)
            nn.init.zeros_(self.pop_bias.weight)

        self.geo_temp = nn.Parameter(torch.tensor(1.0))

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> P [Q,d], the pooling-weighted spherical mean of the walk-token cloud."""
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
        score = geo_temp * cos [+ pop_bias[v]]  (similarity scaled by geo_temp; the learned per-node
        popularity scalar, when on, added at fixed unit weight). NOTE the sign: cosine similarity is
        HIGHER = closer, so there is no negation here, unlike the ball head's geo_temp * (-d_H)."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        sim = self.geom.similarity(p_u.unsqueeze(1), p_v)                       # [B, C] cosine similarity
        score = self.geo_temp * sim                                             # [B, C] scaled similarity
        if self.use_pop_bias:
            v_nodes = cand_tokens.seeds.view(b, c)                              # [B, C] candidate node ids
            score = score + self.pop_bias(v_nodes).squeeze(-1)
        return score
