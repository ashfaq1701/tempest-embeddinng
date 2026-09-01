"""Centroid-to-centroid head on the Poincaré ball:

    s(u,v) = geo_temp * ( -d_H(P_u, P_v) / (rad(P_u) + rad(P_v)) )

A bounded, scale-invariant tree-similarity, sharpened by ONE global learned temperature. This is
the SimCLR/CLIP/InfoNCE shape -- bounded similarity divided by a temperature -- with the normalised
Gromov product standing in for cosine.

WHAT THE SIMILARITY IS. The Gromov product (u|v)_o = (rad_u + rad_v - d)/2 is, in a tree, the depth
of u and v's lowest common ancestor as seen from the origin. Verified identity:

    d / (rad_u + rad_v)  ==  1 - 2*(u|v)_o / (rad_u + rad_v)

so minimising it maximises the SHARE of each node's root-path that the two have in common. That is
the tree analogue of cosine similarity, and the natural similarity for a hyperbolic embedding.

WHY EVERY TERM EARNS ITS PLACE.
  scale-invariant -- numerator and denominator are both degree-1, so contracting or expanding E
    changes the logits by nothing. Neither of the two exploits measured on earlier heads exists:
    -rad_u*rad_v*d sharpened by EXPANDING (margin ~ k^3; it fired -- link -50% for +0.069 test),
    -d/(rad_u*rad_v) could sharpen by CONTRACTING (margin ~ 1/k).
  rad_u is RANK-RELEVANT here, unlike in the product/quotient forms where it was a per-row constant
    that could not reorder anything and existed only to leak temperature. Additively it does not
    factor out: d/dr_u of d_i/(r_u+r_v_i) = -d_i/(r_u+r_v_i)^2 differs per candidate.

WHY THE TEMPERATURE IS MANDATORY, NOT OPTIONAL. The similarity is bounded in [0,1] by the triangle
inequality through the origin, and the EFFECTIVE spread over real pairs is far smaller -- measured
~0.25 at moderate radii, falling to 0.10 near the boundary. Un-tempered, that caps cross-entropy
about 0.16-0.24 below chance (log 6 = 1.792), i.e. structurally incapable of confidence. The
preceding arm removed the temperature entirely and stalled at link 1.60 / test 0.2529, the worst of
four heads, with headroom it could not use. Bounded similarity through a softmax needs a
temperature for exactly the reason cosine does in contrastive learning.

WHY A GLOBAL SCALAR IS SAFE WHERE A PER-ROW RADIUS WAS NOT. A per-row temperature lets each query
move its own knob in its own favour and nothing cancels -- that is the free lunch. One shared
scalar is self-regulating: correctly-ranked rows push it up, incorrectly-ranked rows push it down,
on the same parameter (verified: 6 correct rows want T up, 2 wrong rows outvote them). It also
supplies the focusing the un-tempered head lacked -- gradient mass on an unsolved row goes
23% -> 98% -> 100% as T goes 1 -> 5 -> 15.

geo_temp is LINEAR, init 1.0, exactly as the geo_temp*(-d) baseline parameterises it, so this head
differs from that baseline in the SIMILARITY FUNCTION ALONE.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over an
MLP of [log1p(age) | raw position | rad] at a fixed hidden width. Nothing is standardised: log1p
is a fixed function of the age alone, so no batch-dependent or dataset-derived quantity enters
the pooler. Learned head params: geo_temp and the MLP pooler."""


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


class BagWeights(nn.Module):
    """Pooling weights = softmax(MLP([age_norm | onehot(position) | rad])).

    Features (`rad` is detached, so the pooler reads geometry but does not backprop through it):
      age      -- log1p(age). NOT standardised: log1p is a fixed function of the age alone, so
                  the feature stays absolutely anchored and no batch-dependent quantity enters
                  the pooler.
      pos      -- the RAW hop index as one scalar, 1..max_walk_len, padding 0. Padded slots are
                  masked out of the softmax, so their value never matters.
      rad      -- geodesic distance from the origin: the token's hyperbolic radius

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
        # 1 normalised age + 1 raw position scalar + 1 radius.
        self.n_feat = 3
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, geom: "PoincareManifold", tokens: WalkTokens, x: torch.Tensor,
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
        rad = geom.dist0(x.detach()).unsqueeze(-1)                              # [Q, T, 1]
        feat = torch.cat([age, pos, rad], dim=-1).to(x.dtype)                   # [Q, T, 3]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; the pooling weights carry no learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int, max_walk_len: int,
                 init_irange: float = 1e-3, hidden_dim: int = 32):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(max_walk_len, hidden_dim)

        # Near-origin init: uniform(-irange, irange) per coord -> r ~ 2*irange*sqrt(d/3).
        #
        # irange stays at 1e-3, unchanged from the multiplicative head, so this arm is a
        # SINGLE-VARIABLE change against it (product -> quotient) with the init held fixed.
        # Be aware of what that costs, measured at d=64 through the real pooled path (rad of the
        # POOLED centroid, which runs ~1/3 of the per-node rad since a midpoint of ~10 random
        # tokens sits closer to the origin; maxprob = largest softmax weight over a 1+5 row,
        # uniform = 0.167):
        #     irange 1e-3 -> rad(pooled) 0.021, maxprob 0.976   <- SATURATED at step 0
        #     irange 2e-2 -> rad(pooled) 0.064, maxprob 0.569
        #     irange 5e-2 -> rad(pooled) 0.155, maxprob 0.344
        # rad(P_v) is in a DENOMINATOR, so a small init still means large radial gradients
        # (d/dr_v of d/r_v = -d/r_v^2). Dropping rad(P_u) removes one factor of 1/r, so this arm
        # opens FLATTER than the two-radius quotient did -- measured below. The two-radius arm
        # survived irange 1e-3 without NaN, so this one should comfortably.
        # If epoch 1 diverges or NaNs, raise this before concluding anything about the score.
        self.E = nn.Embedding(self.num_nodes, self.d_emb)
        with torch.no_grad():
            init = self.geom.manifold.projx(
                (torch.rand(self.num_nodes, self.d_emb) * 2 - 1) * float(init_irange))
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        # ONE global temperature. Linear, init 1.0 -- identical parameterisation to the
        # geo_temp*(-d) baseline, so the similarity function is the only variable between them.
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

        score = geo_temp * ( -d_H(P_u,P_v) / (rad(P_u) + rad(P_v)) ). Neither radius is detached
        here (unlike the pooler's `rad` feature): both carry gradient, and both are rank-relevant
        because the sum does not factor out of the row."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        r_u = self.geom.dist0(p_u).unsqueeze(1)                                 # [B, 1] source radius
        r_v = self.geom.dist0(p_v)                                              # [B, C] candidate radii
        # clamp_min is a NaN guard for BOTH centroids landing exactly on the origin, the only way
        # the sum can vanish. It does not set any scale: the score is degree-0, so the embedding
        # scale cancels and irange cannot affect sharpness the way it did on the quotient head.
        sim = geo / (r_u + r_v).clamp_min(1e-9)                                 # [B, C] in [0,1]
        return self.geo_temp * (-sim)                                           # [B, C]
