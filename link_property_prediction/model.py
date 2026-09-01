"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = -d_H(P_u, P_v) / rad(P_v).

The geodesic distance normalised by the CANDIDATE's hyperbolic radius alone. No learned scalar of
any kind, and no source-radius term.

Why rad(P_u) is absent. It is a per-row positive constant, so it cannot reorder candidates: the
argsort of -d/(rad_u*rad_v) and of -d/rad_v are IDENTICAL. It contributes exactly nothing to the
ranking. What it does contribute is the last free-sharpening channel. Softmax is only
SHIFT-invariant, not scale-invariant, so any per-row positive factor moves the cross-entropy while
leaving the MRR alone -- and with rad(P_u) in the denominator a GLOBAL rescale of E sharpens the
softmax for free (verified float64, scale every embedding by k, argsort unchanged throughout:
    k     = 1      0.2     0.01    1e-3
    d/(rad_u*rad_v) spread = 2.35   11.89   237.9   2378.6     <- runaway
    d/rad_v         spread = 1.4517 1.4280  1.4271  1.4271     <- pinned
). Dropping rad(P_u) makes the score DEGREE-0: numerator and denominator both scale linearly, so
contracting or expanding the embedding changes the logits by nothing. Exact in the small-radius
limit; ~2% drift by |x| ~ 0.45, where hyperbolic scaling stops behaving like a similarity.

The two forms this replaces both had a rank-neutral scale channel and both used it:
  -rad_u * rad_v * d   sharpens by EXPANDING  (margin ~ k^3). Measured: link -50% for +0.069 test.
  -d / (rad_u * rad_v) sharpens by CONTRACTING (margin ~ 1/k). Did not actually fire, but was open.

The price, which is real and has no mitigation here: there is now NO temperature channel at all.
Sharpness is fixed by the configuration's shape, and the logit spread is bounded by
2*rad(P_u)/rad(P_v) (triangle inequality through the origin). So the head cannot sharpen over
training, and it cannot do what a learned global temperature does -- concentrate gradient on
unsolved queries as it rises (measured on the geo_temp head: gradient mass on an unsolved row goes
23% -> 98% -> 100% as T goes 1 -> 5 -> 15). Nothing here focuses on hard examples. If this arm
underperforms with a healthy geometry, restoring a single GLOBAL learned T -- s = -T*d/rad(P_v) --
is the indicated next step: global T is self-regulating where a per-row radius is not, because
correctly- and incorrectly-ranked rows pull it in opposite directions on one shared scalar.

rad(P_v) itself IS rank-relevant and is the point: ranking is by log d(P_u,P_v) - log rad(P_v), a
distance ranking plus an additive per-candidate radial bias. Near-origin candidates are penalised
(d/rad_v -> infinity as rad_v -> 0), so unlike the product form this favours peripheral candidates.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over an
MLP of [log1p(age) | raw position | rad] at a fixed hidden width. Nothing is standardised: log1p
is a fixed function of the age alone, so no batch-dependent or dataset-derived quantity enters
the pooler. Learned head params: the MLP pooler alone."""


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

        score = -d_H(P_u, P_v) / rad(P_v). rad(P_v) is NOT detached here (unlike the pooler's `rad`
        feature): it carries gradient. There is no rad(P_u) term -- it is a per-row constant that
        cannot reorder candidates, and removing it makes the score scale-invariant."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        # clamp_min is a NaN guard for a centroid landing exactly on the origin (a bag whose tokens
        # cancel). At irange 1e-3 the pooled radii run ~0.02, so 1e-9 never binds in normal
        # operation -- the scale is set by irange, not by this floor.
        r_v = self.geom.dist0(p_v).clamp_min(1e-9)                              # [B, C] candidate radii
        return -(geo / r_v)                                                     # [B, C]
