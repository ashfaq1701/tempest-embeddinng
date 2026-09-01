"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = -d_H(P_u, P_v) / (rad(P_u) * rad(P_v)).

RADIUS-NORMALISED distance. No learned scalar of any kind. The radii divide rather than multiply,
which is the whole point: it closes the rank-neutral temperature channel that the multiplicative
form leaves open.

Why the division and not the product (measured in float64, see the commit message):
  Ranking within a query row is invariant to any POSITIVE MULTIPLICATIVE row constant, so rad(P_u)
  -- fixed across the row's candidates -- cannot change the MRR either way. But softmax is only
  SHIFT-invariant, not scale-invariant, so rad(P_u) DOES move the cross-entropy. Under the product
  form that is a free lunch: grow rad(P_u) on queries already ranked correctly and the loss falls
  with zero ranking gain (verified: r_u 2e-3 -> 2.9 takes max softmax prob 0.167 -> 0.982 with the
  argsort unchanged; and the YouTube run drove link -50% for +0.069 test).
  Under the QUOTIENT form the same move cancels. As P_u -> origin the temperature 1/rad(P_u) blows
  up, but d_H(P_u,P_v) -> rad(P_v), so d/rad(P_v) -> 1 for EVERY candidate and the ranking signal
  collapses at the same rate. Verified: rad(P_u) swept over 1e-9..1.1 (temperature 0.9 -> 5e8)
  leaves the logit spread pinned at 0.037 and max prob at 0.170. The loss can then only be reduced
  by genuinely improving the ranking.

The price, which is real: the triangle inequality through the origin gives
|rad_u - rad_v| <= d <= rad_u + rad_v, so the achievable logit spread is bounded by 2 / rad(P_v).
Sharpness therefore DEGRADES as the embedding expands (ceiling 9.97 at |x|=0.1, 1.82 at |x|=0.5,
0.68 at |x|=0.9). This head can only be sharp while E stays compact -- the opposite pressure to
the product form, which sought the boundary.

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
        # The radii are in a DENOMINATOR now, so at 1e-3 the head opens as a near-hard argmax:
        # gradients vanish on all but the top candidate, and d/dr_u of d/(r_u*r_v) =
        # -d/(r_u^2 * r_v) is ~1e6, which is a blow-up risk on a manifold parameter in epoch 1.
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

        score = -d_H(P_u, P_v) / (rad(P_u) * rad(P_v)). The radii are NOT detached here (unlike the
        pooler's `rad` feature): they carry gradient. rad(P_u) is constant across a row so it
        cannot move the ranking; rad(P_v) is the rank-relevant radial channel."""
        emb = self.E.weight
        p_u = self.pool(src_tokens, emb)                                        # [B, d]
        p_v = self.pool(cand_tokens, emb)                                       # [B*C, d]
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)                                                 # [B, C, d]
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)                            # [B, C] geodesic distance
        # clamp_min is a NaN guard for a centroid landing exactly on the origin (a bag whose tokens
        # cancel). At irange 1e-3 the pooled radii run ~0.02, so 1e-9 never binds in normal
        # operation -- the scale is set by irange above, not by this floor.
        r_u = self.geom.dist0(p_u).clamp_min(1e-9).unsqueeze(1)                 # [B, 1] source radius
        r_v = self.geom.dist0(p_v).clamp_min(1e-9)                              # [B, C] candidate radii
        return -(geo / (r_u * r_v))                                             # [B, C]
