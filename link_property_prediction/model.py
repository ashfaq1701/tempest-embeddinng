"""Centroid-to-centroid head on the Poincaré ball -- the HYPERBOLIC TRIANGLE RATIO:

    s(u,v) = ( rad(P_v) - d_H(P_u, P_v) ) / rad(P_u)

161 head params: the pooler MLP alone. No learned scalar of any kind.

WHAT IT IS. The numerator is the Gromov product with the source term dropped: the exact product is
(u|v)_o = (rad_u + rad_v - d)/2, and rad_u is constant within a query, so under the per-query
softmax CE it adds the same constant to every candidate and cancels from the loss AND the gradient
(sum_c dL/ds_c = 0 for softmax CE). What remains, rad_v - d, ranks by candidate DEPTH minus
distance: a deep candidate is allowed to be proportionally farther away.

The denominator then reintroduces rad_u as a PER-QUERY TEMPERATURE. It cannot reorder candidates
-- it is a per-row positive divisor -- but softmax is only shift-invariant, not scale-invariant, so
it modulates the entropy of each query's distribution by the source's depth.

MEASURED PROPERTIES, so this run is read against what was actually checked rather than the claim.

  Scale-response: NOT exactly degree-0. Rescaling all of E by k gives spread 0.4547 / 0.4383 /
  0.3720 / 0.1648 at k = 0.25 / 0.5 / 1 / 2 -- a 2.8x fall as the embedding expands. Degree-0
  holds only in the Euclidean limit; at real radii d grows faster than linearly, so there is a
  mild CONTRACTION incentive. Weaker than -d/(rad_u*rad_v), but present.

  The depth modulation is REAL BUT SMALL. Over 100 candidates with the source moved from
  |p_u| = 0.05 to 0.95 (rad_u 0.10 -> 3.66) the entropy runs 4.5958 -> 4.6047 against a uniform
  4.6052: directionally as claimed (deeper source -> flatter ranking) but a 0.009-nat effect.
  What rad_u genuinely does is INVERT the spread ordering across source depths -- without the
  division spread runs 0.056 -> 0.428 -> 0.485 as the source deepens, with it 0.552 -> 0.532 ->
  0.130. That is a real change to the ranking function, not a cosmetic one.

  It is NOT the free-lunch channel a per-row temperature usually is. The model cannot move rad_u
  independently of the numerator: as P_u -> origin, d -> rad_v so (rad_v - d) -> 0 at rate
  rad_u*cos(theta), and the ratio tends to cos(theta). Measured, rad_u swept 6.2e-1 down to
  2.0e-7: spread stays 0.4644 / 0.4527 / 0.4514 and maxprob 0.1442 / 0.1408 / 0.1408. Self-
  limiting, unlike -rad_u*rad_v*d where the same knob ran away (maxprob 0.167 -> 0.982).

THE RISK, stated before launch. Absolute spread is only ~0.13-0.55 across 100 candidates, i.e. a
near-uniform softmax. Both previous un-tempered heads sat in exactly that regime and underfit:
-d/(rad_u+rad_v) scored 0.2969 and -d/rad_v 0.2529, each stalling by ep6 with link parked ~0.3
below chance and nothing focusing gradient on unsolved queries. A global temperature
(s = T * (rad_v - d)/rad_u) would address that and is deliberately NOT included here -- this arm
runs the formula as stated, so the temperature becomes a clean one-variable follow-up.

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

        score = (rad(P_v) - d_H(P_u,P_v)) / rad(P_u). Neither radius is detached here (unlike the
        pooler's `rad` feature): both carry gradient. rad(P_v) is rank-relevant; rad(P_u) is a
        per-row divisor that sets the query's softmax temperature but cannot reorder."""
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
        # clamp_min guards a source centroid sitting exactly at the origin (rad_u = 0), the only
        # divide-by-zero here. Pooled radii run ~0.02 at irange 1e-3, so 1e-9 never binds in
        # normal operation.
        return (r_v - geo) / r_u.clamp_min(1e-9)                                # [B, C]
