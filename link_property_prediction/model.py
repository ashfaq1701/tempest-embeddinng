"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = geo_temp * (-d_H(P_u, P_v)) [+ pop_bias[v]].

The distance term is scaled by a learned geo_temp (init 1.0). When the popularity channel is on, a
learned per-node scalar pop_bias[v] (zero-init, so it contributes exactly 0 at step 0) is added at
FIXED unit weight -- the model sharpens the geometry via geo_temp while popularity rides alongside at
a constant scale.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over an
MLP of [normalised age | one-hot position | rad] at a fixed hidden width. Ages are min-max
normalised per query PAIR against a min/max shared by the source and candidate bags, so the
pooler carries no dataset-derived time constant. Learned head params: geo_temp, the MLP pooler,
and num_nodes popularity scalars when on."""


import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import replace

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
      age_norm -- the token age min-max normalised to [0, 1] across BOTH bags of the query pair
                  by LinkPredHead.normalize_ages. It arrives already normalised; this module
                  applies no scale of its own, so there is no dataset constant here at all.
      pos      -- ONE-HOT over the hop index 1..max_walk_len. Padding (index 0) is dropped, so a
                  padded slot is the all-zero vector rather than a category of its own.
      rad      -- geodesic distance from the origin: the token's hyperbolic radius

    Position is one-hot rather than a learned embedding or a sinusoid: there are only
    max_walk_len distinct values (5 in the default config), so one-hot is exact and carries no
    parameters -- the first Linear learns whatever per-hop weight it wants directly.

    `hidden` is the MLP width, pinned independently of the feature count: under an earlier rule
    where it scaled with the feature count, every feature change silently moved pooler capacity
    too and the two effects could not be separated (see the ablation in CLAUDE.md).
    """

    def __init__(self, max_walk_len: int, hidden_dim: int = 32):
        super().__init__()
        self.max_walk_len = int(max_walk_len)
        self.hidden = int(hidden_dim)
        # 1 normalised age + max_walk_len one-hot position slots + 1 radius.
        self.n_feat = 1 + self.max_walk_len + 1
        self.net = nn.Sequential(nn.Linear(self.n_feat, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, 1))

    def forward(self, geom: "PoincareManifold", tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        # ages arrive ALREADY normalised to [0,1] by LinkPredHead.normalize_ages; padding keeps
        # its -1 sentinel and is masked out of the softmax below, so its value never matters.
        age = tokens.ages.unsqueeze(-1).to(x.dtype)                             # [Q, T, 1]
        # One-hot over hop index 1..max_walk_len. Column 0 (padding) is dropped, so a padded slot
        # is the all-zero vector rather than a category of its own.
        pos = tokens.positions.clamp(0, self.max_walk_len).long()               # [Q, T]
        p_oh = F.one_hot(pos, self.max_walk_len + 1)[..., 1:].to(x.dtype)       # [Q, T, L]
        rad = geom.dist0(x.detach()).unsqueeze(-1)                              # [Q, T, 1]
        feat = torch.cat([age, p_oh, rad], dim=-1).to(x.dtype)                  # [Q, T, n_feat]
        logits = self.net(feat).squeeze(-1)
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; the pooling weights carry no learned parameters."""

    def __init__(self, num_nodes: int, d_emb: int, max_walk_len: int,
                 init_irange: float = 1e-3, use_pop_bias: bool = False,
                 hidden_dim: int = 32):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.use_pop_bias = bool(use_pop_bias)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(max_walk_len, hidden_dim)

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

    @staticmethod
    def normalize_ages(src_tokens: WalkTokens, cand_tokens: WalkTokens):
        """Min-max normalise ages to [0, 1] using a min/max SHARED by both bags.

        Only non-padded slots take part. Padding keeps its -1 sentinel, so the token invariants
        the rest of the module relies on -- padding is -1, mask marks real slots -- survive.

        The min and max are pooled across BOTH bags on purpose: normalising each bag separately
        would put the source and the candidate on different scales, and an age of 0.5 would mean
        different things on the two sides of the same comparison.

        Note the seed slots are non-padded and structurally age 0, so in practice the minimum is
        0 and this reduces to dividing by the largest age in the pair. The subtraction is kept
        because it costs nothing and stops that from being a silent assumption.
        """
        sm, cm = src_tokens.mask, cand_tokens.mask
        vals = torch.cat([src_tokens.ages[sm], cand_tokens.ages[cm]])
        if vals.numel() == 0:
            return src_tokens, cand_tokens
        lo = vals.min().to(torch.float32)
        hi = vals.max().to(torch.float32)
        den = torch.clamp(hi - lo, min=1.0)          # degenerate all-equal case -> all zeros
        def _norm(tok, m):
            a = tok.ages.to(torch.float32)
            return replace(tok, ages=torch.where(m, (a - lo) / den, a))
        return _norm(src_tokens, sm), _norm(cand_tokens, cm)

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
        src_tokens, cand_tokens = self.normalize_ages(src_tokens, cand_tokens)
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
