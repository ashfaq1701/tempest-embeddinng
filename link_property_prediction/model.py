"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = geo_temp * (-d_H(P_u, P_v)) [+ pop_bias[v]].

The distance term is scaled by a learned geo_temp (init 1.0). When the popularity channel is on, a
learned per-node scalar pop_bias[v] (zero-init, so it contributes exactly 0 at step 0) is added at
FIXED unit weight -- the model sharpens the geometry via geo_temp while popularity rides alongside at
a constant scale.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over an
MLP of [standardised log-age | standardised position | rad] at a fixed hidden width. Ages are min-max
log1p'd and standardised against statistics pooled over both bags of the batch, so the pooler
carries no dataset-derived time constant. Learned head params: geo_temp, the MLP pooler,
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
      age_norm -- log1p(age), standardised to mean 0 / std 1 against statistics pooled over both
                  bags, by LinkPredHead.normalize_tokens. It arrives already normalised; this
                  module applies no scale of its own, so no dataset constant enters here.
      pos      -- the hop index as one scalar, z-scored across both bags by
                  LinkPredHead.normalize_tokens. Padded slots are masked out of the softmax.
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
        # ages arrive ALREADY log1p'd and standardised by LinkPredHead.normalize_tokens; padding keeps
        # its -1 sentinel and is masked out of the softmax below, so its value never matters.
        age = tokens.ages.unsqueeze(-1).to(x.dtype)                             # [Q, T, 1]
        # Hop index, ALREADY z-scored by normalize_tokens (so it may be negative -- do NOT
        # clamp). Padding keeps its 0 sentinel and is masked out below.
        pos = tokens.positions.unsqueeze(-1).to(x.dtype)                        # [Q, T, 1]
        rad = geom.dist0(x.detach()).unsqueeze(-1)                              # [Q, T, 1]
        feat = torch.cat([age, pos, rad], dim=-1).to(x.dtype)                   # [Q, T, 3]
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
    def normalize_tokens(src_tokens: WalkTokens, cand_tokens: WalkTokens):
        """Standardise both scalar token features against statistics pooled ACROSS BOTH BAGS.

          age -- log1p first, then z-score. log1p changes the SHAPE: ages span orders of
                 magnitude and are heavily right-skewed (measured on real YouTube tokens, skew
                 +1.83 raw, mean 2-3x median on every dataset in the suite). A linear rescale
                 cannot touch that -- min-max and z-score are both affine and leave relative
                 spacing identical -- so log1p is the only step here that improves resolution
                 among the bulk of tokens; it takes skew to -1.07. The z-score then sets scale.
                 Measured: dropping the z-score and keeping log1p alone costs 0.0176 max test on
                 YouTube (0.5793 -> 0.5617), so both steps earn their place.

          pos -- z-score only. The hop index is 1..max_walk_len, a small bounded integer with no
                 dynamic range to compress, so log1p would do nothing useful. Standardising it
                 puts it on the same footing as the age channel; left raw it enters at 1..5
                 against an age channel of mean 0 / std 1.

        Statistics are pooled over both bags and the whole batch. Pooling the two bags is
        necessary -- normalising them separately would put source and candidate on different
        scales, so a value would mean different things on the two sides of one comparison.
        Batch-wide is safe for mean/std because they are BULK statistics over ~1e5 non-padded
        slots; it is NOT safe for a max, which is an extreme order statistic -- measured over 8
        independent batches the batch max age was T_train, the ceiling, in 8 of 8, which is why
        an earlier min-max version silently degenerated into dividing by T_train.

        Both are mildly K-dependent, since the candidate bag is B*(1+K) rows: measured on
        YouTube, log1p(age) mean moves 10.6375 -> 10.9664 and std 6.1026 -> 6.3329 as K goes
        1 -> 20, i.e. about 0.05 of a standard deviation over a 20x change, and ~0.003 between
        K=5 and K=10. Small enough to ignore; if exact K-invariance is ever needed, take the
        statistics from the source bags alone, which are always B rows.

        Padding keeps its sentinels untouched -- ages -1 (log1p(-1) is -inf, so it must not be
        transformed) and positions 0 -- and padded slots are masked out of the softmax anyway.
        Tokens are rebuilt rather than mutated, so the caller's tensors are unchanged.
        """
        sm, cm = src_tokens.mask, cand_tokens.mask
        av = torch.cat([src_tokens.ages[sm], cand_tokens.ages[cm]])
        if av.numel() == 0:
            return src_tokens, cand_tokens
        la = torch.log1p(av.clamp_min(0).to(torch.float32))
        a_mu, a_sd = la.mean(), torch.clamp(la.std(), min=1e-3)
        pv = torch.cat([src_tokens.positions[sm], cand_tokens.positions[cm]]).to(torch.float32)
        p_mu, p_sd = pv.mean(), torch.clamp(pv.std(), min=1e-3)

        def _norm(tok, m):
            a = tok.ages.to(torch.float32)
            p = tok.positions.to(torch.float32)
            return replace(
                tok,
                ages=torch.where(m, (torch.log1p(a.clamp_min(0)) - a_mu) / a_sd, a),
                positions=torch.where(m, (p - p_mu) / p_sd, p),
            )
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
        src_tokens, cand_tokens = self.normalize_tokens(src_tokens, cand_tokens)
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
