"""Centroid-to-centroid head on the Poincaré ball: s(u,v) = geo_temp * (-d_H(P_u, P_v)) [+ pop_bias[v]].

The distance term is scaled by a learned geo_temp (init 1.0). When the popularity channel is on, a
learned per-node scalar pop_bias[v] (zero-init, so it contributes exactly 0 at step 0) is added at
FIXED unit weight -- the model sharpens the geometry via geo_temp while popularity rides alongside at
a constant scale.

P_x is the weighted gyro-midpoint of x's walk-token bag; the pooling weights are a softmax over
pooling_temp * (-age/mnia - (pos-1)), both priors RAW (no log1p). Learned head params: geo_temp,
the pooling temperature, and num_nodes popularity scalars when the channel is on."""

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

    def midpoint(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Weighted gyro-midpoint: x [Q,T,d], w [Q,T] -> [Q,d]."""
        return self.manifold.weighted_midpoint(x, weights=w, reducedim=[-2], dim=-1, keepdim=False)


class BagWeights(nn.Module):
    """Pooling weights = softmax(prior + MLP_residual([-(age/mnia), -pos, spr])), where the fixed
    PRIOR = pos + spr is a hand-crafted logit and the MLP learns a RESIDUAL correction on top of it.

    The MLP output layer is ZERO-INITIALISED, so at step 0 the residual is exactly 0 and the pooler
    starts as the pure prior (recent-position + high-spread tokens up-weighted); it then learns to
    deviate. hidden = 8*N_FEAT. This anchors the free MLP -- which alone under-performed on YouTube --
    to a known-good starting point instead of a random init.

    N_FEAT = 3: the cosine feature `ang` is OUT, by request."""

    N_FEAT = 3

    def __init__(self, mnia: float):
        super().__init__()
        self.mnia = float(mnia)                 # fixed age scale
        hidden = 8 * self.N_FEAT                # 8x expansion
        self.net = nn.Sequential(nn.Linear(self.N_FEAT, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # Zero-init the residual head: training starts at the pure prior (residual == 0).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def spread(geom: "PoincareManifold", x: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
        """Token spread relative to the bag's unweighted midpoint C, per-bag level removed:
        spr = d(x_p, C) / mean_q d(x_q, C)  ->  [Q, T]."""
        vf = valid.to(x.dtype)
        n = vf.sum(dim=-1, keepdim=True).clamp_min(1.0)
        c = geom.midpoint(x, vf / n).unsqueeze(-2)                              # [Q, 1, d]
        dc = geom.dist(c, x) * vf
        return dc / (dc.sum(dim=-1, keepdim=True) / n).clamp_min(1e-6)

    def forward(self, geom: "PoincareManifold", tokens: WalkTokens, x: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """x [Q,T,d], valid [Q,T] -> w [Q,T] summing to 1, 0 on padding."""
        rec = -(tokens.ages.clamp_min(0).float() / self.mnia)                   # -(age/mnia)
        pos = -(tokens.positions.clamp_min(1).float() - 1.0)                    # -pos
        spr = self.spread(geom, x.detach(), valid)
        prior = (pos + spr).to(x.dtype)                                         # [Q, T] fixed prior logit
        feat = torch.stack([rec, pos, spr], dim=-1).to(x.dtype)                 # [Q, T, 3]
        residual = self.net(feat).squeeze(-1)                                   # [Q, T] learned correction
        logits = prior + residual                                              # prior anchored, MLP corrects
        return torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)


class LinkPredHead(nn.Module):
    """E is a ManifoldParameter; BagWeights is the only other trained module."""

    def __init__(self, num_nodes: int, d_emb: int, mean_node_inter_arrival: float,
                 init_irange: float = 1e-3, use_pop_bias: bool = False):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_emb = int(d_emb)
        self.use_pop_bias = bool(use_pop_bias)
        self.geom = PoincareManifold()
        self.bag_weights = BagWeights(mean_node_inter_arrival)

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

    def pool(self, tokens: WalkTokens, emb: torch.Tensor) -> torch.Tensor:
        """Bag -> P [Q,d], the pooling-weighted gyro-midpoint of the walk-token cloud."""
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)                                               # all-padding walk -> use seed
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True

        x = F.embedding(nodes, emb)                                             # [Q, T, d]
        w = self.bag_weights(self.geom, tokens, x, valid)                                  # [Q, T] sums to 1, 0 on padding
        return self.geom.midpoint(x, w)                                         # [Q, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src = B source queries; cand = B*C candidate queries, query-major. -> [B, C].
        score = geo_temp * (-geo) [+ pop_bias[v]]  (distance scaled by geo_temp; the learned per-node
        popularity scalar, when on, added at fixed unit weight)."""
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
