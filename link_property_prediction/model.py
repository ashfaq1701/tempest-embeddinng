"""Link head — sphere node embeddings + a deep neighbourhood-projection channel, in one module.

    logit = <P[u], E[v]>                               is v near u's neighbourhood?

where P[u] = exp_{E[u]}(mu_u) pushes the source off E[u] toward mu_u — the deep pooling of u's
walk-token tangents (in the tangent space at E[u]) + a TPNet Time2Vec of each token's age, produced
by NeighborhoodProjection. One-sided: only u is walked/projected; each candidate v enters through
its static embedding E[v]. The head owns self.E (a ManifoldParameter on a geoopt.Sphere, link-
trained on it); geometry goes through self.geom (SphereManifold): dist/logmap proxy to geoopt, expmap
is ours (geoopt's expmap NaNs the gradient at the cold-row zero tangent).

(The dual-sided variant — walk every candidate too and score <P[u], P[v]> — was falsified on wiki:
at matched walks it lost to one-sided at ~8x the cost. It lives one `git revert` away; see the
"Important: revert this commit to bring back dual side walks" commit.)
"""
import math
from typing import Optional

import geoopt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class SphereManifold:
    """Unit-sphere geometry over geoopt.Sphere. dist and logmap PROXY straight to geoopt; expmap is
    OURS — geoopt's Sphere.expmap uses sin(‖u‖)/‖u‖ under a torch.where and leaks a NaN gradient at
    ‖u‖ = 0 (the cold-row tangent mu = 0, a node with no walk tokens — common on wiki). E is kept
    on-sphere by RiemannianAdam, so no read-time re-projection is needed."""

    def __init__(self):
        self.manifold = geoopt.Sphere()

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance arccos(⟨x, y⟩) on the sphere — geoopt.Sphere.dist. LOWER = closer."""
        return self.manifold.dist(x, y)

    def logmap(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """Log map at `base` of `point` — geoopt.Sphere.logmap (its gradient is finite at coincidence)."""
        return self.manifold.logmap(base, point)

    def expmap(self, base: torch.Tensor, tangent: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Exponential map: move from `base` along `tangent`.  exp(base, u) = cos(‖u‖)·base +
        sinc(‖u‖/π)·u.  torch.sinc is a smooth primitive (no explicit /‖u‖), so ‖u‖ = 0 is an exact
        differentiable no-op — returns `base` with a FINITE gradient. (geoopt's expmap NaNs the
        gradient there.) The squared norm is clamped off 0 before the sqrt for extra safety."""
        angle = (tangent ** 2).sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()   # ‖u‖, finite grad at 0
        return torch.cos(angle) * base + torch.sinc(angle / math.pi) * tangent


class TimeEncoder(nn.Module):
    """Time2Vec time encoding, ported verbatim from TPNet (TGB_TPNet/models/modules.py)."""

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class NeighborhoodProjection(nn.Module):
    """MEAN-pooling of a source's walk-token descriptors into one tangent vector mu_u at E[u].

    No attention, no displacement. Each CONTEXT token (the walk, EXCLUDING the seed) is described by

        [ Log_{E[u]}(E[token]) ‖ node_feat ‖ Time2Vec(age) ‖ log1p(hop) ‖ edge_feat ]

    and mapped by a single linear layer w_token -> R^d; mu_u is the UNIFORM MEAN of those projections
    over the context tokens. mu_u is then projected onto the tangent space at E[u] (mu -= ⟨mu,E[u]⟩·E[u])
    and fed to exp_{E[u]} downstream to give P[u]. Candidate-independent (never sees E[v]); cold rows
    (no context token) -> mu_u = 0 -> P[u] = E[u]."""

    def __init__(self, d_emb: int, t2v_dim: int = 16, d_ef: int = 0, d_nf: int = 0):
        super().__init__()
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.d_nf = d_nf
        self.geom = SphereManifold()                   # stateless; used for the token log-map

        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        # Per-token descriptor -> d_emb: [token_tangent, node_feat, Time2Vec(age), log1p(hop), edge_feat].
        desc_in = d_emb + d_nf + t2v_dim + 1 + d_ef
        self.w_token = nn.Linear(desc_in, d_emb)

    def forward(self, walk_bag: WalkTokens, emb: torch.Tensor,
                node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """walk_bag: the flattened WalkTokens bag. emb: the FULL node-embedding table [num_nodes,d_emb]
        on the sphere. node_features: the FULL static node-feature table [num_nodes, d_nf] (None if the
        dataset has none). Returns mu_u [B,d_emb], projected onto T_{E[u]} (⊥ E[u]); cold rows -> 0."""
        node_ids = walk_bag.nodes.clamp_min(0)                                    # [B,T]  (padding → row 0)
        source = F.embedding(walk_bag.seeds, emb)                                 # E[u]  [B,d_emb]
        token_tangents = self.geom.logmap(source.unsqueeze(-2), F.embedding(node_ids, emb))  # [B,T,d_emb] ⊥ E[u]
        b, t, _ = token_tangents.shape

        mask = walk_bag.mask & ~walk_bag.seed_mask                               # [B,T]  context (non-seed real)
        ages = walk_bag.ages.clamp_min(0).to(token_tangents.dtype)               # [B,T]

        # Static node features per token (padding zeroed); empty channel if the dataset has none.
        if node_features is None:
            nf_token = token_tangents.new_zeros(b, t, self.d_nf)                  # [B,T,0]
        else:
            nf_token = F.embedding(node_ids, node_features) * walk_bag.mask.unsqueeze(-1)   # [B,T,d_nf]

        # Edge features (seed/padding already zeroed on the bag); empty channel if absent.
        edge_features = walk_bag.edge_features                                    # [B,T,d_ef] or None
        if edge_features is None:
            edge_features = token_tangents.new_zeros(b, t, self.d_ef)             # [B,T,0]

        # TPNet scales delta-times by log(Δt + 1) before the time encoder.
        t2v = self.time_encoder(torch.log1p(ages))                               # [B,T,t2v_dim]
        log_hop = torch.log1p(walk_bag.positions.clamp_min(0).to(t2v.dtype)).unsqueeze(-1)  # [B,T,1]

        # Per-token descriptor -> w_token -> R^d; UNIFORM MEAN over the context tokens.
        desc = torch.cat([token_tangents, nf_token, t2v, log_hop, edge_features], dim=-1)   # [B,T,desc_in]
        proj = self.w_token(desc)                                                # [B,T,d_emb]
        is_context = mask.unsqueeze(-1).to(proj.dtype)                           # [B,T,1]  1.0 at context tokens
        n_context = is_context.sum(dim=-2).clamp_min(1.0)                        # [B,1]    #context tokens (cold->1)
        mu = (proj * is_context).sum(dim=-2) / n_context                        # [B,d_emb]  mean; cold row -> 0

        # Project onto the tangent space at E[u] (⊥ E[u]); exp_{E[u]} applied downstream -> P[u].
        return mu - (mu * source).sum(-1, keepdim=True) * source                 # Π_{E[u]}: ⊥ E[u]


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int,
                 t2v_dim: int = 16, d_ef: int = 0,
                 node_features: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.d_nf = 0 if node_features is None else int(node_features.shape[1])
        # Static per-node feature table [num_nodes, d_nf] (dataset-derived, NOT learned). A buffer so
        # it rides model.to(device) and stays out of the optimizer; non-persistent (derivable from the
        # dataset, so kept out of checkpoints). None when the dataset has no node features.
        self.register_buffer("node_features", node_features, persistent=False)
        self.geom = SphereManifold()

        # E lives on the unit sphere: init uniformly at random on it (geoopt.Sphere.random_uniform),
        # then wrap as a ManifoldParameter so RiemannianAdam keeps it there.
        self.E = nn.Embedding(num_nodes, d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random_uniform(num_nodes, d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        self.neighbourhood = NeighborhoodProjection(
            d_emb=d_emb, t2v_dim=t2v_dim, d_ef=d_ef, d_nf=self.d_nf)

        # Scorer: MLP over the 4 pairwise GEODESIC DISTANCES between {E[x], P[x]} of u and v, plus the two
        # nodes' static node features nf[u], nf[v]. The distances are isometry-invariant (no raw coords);
        # nf[·] are external, frame-free channels. LOWER distance = closer (the MLP learns the sign).
        score_in = 4 + 2 * self.d_nf                             # 4 distances + nf[u] + nf[v]
        self.scorer = nn.Sequential(
            nn.Linear(score_in, 32), nn.GELU(), nn.Linear(32, 1))

    def _project(self, tokens: WalkTokens) -> torch.Tensor:
        """Neighbourhood projection P[x] = exp_{E[seed]}(mu) [N, d] for a bag of N queries. The
        neighbourhood takes the bag + the full E table and derives E[u] / token tangents / stable
        feats / masks itself; E[seed] is looked up separately by the caller."""
        e_seed = F.embedding(tokens.seeds, self.E.weight)                            # E[seed]  [N, d]
        mu = self.neighbourhood(tokens, self.E.weight, self.node_features)           # [N, d]
        return self.geom.expmap(e_seed, mu)                                          # P[x]  [N, d]

    def _node_feats(self, seeds: torch.Tensor) -> torch.Tensor:
        """Static node features [N, d_nf] for the given seed ids; a zero-width [N, 0] tensor when the
        dataset has no node features (so it concatenates as a no-op — no branching at the call site)."""
        if self.node_features is None:
            return self.E.weight.new_zeros(seeds.shape[0], 0)
        return F.embedding(seeds, self.node_features)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in query-major order, each walked with its query's cutoff. Logit = MLP over
        [ d(E_u,E_v), d(E_u,P_v), d(P_u,E_v), d(P_u,P_v), nf[u], nf[v] ] per (u, candidate v). Returns
        logits [B, C]."""
        seed_u = F.embedding(src_tokens.seeds, self.E.weight)                 # E[u]   [B, d]
        nbhd_u = self._project(src_tokens)                                    # P[u]   [B, d]
        nf_u = self._node_feats(src_tokens.seeds)                             # nf[u]  [B, d_nf]
        seed_v = F.embedding(cand_tokens.seeds, self.E.weight)                # E[v]   [B*C, d]
        nbhd_v = self._project(cand_tokens)                                   # P[v]   [B*C, d]
        nf_v = self._node_feats(cand_tokens.seeds)                            # nf[v]  [B*C, d_nf]

        b, d = seed_u.shape
        c = seed_v.shape[0] // b
        # candidate (v) side -> [B, C, ·]; source (u) side broadcast over the C candidates -> [B, C, ·].
        seed_v = seed_v.reshape(b, c, d)                                      # [B, C, d]
        nbhd_v = nbhd_v.reshape(b, c, d)                                      # [B, C, d]
        nf_v = nf_v.reshape(b, c, self.d_nf)                                  # [B, C, d_nf]
        seed_u = seed_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]
        nbhd_u = nbhd_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]
        nf_u = nf_u.unsqueeze(1).expand(b, c, self.d_nf)                      # [B, C, d_nf]

        # Four pairwise geodesic distances (self.geom.dist) between u's and v's identity / nbhd points.
        distances = torch.stack([
            self.geom.dist(seed_u, seed_v),                                  # d(E[u], E[v])  identity affinity
            self.geom.dist(seed_u, nbhd_v),                                  # d(E[u], P[v])  is u in v's nbhd
            self.geom.dist(nbhd_u, seed_v),                                  # d(P[u], E[v])  is v in u's nbhd
            self.geom.dist(nbhd_u, nbhd_v),                                  # d(P[u], P[v])  nbhd overlap
        ], dim=-1)                                                            # [B, C, 4]
        feats = torch.cat([distances, nf_u, nf_v], dim=-1)                    # [B, C, 4 + 2*d_nf]
        return self.scorer(feats).squeeze(-1)                                 # [B, C]
