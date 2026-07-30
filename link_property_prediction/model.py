"""Link head — Poincaré-ball node embeddings + a deep neighbourhood-projection channel, in one module.

    logit = <P[u], E[v]>                               is v near u's neighbourhood?

where P[u] = exp_{E[u]}(mu_u) pushes the source off E[u] toward mu_u — the deep pooling of u's
walk-token tangents (in the tangent space at E[u]) + a TPNet Time2Vec of each token's age, produced
by NeighborhoodProjection. One-sided: only u is walked/projected; each candidate v enters through
its static embedding E[v]. The head owns self.E (a ManifoldParameter on a geoopt.PoincareBall, link-
trained on it); geometry goes through self.geom (PoincareManifold): dist/logmap/expmap all proxy to
geoopt (the ball's expmap has a finite gradient at the cold-row zero tangent).

(The dual-sided variant — walk every candidate too and score <P[u], P[v]> — was falsified on wiki:
at matched walks it lost to one-sided at ~8x the cost. It lives one `git revert` away; see the
"Important: revert this commit to bring back dual side walks" commit.)
"""
from typing import Optional

import geoopt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class PoincareManifold:
    """Poincaré-ball geometry over geoopt.PoincareBall(c=1). dist / logmap / expmap all PROXY straight to
    geoopt — the ball's expmap already has a FINITE gradient at the zero tangent (the cold-row mu = 0, a
    node with no walk tokens), so no custom expmap is needed (unlike the sphere). The tangent space at any
    point is all of R^d — no orthogonal projection. E is kept in the ball by RiemannianAdam."""

    def __init__(self, c: float = 1.0):
        self.manifold = geoopt.PoincareBall(c=c)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic (hyperbolic) distance on the ball — geoopt.PoincareBall.dist. LOWER = closer."""
        return self.manifold.dist(x, y)

    def logmap(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """Log map at `base` of `point` — geoopt.PoincareBall.logmap (finite gradient at coincidence)."""
        return self.manifold.logmap(base, point)

    def expmap(self, base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        """Exp map from `base` along `tangent` — geoopt.PoincareBall.expmap. Finite gradient at u = 0, so
        the cold-row zero tangent is a safe differentiable no-op returning `base`."""
        return self.manifold.expmap(base, tangent)

    def pairwise_dist(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """All-pairs geodesic distances between rows of X [n,d] and Y [m,d] (both in the c=1 ball), as
        [n, m] — WITHOUT materialising an [n,m,d] tensor. Everything factors through the Gram matrix
        ⟨x,y⟩ = X @ Yᵀ:  d(x,y) = arccosh(1 + 2‖x-y‖²/((1-‖x‖²)(1-‖y‖²))),  ‖x-y‖² = ‖x‖²+‖y‖²-2⟨x,y⟩.
        Differentiable in E (grad flows through the matmul). Matches geoopt.PoincareBall(c=1).dist to ~1e-7."""
        xn = (X * X).sum(-1)                                              # ‖x‖²  [n]
        yn = (Y * Y).sum(-1)                                              # ‖y‖²  [m]
        sq = xn[:, None] + yn[None, :] - 2.0 * (X @ Y.t())               # ‖x-y‖²  [n,m]
        arg = 1.0 + 2.0 * sq / ((1.0 - xn)[:, None] * (1.0 - yn)[None, :])
        return torch.arccosh(arg.clamp_min(1.0 + 1e-7))                  # [n,m]


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
    """Weighted-sum pooling of a source's walk-token values into one tangent vector mu_u at E[u].

    Two nets, one weighted sum:
      VALUE  v_p = (W_v · g_p) ⊙ (1 + γ ⊙ tanh(gate([edge_p, nf_p])))
             g_p = Log_{E[u]}(E[token_p]) is the E-DEPENDENT geometric content; W_v has NO bias, so there
             is no E-independent additive term. External features [edge, nf] enter ONLY as a MULTIPLICATIVE
             per-channel gate (γ = LayerScale, init 0 -> starts ×1), so they modulate the geometry but
             cannot bypass E. gate is None when the dataset has no edge/node features (then v_p = W_v · g_p).
      WEIGHT a_p = weight_net([Time2Vec(age_p), log1p(hop_p)]) — a scalar from time/position ONLY
             (isometry-invariant, content-free), softmax-normalised over the CONTEXT tokens. Because the
             weight never sees identity it CANNOT select-and-copy a token; it only up-weights recent/close ones.
      mu_u = Σ_p softmax(a)_p · v_p.

    Every external signal (age/hop, edge/nf) enters MULTIPLICATIVELY; the tangent is the sole additive base,
    so nothing predicts without routing through E's geometry (free-lunch-resistant by construction). mu_u is
    projected onto T_{E[u]} (⊥ E[u]) and fed to exp_{E[u]} downstream -> P[u]. Candidate-independent (never
    sees E[v]); cold rows (no context token) -> mu_u = 0 -> P[u] = E[u]."""

    def __init__(self, d_emb: int, t2v_dim: int = 16, d_ef: int = 0, d_nf: int = 0):
        super().__init__()
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.d_nf = d_nf
        self.geom = PoincareManifold()                 # stateless; used for the token log-map

        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        # VALUE base: W_v · g_p (tangent). NO bias -> no E-independent additive term (keeps E load-bearing).
        self.w_v = nn.Linear(d_emb, d_emb, bias=False)
        # Channel GATE from external features [edge, nf] -> modulates the value MULTIPLICATIVELY (never adds).
        # gate_in == 0 (e.g. wiki, no edge/nf) -> no gate at all (v_p = W_v · g_p).
        gate_in = d_ef + d_nf
        self.gate = nn.Linear(gate_in, d_emb) if gate_in > 0 else None
        self.gate_scale = nn.Parameter(torch.zeros(d_emb))     # γ LayerScale, init 0 -> gate starts as ×1
        # WEIGHT net: [Time2Vec(age), log1p(hop)] -> scalar per token (softmax over context tokens).
        w_in = t2v_dim + 1
        self.weight_net = nn.Sequential(
            nn.Linear(w_in, w_in), nn.GELU(), nn.Linear(w_in, 1))

    def forward(self, walk_bag: WalkTokens, emb: torch.Tensor,
                node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """walk_bag: the flattened WalkTokens bag. emb: the FULL node-embedding table [num_nodes,d_emb]
        in the Poincaré ball. node_features: the FULL static node-feature table [num_nodes, d_nf] (None if
        the dataset has none). Returns mu_u [B,d_emb], a tangent vector at E[u]; cold rows -> 0."""
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

        # VALUE: geometric base (W_v · g_p), channel-gated by [edge, nf] (multiplicative, ≈1 at init).
        value = self.w_v(token_tangents)                                         # [B,T,d]  E-dependent base
        if self.gate is not None:
            feats = torch.cat([edge_features, nf_token], dim=-1)                  # [B,T, d_ef+d_nf]
            value = value * (1.0 + self.gate_scale * torch.tanh(self.gate(feats)))

        # WEIGHT: scalar per token from [Time2Vec(age), log-hop]; softmax over the context tokens.
        w_logit = self.weight_net(torch.cat([t2v, log_hop], dim=-1)).squeeze(-1)  # [B,T]
        w_logit = w_logit.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(w_logit, dim=-1), nan=0.0)       # [B,T]; cold row -> 0

        mu = (weights.unsqueeze(-1) * value).sum(dim=-2)                          # [B,d]  weighted sum

        # On the Poincaré ball the tangent space at E[u] is all of R^d (no ⊥-projection); mu is fed to
        # exp_{E[u]} downstream -> P[u].  (cold rows: mu = 0 -> P[u] = E[u].)
        return mu


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
        self.geom = PoincareManifold()

        # E lives in the Poincaré ball: init near the origin (small-std wrapped normal) so the conformal
        # metric — which blows up near the boundary — stays well-conditioned early; then wrap as a
        # ManifoldParameter so RiemannianAdam keeps it in the ball.
        self.E = nn.Embedding(num_nodes, d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random_normal(num_nodes, d_emb, std=1e-2)
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
        emb = self.E.weight.detach()                                                 # link head reads E DETACHED
        e_seed = F.embedding(tokens.seeds, emb)                                      # E[seed]  [N, d]
        mu = self.neighbourhood(tokens, emb, self.node_features)                     # [N, d]
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
        emb = self.E.weight.detach()                                          # link head reads E DETACHED
        seed_u = F.embedding(src_tokens.seeds, emb)                           # E[u]   [B, d]
        nbhd_u = self._project(src_tokens)                                    # P[u]   [B, d]
        nf_u = self._node_feats(src_tokens.seeds)                             # nf[u]  [B, d_nf]
        seed_v = F.embedding(cand_tokens.seeds, emb)                          # E[v]   [B*C, d]
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
