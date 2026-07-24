"""Link head — sphere node embeddings + a deep neighbourhood-projection channel, in one module.

    logit = <P[u], E[v]>                               is v near u's neighbourhood?

where P[u] = exp_{E[u]}(mu_u) pushes the source off E[u] toward mu_u — the deep pooling of u's
walk-token tangents (in the tangent space at E[u]) + a TPNet Time2Vec of each token's age, produced
by NeighborhoodProjection. One-sided: only u is walked/projected; each candidate v enters through
its static embedding E[v]. The head owns self.E (a ManifoldParameter on a geoopt.Sphere, link-
trained on it); the log/exp maps come straight from geoopt.Sphere (self.manifold).

(The dual-sided variant — walk every candidate too and score <P[u], P[v]> — was falsified on wiki:
at matched walks it lost to one-sided at ~8x the cost. It lives one `git revert` away; see the
"Important: revert this commit to bring back dual side walks" commit.)
"""
import math

import geoopt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens, flatten_tokens


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
    """Single attention-pooling of a source's walk-token tangents into one tangent vector mu_u.

    A source-conditioned query attends over the tokens; the attention scores ARE the pooling weights,
    and mu_u is the weighted centroid of the RAW token tangents:

        q_u  = W_k([ E[u] ; 0 ; 0 ; 0 ])                  [B, d_a]     (seed as a token: time/hop/edge zero-padded;
                                                                       SAME W_k as the keys, d_a = d_emb // 2)
        k_p  = W_k([ Log_{E[u]}(E[token_p]) ; Time2Vec(age_p) ; log1p(hop_p) ; edge_p ])   [B, T, d_a]
        w_p  = softmax_p( (q_u . k_p) / sqrt(d_a) )       [B, T]   (padding masked out)
        mu_u = Sum_p w_p * token_tangent_p                 [B, d_emb]

    The value is the token tangent itself, so mu_u stays a genuine weighted centroid (in the span of
    u's neighbour tangents) — a strict, learnable generalisation of the fixed softmax(-lambda*age)
    weighting. Candidate-independent (never sees E[v]); cold rows (no token) -> mu_u = 0.
    """

    def __init__(self, d_emb: int, t2v_dim: int = 16, d_ef: int = 0):
        super().__init__()
        self.d_emb = d_emb
        self.d_ef = d_ef
        d_a = d_emb // 2                     # attention (query/key) dim, derived from d_emb
        self.scale = 1.0 / math.sqrt(d_a)

        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        # ONE shared projection to d_a for BOTH query and keys (a 2-layer projection overfits on wiki —
        # depth is the regression, not width). The per-token descriptor is content + time + log-hop
        # position (1 scalar) + edge feats. The QUERY is the seed run through this SAME projection with
        # the time/hop/edge components ZERO-padded (the seed has age 0, hop 0, no edge), so query and
        # keys live in one aligned subspace. The hop enters as log1p(hop) inside the descriptor (not an
        # additive PE), so it can be learned/silenced; length-agnostic.
        k_in = d_emb + t2v_dim + 1 + d_ef
        self.w_k = nn.Linear(k_in, d_a)

    def forward(self, source: torch.Tensor, token_tangents: torch.Tensor,
                ages: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """source [B,d_emb] (E[u]); token_tangents [B,T,d_emb] (Log_{E[u]}(E[token]), tangent at
        E[u]); ages [B,T]; mask [B,T] bool (True = real token); positions [B,T] int hop-from-seed
        (1=seed, 0=pad); edge_features [B,T,d_ef] per-token edge features (empty [B,T,0] when the
        dataset has none). Returns mu_u [B,d_emb] tangent at E[u]; cold rows (no token) -> 0."""
        # TPNet scales delta-times by log(Δt + 1) before the time encoder.
        t2v = self.time_encoder(torch.log1p(ages.clamp_min(0.0)))             # [B,T,t2v_dim]
        log_hop = torch.log1p(positions.clamp_min(0).to(t2v.dtype)).unsqueeze(-1)   # [B,T,1] log-hop position
        keys = self.w_k(torch.cat([token_tangents, t2v, log_hop, edge_features], dim=-1))   # [B,T,d_a]
        # Query = the seed AS a token: content = source, but age 0 / hop 0 / no edge, so those three
        # components are zero-padded. Same w_k as the keys → query and keys align in one subspace.
        b = source.shape[0]
        query = self.w_k(torch.cat([
            source,                                       # [B, d_emb]  seed content
            source.new_zeros(b, t2v.shape[-1]),           # [B, t2v_dim]  age 0
            source.new_zeros(b, 1),                       # [B, 1]        hop 0
            source.new_zeros(b, edge_features.shape[-1]), # [B, d_ef]     no edge
        ], dim=-1))                                       # [B, d_a]

        scores = (query.unsqueeze(1) * keys).sum(-1) * self.scale            # [B,T]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)   # cold row -> all 0

        return (weights.unsqueeze(-1) * token_tangents).sum(dim=-2)          # [B,d_emb]


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int,
                 t2v_dim: int = 16, d_ef: int = 0):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.manifold = geoopt.Sphere()

        # E lives on the unit sphere: init uniformly at random on it (geoopt.Sphere.random_uniform),
        # then wrap as a ManifoldParameter so RiemannianAdam keeps it there.
        self.E = nn.Embedding(num_nodes, d_emb)
        with torch.no_grad():
            init = self.manifold.random_uniform(num_nodes, d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.manifold)

        self.neighbourhood = NeighborhoodProjection(
            d_emb=d_emb, t2v_dim=t2v_dim, d_ef=d_ef)

        # Combiner MLP over the 4 pairwise GEODESIC DISTANCES on the sphere between both sides'
        # identity (E[x]) and neighbourhood (P[x]) points. Distances are rotation-invariant, so the
        # scorer stays sphere-faithful (no raw coordinates); LOWER distance = closer.
        self.scorer = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))

    def _token_edge_features(self, tokens: WalkTokens, q: int) -> torch.Tensor:
        """Per-token edge features [Q, T, d_ef] aligned with the flattened token bag. tokens holds
        edge_features as [Q, K, L*d_ef]; reshape to [Q, K*L, d_ef]. When the dataset has no edge
        features (d_ef == 0) this is an empty [Q, T, 0] tensor (a no-op in the key concat)."""
        _, k, length = tokens.nodes.shape
        if tokens.edge_features is not None:
            return tokens.edge_features.reshape(q, k, length, self.d_ef).reshape(q, k * length, self.d_ef)
        return tokens.nodes.new_zeros((q, k * length, self.d_ef), dtype=torch.float32)

    def _project(self, tokens: WalkTokens):
        """Project one bag of N queries. Returns (e_seed, p): e_seed [N, d] = E[seed] (the identity on
        the sphere), p [N, d] = exp_{E[seed]}(mu) (seed pushed toward its walk-token centroid). Both
        on-sphere."""
        e_weight = self.E.weight
        e_seed = F.embedding(tokens.seeds, e_weight)                                  # E[x]  [N, d] (E is on-sphere)
        n = e_seed.shape[0]

        token_ids, token_mask, token_pos = flatten_tokens(
            tokens, exclude_seed_positions=True)
        token_ages = tokens.ages.reshape(n, -1).clamp_min(0)                          # ages read from the instance
        token_ef = self._token_edge_features(tokens, n)                              # [N, T, d_ef]
        token_emb = F.embedding(token_ids.clamp_min(0), e_weight)                     # [N, T, d]
        token_tangent = self.manifold.logmap(e_seed.unsqueeze(-2), token_emb)         # [N, T, d] tangent
        mu = self.neighbourhood(
            e_seed, token_tangent, token_ages.to(e_seed.dtype), token_mask, token_pos, token_ef)  # [N, d]
        return e_seed, self.manifold.expmap(e_seed, mu)                               # (E[x], P[x])  [N, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in query-major order, each walked with its query's cutoff. Score = MLP over
        the FOUR pairwise geodesic distances on the sphere between both sides' identity (seed = E[x])
        and neighbourhood (nbhd = P[x]) points (rotation-invariant, so the scorer is sphere-faithful —
        no raw coordinates; lower distance = closer). Returns logits [B, C]."""
        seed_u, nbhd_u = self._project(src_tokens)                            # E[u], P[u]  [B, d]
        seed_v, nbhd_v = self._project(cand_tokens)                           # E[v], P[v]  [B*C, d]
        b, d = seed_u.shape
        c = seed_v.shape[0] // b
        seed_v = seed_v.reshape(b, c, d)                                      # [B, C, d]
        nbhd_v = nbhd_v.reshape(b, c, d)                                      # [B, C, d]
        seed_u = seed_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]
        nbhd_u = nbhd_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]

        # Four sphere geodesic distances between u's and v's identity / neighbourhood points.
        distances = torch.stack([
            self.manifold.dist(seed_u, seed_v),                              # d(E[u], E[v])  identity affinity
            self.manifold.dist(seed_u, nbhd_v),                              # d(E[u], P[v])  is u in v's neighbourhood
            self.manifold.dist(nbhd_u, seed_v),                              # d(P[u], E[v])  is v in u's neighbourhood
            self.manifold.dist(nbhd_u, nbhd_v),                              # d(P[u], P[v])  neighbourhood overlap
        ], dim=-1)                                                            # [B, C, 4]
        return self.scorer(distances).squeeze(-1)                            # [B, C]
