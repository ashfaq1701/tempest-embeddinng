"""Link head — sphere node embeddings + a deep neighbourhood-projection channel, in one module.

    logit = <P[u], E[v]>                               is v near u's neighbourhood?

where P[u] = exp_{E[u]}(mu_u) pushes the source off E[u] toward mu_u — the deep pooling of u's
walk-token tangents (in the tangent space at E[u]) + a TPNet Time2Vec of each token's age, produced
by NeighborhoodProjection. One-sided: only u is walked/projected; each candidate v enters through
its static embedding E[v]. The head owns self.E (link-trained on the sphere); geometry goes through
self.geom.

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


class SphereManifold:
    """Unit-sphere geometry behind a manifold-agnostic contract (swap the class to swap the space):
    manifold, log_map, exp_map, similarity. similarity is HIGHER = closer (inner product =
    cosine on the sphere; a distance manifold would return -dist). (E is kept on-sphere by
    RiemannianAdam, so no read-time re-projection is needed.)"""
    eps = 1e-6

    def __init__(self):
        self.manifold = geoopt.Sphere()

    def log_map(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        cos_angle = (base * point).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        perp = point - cos_angle * base
        return torch.arccos(cos_angle) * perp / perp.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def exp_map(self, base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        norm = tangent.norm(dim=-1, keepdim=True)
        angle = norm.clamp(max=math.pi - self.eps)
        return torch.cos(angle) * base + torch.sin(angle) / norm.clamp_min(self.eps) * tangent

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(-1)


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

        q_u  = W_q(E[u])                                  [B, d_a]
        k_p  = W_k([ Log_{E[u]}(E[token_p]) ; Time2Vec(age_p) ])   [B, T, d_a]
        w_p  = softmax_p( (q_u . k_p) / sqrt(d_a) )       [B, T]   (padding masked out)
        mu_u = Sum_p w_p * token_tangent_p                 [B, d_emb]

    The value is the token tangent itself, so mu_u stays a genuine weighted centroid (in the span of
    u's neighbour tangents) — a strict, learnable generalisation of the fixed softmax(-lambda*age)
    weighting. Candidate-independent (never sees E[v]); cold rows (no token) -> mu_u = 0.
    """

    def __init__(self, d_emb: int, d_a: int = 128, t2v_dim: int = 16,
                 max_walk_len: int = 5):
        super().__init__()
        self.d_emb = d_emb
        self.scale = 1.0 / math.sqrt(d_a)

        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        self.w_q = nn.Linear(d_emb, d_a)                       # source query
        self.w_k = nn.Linear(d_emb + t2v_dim, d_a)             # per-token key: content + time
        # Learned hop-position embedding (pos in [0..max_walk_len]; 0 = padding, 1 = seed, 2.. =
        # predecessors). Added to the KEY (feeds the attention scores), NOT the pooled tangent values.
        self.pos_emb = nn.Embedding(max_walk_len + 1, d_a)

    def forward(self, source: torch.Tensor, token_tangents: torch.Tensor,
                ages: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """source [B,d_emb] (E[u]); token_tangents [B,T,d_emb] (Log_{E[u]}(E[token]), tangent at
        E[u]); ages [B,T]; mask [B,T] bool (True = real token); positions [B,T] int hop-from-seed
        (1=seed, 0=pad). Returns mu_u [B,d_emb] tangent at E[u]; cold rows (no token) -> 0."""
        # TPNet scales delta-times by log(Δt + 1) before the time encoder.
        t2v = self.time_encoder(torch.log1p(ages.clamp_min(0.0)))             # [B,T,t2v_dim]
        keys = self.w_k(torch.cat([token_tangents, t2v], dim=-1))            # [B,T,d_a]
        keys = keys + self.pos_emb(positions)                               # + learned hop-position PE
        query = self.w_q(source)                                             # [B,d_a]

        scores = (query.unsqueeze(1) * keys).sum(-1) * self.scale            # [B,T]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)   # cold row -> all 0

        return (weights.unsqueeze(-1) * token_tangents).sum(dim=-2)          # [B,d_emb]


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int,
                 proj_dim: int = 128, t2v_dim: int = 16,
                 max_walk_len: int = 5):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.eps = 1e-6
        self.geom = SphereManifold()

        self.E = nn.Embedding(num_nodes, d_emb)
        nn.init.normal_(self.E.weight, mean=0.0, std=1.0 / math.sqrt(d_emb))
        with torch.no_grad():
            unit_rows = self.E.weight.data / self.E.weight.data.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.E.weight = geoopt.ManifoldParameter(unit_rows, manifold=self.geom.manifold)

        self.neighbourhood = NeighborhoodProjection(
            d_emb=d_emb, d_a=proj_dim, t2v_dim=t2v_dim, max_walk_len=max_walk_len)

    def _project(self, tokens: WalkTokens):
        """Project one bag of N queries. Returns (p, e_seed): p [N, d] = exp_{E[seed]}(mu) on the
        sphere (seed pushed toward its walk-token centroid), e_seed [N, d] = E[seed] on the sphere."""
        e_weight = self.E.weight
        e_seed = F.embedding(tokens.seeds, e_weight)                                  # E[x]  [N, d] (E is on-sphere)

        token_ids, token_mask, token_ages, token_pos = flatten_tokens(
            tokens, exclude_seed_positions=True, exclude_seed_tokens=False)
        token_emb = F.embedding(token_ids.clamp_min(0), e_weight)                     # [N, T, d]
        token_tangent = self.geom.log_map(e_seed.unsqueeze(-2), token_emb)           # [N, T, d] tangent
        mu = self.neighbourhood(
            e_seed, token_tangent, token_ages.to(e_seed.dtype), token_mask, token_pos)  # [N, d] tangent
        return self.geom.exp_map(e_seed, mu), e_seed                                 # P[x], E[x]

    def forward(self, src_tokens: WalkTokens, cand_ids: torch.Tensor) -> torch.Tensor:
        """One-sided scoring. src_tokens: B source queries (seeds = u); cand_ids [B, C] candidate
        node ids. Returns logits [B, C] = <P[u], E[v]> — is v near u's neighbourhood?"""
        p_u, _ = self._project(src_tokens)                                    # P[u]  [B, d]
        candidate = F.embedding(cand_ids, self.E.weight)                      # E[v]  [B, C, d] (E on-sphere)
        return self.geom.similarity(p_u.unsqueeze(1), candidate)              # <P[u], E[v]>  [B, C]
