"""Link head — sphere node embeddings + a deep neighbourhood-projection channel, in one module.

    logit = similarity(P[u], E[v])                                 is v near u's neighbourhood?

where P[u] = exp_u(mu_u) is the source pushed off E[u] toward mu_u. mu_u is produced by
NeighborhoodProjection: a deep, learnable pooling of the source's walk-token offsets (in the tangent
space at E[u]) together with a TPNet Time2Vec encoding of each token's age. It is candidate-
independent — mu_u depends only on u, so P[u] is computed once and scored against every candidate by
a plain inner product, exactly as before. The head owns self.E (link-trained on the sphere);
geometry goes through self.geom.
"""
import math

import geoopt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens, flatten_and_exclude_seed


class SphereManifold:
    """Unit-sphere geometry behind a manifold-agnostic contract (swap the class to swap the space):
    manifold, project, log_map, exp_map, similarity. similarity is HIGHER = closer (inner product =
    cosine on the sphere; a distance manifold would return -dist)."""
    eps = 1e-6

    def __init__(self):
        self.manifold = geoopt.Sphere()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def log_map(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        cos_angle = (base * point).sum(-1, keepdim=True).clamp(-1 + self.eps, 1 - self.eps)
        perp = point - cos_angle * base
        return torch.arccos(cos_angle) * perp / perp.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def exp_map(self, base: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        norm = tangent.norm(dim=-1, keepdim=True)
        angle = norm.clamp(max=math.pi - self.eps)
        return self.project(torch.cos(angle) * base + torch.sin(angle) / norm.clamp_min(self.eps) * tangent)

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
    """Single attention-pooling of a source's walk-token offsets into one tangent vector mu_u.

    A source-conditioned query attends over the tokens; the attention scores ARE the pooling weights,
    and mu_u is the weighted centroid of the RAW tangent offsets:

        q_u  = W_q(E[u])                                  [B, d_a]
        k_p  = W_k([ Log_{E[u]}(E[token_p]) ; Time2Vec(age_p) ])   [B, T, d_a]
        w_p  = softmax_p( (q_u . k_p) / sqrt(d_a) )       [B, T]   (padding masked out)
        mu_u = Sum_p w_p * offset_p                        [B, d_emb]

    The value is the offset itself, so mu_u stays a genuine weighted centroid (in the span of u's
    neighbour offsets) — a strict, learnable generalisation of the fixed softmax(-lambda*age)
    weighting. Candidate-independent (never sees E[v]); cold rows (no token) -> mu_u = 0.
    """

    def __init__(self, d_emb: int, d_a: int = 128, dropout: float = 0.0, t2v_dim: int = 100):
        super().__init__()
        self.d_emb = d_emb
        self.scale = 1.0 / math.sqrt(d_a)

        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        self.w_q = nn.Linear(d_emb, d_a)                       # source query
        self.w_k = nn.Linear(d_emb + t2v_dim, d_a)             # per-token key: content + time
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, source: torch.Tensor, offsets: torch.Tensor,
                ages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """source [B,d_emb] (E[u]); offsets [B,T,d_emb] (tangent at E[u]); ages [B,T]; mask [B,T]
        bool (True = real token). Returns mu_u [B,d_emb] tangent at E[u]; cold rows (no token) -> 0."""
        # TPNet scales delta-times by log(Δt + 1) before the time encoder.
        t2v = self.time_encoder(torch.log1p(ages.clamp_min(0.0)))             # [B,T,t2v_dim]
        keys = self.w_k(torch.cat([offsets, t2v], dim=-1))                    # [B,T,d_a]
        query = self.w_q(source)                                             # [B,d_a]

        scores = (query.unsqueeze(1) * keys).sum(-1) * self.scale            # [B,T]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)   # cold row -> all 0
        weights = self.attn_dropout(weights)

        return (weights.unsqueeze(-1) * offsets).sum(dim=-2)                  # [B,d_emb]


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int, t_train: float = 1.0,
                 proj_dim: int = 128, proj_dropout: float = 0.0, t2v_dim: int = 100):
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
            d_emb=d_emb, d_a=proj_dim, dropout=proj_dropout, t2v_dim=t2v_dim)

    def forward(self, src_tokens: WalkTokens, cand_ids: torch.Tensor) -> torch.Tensor:
        e_weight = self.E.weight

        source = self.geom.project(F.embedding(src_tokens.seeds, e_weight))       # E[u]  [B, d]
        candidate = self.geom.project(F.embedding(cand_ids, e_weight))            # E[v]  [B, C, d]

        # neighbourhood: P[u] vs E[v], where P[u] = exp_u(mu_u) and mu_u is the deep projection of
        # u's walk-token offsets (tangent at E[u]) + their Time2Vec ages.
        token_ids, token_mask, token_ages = flatten_and_exclude_seed(src_tokens)
        token_emb = self.geom.project(F.embedding(token_ids.clamp_min(0), e_weight))  # [B, T, d]
        token_offset = self.geom.log_map(source.unsqueeze(-2), token_emb)            # [B, T, d] tangent
        mu_u = self.neighbourhood(
            source, token_offset, token_ages.to(source.dtype), token_mask)           # [B, d] tangent
        p_u = self.geom.exp_map(source, mu_u)                                        # P[u]  [B, d] sphere
        neighbourhood_score = self.geom.similarity(p_u.unsqueeze(1), candidate)      # [B, C]

        return neighbourhood_score
