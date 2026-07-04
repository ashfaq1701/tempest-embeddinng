"""Link head — sphere node embeddings + a single neighbourhood similarity channel, in one module.

    logit = coef_neighbourhood * similarity(P[u], E[v])            is v near u's neighbourhood?

where P[u] = exp_u(mu_u) is the source pushed off E[u] toward mu_u, the recency-weighted centroid
of u's walk-token offsets in the tangent space at E[u]. No direct identity, no extrapolation / no
slope — just the plain neighbourhood centroid scored against each candidate. The head owns self.E
(link-trained on the sphere); geometry goes through self.geom.
"""
import math

import geoopt
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


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int, t_train: float = 1.0):
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

        recency_lambda_init = 10.0 / max(float(t_train), 1.0)      # lambda * age ~ O(1)
        self.log_recency_lambda = nn.Parameter(
            torch.tensor([math.log(math.expm1(recency_lambda_init))], dtype=torch.float32))
        self.coef_neighbourhood = nn.Parameter(torch.ones(1))    # learnable logit scale

    def _neighbourhood_centroid(self, source: torch.Tensor, token_ids: torch.Tensor,
                                token_mask: torch.Tensor, token_ages: torch.Tensor) -> torch.Tensor:
        """mu_u = Sum_p softmax_p(-lambda*age_p) * Log_source(E[token_p]) — the recency-weighted
        centroid of the source's token offsets, in the tangent space at the source. [...,d].
        Cold (all masked) -> 0."""
        token_emb = self.geom.project(F.embedding(token_ids.clamp_min(0), self.E.weight))
        token_offset = self.geom.log_map(source.unsqueeze(-2), token_emb)
        recency_lambda = F.softplus(self.log_recency_lambda)
        weight_logits = (-recency_lambda * token_ages).masked_fill(~token_mask, float("-inf"))
        weight = torch.nan_to_num(torch.softmax(weight_logits, dim=-1), nan=0.0)
        return (weight.unsqueeze(-1) * token_offset).sum(dim=-2)

    def forward(self, src_tokens: WalkTokens, cand_ids: torch.Tensor) -> torch.Tensor:
        e_weight = self.E.weight

        source = self.geom.project(F.embedding(src_tokens.seeds, e_weight))       # E[u]  [B, d]
        candidate = self.geom.project(F.embedding(cand_ids, e_weight))            # E[v]  [B, C, d]

        # neighbourhood: P[u] vs E[v], where P[u] = exp_u(mu_u) is the recency centroid on the sphere
        token_ids, token_mask, token_ages = flatten_and_exclude_seed(src_tokens)
        mu_u = self._neighbourhood_centroid(
            source, token_ids, token_mask, token_ages.to(source.dtype))           # [B, d] tangent
        p_u = self.geom.exp_map(source, mu_u)                                     # P[u]  [B, d] sphere
        neighbourhood_score = self.geom.similarity(p_u.unsqueeze(1), candidate)   # [B, C]

        return self.coef_neighbourhood * neighbourhood_score
