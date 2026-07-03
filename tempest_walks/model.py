"""Velocity link head — sphere node embeddings + one-sided drift scoring, in one module.

Builds a recency-weighted line through the source's walk-token trajectory in the tangent space at
E[u] and scores each candidate against it. Two channels:
    identity  = -distance_weight * ellipse( Log_u(E[v]) - neighbourhood_centroid ; drift_heading )
    velocity  = similarity( exp_u(extrapolated_offset), E[v] )
    logit     = coef_identity * identity + coef_velocity * velocity
coef_velocity inits to 0, so the head starts as the pure centroid-identity baseline and velocity
earns weight. The head owns self.E (link-trained on the sphere); geometry goes through self.geom.
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


class VelocityHead(nn.Module):
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
        self.distance_weight = nn.Parameter(torch.tensor(10.0))
        self.log_axis_along = nn.Parameter(torch.zeros(1))       # ellipse stretch along the drift heading
        self.log_axis_perp = nn.Parameter(torch.zeros(1))        # ... and perpendicular (along == perp -> isotropic)
        self.coef_identity = nn.Parameter(torch.ones(1))
        self.coef_velocity = nn.Parameter(torch.zeros(1))         # earns weight from 0

    def _centroid_and_extrapolation(self, source_emb: torch.Tensor, token_ids: torch.Tensor,
                                    token_mask: torch.Tensor, token_ages: torch.Tensor):
        """Returns (neighbourhood_centroid, extrapolated_offset): the recency centroid of the
        source's token offsets, and the recency-weighted least-squares line evaluated at the query
        time (signed_time = 0). Both [...,d]. Degenerate time -> slope 0 -> the two coincide."""
        source = self.geom.project(source_emb)
        token_emb = self.geom.project(F.embedding(token_ids.clamp_min(0), self.E.weight))
        token_offset = self.geom.log_map(source.unsqueeze(-2), token_emb)

        recency_lambda = F.softplus(self.log_recency_lambda)
        weight_logits = (-recency_lambda * token_ages).masked_fill(~token_mask, float("-inf"))
        weight = torch.nan_to_num(torch.softmax(weight_logits, dim=-1), nan=0.0)

        neighbourhood_centroid = (weight.unsqueeze(-1) * token_offset).sum(dim=-2)

        signed_time = -token_ages.to(token_offset.dtype)
        mean_signed_time = (weight * signed_time).sum(dim=-1)
        time_dev = signed_time - mean_signed_time.unsqueeze(-1)
        time_var = (weight * time_dev * time_dev).sum(dim=-1)
        time_offset_cov = (weight.unsqueeze(-1) * time_dev.unsqueeze(-1) * token_offset).sum(dim=-2)
        slope = time_offset_cov / time_var.clamp_min(self.eps).unsqueeze(-1)
        extrapolated_offset = neighbourhood_centroid - slope * mean_signed_time.unsqueeze(-1)
        return neighbourhood_centroid, extrapolated_offset

    _HEADING_GATE_KNEE = 0.05

    def _drift_heading(self, centroid: torch.Tensor) -> torch.Tensor:
        """Gated unit heading of the centroid offset. g(||c||) = ||c||^2 / (||c||^2 + knee^2) fades
        the frame to 0 (isotropic) when the drift is weak/cold and to the unit heading when strong."""
        norm = centroid.norm(dim=-1, keepdim=True)
        gate = (norm * norm) / (norm * norm + self._HEADING_GATE_KNEE ** 2)
        return gate * centroid / norm.clamp_min(self.eps)

    def _identity_score(self, source: torch.Tensor, candidate: torch.Tensor,
                        prediction_offset: torch.Tensor, heading: torch.Tensor,
                        along_scale: torch.Tensor, perp_scale: torch.Tensor,
                        distance_weight: torch.Tensor) -> torch.Tensor:
        """Anisotropic tangent distance to the centroid prediction: an ellipse oriented by `heading`,
        stretched `along_scale` along the drift direction and `perp_scale` perpendicular to it
        (along_scale == perp_scale -> plain isotropic distance)."""
        gap = self.geom.log_map(source, candidate) - prediction_offset
        along_sq = (gap * heading).sum(-1).pow(2)
        perp_sq = ((gap * gap).sum(-1) - along_sq).clamp_min(0.0)
        distance = (along_scale * along_sq + perp_scale * perp_sq).clamp_min(self.eps).sqrt()
        return -distance_weight * distance

    def forward(self, src_tokens: WalkTokens, cand_ids: torch.Tensor) -> torch.Tensor:
        e_weight = self.E.weight
        batch, num_cand = cand_ids.shape[0], cand_ids.shape[1]
        d = self.d_emb

        source = self.geom.project(F.embedding(src_tokens.seeds, e_weight))       # [B, d]
        candidate = self.geom.project(F.embedding(cand_ids, e_weight))            # [B, C, d]
        distance_weight = self.distance_weight.clamp_min(1e-3)
        along_scale = F.softplus(self.log_axis_along)
        perp_scale = F.softplus(self.log_axis_perp)

        token_ids, token_mask, token_ages = flatten_and_exclude_seed(src_tokens)
        neighbourhood_centroid, extrapolated_offset = self._centroid_and_extrapolation(
            source, token_ids, token_mask, token_ages.to(source.dtype))           # [B, d], [B, d]

        heading = self._drift_heading(neighbourhood_centroid)                    # [B, d]
        source_bc = source.unsqueeze(1).expand(batch, num_cand, d)
        centroid_bc = neighbourhood_centroid.unsqueeze(1).expand(batch, num_cand, d)
        heading_bc = heading.unsqueeze(1).expand(batch, num_cand, d)
        identity_score = self._identity_score(
            source_bc, candidate, centroid_bc, heading_bc, along_scale, perp_scale, distance_weight)

        drifted_source_point = self.geom.exp_map(source, extrapolated_offset)     # [B, d]
        velocity_score = self.geom.similarity(drifted_source_point.unsqueeze(1), candidate)

        return self.coef_identity * identity_score + self.coef_velocity * velocity_score
