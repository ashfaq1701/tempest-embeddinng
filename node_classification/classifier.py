"""The node classifier: z_geo and z_ef in, one logit out.

    geo:   BatchNorm1d(d_geo)                                   -> d_geo
    ef:    BatchNorm1d(d_ef) -> Linear(d_ef, width) -> GELU     -> width
    head:  concat -> Linear(d_geo + width, width) -> GELU -> Dropout -> Linear(width, 1)

BatchNorm, not LayerNorm, on the inputs: LayerNorm rescales each vector by its own norm and
would erase |z_geo|, the radius, which is half of what the geometry encodes. BatchNorm
standardises each feature over the batch and keeps running statistics for eval.

The edge features are projected down before mixing so their 172 dims do not swamp the 64
geometry dims in the first mixing layer. ~20k parameters at (64, 172, 64), close to
DyGLib's MLPClassifier on a 172-d input (~14.6k), so the head is not where we differ.
"""
import torch
import torch.nn as nn


class NodeClassifier(nn.Module):

    def __init__(self, d_geo: int, d_ef: int, width: int = 64, dropout: float = 0.1):
        super().__init__()
        self.geo_norm = nn.BatchNorm1d(d_geo)
        self.ef_proj = nn.Sequential(
            nn.BatchNorm1d(d_ef),
            nn.Linear(d_ef, width),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_geo + width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, z_geo: torch.Tensor, z_ef: torch.Tensor) -> torch.Tensor:
        """[B, d_geo], [B, d_ef] -> logits [B]."""
        h = torch.cat([self.geo_norm(z_geo), self.ef_proj(z_ef)], dim=-1)
        return self.head(h).squeeze(-1)
