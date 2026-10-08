"""The node classifier: two input branches brought to the same width, then mixed.

    geo branch:  BatchNorm1d(2)    -> Linear(2, 16)    -> GELU -> Linear(16, 32) -> GELU   widen
    ef  branch:  BatchNorm1d(d_ef) -> Linear(d_ef, 96) -> GELU -> Linear(96, 32) -> GELU   narrow
    head:        concat [64] -> Linear(64, 32) -> GELU -> Dropout -> Linear(32, 1)

geo is [d0(p_u), sum_i w_i d(x_i, p_u)]; ef is the mean walk edge feature (172-d LIWC on
Wikipedia/Reddit). Each branch gets its own stack and reaches 32 before they meet, so neither
dominates the first mixing layer by sheer width: the 2 geometric numbers are widened, the edge
features narrowed. BatchNorm standardises each input feature over the batch (both are on
arbitrary, unrelated scales) and keeps running statistics for eval.
"""
import torch
import torch.nn as nn


class NodeClassifier(nn.Module):

    def __init__(self, n_geo: int, d_ef: int, width: int = 32, dropout: float = 0.1):
        super().__init__()
        self.geo = nn.Sequential(
            nn.BatchNorm1d(n_geo),
            nn.Linear(n_geo, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, width),
            nn.GELU(),
        )
        self.ef = nn.Sequential(
            nn.BatchNorm1d(d_ef),
            nn.Linear(d_ef, 3 * width),
            nn.GELU(),
            nn.Linear(3 * width, width),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, geo: torch.Tensor, ef: torch.Tensor) -> torch.Tensor:
        """[B, n_geo], [B, d_ef] -> logits [B]."""
        return self.head(torch.cat([self.geo(geo), self.ef(ef)], dim=-1)).squeeze(-1)
