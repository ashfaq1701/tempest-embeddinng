"""The node classifier: 22 standardised scalars in (17 walk-derived + 5 history), one logit out.

    BatchNorm1d(n) -> Linear(n, n) -> GELU -> Linear(n, 32) -> GELU      input stem
    -> Linear(32, 32) -> GELU -> Dropout(0.1) -> Linear(32, 1)            head

BatchNorm standardises each feature over the batch (radii, distances, log-times and counts are
on unrelated scales) and keeps running statistics for eval. Small on purpose: Wikipedia has 156
positive training interactions, and every wider or deeper variant tried tied or overfit.
"""
import torch
import torch.nn as nn


class NodeClassifier(nn.Module):

    def __init__(self, n_in: int, width: int = 32, dropout: float = 0.1):
        super().__init__()
        stem = max(width // 2, n_in)
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_in),
            nn.Linear(n_in, stem),
            nn.GELU(),
            nn.Linear(stem, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_in] -> logits [B]."""
        return self.net(x).squeeze(-1)
