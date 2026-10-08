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
        return self.net(x).squeeze(-1)                                       # [B]
