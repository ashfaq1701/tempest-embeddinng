"""The node classifier: two geometric features of u's walk bag in, one logit out.

    features:  [d0(p_u), sum_i w_i d(x_i, p_u)]   radius of the pooled point, weighted bag spread
    net:       BatchNorm1d(2) -> Linear(2, 32) -> GELU -> Linear(32, 32) -> GELU -> Dropout
               -> Linear(32, 1)

BatchNorm standardises each feature over the batch and keeps running statistics for eval;
both features are distances, on very different scales across datasets and checkpoints.
Two hidden layers so the decision boundary over the (radius, spread) plane can bend.
"""
import torch
import torch.nn as nn


class NodeClassifier(nn.Module):

    def __init__(self, n_feat: int = 2, width: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_feat),
            nn.Linear(n_feat, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """[B, n_feat] -> logits [B]."""
        return self.net(feats).squeeze(-1)
