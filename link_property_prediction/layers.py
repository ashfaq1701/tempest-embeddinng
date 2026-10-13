import torch
import torch.nn as nn


class MaskedBatchNorm(nn.Module):
    """`nn.BatchNorm1d(track_running_stats=False)` over every axis but the last, counting only
    `valid` entries. Both modes normalise by the batch statistics, so the module has no
    train/eval distinction and holds no state beyond its affine. Masked entries are excluded
    from the statistics and returned as zero."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        m = valid.unsqueeze(-1).to(feat.dtype)                               # [..., 1]
        dims = tuple(range(feat.dim() - 1))
        n = m.sum(dim=dims).clamp_min(1.0)                                   # [F]
        mean = (feat * m).sum(dim=dims) / n                                  # [F]
        var = (((feat - mean) ** 2) * m).sum(dim=dims) / n                   # [F]
        out = (feat - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias
        return out * m                                                       # [..., F]
