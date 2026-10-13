import torch
import torch.nn as nn


class MaskedBatchNorm(nn.Module):
    """BatchNorm1d over every axis but the last, counting only `valid` entries. Training
    normalises by the batch statistics and updates running averages with `momentum`; eval
    normalises by the running averages. Masked entries are excluded and returned as zero."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        m = valid.unsqueeze(-1).to(feat.dtype)                               # [..., 1]
        if self.training:
            dims = tuple(range(feat.dim() - 1))
            n = m.sum(dim=dims)                                              # [F]
            mean = (feat * m).sum(dim=dims) / n.clamp_min(1.0)               # [F]
            var = (((feat - mean) ** 2) * m).sum(dim=dims) / n.clamp_min(1.0)
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / (n - 1).clamp_min(1.0), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        out = (feat - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias
        return out * m                                                       # [..., F]
