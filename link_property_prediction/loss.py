"""Training objective. Each term is a named scalar; the trainer sums them 1:1 and logs each one."""
import torch
import torch.nn.functional as F


def link_loss(logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over [B, 1+K] candidate logits, the positive in column 0."""
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)

