"""Training objective. Each term is a named scalar; the trainer sums them 1:1 and logs each one."""
import torch
import torch.nn.functional as F


def link_loss(logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over [B, 1+K] candidate logits, the positive in column 0."""
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def radial_popularity_loss(radii: torch.Tensor, popularity: torch.Tensor) -> torch.Tensor:
    """Pull each node's radius toward the radius its popularity rank earns within the batch.

    radii [N] (with grad) and popularity [N] are one batch's candidates. The target is the batch's
    own radii rearranged so radius falls as popularity rises; it is detached, so the gradient
    moves only the radii, and the term cannot inflate the table's mean radius."""
    target = popularity_ordered_radii(radii.detach(), popularity)
    return ((radii - target) ** 2).mean()


def popularity_ordered_radii(radii: torch.Tensor, popularity: torch.Tensor) -> torch.Tensor:
    """The same radii, reassigned by popularity: the most popular node gets the smallest radius,
    the least popular the largest. Nodes of equal popularity share the mean of their slots."""
    least_popular_first = torch.argsort(popularity, stable=True)
    largest_first = torch.sort(radii, descending=True).values

    reassigned = torch.empty_like(radii)
    reassigned[least_popular_first] = largest_first
    return _mean_within_ties(reassigned, popularity)


def rank_agreement(radii: torch.Tensor, popularity: torch.Tensor) -> float:
    """Spearman correlation between -radius and popularity: +1 when radius falls exactly as
    popularity rises. A diagnostic for the epoch line, never optimised."""
    radius_rank = _average_ranks(-radii)
    popularity_rank = _average_ranks(popularity)
    return float(_pearson(radius_rank, popularity_rank))


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Ranks 0..N-1, ties sharing their mean rank."""
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[torch.argsort(values, stable=True)] = torch.arange(
        values.numel(), dtype=torch.float32, device=values.device)
    return _mean_within_ties(ranks, values)


def _mean_within_ties(x: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Replace each x by the mean of x over the entries sharing its key."""
    _, group = torch.unique(keys, return_inverse=True)
    group_sum = torch.zeros(int(group.max()) + 1, dtype=x.dtype, device=x.device).index_add_(0, group, x)
    group_size = torch.bincount(group).to(x.dtype)
    return (group_sum / group_size)[group]


def _pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation; 0 when either side is constant."""
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(torch.finfo(a.dtype).tiny)
    return (a * b).sum() / denom
