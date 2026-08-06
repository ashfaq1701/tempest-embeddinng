"""Walk-neighbour alignment loss — multi-positive InfoNCE over the TWO token bags, geometry-agnostic.

Clusters E by walk-neighbourhood. Each ANCHOR is a seed, from either the source bag or the candidate
bag, and it is:

    PULLED toward the nodes in its OWN walk   (positives, weighted by the recency/hop prior)
    PUSHED off a batch-wide pool of walk-token nodes drawn from BOTH bags   (negatives)

That is the whole objective. Two "sides" only in the sense that both bags contribute anchors — source
seeds u and candidate seeds v run through identical code. There is NO cross term: nothing pairs B_u
with B_v, and nothing distinguishes a true edge from a sampled candidate. This is walk co-occurrence,
an unsupervised proxy, deliberately kept that way so it can be evaluated on its own.

    log_num(row r)     = logsumexp_{p in positives(r)} [ -dist(E[a_r], E[p]) + log_weight(p) ]  per ROW
    log_denom(node a)  = logsumexp_{x in pool, x != a} [ -dist(E[a],   E[x]) ]                  per NODE
    loss               = mean over anchor ROWS with >= 1 positive of ( log_denom(node(r)) - log_num(r) )

Self-exclusion is BY NODE: a slot whose node equals the anchor is dropped from both terms (it would
contribute the degenerate dist(E[u], E[u]) = 0). Every OTHER node stays in the partition — including
the anchor's own positives, which is standard InfoNCE.

PULL IS WEIGHTED, PUSH IS NOT.
  * The numerator uses `WalkTokens.weight_logits`, i.e. -(log1p(age) + log1p(hop - 1)): maximal (= 0)
    for a just-happened, adjacent token, decaying with age and hop. That prior is the point of the
    objective — how hard a neighbour is pulled should depend on how recent and how near it is.
  * The denominator is UNWEIGHTED and its pool is sampled UNIFORMLY over DISTINCT nodes. How often the
    walks happened to reach a node is not something that should shape repulsion: weighting negatives by
    occurrence count would let heavily-walked hubs absorb the push while the tail — where new-pair
    prediction lives — drifts. Every node in the pool repels exactly as hard as every other.

MEMORY — the two terms are computed SEPARATELY, and that is load-bearing. The obvious implementation
builds ONE [S, P] distance matrix and gathers both terms from it. That matrix is what exhausts memory:
on review at bs 1000 / K_train 20 there are ~22k anchor rows and ~35k distinct context nodes, so an
uncapped [S, P] is ~7.7e8 cells (~3 GB per tensor) and the closed-form arcosh keeps roughly FIVE of
them alive for the backward pass. Tens of GB — to support a numerator that only ever needs R x W
distances (~2 MB: each row against its OWN tokens, never against the pool). So:

  * the NUMERATOR is computed directly as an [R, W] batched distance. It never touches the pool, which
    is also what makes pool sampling possible at all — with a shared matrix a positive dropped from the
    pool would have no column to gather from. Measured cost: ~0.1% of the geometric work.
  * the POOL is capped at `pool_size` distinct nodes, RESAMPLED EVERY CALL, which bounds the [S, P]
    matrix directly. Resampling is what makes the cap variance rather than bias: at ~30% coverage a
    single draw's gradient sits ~55 degrees off the full-pool direction, but averaged over draws it
    recovers to cos ~0.97 with the norm within 3%. A cached pool would be genuinely biased — do not
    cache it.

The denominator is then a single [S, P] matrix with P bounded by the cap. Note it is still the whole
cost of the loss: at S = 11k, P = 10k that is 1.1e8 cells, ~440 MB per tensor and ~2 GB across the
arcosh chain's live intermediates. `pool_size` is the knob — halve it if the backward peaks.

`geom` need only expose `pairwise_dist(X, Y) -> [|X|, |Y|]`, so the same code runs on the sphere and
on the Poincaré ball.
"""
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_NEG_INF = float("-inf")


class _BagContext(NamedTuple):
    """A bag reduced to what the loss needs: seeds + their context (positive) tokens with recency
    weights. `is_context` already drops padding and every slot whose node is the bag's own seed."""
    seeds: torch.Tensor         # [Q]      anchor node id per row
    nodes: torch.Tensor         # [Q, T]   node id per slot (padding clamped to 0; validity in is_context)
    is_context: torch.Tensor    # [Q, T]   True on real, non-self-node slots — the positives
    log_weight: torch.Tensor    # [Q, T]   the recency/hop prior, <= 0 (0 = most recent & adjacent)


def _extract_context(bag: WalkTokens, dtype: torch.dtype) -> _BagContext:
    """Pull (seeds, nodes, positive mask, weights) out of a bag.

    `mask & ~seed_node_mask` drops padding, the seed's own origin slot, AND any mid-walk revisit of the
    seed — all of which would contribute a degenerate zero distance to the anchor."""
    nodes = bag.nodes.clamp_min(0)                                   # [Q, T]  padding(-1) -> row 0
    is_context = bag.mask & ~bag.seed_node_mask                      # [Q, T]  real, not the seed's node
    return _BagContext(bag.seeds, nodes, is_context, bag.weight_logits.to(dtype))


def _pad_cols(t: torch.Tensor, width: int, value) -> torch.Tensor:
    """Right-pad a [Q, T] tensor to [Q, width] (no-op when already that wide)."""
    return t if t.shape[1] == width else F.pad(t, (0, width - t.shape[1]), value=value)


def _sample_pool(context_nodes: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Distinct context nodes, uniformly subsampled to at most `pool_size`.

    UNIFORM OVER DISTINCT NODES: `torch.unique` collapses every occurrence first, so a node the walks
    reached a thousand times has exactly the same chance of entering the pool as one they reached once.
    Sampling the raw token stream instead would reintroduce frequency weighting through the back door.

    Resampled on every call, which is what keeps the cap unbiased in expectation."""
    pool = torch.unique(context_nodes)                               # [P_all] distinct, occurrence-blind
    if pool.numel() <= pool_size:
        return pool
    keep = torch.randperm(pool.numel(), device=pool.device)[:pool_size]
    return pool[keep]                                                # [pool_size]


def _row_token_dist(emb: torch.Tensor, geom, seeds: torch.Tensor,
                    token_nodes: torch.Tensor) -> torch.Tensor:
    """Distance from each anchor row's seed to that row's OWN tokens -> [R, W].

    The numerator's entire geometric need, and it is tiny: R x W (~550k at bs 1000) against the
    denominator's S x P (~7.7e8), a ratio of ~1400x."""
    e_seed = F.embedding(seeds, emb)                                  # [R, d]
    e_tok = F.embedding(token_nodes, emb)                             # [R, W, d]
    return geom.pairwise_dist(e_seed[:, None, :], e_tok).squeeze(-2)  # [R, W]


def _log_denom(emb: torch.Tensor, geom, seed_nodes: torch.Tensor,
               pool_nodes: torch.Tensor) -> torch.Tensor:
    """UNWEIGHTED push-off term per unique anchor NODE -> [S].

    One [S, P] matrix. P is bounded by the caller's `pool_size` cap, which is what keeps this from
    exhausting memory — an uncapped pool on review is ~35k distinct nodes and the backward pass holds
    several [S, P] intermediates at once."""
    d = geom.pairwise_dist(F.embedding(seed_nodes, emb),
                           F.embedding(pool_nodes, emb))              # [S, P]
    is_self = pool_nodes[None, :] == seed_nodes[:, None]              # [S, P] drop dist(E[a], E[a]) = 0
    return torch.logsumexp(d.neg().masked_fill(is_self, _NEG_INF), dim=1)   # [S]


def alignment_loss(src_bag: WalkTokens, cand_bag: WalkTokens, emb: torch.Tensor, geom,
                   pool_size: int = 15_000) -> torch.Tensor:
    """src_bag / cand_bag: the two flattened WalkTokens (seeds = query sources u / candidates v).
    emb: the full node-embedding table [num_nodes, d] on the manifold. geom: exposes
    `pairwise_dist(X, Y) -> [|X|, |Y|]`.

    pool_size:   cap on DISTINCT negative nodes, uniformly resampled each call. It sets the width of
                 the [S, P] denominator matrix, so cost and peak memory are linear in it; raise it for
                 a lower-variance push. The whole pool is used when it is already smaller than the cap
                 (wiki's is; review's is not).

    Returns a scalar (exactly 0 if no anchor has a context token)."""
    device = emb.device
    src = _extract_context(src_bag, emb.dtype)
    cand = _extract_context(cand_bag, emb.dtype)

    # 1. Stack both bags row-wise into one set of anchors. The bags may have different token widths
    #    (T = K*L is per-call), so pad to the wider one.
    width = max(src.nodes.shape[1], cand.nodes.shape[1])
    seeds = torch.cat([src.seeds, cand.seeds])                                                    # [R]
    nodes = torch.cat([_pad_cols(src.nodes, width, 0),
                       _pad_cols(cand.nodes, width, 0)])                                          # [R, W]
    is_context = torch.cat([_pad_cols(src.is_context, width, False),
                            _pad_cols(cand.is_context, width, False)])                            # [R, W]
    log_weight = torch.cat([_pad_cols(src.log_weight, width, 0.0),
                            _pad_cols(cand.log_weight, width, 0.0)])                              # [R, W]

    if not bool(is_context.any()):
        return emb.sum() * 0.0

    # 2. Negative pool: distinct context nodes across both bags, uniformly capped at pool_size.
    context_nodes = nodes[is_context]                                                # [n_ctx_tokens]
    n_distinct = int(torch.unique(context_nodes).numel())
    pool_nodes = _sample_pool(context_nodes, pool_size)                              # [P]
    seed_nodes = torch.unique(seeds)                                                 # [S]

    node_to_row = emb.new_full((emb.shape[0],), -1, dtype=torch.long)
    node_to_row[seed_nodes] = torch.arange(seed_nodes.shape[0], device=device)       # node id -> seed row

    # 3. Denominator (push-off) per unique anchor NODE — one [S, P] matrix, P bounded by pool_size.
    log_denom = _log_denom(emb, geom, seed_nodes, pool_nodes)                        # [S]
    # Uniform sampling makes sum_sampled exp(-d) an unbiased estimate of (n/P_all) * sum_all exp(-d),
    # so this restores the full-pool SCALE. Constant w.r.t. E — zero gradient — and present purely so
    # the logged loss stays comparable across batches whose pools differ in size.
    if pool_nodes.numel() < n_distinct:
        log_denom = log_denom + math.log(n_distinct / pool_nodes.numel())

    # 4. Numerator (pull-in): each anchor ROW toward its OWN walk positives, recency/hop weighted.
    token_dist = _row_token_dist(emb, geom, seeds, nodes)                            # [R, W]
    num_logits = (-token_dist + log_weight).masked_fill(~is_context, _NEG_INF)       # [R, W]
    log_num = torch.logsumexp(num_logits, dim=1)                                     # [R]

    # 5. InfoNCE per anchor row, averaged over rows with >= 1 positive. The denominator is per anchor
    #    NODE, so gather it back to rows: a node used by K rows contributes K times, which preserves the
    #    exact gradient on E. A row whose anchor was the pool's only entry has log_denom = -inf; it is
    #    weightless and must be DROPPED before the mean rather than multiplied by zero (0 * inf = nan).
    row_denom = log_denom[node_to_row[seeds]]                                        # [R]
    keep = is_context.any(dim=1) & torch.isfinite(row_denom)                         # [R]
    if not bool(keep.any()):
        return emb.sum() * 0.0
    return (row_denom - log_num)[keep].mean()
