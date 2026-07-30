"""Walk-neighbour alignment loss — standard multi-positive InfoNCE over TWO token bags, geometry-agnostic.

Clusters E by walk-neighbourhood: each anchor (a seed from EITHER the source bag or the candidate bag) is
pulled toward the nodes in its OWN walk (positives) and pushed off ONE batch-wide, frequency-weighted pool
of every walk token in both bags (negatives). STANDARD InfoNCE: the anchor node is excluded from BOTH its
numerator and its denominator (the degenerate self term dist(E[u], E[u]) = 0); every OTHER node — including
the anchor's own positives — stays in the partition.

The whole loss reduces to gathers into ONE shared distance matrix. The exact trick: a geodesic distance
depends only on the NODES, so every occurrence of a node collapses to a single term. With

    pool_nodes  = unique CONTEXT-token nodes across both bags   (negative pool / numerator columns)  [P]
    seed_nodes  = unique ANCHOR nodes across both bags          (the only rows we query distances from) [S]
    dist        = geodesic(E[seed_nodes], E[pool_nodes])   [S, P]   (matmul-form; no [S,P,d] tensor)

and logit(anchor a, node x) = -dist[a, x]:

    log_num(row r)      = logsumexp_{p ∈ positives(r)}    [ -dist[a, col(p)] + log_weight(p) ]   # per anchor ROW
    log_denom(node a)   = logsumexp_{x ∈ pool, x ≠ a}     [ -dist[a, x]      + log_pool_weight[x] ] # per anchor NODE
    log_pool_weight[x]  = logsumexp over ALL context occurrences of node x (both bags) of log_weight        (EXACT)
    loss                = mean over anchor ROWS with ≥1 positive of ( log_denom(node(r)) - log_num(r) )

Self-exclusion is by NODE (drop node == anchor from both terms). The recency/hop weight
    log_weight(p) = -(log1p(age_p) + log1p(hop_p - 1))
is MAX (= 0) for a just-happened, adjacent token and decays with age/hop. Weights are unnormalised (they
appear in both terms), so the InfoNCE ratio is scale-insensitive. `geom` need only expose
`pairwise_dist(X, Y) -> [|X|, |Y|]`, so the SAME code runs on the sphere and the Poincaré ball.
"""
from typing import NamedTuple

import torch
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class _BagContext(NamedTuple):
    """A walk bag reduced to what the loss needs: seeds + their context (positive) tokens with recency
    weights. `is_context` already drops padding, the seed slot, and self-node revisits."""
    seeds: torch.Tensor         # [Q]      anchor node id per row
    token_nodes: torch.Tensor   # [Q, T]   node id per slot (padding clamped to 0; validity lives in is_context)
    is_context: torch.Tensor    # [Q, T]   True on real, non-seed, non-self-node tokens (the positives)
    log_weight: torch.Tensor    # [Q, T]   recency/hop log-weight, <= 0 (0 = most recent & adjacent)


def _extract_context(bag: WalkTokens) -> _BagContext:
    """Pull the (seeds, token nodes, positive mask, recency weights) a bag contributes to the loss."""
    token_nodes = bag.nodes.clamp_min(0)                                           # padding(-1) -> row 0
    is_context = bag.mask & ~bag.seed_mask & (token_nodes != bag.seeds[:, None])   # real, non-seed, non-self
    age = bag.ages.clamp_min(0).to(torch.float32)                                  # [Q, T]
    hop = bag.positions.clamp_min(1).to(torch.float32)                             # [Q, T]  seed=1, closest ctx=2, ...
    log_weight = -(torch.log1p(age) + torch.log1p(hop - 1.0))                      # [Q, T]  <= 0
    return _BagContext(bag.seeds, token_nodes, is_context, log_weight)


def _pad_cols(t: torch.Tensor, width: int, value) -> torch.Tensor:
    """Right-pad a [Q, T] tensor to [Q, width] (no-op when already that wide)."""
    return t if t.shape[1] == width else F.pad(t, (0, width - t.shape[1]), value=value)


def alignment_loss(src_bag: WalkTokens, cand_bag: WalkTokens, emb: torch.Tensor, geom) -> torch.Tensor:
    """src_bag / cand_bag: the two flattened WalkTokens (seeds = query sources u / candidates v). emb: the
    full node-embedding table [num_nodes, d] on the manifold. geom: exposes `pairwise_dist(X, Y) -> [|X|,|Y|]`.
    Returns a scalar (0 if no anchor in either bag has a context token)."""
    device = emb.device
    src, cand = _extract_context(src_bag), _extract_context(cand_bag)

    # 1. Stack both bags row-wise into one set of anchors (pad to a common token width if the bags differ).
    width = max(src.token_nodes.shape[1], cand.token_nodes.shape[1])
    seeds = torch.cat([src.seeds, cand.seeds])                                                # [R]
    token_nodes = torch.cat([_pad_cols(src.token_nodes, width, 0), _pad_cols(cand.token_nodes, width, 0)])   # [R, W]
    is_context = torch.cat([_pad_cols(src.is_context, width, False), _pad_cols(cand.is_context, width, False)])  # [R, W]
    log_weight = torch.cat([_pad_cols(src.log_weight, width, 0.0), _pad_cols(cand.log_weight, width, 0.0)]).to(emb.dtype)  # [R, W]
    n_rows = seeds.shape[0]

    if not is_context.any():
        return emb.sum() * 0.0

    # 2. Vocabularies + the one shared distance matrix.
    #    pool_nodes = unique context nodes -> shared negative pool AND numerator columns.
    #    seed_nodes = unique anchor nodes  -> the only rows we ever query distances from.
    context_token_nodes = token_nodes[is_context]                                            # [n_ctx_tokens]
    pool_nodes, token_pool_col = torch.unique(context_token_nodes, return_inverse=True)       # [P], [n_ctx_tokens]
    seed_nodes = torch.unique(seeds)                                                          # [S]
    n_pool, n_seed = pool_nodes.shape[0], seed_nodes.shape[0]

    num_nodes = emb.shape[0]
    node_to_col = emb.new_full((num_nodes,), -1, dtype=torch.long)
    node_to_col[pool_nodes] = torch.arange(n_pool, device=device)                            # node id -> pool column
    node_to_row = emb.new_full((num_nodes,), -1, dtype=torch.long)
    node_to_row[seed_nodes] = torch.arange(n_seed, device=device)                            # node id -> seed row

    dist = geom.pairwise_dist(F.embedding(seed_nodes, emb), F.embedding(pool_nodes, emb))     # [S, P] (matmul form)

    # 3. Denominator (push-off): shared pool, computed per unique anchor NODE.
    #    Each pool node's weight aggregates ALL its context occurrences — exact, since a geodesic depends only
    #    on the node. log_weight <= 0 => exp in (0, 1] => scatter-add then log is numerically safe (no shift).
    pool_weight = torch.zeros(n_pool, device=device, dtype=emb.dtype)
    pool_weight.scatter_add_(0, token_pool_col, torch.exp(log_weight[is_context]))
    log_pool_weight = pool_weight.log()                                                      # [P]  (finite: pool from context)

    is_self = pool_nodes[None, :] == seed_nodes[:, None]                                      # [S, P] col node == row anchor
    denom_logits = (-dist + log_pool_weight[None, :]).masked_fill(is_self, float("-inf"))     # [S, P]
    log_denom_per_seed_node = torch.logsumexp(denom_logits, dim=1)                            # [S]

    # 4. Numerator (pull-in): each anchor ROW toward its OWN walk positives (self already out of is_context).
    row_seed_row = node_to_row[seeds]                                                         # [R]  anchor's seed row
    token_col = node_to_col[token_nodes].clamp_min(0)                                         # [R, W]  (masked below)
    token_dist = dist[row_seed_row[:, None].expand(n_rows, width), token_col]                 # [R, W]
    num_logits = (-token_dist + log_weight).masked_fill(~is_context, float("-inf"))           # [R, W]
    log_num_per_row = torch.logsumexp(num_logits, dim=1)                                       # [R]

    # 5. InfoNCE per anchor row, averaged over rows with >=1 positive. The denominator is per anchor NODE, so
    #    gather it back to rows — a node used by K rows then contributes K times (preserves the exact E grad).
    has_positive = is_context.any(dim=1)                                                     # [R]
    per_row_loss = (log_denom_per_seed_node[row_seed_row] - log_num_per_row)[has_positive]
    if per_row_loss.numel() == 0:
        return emb.sum() * 0.0
    return per_row_loss.mean()
