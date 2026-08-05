"""Walk-neighbour alignment loss — multi-positive InfoNCE over the TWO token bags, geometry-agnostic.

Clusters E by walk-neighbourhood. Each ANCHOR is a seed, from either the source bag or the candidate
bag, and it is:

    PULLED toward the nodes in its OWN walk        (positives, weighted by the recency/hop prior)
    PUSHED off ONE batch-wide pool of every walk token in BOTH bags   (negatives)

That is the whole objective. Two "sides" only in the sense that both bags contribute anchors — the
source seeds u and the candidate seeds v are handled by identical code. There is NO cross term here:
nothing pairs B_u with B_v, and nothing distinguishes a true edge from a sampled candidate. This is
walk co-occurrence, i.e. an unsupervised proxy, and it is deliberately kept that way so it can be
compared against the ranking objective on its own.

THE TRICK that makes it cheap: a geodesic distance depends only on the NODES, so every occurrence of
a node collapses to a single term. With

    pool_nodes  = unique CONTEXT-token nodes across both bags  (negative pool AND numerator columns) [P]
    seed_nodes  = unique ANCHOR nodes across both bags         (the only rows we query distances from)[S]
    dist        = geodesic(E[seed_nodes], E[pool_nodes])       [S, P]  (matmul form; no [S, P, d] tensor)

and logit(anchor a, node x) = -dist[a, x]:

    log_num(row r)     = logsumexp_{p in positives(r)} [ -dist[a_r, col(p)] + log_weight(p) ]   per ROW
    log_denom(node a)  = logsumexp_{x in pool, x != a} [ -dist[a,   x]      + log_pool_w[x]  ]   per NODE
    log_pool_w[x]      = log of the summed weight of ALL context occurrences of node x (both bags) — EXACT
    loss               = mean over anchor ROWS with >= 1 positive of ( log_denom(node(r)) - log_num(r) )

Self-exclusion is BY NODE: a slot whose node equals the anchor is dropped from both the numerator and
the denominator (it would contribute the degenerate dist(E[u], E[u]) = 0). Every OTHER node stays in
the partition — including the anchor's own positives, which is standard InfoNCE.

The weights are the shared prior `WalkTokens.weight_logits`:

    log_weight(p) = -( log1p(age_p) + log1p(hop_p - 1) )        maximal (= 0) for a just-happened,
                                                               adjacent token; decays with age and hop

They are used UNNORMALISED. They appear in the numerator and in the pool weights alike, so the InfoNCE
ratio is insensitive to their overall scale, and leaving them raw keeps this a faithful reproduction of
the original loss (normalising per row would change how much total mass a dense bag contributes
relative to a sparse one — a different objective, not a cosmetic difference).

`geom` need only expose `pairwise_dist(X, Y) -> [|X|, |Y|]`, so the same code runs on the sphere and on
the Poincaré ball.
"""
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


def alignment_loss(src_bag: WalkTokens, cand_bag: WalkTokens, emb: torch.Tensor, geom) -> torch.Tensor:
    """src_bag / cand_bag: the two flattened WalkTokens (seeds = query sources u / candidates v).
    emb: the full node-embedding table [num_nodes, d] on the manifold. geom: exposes
    `pairwise_dist(X, Y) -> [|X|, |Y|]`. Returns a scalar (exactly 0 if no anchor has a context token).
    """
    device = emb.device
    src = _extract_context(src_bag, emb.dtype)
    cand = _extract_context(cand_bag, emb.dtype)

    # ── 1. Stack both bags row-wise into one set of anchors. ────────────────────────────────────────
    # The two bags may have different token widths (T = K*L is per-call), so pad to the wider one.
    width = max(src.nodes.shape[1], cand.nodes.shape[1])
    seeds = torch.cat([src.seeds, cand.seeds])                                                    # [R]
    nodes = torch.cat([_pad_cols(src.nodes, width, 0),
                       _pad_cols(cand.nodes, width, 0)])                                          # [R, W]
    is_context = torch.cat([_pad_cols(src.is_context, width, False),
                            _pad_cols(cand.is_context, width, False)])                            # [R, W]
    log_weight = torch.cat([_pad_cols(src.log_weight, width, 0.0),
                            _pad_cols(cand.log_weight, width, 0.0)])                              # [R, W]
    n_rows = seeds.shape[0]

    if not bool(is_context.any()):
        return emb.sum() * 0.0

    # ── 2. Vocabularies + the ONE shared distance matrix. ───────────────────────────────────────────
    # pool_nodes = unique context nodes -> both the negative pool AND the numerator's columns.
    # seed_nodes = unique anchor nodes  -> the only rows we ever query distances from.
    context_nodes = nodes[is_context]                                                # [n_ctx_tokens]
    pool_nodes, token_pool_col = torch.unique(context_nodes, return_inverse=True)     # [P], [n_ctx_tokens]
    seed_nodes = torch.unique(seeds)                                                  # [S]
    n_pool, n_seed = pool_nodes.shape[0], seed_nodes.shape[0]

    node_to_col = emb.new_full((emb.shape[0],), -1, dtype=torch.long)
    node_to_col[pool_nodes] = torch.arange(n_pool, device=device)                    # node id -> pool column
    node_to_row = emb.new_full((emb.shape[0],), -1, dtype=torch.long)
    node_to_row[seed_nodes] = torch.arange(n_seed, device=device)                    # node id -> seed row

    dist = geom.pairwise_dist(F.embedding(seed_nodes, emb),
                              F.embedding(pool_nodes, emb))                          # [S, P] matmul form

    # ── 3. Denominator (push-off): the shared pool, computed once per unique anchor NODE. ───────────
    # A pool node's weight sums ALL of its context occurrences across both bags — exact, since the
    # geodesic depends only on the node. log_weight <= 0 => exp in (0, 1], so scatter-add-then-log is
    # numerically safe with no shift.
    pool_weight = torch.zeros(n_pool, device=device, dtype=emb.dtype)
    pool_weight.scatter_add_(0, token_pool_col, torch.exp(log_weight[is_context]))
    log_pool_weight = pool_weight.log()                                              # [P] finite by construction

    is_self = pool_nodes[None, :] == seed_nodes[:, None]                             # [S, P] col node == anchor
    denom_logits = (-dist + log_pool_weight[None, :]).masked_fill(is_self, _NEG_INF)  # [S, P]
    log_denom_per_seed_node = torch.logsumexp(denom_logits, dim=1)                    # [S]

    # ── 4. Numerator (pull-in): each anchor ROW toward its OWN walk positives. ──────────────────────
    row_seed_row = node_to_row[seeds]                                                # [R] anchor's seed row
    token_col = node_to_col[nodes].clamp_min(0)                                      # [R, W] (masked below)
    token_dist = dist[row_seed_row[:, None].expand(n_rows, width), token_col]        # [R, W] gather only
    num_logits = (-token_dist + log_weight).masked_fill(~is_context, _NEG_INF)       # [R, W]
    log_num_per_row = torch.logsumexp(num_logits, dim=1)                             # [R]

    # ── 5. InfoNCE per anchor row, averaged over rows with >= 1 positive. ───────────────────────────
    # The denominator is per anchor NODE, so gather it back to rows: a node used by K rows contributes
    # K times, which is what preserves the exact gradient on E.
    has_positive = is_context.any(dim=1)                                             # [R]
    per_row_loss = (log_denom_per_seed_node[row_seed_row] - log_num_per_row)[has_positive]
    if per_row_loss.numel() == 0:
        return emb.sum() * 0.0
    return per_row_loss.mean()
