"""Walk-neighbour alignment loss — standard multi-positive InfoNCE over TWO token bags, geometry-agnostic.

Clusters E by walk-neighbourhood: each seed u (from EITHER the source bag or the candidate bag) is pulled
toward the nodes in its OWN walk (positives) and pushed off ONE batch-wide pool of every walk token in both
bags (negatives), frequency-weighted. STANDARD InfoNCE: the anchor u is excluded from BOTH its numerator and
its denominator (the degenerate self term dist(E[u],E[u]) = 0); every OTHER node — including u's own
positives — stays in the partition.

The whole loss is gathers into ONE shared distance matrix. The exact reduction that makes it cheap: a
geodesic distance depends only on the NODES, so every occurrence of a node collapses to a single term. With

    U   = unique CONTEXT-token nodes across both bags   (negative pool / numerator columns)   [nU]
    SU  = unique SEED nodes across both bags             (the only rows ever queried)          [nSU]
    D   = geodesic(E[SU], E[U])   [nSU, nU]              (matmul-form; no [n,m,d] tensor)

and logit s(seed s, node x) = -D[s, x]:

    log_num(row i) = logsumexp_{p ∈ ctx(i)}  [ -D[s, u(p)] + log_w(p) ]     # per seed ROW (own walk positives)
    log_denom(s)   = logsumexp_{u ∈ U, u ≠ s}[ -D[s, u]    + log_W[u] ]     # per seed NODE (shared pool)
    log_W[u]       = logsumexp over ALL context occurrences of node u (both bags) of log_w                (EXACT)
    L              = mean over seed ROWS with ≥1 positive of ( log_denom(node(i)) - log_num(i) )

Self-exclusion is by NODE (drop node(p) == s from ctx, drop col u == s from the denominator). Recency/hop
weight  log_w(p) = -(log1p(age_p) + log1p(hop_p - 1))  is MAX (=0) for a just-happened, adjacent token and
decays with age/hop. Weights are unnormalised (they appear in both terms), so the InfoNCE ratio is
scale-insensitive. `geom` need only expose `pairwise_dist(X, Y) -> [|X|, |Y|]` — the SAME code runs on the
sphere and the Poincaré ball.
"""
import torch
import torch.nn.functional as F

from .walk_tokens import WalkTokens


def _bag_context(bag: WalkTokens):
    """WalkTokens -> (seeds [Q], node_ids [Q,T], context [Q,T] bool, log_w [Q,T]). context = real, non-seed,
    non-self-node token (self = node == seed, dropped for standard InfoNCE). Padding (-1) clamped to row 0
    and excluded by `mask`."""
    node_ids = bag.nodes.clamp_min(0)                                        # [Q,T]  padding(-1) -> 0
    context = bag.mask & ~bag.seed_mask & (node_ids != bag.seeds[:, None])   # drop seed slot, revisits of u, padding
    ages = bag.ages.clamp_min(0).to(torch.float32)                          # [Q,T]
    hop = bag.positions.clamp_min(1).to(torch.float32)                      # [Q,T]  seed=1, closest ctx=2, ...
    log_w = -(torch.log1p(ages) + torch.log1p(hop - 1.0))                   # [Q,T]  <= 0
    return bag.seeds, node_ids, context, log_w


def alignment_loss(src_bag: WalkTokens, cand_bag: WalkTokens, emb: torch.Tensor, geom) -> torch.Tensor:
    """src_bag / cand_bag: the two flattened WalkTokens (seeds = query sources u / candidates v). emb: the
    full node-embedding table [num_nodes, d] on the manifold. geom: exposes `pairwise_dist(X, Y) -> [|X|,|Y|]`.
    Returns a scalar (0 if no seed in either bag has a context token)."""
    dev = emb.device
    ss, sn, sc, sw = _bag_context(src_bag)
    cs, cn, cc, cw = _bag_context(cand_bag)

    # --- combine the two bags along the row (Q) axis; pad to a common T if the bags differ ---
    if sn.shape[1] != cn.shape[1]:
        T = max(sn.shape[1], cn.shape[1])
        pad = lambda x, v: F.pad(x, (0, T - x.shape[1]), value=v)
        sn, cn = pad(sn, 0), pad(cn, 0)
        sc, cc = pad(sc, False), pad(cc, False)
        sw, cw = pad(sw, 0.0), pad(cw, 0.0)
    seeds = torch.cat([ss, cs])                                             # [S]
    node_ids = torch.cat([sn, cn])                                         # [S, T]
    context = torch.cat([sc, cc])                                         # [S, T]
    log_w = torch.cat([sw, cw]).to(emb.dtype)                              # [S, T]
    S, T = node_ids.shape

    if not context.any():
        return emb.sum() * 0.0

    # --- vocabularies: U = unique CONTEXT nodes (neg pool + numerator cols); SU = unique SEED nodes (rows) ---
    ctx_nodes = node_ids[context]                                         # [nCtxTok]  flat context node ids
    U, ctx_u = torch.unique(ctx_nodes, return_inverse=True)              # U [nU]; ctx_u = each ctx token's U-index
    SU = torch.unique(seeds)                                             # [nSU]
    nU, nSU = U.shape[0], SU.shape[0]

    num_nodes = emb.shape[0]
    to_col = emb.new_full((num_nodes,), -1, dtype=torch.long); to_col[U] = torch.arange(nU, device=dev)
    to_row = emb.new_full((num_nodes,), -1, dtype=torch.long); to_row[SU] = torch.arange(nSU, device=dev)

    # --- the ONE shared distance matrix: rows = unique seed nodes, cols = unique context nodes ---
    D = geom.pairwise_dist(F.embedding(SU, emb), F.embedding(U, emb))     # [nSU, nU]

    # --- denominator weights: log_W[u] = logsumexp over ALL context occurrences of node u of log_w (exact).
    # log_w <= 0 so exp(log_w) in (0, 1] -> scatter-add of exps then log is numerically safe (no max-shift). ---
    w_lin = torch.zeros(nU, device=dev, dtype=emb.dtype).scatter_add_(0, ctx_u, torch.exp(log_w[context]))
    log_W = w_lin.log()                                                  # [nU]  (> -inf: every u came from context)

    # --- denominator, per unique seed NODE (depends only on E[seed]); self column (u == seed) -> -inf ---
    self_mask = U[None, :] == SU[:, None]                                # [nSU, nU]  col node == row seed node
    log_denom_node = torch.logsumexp((-D + log_W[None, :]).masked_fill(self_mask, float("-inf")), dim=1)  # [nSU]

    # --- numerator, per seed ROW: gather this seed's own context distances from D; self already out of ctx ---
    su_row = to_row[seeds]                                               # [S]  each seed's row in SU
    tok_col = to_col[node_ids].clamp_min(0)                             # [S, T]  each token's col (masked below)
    d_st = D[su_row[:, None].expand(S, T), tok_col]                     # [S, T]  -logit magnitudes
    log_num = torch.logsumexp((-d_st + log_w).masked_fill(~context, float("-inf")), dim=1)  # [S]

    # --- combine: mean over seed ROWS with >=1 positive; denom gathered node -> row (carries row multiplicity) ---
    has_pos = context.any(dim=1)                                        # [S]
    per_row = (log_denom_node[su_row] - log_num)[has_pos]
    if per_row.numel() == 0:
        return emb.sum() * 0.0
    return per_row.mean()
