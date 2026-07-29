"""Walk-neighbour alignment loss (recency/frequency-weighted InfoNCE) — geometry-agnostic.

Goal: cluster nodes by their walk-neighbourhoods. For each seed u, pull E[u] toward the nodes in u's
OWN backward walks (positives) and push it off the FULL batch bag of walk tokens (every query's context
tokens). All closeness goes through the passed manifold `geom`, so the SAME loss runs on the sphere and
the Poincaré ball unchanged (sphere -> arccos, ball -> hyperbolic distance).

Per seed u (logit  s(u,x) = −geom.dist(E[u], E[x])  — higher = closer):

    L_u =  logsumexp_{n ∈ FULL bag, node(n) ≠ u} [ s(u,n) + log_w(n) ]      ← denominator: push off
         − logsumexp_{p ∈ ctx(u)}                [ s(u,p) + log_w(p) ]      ← numerator:  pull in

  ctx(u)   = u's context tokens (mask & ~seed_mask).
  FULL bag = every context token of EVERY query in the batch — NOT deduplicated, so a node's
             multiplicity IS its in-batch count (frequent nodes weigh more) — minus tokens whose node
             is u itself (the degenerate self-term dist(E[u],E[u])=0). Seeds are not added to the bag.
  log_w(x) = −( log1p(age_x) + log1p(position_x − 1) )      per-token recency/closeness weight: MAX
             (log_w = 0) for a just-happened, immediately-adjacent token; decays as age or hop grows.
             (positions: seed = 1, closest context = 2, …; so position − 1 is the 1-indexed hop rank.)

Minimising L_u pulls each seed toward its own age/hop-weighted walk neighbourhood and away from the
frequency-weighted batch node distribution → nodes that share walk-neighbourhoods cluster together.

Weights are unnormalised: they appear in both numerator and denominator, so the InfoNCE ratio is
insensitive to a global scale — no per-seed renormalisation is needed.
"""
import torch
import torch.nn.functional as F

from .walk_tokens import WalkTokens


def alignment_loss(walk_bag: WalkTokens, emb: torch.Tensor, geom) -> torch.Tensor:
    """`walk_bag`: the SOURCE-side flattened WalkTokens (seeds = query nodes u). `emb`: the full node
    embedding table [num_nodes, d] on the manifold. `geom`: the manifold (must expose `.dist(x, y)`);
    the SAME code works for SphereManifold and PoincareManifold. Returns a scalar (0 if the batch is
    empty or no query has a context token)."""
    q, t = walk_bag.nodes.shape
    if q == 0:
        return emb.sum() * 0.0
    neg_inf = torch.finfo(emb.dtype).min

    seeds = walk_bag.seeds                                                # [Q]
    node_ids = walk_bag.nodes.clamp_min(0)                               # [Q, T]  padding (-1) → row 0
    context = walk_bag.mask & ~walk_bag.seed_mask                        # [Q, T]  valid non-seed tokens

    # Per-token recency/closeness weight in log-space: log_w = −(log1p(age) + log1p(hop − 1)).
    # MAX (log_w = 0) for the most-recent, closest token; decays as age (cutoff − t_edge) or hop grows.
    ages = walk_bag.ages.clamp_min(0).to(emb.dtype)                     # [Q, T]  ≥ 1 on context
    hop = walk_bag.positions.clamp_min(1).to(emb.dtype)                 # [Q, T]  seed = 1, closest ctx = 2, …
    log_w = -(torch.log1p(ages) + torch.log1p(hop - 1.0))              # [Q, T]  ≤ 0

    e_seed = F.embedding(seeds, emb)                                    # [Q, d]
    e_tok = F.embedding(node_ids, emb)                                 # [Q, T, d]
    d = e_seed.shape[-1]

    # ── Numerator: each seed ↔ its OWN context tokens (positives), recency/closeness-weighted. ──
    #   s(u, own token) = −geom.dist(E[u], E[token])   — own bag only, so [Q, T] (no cross product).
    s_own = -geom.dist(e_seed.unsqueeze(1), e_tok)                     # [Q, T]
    log_num = torch.logsumexp(
        (s_own + log_w).masked_fill(~context, neg_inf), dim=1)         # [Q]

    # ── Denominator: each seed ↔ EVERY context token in the batch (NOT deduplicated), self-node out. ──
    e_flat = e_tok.reshape(q * t, d)                                    # [M, d]  M = Q·T (all token slots)
    flat_nodes = node_ids.reshape(1, q * t)                            # [1, M]
    flat_valid = context.reshape(1, q * t)                            # [1, M]  real non-seed tokens
    flat_log_w = log_w.reshape(1, q * t)                              # [1, M]
    # NOTE: geom.dist broadcasts [Q,1,d] vs [1,M,d] → [Q, M] (materialises a [Q, M, d] intermediate; if
    # M is very large this is the memory hot spot — chunk over M then).
    s_all = -geom.dist(e_seed.unsqueeze(1), e_flat.unsqueeze(0))       # [Q, M]
    not_self = flat_nodes != seeds.unsqueeze(1)                        # [Q, M]  drop node(n) == u
    denom_mask = flat_valid & not_self                                # [Q, M]
    log_denom = torch.logsumexp(
        (s_all + flat_log_w).masked_fill(~denom_mask, neg_inf), dim=1)  # [Q]

    # ── Per-seed InfoNCE; seeds with no context token contribute nothing. ──
    has_pos = context.any(dim=1)                                       # [Q]
    per_seed = (log_denom - log_num)[has_pos]
    if per_seed.numel() == 0:
        return emb.sum() * 0.0
    return per_seed.mean()
