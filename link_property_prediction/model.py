"""Stateless, geometry-free temporal link head.

No learned node-embedding table, no manifold. A pair (u, v) is scored purely from Tempest's
MULTI-HOP temporal walks (the differentiator vs GraphMixer/TPNet's 1-hop neighbours), turned into
structural signal by a batch-local, anonymized `NodeEncoding` and pooled by a light attention head:

    logit(u, v) = MLP( ⟨seed_u, seed_v⟩, ⟨seed_u, nbhd_v⟩, ⟨nbhd_u, seed_v⟩, ⟨nbhd_u, nbhd_v⟩ )

where seed_x = the node's own structural code (seed row of NodeEncoding) and nbhd_x = a recency/
structure-weighted aggregation of x's walk neighbourhood (WalkNeighborhoodEncoder). Both bags (source +
candidate) are JOINTLY encoded so they share one batch-local graph and one random basis.

THE BASIS CONSTRAINT (load-bearing): NodeEncoding draws a fresh random X0 each batch, so a node's
raw code lives in a random subspace that rotates batch-to-batch. Learned dense layers on raw codes
would see random inputs and can't learn a stable function. Codes are therefore consumed ONLY through
inner products — attention scores (cos(seed, token)) and the final pairwise dot products — which are
invariant to the random basis. Time/hop/edge features (stable across batches) are the only inputs to
learned linears; they drive the attention logits.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens, flatten_tokens


class NodeEncoding(nn.Module):
    """Stateless, batch-local node embedding — a drop-in replacement for a learned E table.

    Per batch, builds the local walk graph from the WalkTokens, then encodes every node that appears
    (walk nodes + query seeds) by a MULTI-HOP diffusion of random features over that graph:

        A        = batch-local recency-weighted (undirected) adjacency [U, U] from consecutive walk steps
        Â        = D^-1 A                                    (random-walk transition)
        X0       = random Gaussian features [U, dim]         (JL basis; fresh each batch — anonymized)
        node_enc = [ X0 ; Â X0 ; Â² X0 ; … ; Âⁿ X0 ]         [U, (n_hops+1)·dim]

    By Johnson–Lindenstrauss, <Xk[i], Xk[j]> ≈ <Âᵏ[i,:], Âᵏ[j,:]> = the k-hop co-reachability of nodes
    i and j — so the encoding carries multi-hop structure at fixed width, with no learned state and no
    global node ids (identity-free / anonymized). Nodes that appear with no edges get [X0, 0, …, 0].

    forward(tokens) -> (assoc, node_enc):
        assoc     [num_nodes] int64  global id -> local row in [0, U); -1 for nodes absent this batch.
        node_enc  [U, (n_hops+1)·dim] float  one fixed-width row per present node.
    Use NodeEncoding.gather(assoc, node_enc, ids) to look up any id tensor (tokens / seeds / …).
    """

    def __init__(self, num_nodes: int, dim: int, n_hops: int,
                 recency_lambda: float = 0.0, undirected: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.n_hops = n_hops
        self.recency_lambda = recency_lambda
        self.undirected = undirected

    def forward(self, tokens: WalkTokens, other: Optional[WalkTokens] = None):
        """Encode one bag, or JOINTLY encode two bags (e.g. source + candidate walks). When `other` is
        given, both bags share ONE batch-local graph, ONE random X0, and hence ONE comparable encoding
        — necessary because encodings from separate forwards live in independent random subspaces
        (⟨X¹,X²⟩≈0). Gather both sides' ids from the returned node_enc. Returns (assoc, node_enc)."""
        bags = [tokens] if other is None else [tokens, other]
        device = tokens.nodes.device

        # 1. joint batch-local node set (all bags' walk nodes + seeds) + global→local map.
        present = torch.unique(torch.cat(
            [t.nodes[t.nodes_mask] for t in bags] + [t.seeds for t in bags]))           # [U] sorted global ids
        u = int(present.numel())
        assoc = tokens.nodes.new_full((self.num_nodes,), -1)                            # [num_nodes]
        assoc[present] = torch.arange(u, device=device)

        # 2. joint edges (l, l+1) from every bag. Ages are node-aligned to the OLDER endpoint l (walks
        #    stored oldest→seed), so each edge's recency is read from the [..., :-1] slice.
        si_list, di_list, w_list = [], [], []
        for t in bags:
            m = t.nodes_mask
            em = m[..., :-1] & m[..., 1:]                                               # both endpoints real
            si_list.append(assoc[t.nodes[..., :-1][em]])                               # local src
            di_list.append(assoc[t.nodes[..., 1:][em]])                                # local dst
            w_list.append(torch.exp(-self.recency_lambda * t.ages[..., :-1][em].to(torch.float32)))
        si, di, w = torch.cat(si_list), torch.cat(di_list), torch.cat(w_list)
        if self.undirected:
            si, di = torch.cat([si, di]), torch.cat([di, si])
            w = torch.cat([w, w])

        # 3. random base features, then multi-hop diffusion Â X over the (joint) batch-local graph.
        x0 = torch.randn(u, self.dim, device=device)                                   # [U, dim] JL basis
        blocks = [x0]
        if si.numel() > 0:
            adj = torch.sparse_coo_tensor(torch.stack([si, di]), w, (u, u)).coalesce()  # [U,U] weighted
            deg = torch.sparse.sum(adj, dim=1).to_dense()                               # [U] weighted degree
            inv_deg = torch.where(deg > 0, deg.reciprocal(), torch.zeros_like(deg))     # exact 1/deg; 0 if isolated
            xk = x0
            for _ in range(self.n_hops):
                xk = torch.sparse.mm(adj, xk) * inv_deg.unsqueeze(-1)                   # Â xk = D⁻¹A xk (k-hop)
                blocks.append(xk)
        else:
            blocks += [torch.zeros_like(x0) for _ in range(self.n_hops)]               # no edges → 0 blocks

        node_enc = torch.cat(blocks, dim=-1)                                           # [U, (n_hops+1)*dim]
        return assoc, node_enc

    @staticmethod
    def gather(assoc: torch.Tensor, node_enc: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Look up rows for any id tensor. ids [*] → [*, (n_hops+1)*dim]. Padding/absent ids (-1) map to
        row 0 (garbage) and must be masked out by the caller via the tokens' mask."""
        return node_enc[assoc[ids.clamp_min(0)].clamp_min(0)]

    def normalize_blocks(self, node_enc: torch.Tensor) -> torch.Tensor:
        """Per-HOP-block L2 normalisation — a SCORING-side transform kept out of forward() so forward
        stays the exact, testable raw diffusion. Raw block magnitudes shrink with diffusion depth (each
        hop is a mean-of-means), so an inner product over the concatenated code would be dominated by
        hop-0 (exact identity) and the deeper co-reachability hops would be numerically drowned out.
        Making each dim-block unit-norm turns every hop's ⟨block_i, block_j⟩ into a cosine in [-1, 1],
        so all hops enter downstream inner products on equal footing. [U, (n_hops+1)*dim] → same shape."""
        u = node_enc.shape[0]
        blocks = F.normalize(node_enc.view(u, self.n_hops + 1, self.dim), dim=-1)
        return blocks.reshape(u, (self.n_hops + 1) * self.dim)


class TimeEncoder(nn.Module):
    """Time2Vec time encoding, ported verbatim from TPNet (TGB_TPNet/models/modules.py)."""

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=2)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class WalkNeighborhoodEncoder(nn.Module):
    """Pool a query's walk-token bag into (seed_code, nbhd_code): seed_code = the query node's OWN
    structural code; nbhd_code = a recency/structure-weighted aggregation of its NEIGHBOURHOOD tokens
    (the seed's own walk-origin slot is excluded, so nbhd_code is genuinely the neighbourhood, not a
    self-copy).

    Single seed-query attention. The attention logit combines a structural co-reachability term
    cos(seed_code, token_code) — the ONLY way node codes may enter, since their basis is random per
    batch — with a NONLINEAR (2-layer GELU MLP) score over each token's time features
    [Time2Vec(age) ‖ log1p(hop) ‖ edge_feats]. Those time features are basis-independent (consistent
    across batches), so a learned layer on them is legitimate. The nonlinearity was a clean +0.007 test
    win on wiki over a plain linear logit (sweep 2026-07-21); it does NOT overfit because it acts on the
    time features / attention weighting, not on the random codes or scorer capacity (both of which
    overfit). The pooled value is the token code itself, so nbhd_code stays in the (same-batch) code
    subspace and can be consumed by inner products downstream. All four requested token signals are
    present: the code via the co-reachability term + pooled value, and time/hop/edge via the logit.
    """

    def __init__(self, t2v_dim: int, d_ef: int):
        super().__init__()
        self.d_ef = d_ef
        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        # Attention logit = coreach_weight * (seed–token co-reachability) + attn_bias_mlp(token time feats).
        self.coreach_weight = nn.Parameter(torch.tensor(1.0))        # scalar weight on the co-reachability term
        self.attn_bias_mlp = nn.Sequential(                         # nonlinear bias from each token's time/hop/edge feats
            nn.Linear(t2v_dim + 1 + d_ef, 16), nn.GELU(), nn.Linear(16, 1))

    def _token_edge_features(self, tokens: WalkTokens, q: int, t: int) -> torch.Tensor:
        """Per-token edge features [Q, T, d_ef] aligned with the flattened bag ([Q, K, L*d_ef] →
        [Q, K*L, d_ef]); an empty [Q, T, 0] tensor when the dataset has no edge features."""
        if tokens.edge_features is not None:
            _, k, length = tokens.nodes.shape
            return tokens.edge_features.reshape(q, k, length, self.d_ef).reshape(q, t, self.d_ef)
        return tokens.nodes.new_zeros((q, t, 0), dtype=torch.float32)

    def forward(self, assoc: torch.Tensor, node_enc: torch.Tensor,
                tokens: WalkTokens) -> tuple:
        """assoc/node_enc from a (joint) NodeEncoding forward; tokens = one bag of Q queries.
        Returns (seed_code, nbhd_code), each [Q, De]: seed_code = the query node's own structural code;
        nbhd_code = the attention-pooled code of its walk neighbourhood. Cold queries (no neighbourhood
        token) get nbhd_code = 0."""
        q = tokens.nodes.shape[0]
        seed_code = NodeEncoding.gather(assoc, node_enc, tokens.seeds)     # [Q, De]  the query node's own code

        # Flatten the query's walk bag to one [Q, T] token list (padding + the seed's own slot masked out).
        token_ids, token_mask, token_hop = flatten_tokens(tokens, exclude_seed_positions=True)   # [Q, T] each
        num_tokens = token_ids.shape[1]
        token_code = NodeEncoding.gather(assoc, node_enc, token_ids)       # [Q, T, De]  each token's structural code
        token_age = tokens.ages.reshape(q, num_tokens).clamp_min(0)        # [Q, T]      cutoff − t_edge

        # Per-token time/position/edge features (basis-independent — the only inputs to a learned layer).
        age_enc = self.time_encoder(torch.log1p(token_age.to(node_enc.dtype)))            # [Q, T, t2v]
        hop_enc = torch.log1p(token_hop.clamp_min(0).to(node_enc.dtype)).unsqueeze(-1)    # [Q, T, 1]
        edge_feats = self._token_edge_features(tokens, q, num_tokens)                     # [Q, T, d_ef]
        token_time_feats = torch.cat([age_enc, hop_enc, edge_feats], dim=-1)             # [Q, T, t2v+1+d_ef]

        # Attention weight per token = seed↔token co-reachability + a learned bias from its time features.
        coreach = (seed_code.unsqueeze(1) * token_code).sum(-1)                          # [Q, T] seed·token (cosine)
        attn_logit = self.coreach_weight * coreach + self.attn_bias_mlp(token_time_feats).squeeze(-1)   # [Q, T]
        attn_logit = attn_logit.masked_fill(~token_mask, float("-inf"))
        attn_weight = torch.nan_to_num(torch.softmax(attn_logit, dim=-1), nan=0.0)       # cold row → all 0
        nbhd_code = (attn_weight.unsqueeze(-1) * token_code).sum(dim=1)                  # [Q, De] pooled neighbourhood
        return seed_code, nbhd_code


class StatelessLinkHead(nn.Module):
    """Geometry-free, stateless link head (see module docstring). Owns the batch-local NodeEncoding,
    the walk-neighbourhood encoder, and a tiny pairwise scorer over basis-invariant inner products."""

    def __init__(self, num_nodes: int, d_emb: int, n_hops: int = 3,
                 t2v_dim: int = 16, d_ef: int = 0, recency_lambda: float = 0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_ef = d_ef

        self.node_encoding = NodeEncoding(
            num_nodes=num_nodes, dim=d_emb, n_hops=n_hops, recency_lambda=recency_lambda)
        self.encoder = WalkNeighborhoodEncoder(t2v_dim=t2v_dim, d_ef=d_ef)

        # Pairwise scorer: a small MLP over the four basis-invariant inner products between u's and v's
        # (seed_code, nbhd_code). The inner products are scalars (basis-invariant), so a learned MLP on
        # them is legitimate.
        self.scorer = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate queries (seeds = v)
        in query-major order, each walked with its query's cutoff. Returns logits [B, C]."""
        assoc, node_enc = self.node_encoding(src_tokens, cand_tokens)     # joint encode both bags
        node_enc = self.node_encoding.normalize_blocks(node_enc)          # per-hop-block L2 (scoring-side)

        seed_u, nbhd_u = self.encoder(assoc, node_enc, src_tokens)       # source: [B, De] each
        seed_v, nbhd_v = self.encoder(assoc, node_enc, cand_tokens)     # candidates: [B*C, De] each

        b, de = seed_u.shape
        seed_v = seed_v.reshape(b, -1, de)                               # [B, C, De]
        nbhd_v = nbhd_v.reshape(b, -1, de)                               # [B, C, De]
        seed_u = seed_u.unsqueeze(1)                                     # [B, 1, De]
        nbhd_u = nbhd_u.unsqueeze(1)                                     # [B, 1, De]

        # Four basis-invariant inner products between u's and v's (seed, neighbourhood) codes.
        pair_inner_products = torch.stack([
            (seed_u * seed_v).sum(-1),                                   # ⟨seed_u, seed_v⟩  u–v co-reachability
            (seed_u * nbhd_v).sum(-1),                                   # ⟨seed_u, nbhd_v⟩  is u in v's neighbourhood
            (nbhd_u * seed_v).sum(-1),                                   # ⟨nbhd_u, seed_v⟩  is v in u's neighbourhood
            (nbhd_u * nbhd_v).sum(-1),                                   # ⟨nbhd_u, nbhd_v⟩  neighbourhood overlap
        ], dim=-1)                                                       # [B, C, 4]
        return self.scorer(pair_inner_products).squeeze(-1)            # [B, C]
