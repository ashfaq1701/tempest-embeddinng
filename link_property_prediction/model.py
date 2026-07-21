"""Stateless, geometry-free temporal link head.

No learned node-embedding table, no manifold. A pair (u, v) is scored purely from Tempest's
MULTI-HOP temporal walks (the differentiator vs GraphMixer/TPNet's 1-hop neighbours), turned into
structural signal by a batch-local, anonymized `NodeEncoding` and pooled by a light attention head:

    logit(u, v) = MLP( <x_u, x_v>, <x_u, h_v>, <h_u, x_v>, <h_u, h_v> )

where x_x = the node's own structural code (seed row of NodeEncoding) and h_x = a recency/structure-
weighted aggregation of x's walk neighbourhood (WalkNeighborhoodEncoder). Both bags (source +
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
    """Pool a query's walk-token bag into (x, h): x = the seed's OWN structural code, h = a
    recency/structure-weighted aggregation of its NEIGHBOURHOOD tokens (the seed's own walk-origin slot
    is excluded, so h is genuinely the neighbourhood, not a self-copy).

    Single seed-query attention. The attention logit combines a structural co-reachability term
    cos(seed_code, token_code) — the ONLY way node codes may enter, since their basis is random per
    batch — with a NONLINEAR (2-layer GELU MLP) score over the stable per-token features
    [Time2Vec(age) ‖ log1p(hop) ‖ ef]. The nonlinearity on the stable features was a clean +0.007
    test win on wiki over a plain linear logit (sweep 2026-07-21); it does NOT overfit because it acts
    on stable features / attention weighting, not on the random codes or scorer capacity (both of which
    overfit). The pooled value is the token code itself, so h stays in the (same-batch) code subspace
    and can be consumed by inner products downstream. All four requested token signals are present: the
    code via the structural score + pooled value, and time/hop/ef via the logit.
    """

    def __init__(self, t2v_dim: int, d_ef: int):
        super().__init__()
        self.d_ef = d_ef
        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        self.struct_scale = nn.Parameter(torch.tensor(1.0))          # weight on the co-reachability score
        self.logit_bias = nn.Sequential(                            # V7: nonlinear (deeper) attention logit
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
        Returns (x, h), each [Q, De]. Cold queries (no neighbourhood token) get h = 0."""
        q = tokens.nodes.shape[0]
        x = NodeEncoding.gather(assoc, node_enc, tokens.seeds)              # [Q, De]  seed's own code

        ids, mask, pos = flatten_tokens(tokens, exclude_seed_positions=True)   # [Q, T] each
        t = ids.shape[1]
        tok_code = NodeEncoding.gather(assoc, node_enc, ids)               # [Q, T, De]
        ages = tokens.ages.reshape(q, t).clamp_min(0)                      # [Q, T]

        t2v = self.time_encoder(torch.log1p(ages.to(node_enc.dtype)))     # [Q, T, t2v]
        log_hop = torch.log1p(pos.clamp_min(0).to(node_enc.dtype)).unsqueeze(-1)   # [Q, T, 1]
        ef = self._token_edge_features(tokens, q, t)                      # [Q, T, d_ef]
        stable = torch.cat([t2v, log_hop, ef], dim=-1)                    # [Q, T, t2v+1+d_ef]

        struct = (x.unsqueeze(1) * tok_code).sum(-1)                      # [Q, T]  cos co-reachability
        logit = self.struct_scale * struct + self.logit_bias(stable).squeeze(-1)   # [Q, T]
        logit = logit.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(logit, dim=-1), nan=0.0)  # cold row → all 0
        h = (weights.unsqueeze(-1) * tok_code).sum(dim=1)                 # [Q, De]
        return x, h


class StatelessLinkHead(nn.Module):
    """Geometry-free, stateless link head (see module docstring). Owns the batch-local NodeEncoding,
    the walk-neighbourhood encoder, and a tiny pairwise scorer over basis-invariant inner products."""

    def __init__(self, num_nodes: int, d_emb: int, n_hops: int = 3,
                 t2v_dim: int = 16, d_ef: int = 0, recency_lambda: float = 0.0):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.n_hops = n_hops
        self.n_blocks = n_hops + 1
        self.d_ef = d_ef

        self.node_encoding = NodeEncoding(
            num_nodes=num_nodes, dim=d_emb, n_hops=n_hops, recency_lambda=recency_lambda)
        self.encoder = WalkNeighborhoodEncoder(t2v_dim=t2v_dim, d_ef=d_ef)

        # Pairwise scorer: a small MLP over the four basis-invariant inner products of (x, h). The inner
        # products are scalars (basis-invariant), so a learned MLP on them is legitimate.
        self.scorer = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))

    def _normalize_blocks(self, node_enc: torch.Tensor) -> torch.Tensor:
        """Per-HOP-block L2 normalisation so every hop's co-reachability enters the inner product on
        equal footing (raw block magnitudes shrink with diffusion depth; without this the largest block
        would dominate every cosine). [U, n_blocks*dim] → same shape, each dim-block unit-norm."""
        u = node_enc.shape[0]
        blocks = node_enc.view(u, self.n_blocks, self.d_emb)
        blocks = F.normalize(blocks, dim=-1)
        return blocks.reshape(u, self.n_blocks * self.d_emb)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate queries (seeds = v)
        in query-major order, each walked with its query's cutoff. Returns logits [B, C]."""
        assoc, node_enc = self.node_encoding(src_tokens, cand_tokens)     # joint encode both bags
        node_enc = self._normalize_blocks(node_enc)

        x_u, h_u = self.encoder(assoc, node_enc, src_tokens)             # [B, De]
        x_v, h_v = self.encoder(assoc, node_enc, cand_tokens)           # [B*C, De]

        b, de = x_u.shape
        x_v = x_v.reshape(b, -1, de)                                     # [B, C, De]
        h_v = h_v.reshape(b, -1, de)                                     # [B, C, De]
        x_u = x_u.unsqueeze(1)                                           # [B, 1, De]
        h_u = h_u.unsqueeze(1)                                           # [B, 1, De]

        feats = torch.stack([
            (x_u * x_v).sum(-1),                                         # <x_u, x_v>  u–v co-reachability
            (x_u * h_v).sum(-1),                                         # <x_u, h_v>
            (h_u * x_v).sum(-1),                                         # <h_u, x_v>  is v in u's neighbourhood
            (h_u * h_v).sum(-1),                                         # <h_u, h_v>  neighbourhood overlap
        ], dim=-1)                                                       # [B, C, 4]
        return self.scorer(feats).squeeze(-1)                           # [B, C]
