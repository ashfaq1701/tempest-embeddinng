"""Stateless, geometry-free temporal link head.

No learned node-embedding table, no manifold. A pair (u, v) is scored purely from Tempest's
MULTI-HOP temporal walks (the differentiator vs GraphMixer/TPNet's 1-hop neighbours):

    logit(u, v) = MergeLayer( [seed_u ‖ nbhd_u ‖ seed_v ‖ nbhd_v]  ‖  the four ⟨·,·⟩ inner products )

where seed_x / nbhd_x are the node's own / neighbourhood-pooled LEARNED embeddings from
WalkNeighborhoodEncoder, and the scorer is a pre-norm MLP over both sides' embeddings concatenated
(TPNet's decoder) plus the four inner products (the affinity prior). Both bags (source + candidate)
are jointly encoded so they share one batch-local graph.

FIXED BASIS: NodeEncoding uses a PERMANENT per-node-id random fingerprint (drawn once), so a node's
coordinates are stable across batches. Unlike the anonymized fresh-basis variant (where the basis
rotated each batch and codes were usable ONLY via inner products), the fixed basis lets the codes be
MLP-usable and carry cross-batch recurrence — so WalkNeighborhoodEncoder embeds the FULL per-token
descriptor [node_enc ‖ time ‖ hop ‖ edge ‖ node feats] with a learned encoder (projection + residual
FFN blocks; the fixed-basis unlock), then attends the seed over the token embeddings to pool the nbhd.
"""
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .walk_tokens import WalkTokens, flatten_tokens


class NodeEncoding(nn.Module):
    """Stateless, batch-local node embedding — a drop-in replacement for a learned E table.

    Per batch, builds the local walk graph from the WalkTokens, then encodes every node that appears
    (walk nodes + query seeds) by a MULTI-HOP diffusion of random features over that graph:

        A        = batch-local recency-weighted (undirected) adjacency [U, U] from consecutive walk steps
        Â        = D^-1 A                                    (random-walk transition)
        X0       = per-id fixed fingerprints [U, dim]        (rows gathered from x0_table by global id)
        node_enc = [ X0 ; Â X0 ; Â² X0 ; … ; Âⁿ X0 ]         [U, (n_hops+1)·dim]

    By Johnson–Lindenstrauss, <Xk[i], Xk[j]> ≈ <Âᵏ[i,:], Âᵏ[j,:]> = the k-hop co-reachability of nodes
    i and j — so the encoding carries multi-hop structure at fixed width with no learned state. The basis
    is FIXED (one permanent random row per global node id, drawn once), so a node's coordinates are stable
    across batches — usable by learned MLPs, and carrying cross-batch recurrence. Nodes that appear with
    no edges get [X0, 0, …, 0].

    forward(tokens) -> (assoc, node_enc):
        assoc     [num_nodes] int64  global id -> local row in [0, U); -1 for nodes absent this batch.
        node_enc  [U, (n_hops+1)·dim] float  one fixed-width row per present node.
    Use NodeEncoding.gather(assoc, node_enc, ids) to look up any id tensor (tokens / seeds / …).
    """

    def __init__(self, num_nodes: int, dim: int, n_hops: int,
                 recency_lambda: float = 0.0, undirected: bool = True, basis_seed: int = 0):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.n_hops = n_hops
        self.recency_lambda = recency_lambda
        self.undirected = undirected
        # Permanent per-node-id random basis: one fixed row per node, drawn once (seeded). A node's code
        # is therefore stable across batches — the coordinates mean the same thing every batch, so the
        # codes are usable by learned MLPs and carry cross-batch recurrence. Non-learned buffer.
        gen = torch.Generator().manual_seed(basis_seed)
        self.register_buffer("x0_table", torch.randn(num_nodes, dim, generator=gen))

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

        # 3. base features — each present node's PERMANENT fingerprint (gathered by global id) — then
        #    multi-hop diffusion Â X over the (joint) batch-local graph.
        x0 = self.x0_table[present]                                                     # [U, dim] per-id fingerprint
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


class ResidualFFN(nn.Module):
    """Pre-norm gated residual block: x + γ ⊙ FFN(LayerNorm(x)), where FFN = Linear(d_h → e·d_h)
    → GELU → Linear(e·d_h → d_h) → Dropout. The trunk is a pure identity — nothing operates on x
    itself; the branch is normalized on entry, so blocks stack without scale drift. γ (LayerScale,
    per-channel, init ε) starts every block as a near-no-op — the stack begins ≈ the stem's linear
    encoding and earns its FFN capacity channel-by-channel; trained |γ| doubles as a per-channel
    utilization probe. Biases dropped in the FFN: LayerNorm's affine already supplies the input
    offset, and the residual supplies the output offset."""

    def __init__(self, d_h: int, dropout: float, expansion: int = 2, gamma_init: float = 1e-2):
        super().__init__()
        self.norm = nn.LayerNorm(d_h)
        self.ffn = nn.Sequential(
            nn.Linear(d_h, expansion * d_h, bias=False),
            nn.GELU(),
            nn.Linear(expansion * d_h, d_h, bias=False),
            nn.Dropout(dropout),
        )
        self.gamma = nn.Parameter(torch.full((d_h,), float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.ffn(self.norm(x))


class WalkNeighborhoodEncoder(nn.Module):
    """Encode a query's walk-token bag into (seed_emb, nbhd_emb), each [Q, enc_dim].

    Under the FIXED basis a token's node_enc is a STABLE feature, so — unlike the fresh-basis version,
    which could only take raw inner products of the codes — the FULL per-token descriptor
    [node_enc ‖ Time2Vec(age) ‖ log1p(hop) ‖ edge_feat ‖ node_feat] is embedded by a learned encoder
    (this is the fixed-basis unlock): a projection to enc_dim, then n_layers pre-norm RESIDUAL FFN
    blocks. The seed gets its own embedding the same way (age 0, hop 0, no edge; its own node feature),
    then attends over the token embeddings (scaled dot product) and pools them into the neighbourhood
    embedding. Cold queries (no neighbourhood token) get nbhd_emb = 0.
    """

    def __init__(self, d_code: int, t2v_dim: int, d_ef: int, d_nf: int,
                 enc_dim: int = 64, n_layers: int = 2, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_ef = d_ef
        self.d_nf = d_nf
        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        in_dim = d_code + t2v_dim + 1 + d_ef + d_nf         # node_enc ‖ t2v(age) ‖ log_hop ‖ edge ‖ node feats
        # EMBED (in_dim → enc_dim; dim change, no residual), then n_layers pre-norm residual FFN blocks.
        # One encoder shared by both the tokens and the seed.
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, enc_dim)
        self.blocks = nn.ModuleList(ResidualFFN(enc_dim, dropout, expansion) for _ in range(n_layers))
        self.attn_scale = enc_dim ** -0.5

    def _encode(self, feat: torch.Tensor) -> torch.Tensor:
        """Token descriptor [*, in_dim] → embedding [*, enc_dim]: embed, then the residual blocks."""
        x = self.input_proj(self.input_norm(feat))
        for block in self.blocks:
            x = block(x)
        return x

    def _token_edge_features(self, tokens: WalkTokens, q: int, t: int) -> torch.Tensor:
        """Per-token edge features [Q, T, d_ef] aligned with the flattened bag; empty [Q,T,0] if absent."""
        if tokens.edge_features is not None:
            _, k, length = tokens.nodes.shape
            return tokens.edge_features.reshape(q, k, length, self.d_ef).reshape(q, t, self.d_ef)
        return tokens.nodes.new_zeros((q, t, 0), dtype=torch.float32)

    def _token_node_features(self, tokens: WalkTokens, q: int, t: int) -> torch.Tensor:
        """Per-token static node features [Q, T, d_nf] aligned with the flattened bag; empty [Q,T,0]
        when the dataset has no node features."""
        if tokens.node_features is not None:
            _, k, length = tokens.nodes.shape
            return tokens.node_features.reshape(q, k, length, self.d_nf).reshape(q, t, self.d_nf)
        return tokens.nodes.new_zeros((q, t, 0), dtype=torch.float32)

    def _seed_node_features(self, tokens: WalkTokens, q: int) -> torch.Tensor:
        """The seed's own static node feature [Q, d_nf]; empty [Q, 0] when the dataset has none."""
        if tokens.seed_node_features is not None:
            return tokens.seed_node_features
        return tokens.nodes.new_zeros((q, self.d_nf), dtype=torch.float32)

    def forward(self, assoc: torch.Tensor, node_enc: torch.Tensor,
                tokens: WalkTokens) -> tuple:
        """assoc/node_enc from a (joint) NodeEncoding forward; tokens = one bag of Q queries.
        Returns (seed_emb, nbhd_emb), each [Q, enc_dim]."""
        q = tokens.nodes.shape[0]
        dt = node_enc.dtype

        # Each neighbourhood token → an embedding (node_enc is now a legal MLP input under the fixed basis).
        token_ids, token_mask, token_hop = flatten_tokens(tokens, exclude_seed_positions=True)   # [Q, T]
        t = token_ids.shape[1]
        token_age = tokens.ages.reshape(q, t).clamp_min(0)
        token_feat = torch.cat([
            NodeEncoding.gather(assoc, node_enc, token_ids),                         # node_enc(token) [Q,T,d_code]
            self.time_encoder(torch.log1p(token_age.to(dt))),                        # Time2Vec(age)   [Q,T,t2v]
            torch.log1p(token_hop.clamp_min(0).to(dt)).unsqueeze(-1),                # log1p(hop)      [Q,T,1]
            self._token_edge_features(tokens, q, t),                                 # edge feats      [Q,T,d_ef]
            self._token_node_features(tokens, q, t),                                 # node feats      [Q,T,d_nf]
        ], dim=-1)
        token_emb = self._encode(token_feat)                                         # [Q, T, enc_dim]

        # The seed → its own embedding (age 0, hop 0, no edge; its own node feature).
        seed_feat = torch.cat([
            NodeEncoding.gather(assoc, node_enc, tokens.seeds),                      # node_enc(seed)  [Q,d_code]
            self.time_encoder(node_enc.new_zeros(q, 1)).squeeze(1),                  # Time2Vec(0)     [Q,t2v]
            node_enc.new_zeros(q, 1),                                                # log1p(0)        [Q,1]
            node_enc.new_zeros(q, self.d_ef),                                        # no edge         [Q,d_ef]
            self._seed_node_features(tokens, q),                                     # node feats      [Q,d_nf]
        ], dim=-1)
        seed_emb = self._encode(seed_feat)                                           # [Q, enc_dim]

        # The seed attends over its token embeddings (scaled dot product) and pools them.
        logit = (seed_emb.unsqueeze(1) * token_emb).sum(-1) * self.attn_scale        # [Q, T]  seed · token
        logit = logit.masked_fill(~token_mask, float("-inf"))
        weight = torch.nan_to_num(torch.softmax(logit, dim=-1), nan=0.0)             # cold row → all 0
        nbhd_emb = (weight.unsqueeze(-1) * token_emb).sum(dim=1)                     # [Q, enc_dim]
        return seed_emb, nbhd_emb


class StatelessLinkHead(nn.Module):
    """Geometry-free, stateless link head (see module docstring). Owns the batch-local NodeEncoding,
    the walk-neighbourhood encoder, and a tiny pairwise scorer over basis-invariant inner products."""

    def __init__(self, num_nodes: int, d_emb: int, n_hops: int = 3,
                 t2v_dim: int = 16, d_ef: int = 0, d_nf: int = 0, recency_lambda: float = 0.0,
                 enc_dim: int = 64, n_layers: int = 2, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_ef = d_ef
        self.d_nf = d_nf

        self.node_encoding = NodeEncoding(
            num_nodes=num_nodes, dim=d_emb, n_hops=n_hops, recency_lambda=recency_lambda)
        self.encoder = WalkNeighborhoodEncoder(
            d_code=(n_hops + 1) * d_emb, t2v_dim=t2v_dim, d_ef=d_ef, d_nf=d_nf,
            enc_dim=enc_dim, n_layers=n_layers, expansion=expansion, dropout=dropout)

        # MergeLayer scorer (TPNet-style, adapted): a pre-norm MLP over the CONCAT of u's and v's learned
        # (seed, neighbourhood) embeddings, PLUS the four inner products. The concat gives expressiveness;
        # the inner products give the affinity/co-reachability prior for free (we keep seed/nbhd separate,
        # so unlike TPNet's single fused embedding the pairwise signal isn't pre-baked). LayerNorm+dropout
        # harmonise the mixed-scale inputs and regularise.
        scorer_in = 4 * enc_dim + 4                                      # [seed_u‖nbhd_u‖seed_v‖nbhd_v] + 4 ⟨·,·⟩
        self.scorer = nn.Sequential(
            nn.LayerNorm(scorer_in),
            nn.Linear(scorer_in, enc_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(enc_dim, 1))

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate queries (seeds = v)
        in query-major order, each walked with its query's cutoff. Returns logits [B, C]."""
        assoc, node_enc = self.node_encoding(src_tokens, cand_tokens)     # joint encode both bags

        seed_u, nbhd_u = self.encoder(assoc, node_enc, src_tokens)       # source: [B, enc_dim] each
        seed_v, nbhd_v = self.encoder(assoc, node_enc, cand_tokens)     # candidates: [B*C, enc_dim] each

        b, e = seed_u.shape
        c = seed_v.shape[0] // b
        seed_v = seed_v.reshape(b, c, e)                                 # [B, C, enc_dim]
        nbhd_v = nbhd_v.reshape(b, c, e)                                 # [B, C, enc_dim]
        seed_u = seed_u.unsqueeze(1).expand(b, c, e)                     # [B, C, enc_dim]
        nbhd_u = nbhd_u.unsqueeze(1).expand(b, c, e)                     # [B, C, enc_dim]

        # Four inner products between u's and v's learned (seed, neighbourhood) embeddings.
        inner_products = torch.stack([
            (seed_u * seed_v).sum(-1),                                   # ⟨seed_u, seed_v⟩  u–v affinity
            (seed_u * nbhd_v).sum(-1),                                   # ⟨seed_u, nbhd_v⟩  is u in v's neighbourhood
            (nbhd_u * seed_v).sum(-1),                                   # ⟨nbhd_u, seed_v⟩  is v in u's neighbourhood
            (nbhd_u * nbhd_v).sum(-1),                                   # ⟨nbhd_u, nbhd_v⟩  neighbourhood overlap
        ], dim=-1)                                                       # [B, C, 4]

        # MergeLayer: concat both sides' embeddings + the inner products → MLP → scalar.
        feats = torch.cat([seed_u, nbhd_u, seed_v, nbhd_v, inner_products], dim=-1)   # [B, C, 4*enc_dim+4]
        return self.scorer(feats).squeeze(-1)                          # [B, C]
