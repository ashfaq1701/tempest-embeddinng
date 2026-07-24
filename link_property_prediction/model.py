"""Link head — node embeddings + a residual neighbourhood encoder, in one module.

Each node has a learned embedding E[x] (self.E, an EmbeddingTable; its stored weight is free Euclidean
but every lookup is L2-normalized, so E[x] is unit-norm — Method A, plain optimizer, no geoopt).
NeighborhoodEncoder (master's residual encoder, at d_emb — no separate enc_dim) turns each query's
walk-token bag into a FREE d_emb (seed_emb, nbhd_emb): it embeds the seed E[u] and each token
[E[token] ‖ Time2Vec(age) ‖ log1p(hop) ‖ edge] through input_proj + n_layers residual FFN, then the
seed attends over the tokens to pool the neighbourhood.

Two-sided: both u and every candidate v are walked and encoded. The pair (u, v) is scored by an MLP
over the FOUR inner products of the two sides' seed and neighbourhood embeddings:
    <seed_u,seed_v>   <seed_u,nbhd_v>   <nbhd_u,seed_v>   <nbhd_u,nbhd_v>
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens, flatten_tokens


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


class NeighborhoodEncoder(nn.Module):
    """Encode a query's walk-token bag into (seed_emb, nbhd_emb), each FREE [Q, d_emb]. Master's residual
    encoder, ported to d_emb width (no separate enc_dim):

    Each per-token descriptor [E[token] ‖ Time2Vec(age) ‖ log1p(hop) ‖ edge] is embedded by input_proj
    (in_dim → d_emb) then n_layers pre-norm RESIDUAL FFN blocks. The seed is encoded the same way (its
    own E[u], age 0, hop 0, no edge), then attends over the token embeddings (scaled dot product) and
    pools them into the neighbourhood embedding. Both outputs are free Euclidean d_emb vectors. Cold
    queries (no neighbourhood token) get nbhd_emb = 0."""

    def __init__(self, d_emb: int, t2v_dim: int = 16, d_ef: int = 0,
                 n_layers: int = 2, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_ef = d_ef
        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        in_dim = d_emb + t2v_dim + 1 + d_ef                 # E[token] ‖ t2v(age) ‖ log_hop ‖ edge
        # EMBED (in_dim → d_emb; dim change, no residual), then n_layers pre-norm residual FFN blocks.
        # One encoder shared by both the tokens and the seed.
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_proj = nn.Linear(in_dim, d_emb)
        self.blocks = nn.ModuleList(ResidualFFN(d_emb, dropout, expansion) for _ in range(n_layers))
        self.attn_scale = d_emb ** -0.5

    def _encode(self, feat: torch.Tensor) -> torch.Tensor:
        """Descriptor [*, in_dim] → embedding [*, d_emb]: embed, then the residual blocks."""
        x = self.input_proj(self.input_norm(feat))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, e_seed: torch.Tensor, token_emb: torch.Tensor, ages: torch.Tensor,
                mask: torch.Tensor, positions: torch.Tensor, edge_features: torch.Tensor) -> tuple:
        """e_seed [B,d_emb] (E[u]); token_emb [B,T,d_emb] (E[token]); ages/positions [B,T]; mask [B,T]
        bool (True = real token); edge_features [B,T,d_ef]. Returns (seed_emb, nbhd_emb), each free
        [B,d_emb]; cold rows (no token) → nbhd_emb = 0."""
        # TPNet scales delta-times by log(Δt + 1) before the time encoder.
        t2v = self.time_encoder(torch.log1p(ages.clamp_min(0.0)))                       # [B,T,t2v_dim]
        log_hop = torch.log1p(positions.clamp_min(0).to(t2v.dtype)).unsqueeze(-1)       # [B,T,1] log-hop
        token_feat = torch.cat([token_emb, t2v, log_hop, edge_features], dim=-1)        # [B,T,in_dim]
        token_enc = self._encode(token_feat)                                           # [B,T,d_emb]

        # The seed AS a token: content = E[u], age 0 / hop 0 / no edge (zero-padded), SAME encoder.
        b = e_seed.shape[0]
        seed_feat = torch.cat([
            e_seed,                                        # [B, d_emb]  seed content
            e_seed.new_zeros(b, t2v.shape[-1]),            # [B, t2v_dim]  age 0
            e_seed.new_zeros(b, 1),                        # [B, 1]        hop 0
            e_seed.new_zeros(b, edge_features.shape[-1]),  # [B, d_ef]     no edge
        ], dim=-1)                                         # [B, in_dim]
        seed_emb = self._encode(seed_feat)                                             # [B, d_emb]

        # The seed attends over its token embeddings (scaled dot product) and pools them.
        scores = (seed_emb.unsqueeze(1) * token_enc).sum(-1) * self.attn_scale          # [B,T]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)             # cold row → all 0
        nbhd_emb = (weights.unsqueeze(-1) * token_enc).sum(dim=1)                       # [B, d_emb]
        return seed_emb, nbhd_emb


class EmbeddingTable(nn.Module):
    """The learned node-embedding table + a single retrieval method — the one place embeddings are read.

    KEEP-ON-SPHERE (Method A): the stored parameter is a FREE Euclidean vector, but every `lookup`
    L2-normalizes it, so everything the model sees is unit-norm (on the sphere). Backprop through
    F.normalize auto-projects the gradient onto the tangent space, giving Riemannian-style gradients
    under a plain Euclidean optimizer — no geoopt / RiemannianAdam. Pair with weight_decay = 0 on this
    table (decay + normalization interact badly: it shrinks ‖weight‖ and couples into the effective LR).
    The raw (UN-normalized) parameter is exposed as `.weight` for callers that need the whole table
    (probes, dumps) — normalize on export if unit rows are wanted."""

    def __init__(self, num_nodes: int, d_emb: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_nodes, d_emb))
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(d_emb))

    def lookup(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Unit-norm embeddings for `node_ids` [*] -> [*, d_emb] (L2-normalized in the forward pass)."""
        return F.normalize(F.embedding(node_ids, self.weight), dim=-1, eps=1e-8)


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int, t2v_dim: int = 16, d_ef: int = 0,
                 n_layers: int = 2, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.d_ef = d_ef

        # Node embeddings, link-trained; E[x] is unit-norm via the table's forward-normalization.
        self.E = EmbeddingTable(num_nodes, d_emb)

        # Master's residual encoder (ported to d_emb, no separate enc_dim): turns each query's walk-token
        # bag into free d_emb (seed_emb, nbhd_emb) via input_proj + n_layers residual FFN + attention.
        self.neighbourhood = NeighborhoodEncoder(
            d_emb=d_emb, t2v_dim=t2v_dim, d_ef=d_ef,
            n_layers=n_layers, expansion=expansion, dropout=dropout)

        # Combiner MLP over the 4 pairwise inner products of both sides' (seed, neighbourhood) embeddings.
        self.scorer = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))

    def _token_edge_features(self, tokens: WalkTokens, q: int) -> torch.Tensor:
        """Per-token edge features [Q, T, d_ef] aligned with the flattened token bag. tokens holds
        edge_features as [Q, K, L*d_ef]; reshape to [Q, K*L, d_ef]. When the dataset has no edge
        features (d_ef == 0) this is an empty [Q, T, 0] tensor (a no-op in the key concat)."""
        _, k, length = tokens.nodes.shape
        if tokens.edge_features is not None:
            return tokens.edge_features.reshape(q, k, length, self.d_ef).reshape(q, k * length, self.d_ef)
        return tokens.nodes.new_zeros((q, k * length, self.d_ef), dtype=torch.float32)

    def _project(self, tokens: WalkTokens):
        """Encode one bag of N queries into (seed_emb, nbhd_emb), each free [N, d]: the residual encoder
        embeds the seed (E[u]) and each walk token (E[token] + time/hop/edge), then the seed attends over
        the tokens to pool the neighbourhood. Cold rows (no token) -> nbhd_emb = 0."""
        e_seed = self.E.lookup(tokens.seeds)                                          # E[x]  [N, d]
        n = e_seed.shape[0]

        token_ids, token_mask, token_pos = flatten_tokens(
            tokens, exclude_seed_positions=True)
        token_ages = tokens.ages.reshape(n, -1).clamp_min(0)                          # ages read from the instance
        token_ef = self._token_edge_features(tokens, n)                              # [N, T, d_ef]
        token_emb = self.E.lookup(token_ids.clamp_min(0))                             # E[token]  [N, T, d]
        return self.neighbourhood(
            e_seed, token_emb, token_ages.to(e_seed.dtype), token_mask, token_pos, token_ef)  # (seed_emb, nbhd_emb)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in query-major order, each walked with its query's cutoff. Score = MLP over
        the FOUR pairwise inner products of both sides' seed and neighbourhood embeddings (all free d_emb
        from the residual encoder). Returns logits [B, C]."""
        seed_u, nbhd_u = self._project(src_tokens)                            # seed_u, nbhd_u  [B, d]
        seed_v, nbhd_v = self._project(cand_tokens)                           # seed_v, nbhd_v  [B*C, d]
        b, d = seed_u.shape
        c = seed_v.shape[0] // b
        seed_v = seed_v.reshape(b, c, d)                                      # [B, C, d]
        nbhd_v = nbhd_v.reshape(b, c, d)                                      # [B, C, d]
        seed_u = seed_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]
        nbhd_u = nbhd_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]

        # Four inner products between u's and v's seed / neighbourhood embeddings.
        inner_products = torch.stack([
            (seed_u * seed_v).sum(-1),                                        # ⟨seed_u, seed_v⟩  u–v affinity
            (seed_u * nbhd_v).sum(-1),                                        # ⟨seed_u, nbhd_v⟩  is u in v's nbhd
            (nbhd_u * seed_v).sum(-1),                                        # ⟨nbhd_u, seed_v⟩  is v in u's nbhd
            (nbhd_u * nbhd_v).sum(-1),                                        # ⟨nbhd_u, nbhd_v⟩  neighbourhood overlap
        ], dim=-1)                                                            # [B, C, 4]
        return self.scorer(inner_products).squeeze(-1)                        # [B, C]
