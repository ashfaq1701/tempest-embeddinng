"""Link head — EUCLIDEAN node embeddings + a neighbourhood-projection channel, in one module.

Each node has a plain learned embedding E[x]. Its neighbourhood embedding P[x] = E[x] + mu_x moves the
seed by mu_x — the attention-pooled centroid of the DISPLACEMENTS (E[token] - E[x]) of x's walk tokens
(+ a TPNet Time2Vec of each token's age + log-hop + edge feats), produced by NeighborhoodProjection.

Two-sided: both u and every candidate v are walked and projected. The pair (u, v) is scored by an MLP
over the FOUR inner products of the two sides' identity (E) and neighbourhood (P) embeddings:
    <E[u],E[v]>   <E[u],P[v]>   <P[u],E[v]>   <P[u],P[v]>
The head owns self.E (a plain nn.Embedding, link-trained in Euclidean space).
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


class NeighborhoodProjection(nn.Module):
    """Single attention-pooling of a source's walk-token displacements into one vector mu_u.

    A source-conditioned query attends over the tokens; the attention scores ARE the pooling weights,
    and mu_u is the weighted centroid of the RAW token displacements:

        q_u  = W_k([ E[u] ; 0 ; 0 ; 0 ])                  [B, d_a]     (seed as a token: time/hop/edge zero-padded;
                                                                       SAME W_k as the keys, d_a = d_emb // 2)
        k_p  = W_k([ (E[token_p] - E[u]) ; Time2Vec(age_p) ; log1p(hop_p) ; edge_p ])   [B, T, d_a]
        w_p  = softmax_p( (q_u . k_p) / sqrt(d_a) )       [B, T]   (padding masked out)
        mu_u = Sum_p w_p * token_delta_p                   [B, d_emb]

    The value is the token displacement itself, so mu_u stays a genuine weighted centroid (in the span
    of u's neighbour displacements) — a strict, learnable generalisation of the fixed softmax(-lambda*age)
    weighting. Candidate-independent (never sees E[v]); cold rows (no token) -> mu_u = 0.
    """

    def __init__(self, d_emb: int, t2v_dim: int = 16, d_ef: int = 0):
        super().__init__()
        self.d_emb = d_emb
        self.d_ef = d_ef
        d_a = d_emb // 2                     # attention (query/key) dim, derived from d_emb
        self.scale = 1.0 / math.sqrt(d_a)

        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        # ONE shared projection to d_a for BOTH query and keys (a 2-layer projection overfits on wiki —
        # depth is the regression, not width). The per-token descriptor is content + time + log-hop
        # position (1 scalar) + edge feats. The QUERY is the seed run through this SAME projection with
        # the time/hop/edge components ZERO-padded (the seed has age 0, hop 0, no edge), so query and
        # keys live in one aligned subspace. The hop enters as log1p(hop) inside the descriptor (not an
        # additive PE), so it can be learned/silenced; length-agnostic.
        k_in = d_emb + t2v_dim + 1 + d_ef
        self.w_k = nn.Linear(k_in, d_a)

    def forward(self, source: torch.Tensor, token_deltas: torch.Tensor,
                ages: torch.Tensor, mask: torch.Tensor, positions: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """source [B,d_emb] (E[u]); token_deltas [B,T,d_emb] (E[token] - E[u], the displacement from the
        seed); ages [B,T]; mask [B,T] bool (True = real token); positions [B,T] int hop-from-seed
        (1=seed, 0=pad); edge_features [B,T,d_ef] per-token edge features (empty [B,T,0] when the
        dataset has none). Returns mu_u [B,d_emb] the pooled displacement; cold rows (no token) -> 0."""
        # TPNet scales delta-times by log(Δt + 1) before the time encoder.
        t2v = self.time_encoder(torch.log1p(ages.clamp_min(0.0)))             # [B,T,t2v_dim]
        log_hop = torch.log1p(positions.clamp_min(0).to(t2v.dtype)).unsqueeze(-1)   # [B,T,1] log-hop position
        keys = self.w_k(torch.cat([token_deltas, t2v, log_hop, edge_features], dim=-1))   # [B,T,d_a]
        # Query = the seed AS a token: content = source, but age 0 / hop 0 / no edge, so those three
        # components are zero-padded. Same w_k as the keys → query and keys align in one subspace.
        b = source.shape[0]
        query = self.w_k(torch.cat([
            source,                                       # [B, d_emb]  seed content
            source.new_zeros(b, t2v.shape[-1]),           # [B, t2v_dim]  age 0
            source.new_zeros(b, 1),                       # [B, 1]        hop 0
            source.new_zeros(b, edge_features.shape[-1]), # [B, d_ef]     no edge
        ], dim=-1))                                       # [B, d_a]

        scores = (query.unsqueeze(1) * keys).sum(-1) * self.scale            # [B,T]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)   # cold row -> all 0

        return (weights.unsqueeze(-1) * token_deltas).sum(dim=-2)            # [B,d_emb]


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int,
                 t2v_dim: int = 16, d_ef: int = 0):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.d_ef = d_ef

        # Plain Euclidean node embeddings, link-trained (no manifold).
        self.E = nn.Embedding(num_nodes, d_emb)
        nn.init.normal_(self.E.weight, mean=0.0, std=1.0 / math.sqrt(d_emb))

        self.neighbourhood = NeighborhoodProjection(
            d_emb=d_emb, t2v_dim=t2v_dim, d_ef=d_ef)

        # Combiner MLP over the 4 pairwise inner products of both sides' identity (E[x]) and
        # neighbourhood (P[x]) embeddings.
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
        """Project one bag of N queries. Returns (e_seed, p): e_seed [N, d] = E[seed] (the identity), and
        p [N, d] = E[seed] + mu (the seed moved by mu, the attention-pooled centroid of its walk tokens'
        displacements). Cold rows (no token) -> mu = 0 -> p = e_seed."""
        e_weight = self.E.weight
        e_seed = F.embedding(tokens.seeds, e_weight)                                  # E[x]  [N, d]
        n = e_seed.shape[0]

        token_ids, token_mask, token_pos = flatten_tokens(
            tokens, exclude_seed_positions=True)
        token_ages = tokens.ages.reshape(n, -1).clamp_min(0)                          # ages read from the instance
        token_ef = self._token_edge_features(tokens, n)                              # [N, T, d_ef]
        token_emb = F.embedding(token_ids.clamp_min(0), e_weight)                     # [N, T, d]
        token_delta = token_emb - e_seed.unsqueeze(-2)                                # [N, T, d] E[token] - E[seed]
        mu = self.neighbourhood(
            e_seed, token_delta, token_ages.to(e_seed.dtype), token_mask, token_pos, token_ef)  # [N, d]
        return e_seed, e_seed + mu                                                    # (E[x], P[x])  [N, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in query-major order, each walked with its query's cutoff. Score = MLP over
        the FOUR pairwise inner products of both sides' identity (seed = E[x]) and neighbourhood
        (nbhd = P[x]) embeddings. Returns logits [B, C]."""
        seed_u, nbhd_u = self._project(src_tokens)                            # E[u], P[u]  [B, d]
        seed_v, nbhd_v = self._project(cand_tokens)                           # E[v], P[v]  [B*C, d]
        b, d = seed_u.shape
        c = seed_v.shape[0] // b
        seed_v = seed_v.reshape(b, c, d)                                      # [B, C, d]
        nbhd_v = nbhd_v.reshape(b, c, d)                                      # [B, C, d]
        seed_u = seed_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]
        nbhd_u = nbhd_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]

        # Four inner products between u's and v's identity / neighbourhood embeddings.
        inner_products = torch.stack([
            (seed_u * seed_v).sum(-1),                                        # ⟨E[u], E[v]⟩  identity affinity
            (seed_u * nbhd_v).sum(-1),                                        # ⟨E[u], P[v]⟩  is u in v's neighbourhood
            (nbhd_u * seed_v).sum(-1),                                        # ⟨P[u], E[v]⟩  is v in u's neighbourhood
            (nbhd_u * nbhd_v).sum(-1),                                        # ⟨P[u], P[v]⟩  neighbourhood overlap
        ], dim=-1)                                                            # [B, C, 4]
        return self.scorer(inner_products).squeeze(-1)                        # [B, C]
