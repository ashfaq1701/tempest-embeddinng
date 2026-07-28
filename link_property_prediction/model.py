"""Link head — sphere node embeddings + a deep neighbourhood-projection channel, in one module.

    logit = <P[u], E[v]>                               is v near u's neighbourhood?

where P[u] = exp_{E[u]}(mu_u) pushes the source off E[u] toward mu_u — the deep pooling of u's
walk-token tangents (in the tangent space at E[u]) + a TPNet Time2Vec of each token's age, produced
by NeighborhoodProjection. One-sided: only u is walked/projected; each candidate v enters through
its static embedding E[v]. The head owns self.E (a ManifoldParameter on a geoopt.Sphere, link-
trained on it); geometry goes through self.geom (SphereManifold): dist/logmap proxy to geoopt, expmap
is ours (geoopt's expmap NaNs the gradient at the cold-row zero tangent).

(The dual-sided variant — walk every candidate too and score <P[u], P[v]> — was falsified on wiki:
at matched walks it lost to one-sided at ~8x the cost. It lives one `git revert` away; see the
"Important: revert this commit to bring back dual side walks" commit.)
"""
import math
from typing import Optional

import geoopt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens


class SphereManifold:
    """Unit-sphere geometry over geoopt.Sphere. dist and logmap PROXY straight to geoopt; expmap is
    OURS — geoopt's Sphere.expmap uses sin(‖u‖)/‖u‖ under a torch.where and leaks a NaN gradient at
    ‖u‖ = 0 (the cold-row tangent mu = 0, a node with no walk tokens — common on wiki). E is kept
    on-sphere by RiemannianAdam, so no read-time re-projection is needed."""

    def __init__(self):
        self.manifold = geoopt.Sphere()

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance arccos(⟨x, y⟩) on the sphere — geoopt.Sphere.dist. LOWER = closer."""
        return self.manifold.dist(x, y)

    def logmap(self, base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """Log map at `base` of `point` — geoopt.Sphere.logmap (its gradient is finite at coincidence)."""
        return self.manifold.logmap(base, point)

    def expmap(self, base: torch.Tensor, tangent: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Exponential map: move from `base` along `tangent`.  exp(base, u) = cos(‖u‖)·base +
        sinc(‖u‖/π)·u.  torch.sinc is a smooth primitive (no explicit /‖u‖), so ‖u‖ = 0 is an exact
        differentiable no-op — returns `base` with a FINITE gradient. (geoopt's expmap NaNs the
        gradient there.) The squared norm is clamped off 0 before the sqrt for extra safety."""
        angle = (tangent ** 2).sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()   # ‖u‖, finite grad at 0
        return torch.cos(angle) * base + torch.sinc(angle / math.pi) * tangent


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


class NeighborhoodProjection(nn.Module):
    """DISPLACED-tangent attention pooling of a source's walk-token tangents into one tangent vector
    mu_u at E[u]. Single-head.

    Generalises the old "rescale-only" centroid (mu_u = Σ_p w_p·g_p, value = the RAW tangent
    g_p = Log_{E[u]}(E[token_p]), so the only learnable per token was the scalar attention weight —
    mu_u was trapped in the conic hull of the FIXED neighbour tangents) by giving the VALUE a learned
    displacement:

        v_p = g_p + γ ⊙ ( W_v · g_p  +  Enc([node_feat_p, Time2Vec(age_p), log1p(hop_p), edge_p]) )

    a content-LINEAR reshape of the tangent (W_v) plus a DEEP displacement driven only by the STABLE
    features (Enc). Depth lives on the stable path on purpose — a 2-layer projection on the
    embedding/code path overfit on wiki; width/linear is the safe lever there. γ is a per-channel
    LayerScale init 0, so at start v == g (displacement OFF) and the module is a plain tangent
    centroid, earning capacity channel-by-channel. A seed-conditioned query (decoupled from the keys)
    attends over the tokens; mu_u = Σ_p w_p · v_p.

    Sphere validity: each value v_p (and mu_u) is a FREE R^d vector (the displacement leaves the
    tangent subspace); mu_u is projected ONCE onto the tangent space at E[u] (mu -= ⟨mu, E[u]⟩·E[u]),
    which is all exp_{E[u]} needs — the standard retraction / extrinsic-mean pattern (compute freely
    off the tangent space, project back once). Project-once retains strictly MORE information than
    per-step projection (see researches/off-tangent-intermediate-values-and-project-once.txt).
    Candidate-independent (never sees E[v]); cold rows (no token) -> weights 0 -> mu_u = 0 -> P[u] = E[u].

    (The multi-head variant — H decoupled heads + a W_o head-combine — is at commit 889c59f. On coin
    it bought only ~+0.004 val/test over this single-head displacement at 3x the head params and
    ~1.5x the compute; recover it from there if the multi-head lever is wanted again.)
    """

    def __init__(self, d_emb: int, t2v_dim: int = 16, d_ef: int = 0, d_nf: int = 0):
        super().__init__()
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.d_nf = d_nf
        self.d_a = d_emb // 2                           # attention (query/key) dim
        self.scale = 1.0 / math.sqrt(self.d_a)
        self.geom = SphereManifold()                   # stateless; used for the token log-map

        self.time_encoder = TimeEncoder(time_dim=t2v_dim)
        s_in = d_nf + t2v_dim + 1 + d_ef               # stable-feature dim: [node-feat, Time2Vec, log-hop, edge]
        desc_in = d_emb + s_in                         # descriptor: [content, node-feat, t2v, log-hop, edge]

        # Attention: DECOUPLED query (seed descriptor) and key (token descriptor).
        self.w_q = nn.Linear(desc_in, self.d_a)
        self.w_k = nn.Linear(desc_in, self.d_a)

        # Value displacement:
        #   content path — LINEAR reshape of the tangent (depth here overfits; keep it linear);
        #   stable  path — DEEP (safe: driven only by node-feat / time / hop / edge).
        self.w_v = nn.Linear(d_emb, d_emb, bias=False)
        self.disp_enc = nn.Sequential(
            nn.Linear(s_in, d_emb), nn.GELU(), nn.Linear(d_emb, d_emb))
        self.gamma = nn.Parameter(torch.zeros(d_emb))    # LayerScale, init 0 -> displacement off

    def forward(self, walk_bag: WalkTokens, emb: torch.Tensor,
                node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """walk_bag: the flattened WalkTokens bag. emb: the FULL node-embedding table [num_nodes,d_emb]
        on the sphere. node_features: the FULL static node-feature table [num_nodes, d_nf] (None if the
        dataset has none). Builds the per-token KEY descriptor [tangent ‖ nf ‖ t2v(age) ‖ log-hop ‖ edge]
        and the SEED QUERY descriptor [E[u] ‖ nf_seed ‖ 0 ‖ 0 ‖ 0] — the seed keeps its OWN node feature,
        but its time/hop/edge are zero. Returns mu_u [B,d_emb], ⊥ E[u], ‖·‖<π; cold rows -> 0."""
        node_ids = walk_bag.nodes.clamp_min(0)                                    # [B,T]  (padding → row 0)
        source = F.embedding(walk_bag.seeds, emb)                                 # E[u]  [B,d_emb]
        token_tangents = self.geom.logmap(source.unsqueeze(-2), F.embedding(node_ids, emb))  # [B,T,d_emb] ⊥ E[u]
        b, t, _ = token_tangents.shape

        mask = walk_bag.mask & ~walk_bag.seed_mask                               # [B,T]  context (non-seed real)
        ages = walk_bag.ages.clamp_min(0).to(token_tangents.dtype)               # [B,T]

        # Static node features — per token (padding zeroed) AND the seed's own (kept in the query).
        if node_features is None:
            nf_token = token_tangents.new_zeros(b, t, self.d_nf)                  # [B,T,0]
            nf_seed = source.new_zeros(b, self.d_nf)                             # [B,0]
        else:
            nf_token = F.embedding(node_ids, node_features) * walk_bag.mask.unsqueeze(-1)   # [B,T,d_nf]
            nf_seed = F.embedding(walk_bag.seeds, node_features)                            # [B,d_nf]

        # Edge features (seed/padding already zeroed on the bag); empty channel if absent.
        edge_features = walk_bag.edge_features                                    # [B,T,d_ef] or None
        if edge_features is None:
            edge_features = token_tangents.new_zeros(b, t, self.d_ef)             # [B,T,0]

        # TPNet scales delta-times by log(Δt + 1) before the time encoder.
        t2v = self.time_encoder(torch.log1p(ages))                               # [B,T,t2v_dim]
        log_hop = torch.log1p(walk_bag.positions.clamp_min(0).to(t2v.dtype)).unsqueeze(-1)  # [B,T,1]

        # Per-token STABLE features: [node-feat, Time2Vec, log-hop, edge].
        stable = torch.cat([nf_token, t2v, log_hop, edge_features], dim=-1)        # [B,T,s_in]

        # ── attention (decoupled Q/K) ──────────────────────────────────────────────────────────
        #   key   = W_k · [tangent ‖ nf_token ‖ t2v ‖ log-hop ‖ edge]
        #   query = W_q · [E[u]    ‖ nf_seed  ‖ 0   ‖ 0       ‖ 0   ]   (seed keeps its node feature)
        keys = self.w_k(torch.cat([token_tangents, stable], dim=-1))              # [B,T,d_a]
        query_zeros = source.new_zeros(b, stable.shape[-1] - self.d_nf)           # [B, t2v+1+d_ef]
        query = self.w_q(torch.cat([source, nf_seed, query_zeros], dim=-1))       # [B,d_a]
        scores = (query.unsqueeze(1) * keys).sum(-1) * self.scale                 # [B,T]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)        # [B,T]; cold row -> 0

        # ── displaced values ───────────────────────────────────────────────────────────────────
        disp = self.w_v(token_tangents) + self.disp_enc(stable)                   # [B,T,d]
        values = token_tangents + self.gamma * disp                               # [B,T,d]; γ init 0

        # ── pool -> project onto T_{E[u]} ───────────────────────────────────────────────────────
        mu = (weights.unsqueeze(-1) * values).sum(dim=-2)                         # [B,d]
        return mu - (mu * source).sum(-1, keepdim=True) * source                  # Π_{E[u]}: ⊥ E[u]


class LinkPredHead(nn.Module):
    def __init__(self, num_nodes: int, d_emb: int,
                 t2v_dim: int = 16, d_ef: int = 0,
                 node_features: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_emb = d_emb
        self.d_ef = d_ef
        self.d_nf = 0 if node_features is None else int(node_features.shape[1])
        # Static per-node feature table [num_nodes, d_nf] (dataset-derived, NOT learned). A buffer so
        # it rides model.to(device) and stays out of the optimizer; non-persistent (derivable from the
        # dataset, so kept out of checkpoints). None when the dataset has no node features.
        self.register_buffer("node_features", node_features, persistent=False)
        self.geom = SphereManifold()

        # E lives on the unit sphere: init uniformly at random on it (geoopt.Sphere.random_uniform),
        # then wrap as a ManifoldParameter so RiemannianAdam keeps it there.
        self.E = nn.Embedding(num_nodes, d_emb)
        with torch.no_grad():
            init = self.geom.manifold.random_uniform(num_nodes, d_emb)
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom.manifold)

        self.neighbourhood = NeighborhoodProjection(
            d_emb=d_emb, t2v_dim=t2v_dim, d_ef=d_ef, d_nf=self.d_nf)

        # Combiner MLP over the 4 pairwise GEODESIC DISTANCES (geoopt.Sphere.dist = arccos⟨·,·⟩) on the
        # sphere between both sides' identity (E[x]) and neighbourhood (P[x]) points. Rotation-invariant,
        # so the scorer stays sphere-faithful (no raw coordinates); LOWER distance = closer (the MLP
        # learns the sign). (Expmap NaN was fixed in SphereManifold, so geoopt.dist is safe to score on.)
        self.scorer = nn.Sequential(
            nn.Linear(4, 32), nn.GELU(), nn.Linear(32, 1))

    def _project(self, tokens: WalkTokens) -> torch.Tensor:
        """Neighbourhood projection P[x] = exp_{E[seed]}(mu) [N, d] for a bag of N queries. The
        neighbourhood takes the bag + the full E table and derives E[u] / token tangents / stable
        feats / masks itself; E[seed] is looked up separately by the caller."""
        e_seed = F.embedding(tokens.seeds, self.E.weight)                            # E[seed]  [N, d]
        mu = self.neighbourhood(tokens, self.E.weight, self.node_features)           # [N, d]
        return self.geom.expmap(e_seed, mu)                                          # P[x]  [N, d]

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Two-sided scoring. src_tokens: B source queries (seeds = u). cand_tokens: the B*C candidate
        queries (seeds = v) in query-major order, each walked with its query's cutoff. Score = MLP over
        the FOUR pairwise similarities (inner products = cosines) on the sphere between both sides'
        identity (seed = E[x]) and neighbourhood (nbhd = P[x]) points (rotation-invariant, so the
        scorer is sphere-faithful — no raw coordinates; higher = closer). Returns logits [B, C]."""
        seed_u = F.embedding(src_tokens.seeds, self.E.weight)                 # E[u]  [B, d]
        nbhd_u = self._project(src_tokens)                                    # P[u]  [B, d]
        seed_v = F.embedding(cand_tokens.seeds, self.E.weight)                # E[v]  [B*C, d]
        nbhd_v = self._project(cand_tokens)                                   # P[v]  [B*C, d]
        b, d = seed_u.shape
        c = seed_v.shape[0] // b
        seed_v = seed_v.reshape(b, c, d)                                      # [B, C, d]
        nbhd_v = nbhd_v.reshape(b, c, d)                                      # [B, C, d]
        seed_u = seed_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]
        nbhd_u = nbhd_u.unsqueeze(1).expand(b, c, d)                          # [B, C, d]

        # Four sphere geodesic distances (geoopt.Sphere.dist = arccos⟨·,·⟩) between u's and v's
        # identity / neighbourhood points.
        distances = torch.stack([
            self.geom.dist(seed_u, seed_v),                                 # d(E[u], E[v])  identity affinity
            self.geom.dist(seed_u, nbhd_v),                                 # d(E[u], P[v])  is u in v's neighbourhood
            self.geom.dist(nbhd_u, seed_v),                                 # d(P[u], E[v])  is v in u's neighbourhood
            self.geom.dist(nbhd_u, nbhd_v),                                 # d(P[u], P[v])  neighbourhood overlap
        ], dim=-1)                                                            # [B, C, 4]
        return self.scorer(distances).squeeze(-1)                            # [B, C]
