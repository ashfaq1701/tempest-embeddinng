"""Strict-causal training + eval loop.

Single Trainer class. Per-batch ordering:

  TRAINING:
    1. walks = walk_gen.walks_for_nodes(seeds)       ← pre-ingest state
       seeds = unique(B_src ∪ B_tgt)
    2. L_align = alignment_loss(walks, τ)            ← InfoNCE scalar
    3. neg = neg_sampler.sample(batch)               ← pre-observe state
    4. logits = link_head(E[u].detach(), E[v].detach()) for pos + neg
       L_bce = BCE(logits, [1×B, 0×B*K])
    5. L_total = L_align + L_bce
    6. optimizer.zero_grad(set_to_none=True); L_total.backward(); optimizer.step()
    7. neg_sampler.observe(B_src, B_tgt)             ← post-scoring
    8. walk_gen.add_edges(B_src, B_tgt, B_ts, B_ef)  ← post-scoring, last

  EVAL (within torch.no_grad()):
    1. neg_dst_list = tgb_neg_sampler.sample(batch)  ← per-positive negs
    2. Score positives and negatives via link_head(E[u], E[v]).
       Undirected eval: average forward(u, v) + forward(v, u).
    3. evaluator.score_to_metric(pos, neg) per positive (TGB MRR).
    4. walk_gen.add_edges(batch)                     ← Tempest state
                                                       carries forward.
    NOTE: reservoir not updated, walks not sampled at eval.
          Model parameters frozen.

Epoch boundary:
  - walk_gen.reset()
  - neg_sampler.reset() (if Historical)
  - Model parameters and optimiser state are NOT reset.

Final test eval:
  - walk_gen.reset() then re-ingest training edges so Tempest reflects
    "all of training" before val/test scoring begins.

Early stop:
  - Snapshot best-val model + projection + head state_dicts.
  - Restore before final eval.
  - Optimiser state not snapshotted (not needed after training stops).
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .data import Batch
from .evaluator import Evaluator
from .losses import alignment_loss
from .model import (
    CrossAttentionLinkHead,
    EmbeddingTable,
    HybridLinkHead,
    LinkHead,
    ProjectionHead,
)
from .negatives import (
    HistoricalNegativeSampler,
    NegativeSampler,
    UniformNegativeSampler,
)
from .utils import make_lr_lambda
from .walk_encoder import AttentionWalkEncoder, WalkEncoder, lookup_h_seed
from .walks import WalkGenerator, slice_walks_by_seeds


@dataclass
class TrainerConfig:
    # Dataset-derived (passed in by train.py).
    num_nodes: int
    is_directed: bool
    is_bipartite: bool
    dst_pool: np.ndarray
    t_train_span: float
    d_node_feat: Optional[int] = None
    d_edge_feat: Optional[int] = None       # dataset's per-edge feature
                                            # dim; None if absent. Only
                                            # consulted by the walk
                                            # encoder when enabled.

    # Model.
    d_emb: int = 128
    d_proj: int = 128

    # Loss-formulation.
    tau: float = 0.5            # InfoNCE temperature
    beta_time: float = 1.0      # hop/time weight exponent
    num_align_negatives: int = 128    # Sampled negatives per seed in
                                      # the InfoNCE partition function.
                                      # Frequency-weighted (count^0.75)
                                      # from the pool's unique nodes.
                                      # 128 chosen from the wiki K sweep
                                      # (3 seeds × 50 ep): knee of the
                                      # diminishing-returns curve — gains
                                      # ~98% of K=512's test MRR at ~2.6×
                                      # less compute and ~half the val std.
                                      # Also the largest K that fits in
                                      # ~7 GB at comment-scale NK≈15K on
                                      # an 8 GB GPU (K=256+ OOMs there).

    # Walks.
    num_walks_per_node: int = 5
    max_walk_len: int = 20
    walk_bias: str = "ExponentialWeight"
    start_bias: str = "Uniform"

    # Walk encoder for the LINK HEAD only (feature-flagged; default
    # OFF). When enabled, h_seed (encoder output) replaces E[seed] at
    # the link head. InfoNCE alignment_loss is UNCHANGED and continues
    # to operate on E + projection heads. Encoder is trained ONLY by
    # BCE; E lookups inside the encoder are detached so BCE does not
    # reach E via the encoder.
    use_walk_encoder: bool = False
    encoder_arch: str = "gru"       # "gru" (default) or "attn"
    encoder_n_heads: int = 4        # attn-only: # of attention heads
    encoder_n_layers: int = 1       # attn-only: # of transformer layers
    encoder_exclude_seed: bool = False  # attn-only: drop E[seed] from
                                        # both the last-edge tgt slot
                                        # (replaced by a [SEED] marker)
                                        # AND the final MLP_seed concat.
                                        # Makes h_seed purely
                                        # neighbourhood-derived.
    link_head_type: str = "standard"   # "standard" -> link_head(e_u, e_v)
                                       #   when encoder OFF, or
                                       #   link_head(h_u, h_v) when ON.
                                       # "hybrid"   -> link_head accepts
                                       #   concat([E[v].detach(), h_v])
                                       #   per side; ONLY valid when
                                       #   encoder is ON. Tests whether
                                       #   augmenting h with E at the
                                       #   link head helps vs replacing.
                                       # "cross_attn" -> cross-attention
                                       #   link head over per-seed
                                       #   walk-token banks. Encoder
                                       #   must be ON and arch='attn'
                                       #   (token bank is exposed only
                                       #   by AttentionWalkEncoder).
    link_head_n_heads: int = 4         # cross_attn-only: # attn heads
    d_te: int = 32           # time2vec dim
    d_he: int = 16           # hop emb dim
    d_edge: int = 128        # per-edge representation dim
    d_walk: int = 128        # GRU hidden / per-walk representation dim

    # Negatives (training).
    num_neg_per_pos: int = 10
    hist_neg_ratio: float = 0.5
    reservoir_size: int = 32

    # Optimisation.
    lr: float = 1e-3            # peak LR (after warmup). Validated on
                                # wiki bs=200 seed-42 sampled-neg K=64:
                                # lr=1e-3 → val 0.4454 vs lr=1e-2 → 0.4301.
                                # Noisier K=64 gradients prefer smaller steps.
    lr_min: float = 1e-5        # cosine decay floor (~peak/1000; matches
                                # contrastive-SSL cosine-to-near-0 norm).
    warmup_fraction: float = 0.05
    warmup_steps_cap: int = 500
    decay_horizon_epochs: int = 50  # cosine reaches lr_min at this
                                    # epoch count. SEPARATE from
                                    # num_epochs — short runs stay
                                    # near peak; full decay is hit
                                    # only at num_epochs = horizon.
    weight_decay: float = 1e-4
    num_epochs: int = 50
    early_stop_patience: int = 0
    max_grad_norm: Optional[float] = None   # if set, clip global grad
                                            # norm at this value before
                                            # optimizer.step(). Required
                                            # for stable training with
                                            # the attention encoder
                                            # (transformers diverge
                                            # without clipping at lr=1e-2);
                                            # harmless for other configs.

    # System.
    seed: int = 42
    use_gpu: bool = False
    use_gpu_tempest: bool = False    # independent from use_gpu

    # Eval.
    monitor_sample_pct: float = 1.0
    skip_final_full_eval: bool = False


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        node_feat: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
        )

        if node_feat is not None:
            assert config.d_node_feat is not None, (
                "node_feat passed but config.d_node_feat is None"
            )
            assert node_feat.shape == (config.num_nodes, config.d_node_feat), (
                f"node_feat shape {node_feat.shape} != "
                f"({config.num_nodes}, {config.d_node_feat})"
            )
            self.node_feat = torch.from_numpy(node_feat).float().to(self.device)
        else:
            self.node_feat = None

        # Model.
        self.embedding_table = EmbeddingTable(
            num_nodes=config.num_nodes,
            d_emb=config.d_emb,
        ).to(self.device)
        self.p_target = ProjectionHead(
            d_emb=config.d_emb,
            d_proj=config.d_proj,
            d_node_feat=config.d_node_feat,
        ).to(self.device)
        self.p_context = ProjectionHead(
            d_emb=config.d_emb,
            d_proj=config.d_proj,
            d_node_feat=config.d_node_feat,
        ).to(self.device)
        # Link head dispatch.
        if config.link_head_type == "standard":
            self.link_head: nn.Module = LinkHead(d_emb=config.d_emb).to(self.device)
        elif config.link_head_type == "hybrid":
            if not config.use_walk_encoder:
                raise ValueError(
                    "link_head_type='hybrid' requires use_walk_encoder=True "
                    "(hybrid head consumes BOTH E[v] and h_v per side)."
                )
            self.link_head = HybridLinkHead(d_emb=config.d_emb).to(self.device)
        elif config.link_head_type == "cross_attn":
            if not config.use_walk_encoder:
                raise ValueError(
                    "link_head_type='cross_attn' requires use_walk_encoder=True"
                )
            if config.encoder_arch != "attn":
                raise ValueError(
                    "link_head_type='cross_attn' requires encoder_arch='attn' "
                    "(token bank is only exposed by AttentionWalkEncoder)."
                )
            self.link_head = CrossAttentionLinkHead(
                d_emb=config.d_emb, n_heads=config.link_head_n_heads,
            ).to(self.device)
        else:
            raise ValueError(
                f"Unknown link_head_type: {config.link_head_type!r} "
                f"(expected 'standard', 'hybrid', or 'cross_attn')."
            )

        # Walk sampler.
        self.walk_gen = WalkGenerator(
            is_directed=config.is_directed,
            use_gpu=config.use_gpu_tempest,
            walk_bias=config.walk_bias,
            start_bias=config.start_bias,
            max_walk_len=config.max_walk_len,
            num_walks_per_node=config.num_walks_per_node,
        )

        # Optional walk encoder (link-head-only path; default OFF).
        # When enabled, h_seed (encoder output) replaces E[seed] at
        # the link head only. InfoNCE alignment_loss is unchanged.
        if config.use_walk_encoder:
            d_ef = config.d_edge_feat if config.d_edge_feat is not None else 0
            if config.encoder_arch == "gru":
                if config.encoder_exclude_seed:
                    raise ValueError(
                        "encoder_exclude_seed is only supported by "
                        "encoder_arch='attn'."
                    )
                self.walk_encoder: Optional[nn.Module] = WalkEncoder(
                    embedding_table=self.embedding_table,
                    d_emb=config.d_emb,
                    d_ef=d_ef,
                    d_te=config.d_te,
                    d_he=config.d_he,
                    d_edge=config.d_edge,
                    d_walk=config.d_walk,
                    max_walk_len=config.max_walk_len,
                ).to(self.device)
            elif config.encoder_arch == "attn":
                self.walk_encoder = AttentionWalkEncoder(
                    embedding_table=self.embedding_table,
                    d_emb=config.d_emb,
                    d_ef=d_ef,
                    d_te=config.d_te,
                    d_he=config.d_he,
                    d_edge=config.d_edge,
                    d_walk=config.d_walk,
                    max_walk_len=config.max_walk_len,
                    n_heads=config.encoder_n_heads,
                    n_layers=config.encoder_n_layers,
                    exclude_seed=config.encoder_exclude_seed,
                ).to(self.device)
            else:
                raise ValueError(
                    f"Unknown encoder_arch: {config.encoder_arch!r} "
                    f"(expected 'gru' or 'attn')."
                )
        else:
            self.walk_encoder = None

        # Negative samplers (training).
        if config.hist_neg_ratio > 0:
            self.neg_sampler_train: NegativeSampler = HistoricalNegativeSampler(
                num_nodes=config.num_nodes,
                num_neg_per_pos=config.num_neg_per_pos,
                hist_ratio=config.hist_neg_ratio,
                reservoir_size=config.reservoir_size,
                dst_pool=config.dst_pool,
                seed=config.seed,
            )
        else:
            self.neg_sampler_train = UniformNegativeSampler(
                num_neg_per_pos=config.num_neg_per_pos,
                dst_pool=config.dst_pool,
                seed=config.seed,
            )

        # Single optimiser over all trainable parameters. The decoupling
        # between E-training (InfoNCE alignment) and link-head training
        # (BCE) comes from the .detach() at the BCE call site, NOT from
        # separate optimisers.
        params = (
            list(self.embedding_table.parameters())
            + list(self.p_target.parameters())
            + list(self.p_context.parameters())
            + list(self.link_head.parameters())
        )
        if self.walk_encoder is not None:
            params += list(self.walk_encoder.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR scheduler is set up at the start of train() once we know
        # batches_per_epoch (which depends on the train_batches_factory).
        self.lr_scheduler: Optional[LambdaLR] = None
        self._global_step = 0

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _setup_lr_scheduler(self, batches_per_epoch: int) -> None:
        """Build a cosine-decay + linear-warmup LR schedule.

        W = min(warmup_fraction * decay_steps, warmup_steps_cap)
        T = decay_horizon_epochs * batches_per_epoch

        T is based on decay_horizon_epochs (a property of the
        schedule), NOT num_epochs. This decouples short anchor runs
        (5 ep — stays near peak) from full runs (50 ep — completes
        decay).
        """
        peak_lr = float(self.config.lr)
        lr_min = float(self.config.lr_min)

        decay_steps = self.config.decay_horizon_epochs * max(batches_per_epoch, 1)
        warmup_steps = min(
            int(self.config.warmup_fraction * decay_steps),
            self.config.warmup_steps_cap,
        )
        warmup_steps = max(warmup_steps, 1)

        # Set optimizer base lr to peak so the lambda returns a [0, 1]
        # scaling factor relative to peak.
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = peak_lr
            param_group["initial_lr"] = peak_lr

        lr_min_ratio = lr_min / peak_lr if peak_lr > 0 else 0.0
        lr_lambda = make_lr_lambda(warmup_steps, decay_steps, lr_min_ratio)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self._global_step = 0
        print(
            f"  LR schedule: peak={peak_lr:.2e}, min={lr_min:.2e}, "
            f"warmup={warmup_steps} steps, "
            f"decay_horizon={decay_steps} steps "
            f"({self.config.decay_horizon_epochs} epochs × "
            f"{batches_per_epoch} batches)"
        )

    # LinkHead.pair_feats has 6 * d_emb columns and 4 bytes/float, so
    # a batch of N pairs allocates N * 6 * d_emb * 4 bytes at the
    # concat. At 1000 TGB negs/pos and bs=2000, N reaches 2M, which
    # OOMs an 8 GB GPU at d_emb=128. Chunk the score so per-call
    # allocations stay under a fixed budget.
    #
    # Note on the bilinear: nn.Bilinear(d, d, 1) materialises a
    # [chunk, d, d] intermediate during forward. For HybridLinkHead
    # (d_input = 2*d_emb = 256), 50k * 256 * 256 * 4 B = 13 GB →
    # OOM. 10k keeps the peak at ~2.6 GB even for the hybrid head.
    #
    # CrossAttentionLinkHead's MultiheadAttention packs q+k+v as
    # [chunk, T, 3*d] = ~chunk*95*384*4 B which is 730 MB at 5k.
    # Train accumulates multiple chunks' activations before backward
    # → must chunk small (2k) to fit on 8 GB GPU.
    _EVAL_SCORE_CHUNK = 2_000

    def _score_pairs(
        self,
        u_ids: torch.Tensor,
        v_ids: torch.Tensor,
        h_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Score [P] (u, v) pairs through link_head. Inputs come from
        the walk encoder's per-batch h_cache (a dict carrying h_seed
        and, for cross_attn, also tokens + token_mask) when the encoder
        is on, else from raw E. Undirected eval symmetrises by
        averaging forward(u, v) + forward(v, u). Used at eval (no_grad
        context); training has its own inline path with the encoder /
        detach branching.

        Chunks internally to bound peak GPU memory regardless of P.
        """
        cross_attn = (
            h_cache is not None
            and self.config.link_head_type == "cross_attn"
        )

        def _lookup(ids: torch.Tensor) -> torch.Tensor:
            """Standard / hybrid path: return one tensor per side, then
            link_head(e_u, e_v). Not used in cross_attn (which needs
            per-pair token banks)."""
            if h_cache is not None:
                h_seed = h_cache["h_seed"]
                seeds_sorted = h_cache["seeds_sorted"]
                h = lookup_h_seed(h_seed, seeds_sorted, ids)
                if self.config.link_head_type == "hybrid":
                    e = self.embedding_table(ids)
                    return torch.cat([e, h], dim=-1)
                return h
            return self.embedding_table(ids)

        def _score_chunk_cross_attn(u_chunk, v_chunk):
            h_seed = h_cache["h_seed"]
            seeds_sorted = h_cache["seeds_sorted"]
            tokens = h_cache["tokens"]
            token_mask = h_cache["token_mask"]
            h_u = lookup_h_seed(h_seed, seeds_sorted, u_chunk)
            h_v = lookup_h_seed(h_seed, seeds_sorted, v_chunk)
            u_rows = torch.searchsorted(seeds_sorted, u_chunk)
            v_rows = torch.searchsorted(seeds_sorted, v_chunk)
            u_tok = tokens[u_rows]
            u_msk = token_mask[u_rows]
            v_tok = tokens[v_rows]
            v_msk = token_mask[v_rows]
            logits = self.link_head(h_u, h_v, u_tok, u_msk, v_tok, v_msk)
            if not self.config.is_directed:
                logits = 0.5 * (
                    logits
                    + self.link_head(h_v, h_u, v_tok, v_msk, u_tok, u_msk)
                )
            return logits

        def _score_chunk_pointwise(u_chunk, v_chunk):
            e_u = _lookup(u_chunk)
            e_v = _lookup(v_chunk)
            logits = self.link_head(e_u, e_v)
            if not self.config.is_directed:
                logits = 0.5 * (logits + self.link_head(e_v, e_u))
            return logits

        score_chunk = (
            _score_chunk_cross_attn if cross_attn else _score_chunk_pointwise
        )

        P = u_ids.shape[0]
        if P <= self._EVAL_SCORE_CHUNK:
            return score_chunk(u_ids, v_ids)

        out_parts = []
        for start in range(0, P, self._EVAL_SCORE_CHUNK):
            end = min(start + self._EVAL_SCORE_CHUNK, P)
            out_parts.append(score_chunk(u_ids[start:end], v_ids[start:end]))
        return torch.cat(out_parts, dim=0)

    # ──────────────────────────────────────────────────────────────────
    # Per-batch training step
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device

        # Step A: align seeds = unique(src ∪ tgt). Walk seeds always
        # span both endpoints. Tempest respects edge direction
        # internally — on directed graphs, walks from a source follow
        # outgoing edges, walks from a target follow incoming edges.
        # The structural difference between src-walks and tgt-walks
        # captures directedness implicitly; explicit branching on
        # is_directed at the seed-selection step adds nothing.
        align_seeds_np = np.unique(np.concatenate([batch.src, batch.tgt]))

        # Step B: link-pred negatives from PRE-OBSERVE reservoir.
        # Sampled now so the encoder path (when on) can fold neg_tgt
        # into the single Tempest walk sample below.
        neg_src, neg_tgt = self.neg_sampler_train.sample(batch)
        B = len(batch.src)
        K_neg = neg_src.shape[1]

        # Step C: walks from PRE-INGEST state.
        # Encoder ON: ONE Tempest call for all_seeds = unique(src ∪ tgt
        # ∪ neg_tgt). The (src ∪ tgt) subset is sliced out for InfoNCE
        # so alignment_loss sees the same shape it always has, while
        # the encoder consumes the full set for h_seed.
        # Encoder OFF: only (src ∪ tgt) walks are needed; sample those
        # directly (baseline behavior preserved bit-for-bit).
        if self.walk_encoder is not None:
            all_seeds_np = np.unique(np.concatenate([
                align_seeds_np,
                neg_tgt.reshape(-1).astype(np.int64),
            ]))
            walks_all = self.walk_gen.walks_for_nodes(all_seeds_np)
            walks_align = slice_walks_by_seeds(walks_all, align_seeds_np)
        else:
            walks_all = None
            walks_align = self.walk_gen.walks_for_nodes(align_seeds_np)
        t_now = int(batch.t_max)

        # Step D: InfoNCE contrastive alignment over (src ∪ tgt) walks.
        # The softmax denominator over batch contexts ∪ sampled negs
        # is the anti-collapse mechanism (replaces Wang-Isola
        # uniformity). Strict separation of concerns: alignment_loss
        # never sees the walk encoder; it operates on E + projection
        # heads only.
        l_align = alignment_loss(
            embedding_table=self.embedding_table,
            p_target=self.p_target,
            p_context=self.p_context,
            walks=walks_align,
            t_now=t_now,
            T_train=self.config.t_train_span,
            beta=self.config.beta_time,
            tau=self.config.tau,
            node_feat=self.node_feat,
            num_align_negatives=self.config.num_align_negatives,
        )

        # Step E: link head logits.
        all_u = np.concatenate([batch.src, neg_src.reshape(-1).astype(np.int64)])
        all_v = np.concatenate([batch.tgt, neg_tgt.reshape(-1).astype(np.int64)])
        u_t = torch.from_numpy(all_u).long().to(device)
        v_t = torch.from_numpy(all_v).long().to(device)

        if self.walk_encoder is not None:
            # Encoder consumes the full walks_all. For cross_attn the
            # encoder additionally returns a per-seed token bank
            # (only AttentionWalkEncoder supports this — enforced at
            # __init__).
            need_tokens = self.config.link_head_type == "cross_attn"
            if need_tokens:
                h_seed, tokens_all, token_mask_all = self.walk_encoder(
                    walks_all,
                    t_now=t_now,
                    T_train=self.config.t_train_span,
                    return_tokens=True,
                )
            else:
                h_seed = self.walk_encoder(
                    walks_all, t_now=t_now, T_train=self.config.t_train_span,
                )                                            # [N_all, d_emb]
            seeds_sorted = walks_all.seeds.to(device).long()
            h_u = lookup_h_seed(h_seed, seeds_sorted, u_t)
            h_v = lookup_h_seed(h_seed, seeds_sorted, v_t)
            if self.config.link_head_type == "hybrid":
                # Augment with detached E to preserve "BCE does not reach
                # E" — E provides identity prior as a frozen value; h
                # carries BCE gradient through the encoder.
                e_u = self.embedding_table(u_t).detach()
                e_v = self.embedding_table(v_t).detach()
                eh_u = torch.cat([e_u, h_u], dim=-1)
                eh_v = torch.cat([e_v, h_v], dim=-1)
                logits = self.link_head(eh_u, eh_v)
            elif self.config.link_head_type == "cross_attn":
                # Chunk per-pair scoring: gathering [P, T, d] token
                # banks for both sides of P=22000 pairs is ~2 GB and
                # OOMs at bs=2000 on 8 GB. Chunking keeps peak under
                # ~600 MB without changing semantics (gradients
                # accumulate across chunks via autograd).
                P = u_t.shape[0]
                chunk = self._EVAL_SCORE_CHUNK
                if P <= chunk:
                    u_rows = torch.searchsorted(seeds_sorted, u_t)
                    v_rows = torch.searchsorted(seeds_sorted, v_t)
                    u_tok = tokens_all[u_rows]
                    u_mask = token_mask_all[u_rows]
                    v_tok = tokens_all[v_rows]
                    v_mask = token_mask_all[v_rows]
                    logits = self.link_head(h_u, h_v, u_tok, u_mask, v_tok, v_mask)
                else:
                    out_parts = []
                    for start in range(0, P, chunk):
                        end = min(start + chunk, P)
                        u_chunk = u_t[start:end]
                        v_chunk = v_t[start:end]
                        u_rows = torch.searchsorted(seeds_sorted, u_chunk)
                        v_rows = torch.searchsorted(seeds_sorted, v_chunk)
                        u_tok = tokens_all[u_rows]
                        u_mask = token_mask_all[u_rows]
                        v_tok = tokens_all[v_rows]
                        v_mask = token_mask_all[v_rows]
                        h_u_chunk = h_u[start:end]
                        h_v_chunk = h_v[start:end]
                        out_parts.append(
                            self.link_head(
                                h_u_chunk, h_v_chunk,
                                u_tok, u_mask, v_tok, v_mask,
                            )
                        )
                    logits = torch.cat(out_parts, dim=0)
            else:
                logits = self.link_head(h_u, h_v)
        else:
            # Baseline path: detached E inputs (stop-grad on E for BCE).
            e_u = self.embedding_table(u_t).detach()
            e_v = self.embedding_table(v_t).detach()
            logits = self.link_head(e_u, e_v)

        labels = torch.cat([
            torch.ones(B, device=device),
            torch.zeros(B * K_neg, device=device),
        ])
        l_bce = F.binary_cross_entropy_with_logits(logits, labels)

        # Step 5 + 6: total loss + single backward + step.
        l_total = l_align + l_bce
        self.optimizer.zero_grad(set_to_none=True)
        l_total.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                (p for group in self.optimizer.param_groups for p in group["params"]),
                max_norm=self.config.max_grad_norm,
            )
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self._global_step += 1

        # Step 7: observe positives into reservoir (post-scoring).
        if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
            self.neg_sampler_train.observe(batch.src, batch.tgt)

        # Step 8: ingest into Tempest (LAST).
        self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)

        return {
            "align": float(l_align.detach()),
            "bce": float(l_bce.detach()),
            "total": float(l_total.detach()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    # ──────────────────────────────────────────────────────────────────
    # Eval — strict-causal, no_grad
    # ──────────────────────────────────────────────────────────────────

    def _eval(
        self,
        evaluator: Evaluator,
        batches: Iterable[Batch],
        sample_pct: float = 1.0,
    ) -> float:
        """Streaming eval. Reservoir NOT updated. Tempest state advances
        via post-scoring add_edges. When the walk encoder is on, walks
        ARE sampled per batch — needed to derive h_seed for every node
        the link head scores — but Tempest's ingestion still happens
        only after scoring (strict-causal preserved)."""
        self.embedding_table.eval()
        self.p_target.eval()
        self.p_context.eval()
        self.link_head.eval()
        if self.walk_encoder is not None:
            self.walk_encoder.eval()

        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in batches:
                B_full = len(batch.src)
                if 0.0 < sample_pct < 1.0:
                    n_keep = max(1, int(B_full * sample_pct))
                    keep_idx = np.random.choice(B_full, size=n_keep, replace=False)
                    score_batch = Batch(
                        src=batch.src[keep_idx],
                        tgt=batch.tgt[keep_idx],
                        ts=batch.ts[keep_idx],
                        edge_feat=(
                            batch.edge_feat[keep_idx]
                            if batch.edge_feat is not None
                            else None
                        ),
                        t_max=(
                            int(batch.ts[keep_idx].max())
                            if len(keep_idx) > 0
                            else batch.t_max
                        ),
                    )
                else:
                    score_batch = batch

                B = len(score_batch.src)
                if B == 0:
                    # Still advance Tempest state with the FULL batch (strict-causal).
                    self.walk_gen.add_edges(
                        batch.src, batch.tgt, batch.ts, batch.edge_feat,
                    )
                    continue

                # TGB-supplied per-positive negative destinations.
                neg_src_list, neg_tgt_list = evaluator.sample_negatives(score_batch)

                # Flatten negatives once — needed BEFORE encoding so the
                # h_cache covers every node the link head will score.
                counts = [int(arr.shape[0]) for arr in neg_tgt_list]
                if sum(counts) > 0:
                    flat_neg_u_np = np.concatenate(
                        [
                            np.full(counts[i], int(score_batch.src[i]), dtype=np.int64)
                            for i in range(B)
                        ]
                    )
                    flat_neg_v_np = np.concatenate(
                        [nt.astype(np.int64) for nt in neg_tgt_list]
                    )
                else:
                    flat_neg_u_np = None
                    flat_neg_v_np = None

                # Build the walk-encoder cache for this eval batch
                # (only when encoder is on). For cross_attn the cache
                # also carries (tokens, token_mask).
                h_cache: Optional[Dict[str, torch.Tensor]] = None
                if self.walk_encoder is not None:
                    parts = [
                        score_batch.src.astype(np.int64),
                        score_batch.tgt.astype(np.int64),
                    ]
                    if flat_neg_u_np is not None:
                        parts.append(flat_neg_u_np)
                        parts.append(flat_neg_v_np)
                    all_nodes = np.unique(np.concatenate(parts))
                    walks_eval = self.walk_gen.walks_for_nodes(all_nodes)
                    if self.config.link_head_type == "cross_attn":
                        h_seed_eval, tokens_eval, token_mask_eval = self.walk_encoder(
                            walks_eval,
                            t_now=int(score_batch.t_max),
                            T_train=self.config.t_train_span,
                            return_tokens=True,
                        )
                    else:
                        h_seed_eval = self.walk_encoder(
                            walks_eval,
                            t_now=int(score_batch.t_max),
                            T_train=self.config.t_train_span,
                        )
                        tokens_eval = None
                        token_mask_eval = None
                    seeds_sorted_eval = walks_eval.seeds.to(self.device).long()
                    h_cache = {
                        "h_seed": h_seed_eval,
                        "seeds_sorted": seeds_sorted_eval,
                        "tokens": tokens_eval,
                        "token_mask": token_mask_eval,
                    }

                # Positive scores in one shot.
                pos_u = torch.from_numpy(score_batch.src.astype(np.int64)).to(self.device)
                pos_v = torch.from_numpy(score_batch.tgt.astype(np.int64)).to(self.device)
                pos_logits = self._score_pairs(pos_u, pos_v, h_cache=h_cache).cpu().numpy()

                # Score negatives.
                if flat_neg_u_np is not None:
                    neg_u_t = torch.from_numpy(flat_neg_u_np).to(self.device)
                    neg_v_t = torch.from_numpy(flat_neg_v_np).to(self.device)
                    neg_logits = self._score_pairs(neg_u_t, neg_v_t, h_cache=h_cache).cpu().numpy()
                else:
                    neg_logits = np.empty(0, dtype=np.float64)

                cursor = 0
                for i in range(B):
                    K_i = counts[i]
                    m = evaluator.score_to_metric(
                        float(pos_logits[i]),
                        neg_logits[cursor : cursor + K_i],
                    )
                    total += m
                    cursor += K_i
                n += B

                # Strict-causal: advance Tempest with the FULL batch.
                self.walk_gen.add_edges(
                    batch.src, batch.tgt, batch.ts, batch.edge_feat,
                )

        return total / max(n, 1)

    # ──────────────────────────────────────────────────────────────────
    # Snapshot / restore (early-stop)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().to("cpu", copy=True)
            for k, v in module.state_dict().items()
        }

    def _snapshot(self) -> Dict[str, Any]:
        snap = {
            "embedding_table": self._cpu_state_dict(self.embedding_table),
            "p_target":        self._cpu_state_dict(self.p_target),
            "p_context":       self._cpu_state_dict(self.p_context),
            "link_head":       self._cpu_state_dict(self.link_head),
        }
        if self.walk_encoder is not None:
            snap["walk_encoder"] = self._cpu_state_dict(self.walk_encoder)
        return snap

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.embedding_table.load_state_dict(snap["embedding_table"])
        self.p_target.load_state_dict(snap["p_target"])
        self.p_context.load_state_dict(snap["p_context"])
        self.link_head.load_state_dict(snap["link_head"])
        if self.walk_encoder is not None and "walk_encoder" in snap:
            self.walk_encoder.load_state_dict(snap["walk_encoder"])

    def _re_ingest_train(self, train_batches_factory) -> None:
        """Reset Tempest, re-ingest all training edges. Used before
        final val/test eval so Tempest state matches post-training."""
        self.walk_gen.reset()
        for batch in train_batches_factory():
            self.walk_gen.add_edges(
                batch.src, batch.tgt, batch.ts, batch.edge_feat,
            )

    # ──────────────────────────────────────────────────────────────────
    # Train loop
    # ──────────────────────────────────────────────────────────────────

    def train(
        self,
        train_batches_factory,
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory=None,
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory=None,
    ) -> Dict[str, Any]:
        n_epochs = self.config.num_epochs
        patience = self.config.early_stop_patience
        sample_pct = self.config.monitor_sample_pct

        # Set up the LR scheduler. We need batches_per_epoch which
        # we compute by exhausting the factory once (cheap — counts
        # only, doesn't materialise tensors per batch).
        batches_per_epoch = sum(1 for _ in train_batches_factory())
        self._setup_lr_scheduler(batches_per_epoch)

        best_val = -1.0
        best_test = -1.0
        best_epoch = -1
        best_snap: Optional[Dict[str, Any]] = None
        no_improve = 0
        per_epoch_val: List[float] = []
        per_epoch_test: List[float] = []

        for ep in range(1, n_epochs + 1):
            # Epoch boundary: walks + reservoir reset only.
            self.walk_gen.reset()
            if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                self.neg_sampler_train.reset()

            self.embedding_table.train()
            self.p_target.train()
            self.p_context.train()
            self.link_head.train()
            if self.walk_encoder is not None:
                self.walk_encoder.train()

            t0 = time.time()
            sums = {"align": 0.0, "bce": 0.0, "total": 0.0}
            n_batches = 0
            for batch in train_batches_factory():
                metrics = self._train_step(batch)
                for k in sums:
                    sums[k] += metrics[k]
                n_batches += 1
            train_dt = time.time() - t0

            line = (
                f"epoch {ep}/{n_epochs}  "
                f"align={sums['align']/max(n_batches,1):.4f}  "
                f"bce={sums['bce']/max(n_batches,1):.4f}  "
                f"train {train_dt:.1f}s"
            )

            if val_evaluator is not None and val_batches_factory is not None:
                t1 = time.time()
                val_metric = self._eval(
                    val_evaluator, val_batches_factory(), sample_pct,
                )
                eval_dt = time.time() - t1
                per_epoch_val.append(val_metric)

                test_metric = -1.0
                if val_metric > best_val:
                    best_val = val_metric
                    best_epoch = ep
                    best_snap = self._snapshot()
                    no_improve = 0
                    if (
                        test_evaluator is not None
                        and test_batches_factory is not None
                    ):
                        test_metric = self._eval(
                            test_evaluator, test_batches_factory(), sample_pct,
                        )
                        best_test = test_metric
                        per_epoch_test.append(test_metric)
                        line += f"  val {val_metric:.4f}  test {test_metric:.4f} (new best)"
                    else:
                        line += f"  val {val_metric:.4f} (new best)"
                else:
                    no_improve += 1
                    line += (
                        f"  val {val_metric:.4f}  "
                        f"patience {no_improve}/{patience}"
                    )
                line += f"  eval {eval_dt:.1f}s"
            print(line)

            if patience > 0 and no_improve >= patience:
                break

        if best_snap is not None:
            self._restore(best_snap)
            print(
                f"  restored best weights from epoch {best_epoch} "
                f"(val {best_val:.4f}, test {best_test:.4f})"
            )

        if (
            not self.config.skip_final_full_eval
            and val_evaluator is not None
            and val_batches_factory is not None
        ):
            print("=== Final full eval ===")
            self._re_ingest_train(train_batches_factory)
            final_val = self._eval(val_evaluator, val_batches_factory(), 1.0)
            print(f"  val MRR: {final_val:.4f}")
            if test_evaluator is not None and test_batches_factory is not None:
                final_test = self._eval(test_evaluator, test_batches_factory(), 1.0)
                print(f"  test MRR: {final_test:.4f}")
                best_val, best_test = final_val, final_test

        return {
            "stopped_at_epoch": best_epoch if best_snap is not None else n_epochs,
            "best_val_mrr": best_val,
            "best_test_mrr": best_test,
            "per_epoch_val_mrr": per_epoch_val,
            "per_epoch_test_mrr": per_epoch_test,
        }
