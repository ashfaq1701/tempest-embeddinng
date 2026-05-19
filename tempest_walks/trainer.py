"""Strict-causal training + eval loop.

Per batch — IN THIS EXACT ORDER (training and eval both):

  1. Sample walks for seeds from CURRENT (pre-ingest) Tempest state.
  2. Update embeddings from those walks (alignment + uniformity).
  3. Score link prediction (BCE at train, TGB Evaluator at val/test).
  4. Ingest the batch into Tempest — LAST.

Step 1 sees only events strictly before the current batch's edges, so
neither the embedding update nor the link scoring can leak the current
batch's positives into themselves. Compare with the v1 / v2 baselines
that ingested first; their training MRR was leak-inflated.

Two optimizers, run sequentially per batch:
  - embedding_optimizer ← (l_align + η · l_uniform).backward()
  - link_optimizer      ← BCE(logits, labels).backward()
"""

import time
from typing import Iterable, Optional

import numpy as np
import torch

from .config import Config
from .data import Batch
from .evaluator import Evaluator
from .losses import alignment_loss, link_bce, uniformity_loss
from .history import NodeHistory
from .model import (
    CoOccurrenceEncoder,
    CrossPairAttention,
    EmbeddingStore,
    LinkPredictor,
    NodeEncoder,
    WalkEncoder,
    masked_mean_pool,
)
from .negatives import (
    HistoricalNegativeSampler,
    NegativeSampler,
    UniformNegativeSampler,
)
from .walks import WalkData, WalkGenerator


class Trainer:
    def __init__(
        self,
        config: Config,
        train_dst_pool: np.ndarray,
        node_feat: Optional[np.ndarray] = None,   # [n_nodes, d_node_feat] or None
        edge_feat_dim: int = 0,                    # d_edge_feat from train.edge_feat
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu",
        )

        # The EmbeddingStore fuses node + edge features (when present) into
        # the same dual-table representation the link MLP and alignment loss
        # both consume — no special-case code paths anywhere else.
        self.embedding_store = EmbeddingStore(
            n_nodes=config.max_node_count,
            d_emb=config.d_emb,
            node_feat=node_feat,
            edge_feat_dim=edge_feat_dim,
        ).to(self.device)
        # Phase 3: 4-channel link MLP [E(u) ‖ E(v) ‖ W(u) ‖ W(v)] —
        # cross-pair attention does the interaction work BEFORE the MLP
        # sees its inputs, so we don't need Hadamard/L1/cross-table blocks.
        # Optional 5th channel: co-occurrence projection (requires node_history).
        co_feat_enabled = config.use_co_feat and config.use_node_encoder
        self.link_predictor = LinkPredictor(
            config.d_emb,
            config.d_hidden_link,
            dropout=config.link_dropout,
            use_co_feat=co_feat_enabled,
        ).to(self.device)
        self.co_encoder: Optional[CoOccurrenceEncoder] = None
        if co_feat_enabled:
            self.co_encoder = CoOccurrenceEncoder(d_emb=config.d_emb).to(self.device)
        # Cross-pair attention head — pair-conditioned walk summary.
        # Lives with the link side: parameters are scoring-specific
        # (the alignment loss doesn't touch them).
        self.cross_pair_attn: Optional[CrossPairAttention] = None
        if config.use_walk_encoder:
            self.cross_pair_attn = CrossPairAttention(
                d_model=config.d_emb,
                n_heads=config.xpair_n_heads,
                dropout=config.xpair_dropout,
            ).to(self.device)

        # DyGFormer-style dynamic node encoder + history buffer.
        # History writes happen in the post-scoring block (LAST), reads
        # happen at the start of _step / evaluate_batch (BEFORE scoring).
        self.node_history: Optional[NodeHistory] = None
        self.node_encoder: Optional[NodeEncoder] = None
        if config.use_node_encoder:
            self.node_history = NodeHistory(
                n_nodes=config.max_node_count,
                K=config.k_history,
                edge_feat_dim=edge_feat_dim,
            )
            self.node_encoder = NodeEncoder(
                embedding_store=self.embedding_store,
                d_model=config.d_emb,
                n_heads=config.node_enc_n_heads,
                n_layers=config.node_enc_n_layers,
                d_time=config.d_time,
                d_role=config.d_role,
                dropout=config.node_enc_dropout,
                ff_dim=config.node_enc_ff_dim,
            ).to(self.device)
        # Walk encoder (Phase 1): GRU over per-position walk inputs.
        # d_gru is fixed to d_emb so its output can be directly cosine-
        # compared against target(seed) in the alignment loss. The encoder
        # reuses EmbeddingStore's edge_feat_proj and context() lookup so
        # there's no duplicated parameter for the feature projections.
        self.walk_encoder: Optional[WalkEncoder] = None
        if config.use_walk_encoder:
            self.walk_encoder = WalkEncoder(
                embedding_store=self.embedding_store,
                d_gru=config.d_emb,
                d_time=config.d_time,
                d_role=config.d_role,
                dropout=config.walk_encoder_dropout,
            ).to(self.device)
        self.walk_gen = WalkGenerator(
            is_directed=config.is_directed,
            use_gpu=False,                  # Tempest CPU — preserves 8 GB VRAM
            walk_bias=config.walk_bias,
            max_walk_len=config.max_walk_len,
            num_walks_per_node=config.num_walks_per_node,
        )
        self.neg_sampler_train: NegativeSampler
        if config.hist_neg_ratio > 0:
            # TGB-protocol-matched mix: K_hist drawn from each source's
            # reservoir (events strictly before the current batch — observe
            # runs in the post-scoring block below), K_rand uniform random.
            self.neg_sampler_train = HistoricalNegativeSampler(
                num_nodes=config.max_node_count,
                num_neg_per_pos=config.num_neg_per_pos,
                hist_ratio=config.hist_neg_ratio,
                reservoir_size=config.reservoir_size,
                dst_pool=train_dst_pool,
                seed=config.seed,
            )
        else:
            self.neg_sampler_train = UniformNegativeSampler(
                num_neg_per_pos=config.num_neg_per_pos,
                dst_pool=train_dst_pool,
                seed=config.seed,
            )
        emb_params = list(self.embedding_store.parameters())
        if self.walk_encoder is not None:
            emb_params += list(self.walk_encoder.parameters())
        self.emb_optimizer = torch.optim.Adam(emb_params, lr=config.emb_lr)
        link_params = list(self.link_predictor.parameters())
        if self.cross_pair_attn is not None:
            link_params += list(self.cross_pair_attn.parameters())
        if self.node_encoder is not None:
            link_params += list(self.node_encoder.parameters())
        if self.co_encoder is not None:
            link_params += list(self.co_encoder.parameters())
        self.link_optimizer = torch.optim.Adam(link_params, lr=config.link_lr)
        self._time_scale = config.alignment_time_scale  # overridden after dataset load

    # ------------------------------------------------------------------ #
    # Per-batch step (strict-causal: ingest is the LAST thing).
    # ------------------------------------------------------------------ #

    def _compute_seed_seq(
        self,
        walks: WalkData,
        h_walks: torch.Tensor,
        seeds_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-seed averaged walk SEQUENCE for cross-pair attention.

        h_walks: [N*K, L, d] — GRU outputs at every position of every walk.
        Returns:
            h_seq:     [N, L, d]  — at each position p, the mean of
                                    h_walks[seed_i * K : (seed_i+1) * K, p, :]
                                    over walks where lens > p (i.e. position
                                    p is real for that walk).
            valid_mask: [N, L]    — True if at least one walk of that seed
                                    has lens > p (so position p is real).
                                    Cold-start seeds (all-zero) get
                                    valid_mask[i, 0]=True and h_seq[i, 0, :]
                                    = target(seed_i) as a fallback.
        """
        K = walks.K
        N = seeds_tensor.shape[0]
        L = h_walks.shape[1]
        d = h_walks.shape[2]
        walk_lens = walks.lens.to(self.device).long()                              # [N*K]

        positions = torch.arange(L, device=self.device).unsqueeze(0)              # [1, L]
        per_walk_valid = positions < walk_lens.unsqueeze(1)                        # [N*K, L]

        h_walks_re = h_walks.reshape(N, K, L, d)
        valid_re = per_walk_valid.reshape(N, K, L)

        valid_f = valid_re.float().unsqueeze(-1)                                   # [N, K, L, 1]
        h_sum = (h_walks_re * valid_f).sum(dim=1)                                  # [N, L, d]
        h_cnt = valid_f.sum(dim=1).clamp_min(1.0)                                  # [N, L, 1]
        h_seq = h_sum / h_cnt                                                      # [N, L, d]

        valid_mask = valid_re.any(dim=1)                                           # [N, L]

        # Cold-start fallback: seeds whose every walk has lens=0 — no valid
        # position at all. Inject target(seed) into position 0 so cross-pair
        # attention has something to attend FROM and so the masked_mean_pool
        # doesn't return zero.
        no_walk_any = ~valid_mask.any(dim=1)                                       # [N]
        if no_walk_any.any():
            target_seeds = self.embedding_store.target(seeds_tensor)
            h_seq = h_seq.clone()
            valid_mask = valid_mask.clone()
            h_seq[no_walk_any, 0, :] = target_seeds[no_walk_any]
            valid_mask[no_walk_any, 0] = True

        return h_seq, valid_mask

    def _step(self, batch: Batch) -> tuple[float, float, float]:
        """Unified Phase 2 step: walks → encoder → alignment + uniformity + link,
        single backward. Strict-causal: walks and negatives both come from
        state ≤ batch B−1 (the post-scoring block ingests batch B for B+1).

        Ordering inside this method:
          1. Sample negatives from the PRE-batch reservoir.
          2. Build the full seed set: unique(src ∪ tgt ∪ neg_src ∪ neg_tgt).
          3. Sample walks for the seed set from the PRE-ingest Tempest.
          4. Forward through the walk encoder.
          5. Build per-seed walk-summary h (mean over K walks, target() fallback).
          6. Alignment + uniformity + link losses → single backward → both
             optimizers step.
        """
        # ── (1) Negatives from the pre-batch reservoir ────────────────────────
        neg_src, neg_tgt = self.neg_sampler_train.sample(batch)
        B = len(batch.src)
        K_neg = neg_src.shape[1]

        # ── (2) Full seed set ─────────────────────────────────────────────────
        seeds_np = np.unique(np.concatenate([
            batch.src,
            batch.tgt,
            neg_src.reshape(-1).astype(np.int64),
            neg_tgt.reshape(-1).astype(np.int64),
        ]))

        # ── (3) Walks from the pre-ingest Tempest ─────────────────────────────
        walks = self.walk_gen.walks_for_nodes(seeds_np)
        nodes = walks.nodes.to(self.device).long().clamp_min(0)
        timestamps = walks.timestamps.to(self.device).long()
        walk_lens = walks.lens.to(self.device).long()
        edge_feats = (
            walks.edge_feats.to(self.device) if walks.edge_feats is not None else None
        )
        seeds_tensor = walks.seeds.to(self.device).long()
        K = walks.K
        t_query_per_seed = torch.full(
            (seeds_tensor.shape[0],), int(batch.t_max),
            dtype=torch.long, device=self.device,
        )
        t_query_per_walk = t_query_per_seed.repeat_interleave(K)

        # ── (4) Walk-encoder forward (single pass over the union seed set) ────
        if self.walk_encoder is None:
            raise RuntimeError("Phase 2 requires walk_encoder; pass --use-walk-encoder.")
        h_walks = self.walk_encoder(
            walk_nodes=nodes,
            walk_timestamps=timestamps,
            lens=walk_lens,
            walk_edge_feats=edge_feats,
            t_query=t_query_per_walk,
            time_scale=self._time_scale,
        )                                                                          # [N*K, L, d]

        # ── (5) Per-seed walk SEQUENCE (avg over K, target() cold-start fallback)
        seed_seq, seq_mask = self._compute_seed_seq(walks, h_walks, seeds_tensor)
        # seed_seq: [N, L, d]   seq_mask: [N, L]

        # ── (6a) Alignment: target(seed) ↔ GRU outputs along the walk ────────
        e_target_seed = self.embedding_store.target(seeds_tensor)                  # [N, d]
        l_align = alignment_loss(
            e_target_seed=e_target_seed,
            e_context_all=h_walks,
            walks=walks,
            t_query=t_query_per_seed,
            beta=self.config.temporal_decay_exp,
            time_scale=self._time_scale,
        )

        # ── (6b) Uniformity on batch positive nodes ──────────────────────────
        unique_batch_nodes = np.unique(np.concatenate([batch.src, batch.tgt]))
        ub = torch.from_numpy(unique_batch_nodes).long().to(self.device)
        l_uniform = uniformity_loss(
            self.embedding_store.target(ub),
            temperature=self.config.uniformity_temperature,
            cap=self.config.uniformity_cap,
        )

        # ── (6c) Link prediction: 12-block (or 13 with co-feat) link MLP
        # restoring Phase 2's empirically-best cross-table interaction
        # structure. Cross-pair attention was tried in Phase 3 with a
        # collapsed 4-channel head and regressed test MRR by ~0.05 — the
        # Hadamard/L1 cross-table blocks were doing more work than
        # cross-pair attention added, so we drop cross-pair entirely.
        all_u = np.concatenate([batch.src, neg_src.reshape(-1).astype(np.int64)])
        all_v = np.concatenate([batch.tgt, neg_tgt.reshape(-1).astype(np.int64)])
        u_idx_in_seeds = np.searchsorted(seeds_np, all_u)
        v_idx_in_seeds = np.searchsorted(seeds_np, all_v)
        u_idx_t = torch.from_numpy(u_idx_in_seeds).long().to(self.device)
        v_idx_t = torch.from_numpy(v_idx_in_seeds).long().to(self.device)

        # Seed-pooled walk summary (Phase 2 style) — mean over the K walks
        # of each seed at the seed position, with target() fallback for
        # cold-start nodes. Cheaper than cross-pair attention and avoids
        # Phase 3's regression.
        K = walks.K
        arange_w = torch.arange(h_walks.shape[0], device=self.device)
        seed_pos = (walk_lens - 1).clamp_min(0)
        h_at_seed = h_walks[arange_w, seed_pos, :]                      # [N*K, d]
        has_walk = (walk_lens > 0).float().unsqueeze(-1)
        h_at_seed = h_at_seed.reshape(seeds_tensor.shape[0], K, -1)
        has_walk = has_walk.reshape(seeds_tensor.shape[0], K, 1)
        seed_h_sum = (h_at_seed * has_walk).sum(dim=1)
        seed_h_cnt = has_walk.sum(dim=1).clamp_min(1.0)
        seed_h = seed_h_sum / seed_h_cnt                                # [N_unique, d]
        no_walk = (has_walk.sum(dim=1) == 0).float()
        target_seeds_fallback = self.embedding_store.target(seeds_tensor)
        seed_h = seed_h * (1.0 - no_walk) + target_seeds_fallback * no_walk

        w_u = seed_h[u_idx_t]
        w_v = seed_h[v_idx_t]

        u_t = torch.from_numpy(all_u).long().to(self.device)
        v_t = torch.from_numpy(all_v).long().to(self.device)
        e_t_u = self.embedding_store.target(u_t)
        e_t_v = self.embedding_store.target(v_t)
        e_c_u = self.embedding_store.context(u_t)
        e_c_v = self.embedding_store.context(v_t)

        # Dynamic node embedding via DyGFormer-style encoder.
        # STRICT-CAUSAL READ: history reflects events ≤ batch B−1; this
        # batch's events have NOT been written yet (the post-scoring block
        # at the end of train()'s per-batch loop is the only writer).
        # The DyG output is added as a RESIDUAL to target(u) before the
        # cross-table blocks of the link MLP see it. Cold-start rows have
        # node_h=0 → fall back to the static lookup naturally.
        if self.node_encoder is not None and self.node_history is not None:
            # Read the history slice for every node we need to score
            # (deduplicated through seeds_np to amortise the lookup).
            nb, hts, ef_hist, ro, vc = self.node_history.read_windows_for_nodes(seeds_np)
            nb_t = torch.from_numpy(nb).long().to(self.device)
            hts_t = torch.from_numpy(hts).long().to(self.device)
            ef_hist_t = (
                torch.from_numpy(ef_hist).float().to(self.device)
                if ef_hist is not None else None
            )
            ro_t = torch.from_numpy(ro).long().to(self.device)
            vc_t = torch.from_numpy(vc).long().to(self.device)
            tq_hist = torch.full(
                (seeds_tensor.shape[0],), int(batch.t_max),
                dtype=torch.long, device=self.device,
            )
            node_h, _has_hist = self.node_encoder(
                neighbors=nb_t,
                timestamps=hts_t,
                edge_feats=ef_hist_t,
                roles=ro_t,
                valid_cnt=vc_t,
                t_query=tq_hist,
                time_scale=self._time_scale,
            )                                              # [N_unique, d]
            # Gather node_h for u and v of each pair; add as residual on top
            # of the static identity. Cold-start rows have node_h=0 → fall
            # back to e_t_u / e_t_v naturally.
            e_t_u = e_t_u + node_h[u_idx_t]
            e_t_v = e_t_v + node_h[v_idx_t]

        # Optional co-occurrence feature from per-pair history overlap.
        co_feat: Optional[torch.Tensor] = None
        if self.co_encoder is not None and self.node_history is not None:
            # nb_t, vc_t already gathered above when node_encoder ran.
            # If node_encoder is off but co_encoder is on (not currently a
            # supported config), would need a separate read here.
            nb_u = nb_t[u_idx_t]                                            # [P, K]
            nb_v = nb_t[v_idx_t]
            vc_u_p = vc_t[u_idx_t]                                          # [P]
            vc_v_p = vc_t[v_idx_t]
            co_feat = self.co_encoder(nb_u, nb_v, vc_u_p, vc_v_p)           # [P, d_emb]

        logits = self.link_predictor(e_t_u, e_t_v, e_c_u, e_c_v, w_u, w_v, co_feat)
        labels = torch.cat([
            torch.ones(B, device=self.device),
            torch.zeros(B * K_neg, device=self.device),
        ])
        l_link = link_bce(logits, labels)

        # ── Single backward, both optimizers step ──────────────────────────
        # The walk encoder + embedding store get gradient from both alignment
        # and link BCE. The link MLP gets gradient only from link BCE (since
        # alignment doesn't depend on its parameters). Each optimizer.step()
        # only touches its own param group.
        l_total = (
            l_align
            + self.config.eta_uniform * l_uniform
            + l_link
        )
        self.emb_optimizer.zero_grad(set_to_none=True)
        self.link_optimizer.zero_grad(set_to_none=True)
        l_total.backward()
        self.emb_optimizer.step()
        self.link_optimizer.step()
        return float(l_align.detach()), float(l_uniform.detach()), float(l_link.detach())

    # ------------------------------------------------------------------ #
    # Phase loops.
    # ------------------------------------------------------------------ #

    def train(self, batches: Iterable[Batch]) -> None:
        batches = list(batches)
        for epoch in range(self.config.num_epochs):
            self.walk_gen.reset()
            if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                self.neg_sampler_train.reset()
            if self.node_history is not None:
                self.node_history.reset()
            self.embedding_store.train()
            self.link_predictor.train()
            if self.walk_encoder is not None:
                self.walk_encoder.train()
            if self.cross_pair_attn is not None:
                self.cross_pair_attn.train()
            if self.node_encoder is not None:
                self.node_encoder.train()
            if self.co_encoder is not None:
                self.co_encoder.train()
            t0 = time.perf_counter()
            sum_align = sum_uniform = sum_link = 0.0
            n = 0
            for batch in batches:
                l_align, l_uniform, l_link = self._step(batch)
                # ── Post-scoring strict-causal block (feeds batch B+1) ──
                # Reservoir observe AND Tempest ingest run AFTER scoring.
                # Both contain events up through (and including) batch B
                # only when batch B+1's loop begins.
                if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                    self.neg_sampler_train.observe(batch.src, batch.tgt)
                self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
                if self.node_history is not None:
                    # LAST write of the post-scoring block. After this line,
                    # the history reflects events ≤ batch B. Batch B+1's
                    # read at the start of its _step will see this.
                    self.node_history.write_events(
                        src=batch.src,
                        tgt=batch.tgt,
                        ts=batch.ts,
                        edge_feat=batch.edge_feat,
                    )
                sum_align += l_align
                sum_uniform += l_uniform
                sum_link += l_link
                n += 1
            dt = time.perf_counter() - t0
            print(
                f"  epoch {epoch+1}/{self.config.num_epochs}  "
                f"align={sum_align/n:.4f}  uniform={sum_uniform/n:.4f}  "
                f"link={sum_link/n:.4f}  {dt:.1f}s",
                flush=True,
            )

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Batch], evaluator: Evaluator) -> float:
        """Streaming evaluation. Returns dataset-level metric (TGB official)."""
        self.embedding_store.eval()
        self.link_predictor.eval()
        if self.walk_encoder is not None:
            self.walk_encoder.eval()
        if self.cross_pair_attn is not None:
            self.cross_pair_attn.eval()
        if self.node_encoder is not None:
            self.node_encoder.eval()
        if self.co_encoder is not None:
            self.co_encoder.eval()
        total = 0.0
        n = 0
        for batch in batches:
            # 1-3. Score the batch FIRST (pre-ingest, strictly causal).
            m, b = evaluator.evaluate_batch(batch)
            total += m
            n += b
            # 4. Ingest LAST. Eval-time embedding adaptation is intentionally
            # OFF — the model is frozen at val/test, per TGB's streaming
            # convention. The Tempest state still accumulates so subsequent
            # eval batches' walks include earlier eval edges (also TGB-
            # conventional: "previously observed test edges can be accessed
            # by the model but back-propagation [...] is not permitted").
            self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
            if self.node_history is not None:
                # Same protocol as training: history write is the LAST line.
                # Batch B+1's evaluator.evaluate_batch will see events ≤ B.
                self.node_history.write_events(
                    src=batch.src,
                    tgt=batch.tgt,
                    ts=batch.ts,
                    edge_feat=batch.edge_feat,
                )
        return total / max(n, 1)

    # ------------------------------------------------------------------ #
    # Hooks.
    # ------------------------------------------------------------------ #

    def set_time_scale(self, scale: float) -> None:
        """Set the alignment-loss time scale. Call after loading the dataset:
        a sensible default is (t_max_train − t_min_train) / max_walk_len so
        a typical one-step Δt maps to ~1 and the temporal-decay weight
        retains non-trivial mass at deep walk positions."""
        self._time_scale = scale
