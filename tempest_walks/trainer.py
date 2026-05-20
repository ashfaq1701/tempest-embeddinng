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
from .losses import alignment_loss, link_bce, normbrake_loss, uniformity_loss
from .model import EmbeddingStore, LinkPredictor
from .negatives import (
    HistoricalNegativeSampler,
    NegativeSampler,
    UniformNegativeSampler,
)
from .walks import WalkGenerator


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
        self.link_predictor = LinkPredictor(config.d_emb, config.d_hidden_link).to(self.device)
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
        self.emb_optimizer = torch.optim.Adam(
            self.embedding_store.parameters(), lr=config.emb_lr,
        )
        self.link_optimizer = torch.optim.Adam(
            self.link_predictor.parameters(),
            lr=config.link_lr,
            weight_decay=config.weight_decay_link,
        )
        self._time_scale = config.alignment_time_scale  # overridden after dataset load

    # ------------------------------------------------------------------ #
    # Per-batch step (strict-causal: ingest is the LAST thing).
    # ------------------------------------------------------------------ #

    def _embedding_step(self, batch: Batch) -> tuple[float, float]:
        """Alignment + uniformity from walks sampled from the PRE-ingest state."""
        # Seeds: union of src and tgt of the current batch. Walking from
        # BOTH sides matters on bipartite-flavored datasets (e.g. tgbl-wiki:
        # users → pages). Seeding only on batch.src would leave target(page)
        # and context(user) with no alignment-loss signal — the link MLP
        # would then have to train half the embedding tables on BCE alone.
        # Union seeding gives every node touched by the batch a chance to
        # pull its target view via alignment.
        seeds_np = np.unique(np.concatenate([batch.src, batch.tgt]))
        walks = self.walk_gen.walks_for_nodes(seeds_np)
        nodes = walks.nodes.to(self.device).long().clamp_min(0)
        edge_feats = (
            walks.edge_feats.to(self.device) if walks.edge_feats is not None else None
        )
        e_target_seed = self.embedding_store.target(walks.seeds.to(self.device))   # [N, d]
        # context_walk fuses node-feature residuals (via context) AND the
        # edge-feature of the hop leaving each walk position (right-padded
        # at the seed slot, which the alignment loss masks out anyway).
        # Each augmentation is a no-op when the dataset doesn't have it.
        e_context_all = self.embedding_store.context_walk(nodes, edge_feats)      # [N*K, L, d]

        t_query = torch.full(
            (walks.seeds.shape[0],), int(batch.t_max), dtype=torch.long, device=self.device,
        )
        l_align = alignment_loss(
            e_target_seed=e_target_seed,
            e_context_all=e_context_all,
            walks=walks,
            t_query=t_query,
            beta=self.config.temporal_decay_exp,
            time_scale=self._time_scale,
        )

        # Uniformity is on the unique nodes touched by the batch (src + tgt) —
        # spreads the same embedding space the link MLP will consume.
        unique_batch_nodes = np.unique(np.concatenate([batch.src, batch.tgt]))
        ub = torch.from_numpy(unique_batch_nodes).long().to(self.device)
        l_uniform = uniformity_loss(
            self.embedding_store.target(ub),
            temperature=self.config.uniformity_temperature,
            cap=self.config.uniformity_cap,
        )
        l_total = l_align + self.config.eta_uniform * l_uniform

        # Normbrake auxiliary (CLAUDE.md Lesson 18). Composes with the
        # primary loss by addition. Operates on the underlying embedding
        # tables, not on the per-walk lookups — so it can constrain
        # magnitudes even when a given batch only touches a subset of nodes.
        if self.config.lambda_normbrake > 0:
            l_nb = normbrake_loss(
                E_target=self.embedding_store.E_target.weight,
                E_context=self.embedding_store.E_context.weight,
                threshold=self.config.normbrake_threshold,
            )
            l_total = l_total + self.config.lambda_normbrake * l_nb

        self.emb_optimizer.zero_grad(set_to_none=True)
        l_total.backward()
        self.emb_optimizer.step()
        return float(l_align.detach()), float(l_uniform.detach())

    def _link_step(self, batch: Batch) -> float:
        neg_src, neg_tgt = self.neg_sampler_train.sample(batch)
        B = len(batch.src)
        K = neg_src.shape[1]

        all_u = np.concatenate([batch.src, neg_src.reshape(-1).astype(np.int64)])
        all_v = np.concatenate([batch.tgt, neg_tgt.reshape(-1).astype(np.int64)])
        u_t = torch.from_numpy(all_u).long().to(self.device)
        v_t = torch.from_numpy(all_v).long().to(self.device)

        logits = self.link_predictor(
            self.embedding_store.target(u_t),      # node_feat residual folded in
            self.embedding_store.target(v_t),
            self.embedding_store.context(u_t),
            self.embedding_store.context(v_t),
        )
        labels = torch.cat(
            [
                torch.ones(B, device=self.device),
                torch.zeros(B * K, device=self.device),
            ],
        )
        loss = link_bce(logits, labels)
        self.link_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.link_optimizer.step()
        return float(loss.detach())

    # ------------------------------------------------------------------ #
    # Phase loops.
    # ------------------------------------------------------------------ #

    def train(self, batches: Iterable[Batch]) -> None:
        batches = list(batches)
        for epoch in range(self.config.num_epochs):
            self.walk_gen.reset()
            if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                self.neg_sampler_train.reset()
            self.embedding_store.train()
            self.link_predictor.train()
            t0 = time.perf_counter()
            sum_align = sum_uniform = sum_link = 0.0
            n = 0
            for batch in batches:
                l_align, l_uniform = self._embedding_step(batch)
                l_link = self._link_step(batch)
                # ── Post-scoring strict-causal block (feeds batch B+1) ──
                # Reservoir observe AND Tempest ingest run AFTER scoring.
                # Both contain events up through (and including) batch B
                # only when batch B+1's loop begins.
                if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                    self.neg_sampler_train.observe(batch.src, batch.tgt)
                self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
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
