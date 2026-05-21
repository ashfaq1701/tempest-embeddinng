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

import copy
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch

from .config import Config
from .data import Batch
from .evaluator import Evaluator
from .losses import (
    alignment_loss,
    infonce_loss,
    link_bce,
    normbrake_loss,
    sgns_loss,
    triplet_loss,
    uniformity_loss,
)
from .model import EmbeddingStore, LinkPredictor, TimeEncoder
from .negatives import (
    HistoricalNegativeSampler,
    NegativeSampler,
    UniformNegativeSampler,
)
from .timestate import NodeTimeState
from .walks import WalkGenerator


class Trainer:
    def __init__(
        self,
        config: Config,
        train_dst_pool: np.ndarray,
        node_feat: Optional[np.ndarray] = None,   # [n_nodes, d_node_feat] or None
        edge_feat_dim: int = 0,                    # d_edge_feat from train.edge_feat
        train_destinations_full: Optional[np.ndarray] = None,  # raw (non-unique) for SGNS unigram^0.75
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
        # Component 0: time encoding at the link MLP.
        self.use_time_encoding = config.use_time_encoding
        self.link_predictor = LinkPredictor(
            d_emb=config.d_emb,
            hidden=config.d_hidden_link,
            use_time_encoding=config.use_time_encoding,
            d_time=2 * config.time_enc_k,
            head_mode=config.head_mode,
            cross_table_dropout=config.cross_table_dropout,
            n_layers=config.link_mlp_n_layers,
            dropout=config.link_mlp_dropout,
        ).to(self.device)
        # §4.8.2 embedding-side dropout — applied AT the link-MLP read site
        # (in _link_step) so it regularises the cross-table input without
        # touching the embedding-side loss path. 0=off.
        self.emb_dropout = torch.nn.Dropout(config.embedding_dropout) if config.embedding_dropout > 0 else None
        self.time_encoder: Optional[TimeEncoder] = None
        self.time_state: Optional[NodeTimeState] = None
        if config.use_time_encoding:
            # `time_scale` is overridden after the dataset is loaded
            # (see set_time_scale). 1.0 is a placeholder; the real value
            # determines the ω_i geometric init scale.
            self.time_encoder = TimeEncoder(k=config.time_enc_k, time_scale=1.0).to(self.device)
            self.time_state = NodeTimeState(n_nodes=config.max_node_count)
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
        # Embedding optimizer. SGNS uses the Mikolov lr schedule (starts at
        # `sgns_lr_init`, decays linearly to `sgns_lr_final` over the first
        # `sgns_lr_decay_epochs`). Other primaries use Adam at `emb_lr`.
        # Triplet adds weight_decay_emb (the literature default 1e-4 supplies
        # the norm control that uniformity used to provide).
        if config.primary_loss == "sgns":
            emb_init_lr = config.sgns_lr_init
            emb_wd = 0.0
        else:
            emb_init_lr = config.emb_lr
            emb_wd = config.weight_decay_emb if config.primary_loss == "triplet" else 0.0
        self.emb_optimizer = torch.optim.Adam(
            self.embedding_store.parameters(),
            lr=emb_init_lr,
            weight_decay=emb_wd,
        )
        # Pre-cache the training-destination pool tensor (used by triplet /
        # InfoNCE / SGNS to draw context-side negatives).
        self._train_dst_pool_t = torch.from_numpy(
            np.asarray(train_dst_pool, dtype=np.int64),
        ).to(self.device)

        # SGNS unigram^0.75 sampling weights over training destinations
        # (Mikolov 2013). Computed once at construction. P(v) ∝ degree(v)^0.75
        # where degree(v) = count of v in the raw (non-unique) training
        # destinations array. Critical for the Levy & Goldberg PMI
        # factorization result; uniform sampling silently degrades quality.
        self._unigram_pool_t: Optional[torch.Tensor] = None       # [pool_size]
        self._unigram_probs: Optional[torch.Tensor] = None        # [pool_size]
        # Subsampling keep-probability per node (Mikolov 2013). keep(v) =
        # min(1, sqrt(t / f(v))) where f(v) = degree(v) / total. None for
        # non-SGNS paths.
        self._sgns_keep_prob: Optional[torch.Tensor] = None       # [max_node_count]
        if config.primary_loss == "sgns":
            if train_destinations_full is None:
                raise ValueError(
                    "primary_loss='sgns' requires train_destinations_full "
                    "(raw, non-unique destinations array) for unigram^0.75 "
                    "negative sampling — pass it from the entry script."
                )
            dst_full = np.asarray(train_destinations_full, dtype=np.int64)
            uniq, counts = np.unique(dst_full, return_counts=True)
            weights = counts.astype(np.float64) ** 0.75
            weights = weights / weights.sum()
            self._unigram_pool_t = torch.from_numpy(uniq).long().to(self.device)
            self._unigram_probs = torch.from_numpy(weights).float().to(self.device)
            # Subsampling: keep_prob per node. Use degree-from-destinations
            # as a proxy for walk-frequency (high-degree nodes occupy more
            # walk positions). t=1e-5 is the Mikolov default.
            total = float(dst_full.shape[0])
            t = config.sgns_subsample_t
            keep = np.ones(config.max_node_count, dtype=np.float32)  # default keep all
            f = counts.astype(np.float64) / total
            keep_v = np.minimum(1.0, np.sqrt(t / np.maximum(f, 1e-12))).astype(np.float32)
            keep[uniq] = keep_v
            self._sgns_keep_prob = torch.from_numpy(keep).float().to(self.device)
        # TimeEncoder lives in the link-side param group: it's scored-side,
        # not alignment-side, and gets gradient only from link BCE.
        link_params = list(self.link_predictor.parameters())
        if self.time_encoder is not None:
            link_params += list(self.time_encoder.parameters())
        self.link_optimizer = torch.optim.Adam(
            link_params,
            lr=config.link_lr,
            weight_decay=config.weight_decay_link,
        )
        self._time_scale = config.alignment_time_scale  # overridden after dataset load
        # Side-channel for per-epoch logging of the optional normbrake aux.
        # _embedding_step writes this each batch; train() aggregates per epoch.
        self._last_normbrake: float = 0.0

    # ------------------------------------------------------------------ #
    # Per-batch step (strict-causal: ingest is the LAST thing).
    # ------------------------------------------------------------------ #

    def _embedding_step(self, batch: Batch) -> tuple[float, float]:
        """Run the primary embedding-side loss + (optional) normbrake aux.

        Routes on `config.primary_loss`:
          - "alignment" (v2.2 default):  alignment + uniformity, scaled by
                                          lambda_align / eta_uniform.
          - "triplet" / "infonce" / "sgns": v2.3 §4.7 primaries (uniformity
                                            forced off; alignment loss
                                            replaced by the chosen primary).
        All four optionally compose with §4.4 norm-brake when
        `lambda_normbrake > 0`.

        Returns (primary_loss_value, uniformity_loss_value) — uniformity is
        reported as 0.0 for the three new primaries (it's not computed).
        """
        # Walks pre-ingest (strict-causal); union seeding (Lesson 9).
        seeds_np = np.unique(np.concatenate([batch.src, batch.tgt]))
        walks = self.walk_gen.walks_for_nodes(seeds_np)
        nodes = walks.nodes.to(self.device).long().clamp_min(0)
        edge_feats = (
            walks.edge_feats.to(self.device) if walks.edge_feats is not None else None
        )
        e_target_seed = self.embedding_store.target(walks.seeds.to(self.device))
        e_context_all = self.embedding_store.context_walk(nodes, edge_feats)
        t_query = torch.full(
            (walks.seeds.shape[0],), int(batch.t_max), dtype=torch.long, device=self.device,
        )

        primary = self.config.primary_loss
        if primary == "alignment":
            l_primary = alignment_loss(
                e_target_seed=e_target_seed,
                e_context_all=e_context_all,
                walks=walks,
                t_query=t_query,
                beta=self.config.temporal_decay_exp,
                time_scale=self._time_scale,
                weighting=self.config.align_weighting,
            )
            unique_batch_nodes = np.unique(np.concatenate([batch.src, batch.tgt]))
            ub = torch.from_numpy(unique_batch_nodes).long().to(self.device)
            l_uniform = uniformity_loss(
                self.embedding_store.target(ub),
                temperature=self.config.uniformity_temperature,
                cap=self.config.uniformity_cap,
            )
            l_total = (
                self.config.lambda_align * l_primary
                + self.config.eta_uniform * l_uniform
            )
            l_uniform_val = float(l_uniform.detach())

        elif primary == "triplet":
            # One uniform-random destination per walk as the negative.
            NK = walks.nodes.shape[0]
            neg_pool_idx = torch.randint(
                0, self._train_dst_pool_t.shape[0], (NK,), device=self.device,
            )
            neg_node_ids = self._train_dst_pool_t[neg_pool_idx]                 # [N*K]
            e_context_neg = self.embedding_store.context(neg_node_ids)          # [N*K, d]
            l_primary = triplet_loss(
                e_target_seed=e_target_seed,
                e_context_all=e_context_all,
                e_context_neg=e_context_neg,
                walks=walks,
                t_query=t_query,
                beta=self.config.temporal_decay_exp,
                time_scale=self._time_scale,
                weighting=self.config.align_weighting,
                margin=self.config.triplet_margin,
            )
            l_total = l_primary
            l_uniform_val = 0.0

        elif primary == "infonce":
            # Uniform-random destinations for the additional cross-batch
            # negatives (in-batch negatives are sampled inside the loss).
            num_unif = self.config.infonce_num_neg_unif
            if num_unif > 0:
                pool_idx = torch.randint(
                    0, self._train_dst_pool_t.shape[0], (num_unif,), device=self.device,
                )
                neg_ids = self._train_dst_pool_t[pool_idx]
                e_context_unif_neg = self.embedding_store.context(neg_ids)     # [Nu, d]
            else:
                e_context_unif_neg = torch.empty(
                    0, self.config.d_emb, device=self.device,
                )
            l_primary = infonce_loss(
                e_target_seed=e_target_seed,
                e_context_all=e_context_all,
                e_context_unif_neg=e_context_unif_neg,
                walks=walks,
                t_query=t_query,
                beta=self.config.temporal_decay_exp,
                time_scale=self._time_scale,
                weighting=self.config.align_weighting,
                tau=self.config.infonce_tau,
                num_neg_in_batch=self.config.infonce_num_neg_in_batch,
            )
            l_total = l_primary
            l_uniform_val = 0.0

        elif primary == "sgns":
            # k_neg unigram^0.75 negatives per (walk, position).
            NK, L = walks.nodes.shape
            k_neg = self.config.sgns_k_neg
            neg_pool_idx = torch.multinomial(
                self._unigram_probs,
                num_samples=NK * L * k_neg,
                replacement=True,
            )                                                             # [NK*L*k_neg]
            neg_node_ids = self._unigram_pool_t[neg_pool_idx]              # same shape
            e_neg_flat = self.embedding_store.context(neg_node_ids)        # [NK*L*k_neg, d]
            e_context_neg = e_neg_flat.view(NK, L, k_neg, -1)
            l_primary = sgns_loss(
                e_target_seed=e_target_seed,
                e_context_all=e_context_all,
                e_context_neg=e_context_neg,
                walks=walks,
                t_query=t_query,
                beta=self.config.temporal_decay_exp,
                time_scale=self._time_scale,
                weighting=self.config.align_weighting,
                subsample_keep_prob_per_node=self._sgns_keep_prob,
            )
            l_total = l_primary
            l_uniform_val = 0.0

        else:
            raise ValueError(f"unknown primary_loss={primary!r}")

        # Norm-brake auxiliary (composable with any primary). Acts on the
        # underlying embedding tables, not on the per-walk lookups.
        if self.config.lambda_normbrake > 0:
            l_nb = normbrake_loss(
                E_target=self.embedding_store.E_target.weight,
                E_context=self.embedding_store.E_context.weight,
                threshold=self.config.normbrake_threshold,
            )
            l_total = l_total + self.config.lambda_normbrake * l_nb
            self._last_normbrake = float(l_nb.detach())
        else:
            self._last_normbrake = 0.0

        self.emb_optimizer.zero_grad(set_to_none=True)
        l_total.backward()
        # Optional grad-norm logging (debug instrumentation for deep-analysis).
        # Captures the L2 norms of the embedding tables' gradients before the
        # optimizer step — the signal for "how aggressively is the loss
        # pulling the embeddings around this batch?". Aggregated per-epoch
        # by train() into `_debug_grad_acc`.
        if getattr(self, "_debug_grad_acc", None) is not None:
            g_t = self.embedding_store.E_target.weight.grad
            g_c = self.embedding_store.E_context.weight.grad
            self._debug_grad_acc["E_target_grad"] += (
                float(g_t.norm().item()) if g_t is not None else 0.0
            )
            self._debug_grad_acc["E_context_grad"] += (
                float(g_c.norm().item()) if g_c is not None else 0.0
            )
            self._debug_grad_acc["n_batches"] += 1
        self.emb_optimizer.step()
        return float(l_primary.detach()), l_uniform_val

    # ------------------------------------------------------------------ #
    # Component 0 helper: query NodeTimeState + compute Δt features
    # ------------------------------------------------------------------ #

    def _time_features(
        self,
        all_u_np: np.ndarray,
        all_v_np: np.ndarray,
        t_query: int,
    ) -> Optional[tuple]:
        """Read PRE-batch NodeTimeState, compute Φ(Δt) and cold-start bits
        for every (u, v) pair to be scored. Returns None if time encoding
        is disabled in this Trainer.

        STRICT-CAUSAL: this must be called BEFORE the post-scoring `update`
        of NodeTimeState. The state buffers read here reflect events ≤ B-1
        (where B is the current batch) — the post-scoring block writes
        batch B's events AFTER this returns.
        """
        if self.time_encoder is None or self.time_state is None:
            return None
        last_u_np, last_v_np, last_uv_np = self.time_state.query(all_u_np, all_v_np)
        t_q = int(t_query)
        # Δt = t_query - last_event_time. Strict-causal ⇒ Δt ≥ 0 by
        # construction, but clamp defensively.
        # For cold-start (last_*_time == 0), we want a large but bounded Δt
        # passing through Φ; the cold-start binary bit is the real signal.
        clamp_to = float(self._time_scale) * float(self.config.cold_start_dt_clamp_factor)
        # Compute raw Δt on CPU as float, clamp, then move to device.
        dt_u_np = np.clip((t_q - last_u_np).astype(np.float32), 0.0, clamp_to)
        dt_v_np = np.clip((t_q - last_v_np).astype(np.float32), 0.0, clamp_to)
        dt_uv_np = np.clip((t_q - last_uv_np).astype(np.float32), 0.0, clamp_to)
        # Cold-start binary bits (1.0 if never seen, 0.0 otherwise).
        is_cold_u_np = (last_u_np == 0).astype(np.float32)
        is_cold_v_np = (last_v_np == 0).astype(np.float32)
        is_cold_uv_np = (last_uv_np == 0).astype(np.float32)

        dt_u = torch.from_numpy(dt_u_np).to(self.device)
        dt_v = torch.from_numpy(dt_v_np).to(self.device)
        dt_uv = torch.from_numpy(dt_uv_np).to(self.device)
        phi_u = self.time_encoder(dt_u)                                    # [P, 2k]
        phi_v = self.time_encoder(dt_v)
        phi_uv = self.time_encoder(dt_uv)
        cold_u = torch.from_numpy(is_cold_u_np).to(self.device).unsqueeze(-1)   # [P, 1]
        cold_v = torch.from_numpy(is_cold_v_np).to(self.device).unsqueeze(-1)
        cold_uv = torch.from_numpy(is_cold_uv_np).to(self.device).unsqueeze(-1)
        return phi_u, phi_v, phi_uv, cold_u, cold_v, cold_uv

    def _link_step(self, batch: Batch) -> float:
        neg_src, neg_tgt = self.neg_sampler_train.sample(batch)
        B = len(batch.src)
        K = neg_src.shape[1]

        all_u = np.concatenate([batch.src, neg_src.reshape(-1).astype(np.int64)])
        all_v = np.concatenate([batch.tgt, neg_tgt.reshape(-1).astype(np.int64)])
        u_t = torch.from_numpy(all_u).long().to(self.device)
        v_t = torch.from_numpy(all_v).long().to(self.device)

        # Component 0: per-pair time features computed from PRE-batch state.
        # t_query is the batch's max timestamp (same convention as alignment).
        time_feats = self._time_features(all_u, all_v, int(batch.t_max))

        # §4.8.2 embedding-side dropout at the link-MLP read site (training
        # mode only — torch.nn.Dropout is a no-op during eval()).
        e_t_u = self.embedding_store.target(u_t)
        e_t_v = self.embedding_store.target(v_t)
        e_c_u = self.embedding_store.context(u_t)
        e_c_v = self.embedding_store.context(v_t)
        if self.emb_dropout is not None:
            e_t_u = self.emb_dropout(e_t_u)
            e_t_v = self.emb_dropout(e_t_v)
            e_c_u = self.emb_dropout(e_c_u)
            e_c_v = self.emb_dropout(e_c_v)
        if time_feats is not None:
            phi_u, phi_v, phi_uv, cold_u, cold_v, cold_uv = time_feats
            logits = self.link_predictor(
                e_t_u, e_t_v, e_c_u, e_c_v,
                phi_u, phi_v, phi_uv,
                cold_u, cold_v, cold_uv,
            )
        else:
            logits = self.link_predictor(e_t_u, e_t_v, e_c_u, e_c_v)
        labels = torch.cat(
            [
                torch.ones(B, device=self.device),
                torch.zeros(B * K, device=self.device),
            ],
        )
        loss = link_bce(logits, labels)
        self.link_optimizer.zero_grad(set_to_none=True)
        # Joint training (v2.3 Group C, λ_link > 0): let BCE backprop into the
        # embedding tables too. The forward already touched them via
        # embedding_store.target / .context lookups, so the autograd graph is
        # in place — we just need to zero their existing grads (primary-loss
        # grads that were already stepped in _embedding_step), let backward
        # populate them again from BCE, scale by λ_link, then step.
        joint = self.config.lambda_link > 0
        if joint:
            for p in self.embedding_store.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        loss.backward()
        self.link_optimizer.step()
        if joint:
            for p in self.embedding_store.parameters():
                if p.grad is not None:
                    p.grad.mul_(self.config.lambda_link)
            self.emb_optimizer.step()
        return float(loss.detach())

    # ------------------------------------------------------------------ #
    # Phase loops.
    # ------------------------------------------------------------------ #

    def _model_state_snapshot(self) -> Dict[str, Any]:
        """Deep-copy of model weights only (NOT walk_gen / time_state /
        neg_sampler — those are state buffers, restored by reset() each
        epoch). Used by early-stopping to remember the best epoch."""
        snap: Dict[str, Any] = {
            "embedding_store": copy.deepcopy(self.embedding_store.state_dict()),
            "link_predictor": copy.deepcopy(self.link_predictor.state_dict()),
        }
        if self.time_encoder is not None:
            snap["time_encoder"] = copy.deepcopy(self.time_encoder.state_dict())
        return snap

    def _load_model_state(self, snap: Dict[str, Any]) -> None:
        self.embedding_store.load_state_dict(snap["embedding_store"])
        self.link_predictor.load_state_dict(snap["link_predictor"])
        if self.time_encoder is not None and "time_encoder" in snap:
            self.time_encoder.load_state_dict(snap["time_encoder"])

    def train(
        self,
        batches: Iterable[Batch],
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory: Optional[Callable[[], Iterable[Batch]]] = None,
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory: Optional[Callable[[], Iterable[Batch]]] = None,
        early_stop_patience: Optional[int] = None,
        log_debug: bool = False,
        monitor_sample_pct: float = 1.0,
        skip_final_full_eval: bool = False,
    ) -> Dict[str, Any]:
        """Train up to `config.num_epochs`.

        Returns a summary dict. If `val_evaluator` is None, behaves like
        the original train(): just trains num_epochs and returns an empty
        per-epoch curve.

        If `val_evaluator` is provided, runs val eval AFTER each training
        epoch (TGB streaming convention: state at start of val eval is
        end-of-training-epoch). Tracks the best-val-MRR checkpoint and
        deep-copies model weights when val improves.

        If `test_evaluator` is also provided, runs test eval whenever val
        improves and pins `best_test_mrr` to the same epoch as
        `best_val_mrr` (so the two reported numbers come from the same
        model snapshot). Test eval starts from end-of-val state — same
        as the no-early-stop path.

        If `early_stop_patience` is set (and val_evaluator is provided),
        stops after that many epochs without val improvement.

        Best weights are restored before return so subsequent calls
        (e.g. a final evaluator outside this function) read the best
        snapshot. The walk_gen / time_state / neg_sampler buffers remain
        in their end-of-last-epoch state — if you need a clean re-eval
        from end-of-training-only state, call walk_gen.reset() and
        re-ingest training edges separately.
        """
        batches = list(batches)
        n_epochs = self.config.num_epochs
        do_val = val_evaluator is not None and val_batches_factory is not None
        do_test = (
            test_evaluator is not None and test_batches_factory is not None and do_val
        )

        best_val_mrr = -float("inf")
        best_test_mrr: Optional[float] = None
        best_epoch = 0
        best_state: Optional[Dict[str, Any]] = None
        epochs_no_improve = 0
        per_epoch_val: List[float] = []
        per_epoch_test: List[float] = []
        stopped_at_epoch = n_epochs

        # Aggregated per-epoch metrics for §4.7.5 deliverables.
        per_epoch_col_norm: List[float] = []
        per_epoch_normbrake: List[float] = []
        # Debug instrumentation (deep-analysis runs only).
        # When log_debug is True, run test eval EVERY epoch (not just on
        # new best val) and track gradient / weight-norm signals for cliff
        # localization. Also fills _debug_grad_acc so _embedding_step
        # accumulates per-batch grad norms (see end of _embedding_step).
        per_epoch_test_all: List[float] = []
        per_epoch_grad_target: List[float] = []
        per_epoch_grad_context: List[float] = []
        per_epoch_link_w_norm: List[float] = []

        for epoch in range(n_epochs):
            # SGNS lr schedule (Mikolov linear decay over first N epochs).
            if self.config.primary_loss == "sgns":
                lr_init = self.config.sgns_lr_init
                lr_final = self.config.sgns_lr_final
                decay_epochs = max(1, self.config.sgns_lr_decay_epochs)
                if epoch < decay_epochs:
                    frac = epoch / float(decay_epochs)
                    lr_now = lr_init + (lr_final - lr_init) * frac
                else:
                    lr_now = lr_final
                for pg in self.emb_optimizer.param_groups:
                    pg["lr"] = lr_now

            # ── Per-epoch reset — wipes any prior per-epoch eval mutations ──
            self.walk_gen.reset()
            if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                self.neg_sampler_train.reset()
            if self.time_state is not None:
                self.time_state.reset()
            self.embedding_store.train()
            self.link_predictor.train()
            if self.time_encoder is not None:
                self.time_encoder.train()
            t0 = time.perf_counter()
            sum_align = sum_uniform = sum_link = sum_normbrake = 0.0
            n = 0
            # Reset per-batch grad accumulator if log_debug.
            if log_debug:
                self._debug_grad_acc = {
                    "E_target_grad": 0.0, "E_context_grad": 0.0, "n_batches": 0,
                }
            else:
                self._debug_grad_acc = None
            for batch in batches:
                l_align, l_uniform = self._embedding_step(batch)
                l_link = self._link_step(batch)
                # ── Post-scoring strict-causal block (feeds batch B+1) ──
                if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                    self.neg_sampler_train.observe(batch.src, batch.tgt)
                self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
                if self.time_state is not None:
                    self.time_state.update(batch.src, batch.tgt, batch.ts)
                sum_align += l_align
                sum_uniform += l_uniform
                sum_link += l_link
                sum_normbrake += self._last_normbrake
                n += 1
            train_dt = time.perf_counter() - t0

            # § 4.7.5 cliff diagnostic: per-epoch mean column norm of the
            # joint (E_target ∪ E_context) — the direct cause-signal for the
            # over-training cliff. Mean ~0.36 at the wiki 2-ep anchor; growth
            # past ~0.55 is where the cliff starts to bite.
            with torch.no_grad():
                col_t = self.embedding_store.E_target.weight.norm(dim=0)
                col_c = self.embedding_store.E_context.weight.norm(dim=0)
                joint_mean_col = float(((col_t.sum() + col_c.sum()) / (col_t.numel() + col_c.numel())).item())
            per_epoch_col_norm.append(joint_mean_col)
            per_epoch_normbrake.append(float(sum_normbrake / max(n, 1)))
            # Debug instrumentation (log_debug only): aggregate per-batch grad
            # norms into per-epoch means, and capture the link MLP first-Linear's
            # cross-table column-norm mean (the original Phase 0.5 cliff signal).
            if log_debug and self._debug_grad_acc is not None:
                nbatch = max(self._debug_grad_acc["n_batches"], 1)
                per_epoch_grad_target.append(
                    self._debug_grad_acc["E_target_grad"] / nbatch,
                )
                per_epoch_grad_context.append(
                    self._debug_grad_acc["E_context_grad"] / nbatch,
                )
                # Link MLP first-Linear cross-table column norms (mean over
                # the first 8·d_emb input cols if head_mode == "cross_table",
                # else over the full input dim if "component_0_only").
                with torch.no_grad():
                    W = self.link_predictor.net[0].weight.detach()   # [hidden, in_d]
                    col = W.norm(dim=0)
                    if self.link_predictor.head_mode == "cross_table":
                        cross_table_n = 8 * self.config.d_emb
                        link_mean = float(col[:cross_table_n].mean().item())
                    else:
                        link_mean = float(col.mean().item())
                per_epoch_link_w_norm.append(link_mean)

            # ── Per-epoch val + (conditional) test eval ──
            val_mrr: Optional[float] = None
            test_mrr: Optional[float] = None
            eval_dt = 0.0
            if do_val:
                t1 = time.perf_counter()
                # Per-epoch monitoring eval — uses `monitor_sample_pct` for
                # cheap evals on big datasets. Final report eval (after
                # weight-restore, see post-loop block) is always full-precision.
                val_mrr = self.evaluate(
                    val_batches_factory(), val_evaluator, sample_pct=monitor_sample_pct,
                )
                per_epoch_val.append(float(val_mrr))
                if log_debug and do_test:
                    test_all = self.evaluate(
                        test_batches_factory(), test_evaluator, sample_pct=monitor_sample_pct,
                    )
                    per_epoch_test_all.append(float(test_all))
                improved = val_mrr > best_val_mrr
                if improved:
                    best_val_mrr = float(val_mrr)
                    best_epoch = epoch + 1
                    best_state = self._model_state_snapshot()
                    epochs_no_improve = 0
                    if do_test:
                        if log_debug:
                            test_mrr = test_all
                            best_test_mrr = float(test_mrr)
                            per_epoch_test.append(float(test_mrr))
                        else:
                            test_mrr = self.evaluate(
                                test_batches_factory(), test_evaluator,
                                sample_pct=monitor_sample_pct,
                            )
                            best_test_mrr = float(test_mrr)
                            per_epoch_test.append(float(test_mrr))
                else:
                    epochs_no_improve += 1
                eval_dt = time.perf_counter() - t1

            # ── Per-epoch log ──
            tag = f"  epoch {epoch+1}/{n_epochs}  "
            primary_label = self.config.primary_loss[:5]
            tag += f"{primary_label}={sum_align/n:.4f}  uniform={sum_uniform/n:.4f}  nb={sum_normbrake/max(n,1):.4f}  "
            tag += f"link={sum_link/n:.4f}  train {train_dt:.1f}s"
            if val_mrr is not None:
                tag += f"  val {val_mrr:.4f}"
                if test_mrr is not None:
                    tag += f"  test {test_mrr:.4f} (new best)"
                tag += f"  eval {eval_dt:.1f}s"
                tag += f"  patience {epochs_no_improve}/{early_stop_patience or '-'}"
            print(tag, flush=True)

            # ── Early-stop check ──
            if (
                do_val
                and early_stop_patience is not None
                and epochs_no_improve >= early_stop_patience
            ):
                stopped_at_epoch = epoch + 1
                print(
                    f"  early stop at epoch {stopped_at_epoch}; "
                    f"best epoch was {best_epoch} (val {best_val_mrr:.4f})",
                    flush=True,
                )
                break

        # ── Restore best weights (if any val tracking happened) ──
        if best_state is not None:
            self._load_model_state(best_state)
            print(
                f"  restored best weights from epoch {best_epoch} "
                f"(monitor val {best_val_mrr:.4f}"
                + (f", monitor test {best_test_mrr:.4f}" if best_test_mrr is not None else "")
                + ")",
                flush=True,
            )

        # ── Final FULL eval (always sample_pct=1.0) on the restored weights ──
        # Per-epoch numbers above are sampled if monitor_sample_pct < 1; this
        # gives the paper-defensible report value. Skippable on memory-tight
        # datasets (review): pass skip_final_full_eval=True to fall back to
        # the monitor (sampled) values as the reported result.
        final_val_full: Optional[float] = None
        final_test_full: Optional[float] = None
        if (best_state is not None and monitor_sample_pct < 1.0
                and val_evaluator is not None and not skip_final_full_eval):
            print(f"  final full eval (sample_pct=1.0) on restored weights …", flush=True)
            self.walk_gen.reset()
            if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                self.neg_sampler_train.reset()
            if self.time_state is not None:
                self.time_state.reset()
            # Re-ingest training batches' edges (cheap; no model forward).
            for b in batches:
                self.walk_gen.add_edges(b.src, b.tgt, b.ts, b.edge_feat)
                if self.time_state is not None:
                    self.time_state.update(b.src, b.tgt, b.ts)
            t_final = time.perf_counter()
            final_val_full = self.evaluate(val_batches_factory(), val_evaluator, sample_pct=1.0)
            if do_test and test_evaluator is not None:
                final_test_full = self.evaluate(test_batches_factory(), test_evaluator, sample_pct=1.0)
            print(
                f"  FINAL full val {final_val_full:.4f}"
                + (f"  test {final_test_full:.4f}" if final_test_full is not None else "")
                + f"  ({time.perf_counter() - t_final:.1f}s)",
                flush=True,
            )

        return {
            "best_epoch": best_epoch,
            # If we ran a final full eval, report THAT as best_*_mrr (the
            # paper-defensible number); else fall back to the monitor value.
            "best_val_mrr": (final_val_full if final_val_full is not None
                             else (float(best_val_mrr) if do_val else None)),
            "best_test_mrr": (final_test_full if final_test_full is not None
                              else best_test_mrr),
            "monitor_best_val_mrr": float(best_val_mrr) if do_val else None,
            "monitor_best_test_mrr": best_test_mrr,
            "stopped_at_epoch": stopped_at_epoch,
            "per_epoch_val_mrr": per_epoch_val,
            "per_epoch_test_mrr": per_epoch_test,
            "per_epoch_col_norm": per_epoch_col_norm,
            "per_epoch_normbrake": per_epoch_normbrake,
            # Debug instrumentation (populated only when log_debug=True).
            "per_epoch_test_mrr_all": per_epoch_test_all,
            "per_epoch_grad_target": per_epoch_grad_target,
            "per_epoch_grad_context": per_epoch_grad_context,
            "per_epoch_link_w_norm": per_epoch_link_w_norm,
        }

    @torch.no_grad()
    def evaluate(self, batches: Iterable[Batch], evaluator: Evaluator,
                 sample_pct: float = 1.0) -> float:
        """Streaming evaluation. Returns dataset-level metric (TGB official).

        `sample_pct < 1.0` enables sub-sampling positives per batch for
        cheap per-epoch monitoring on large datasets (tgbl-review-v2). State
        ingest (walk_gen / time_state) happens on the FULL batch regardless,
        so state evolution stays exact — only the scoring path subsamples.
        """
        self.embedding_store.eval()
        self.link_predictor.eval()
        if self.time_encoder is not None:
            self.time_encoder.eval()
        total = 0.0
        n = 0
        for batch in batches:
            # 1-3. Score the batch FIRST (pre-ingest, strictly causal).
            m, b = evaluator.evaluate_batch(batch, sample_pct=sample_pct)
            total += m
            n += b
            # 4. Ingest LAST. Eval-time embedding adaptation is intentionally
            # OFF — the model is frozen at val/test, per TGB's streaming
            # convention. The Tempest state still accumulates so subsequent
            # eval batches' walks include earlier eval edges (also TGB-
            # conventional: "previously observed test edges can be accessed
            # by the model but back-propagation [...] is not permitted").
            self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
            if self.time_state is not None:
                # Same protocol as training: time_state.update is the LAST
                # post-scoring line. Eval batch B+1's _time_features will
                # see events ≤ B (training + earlier eval batches).
                self.time_state.update(batch.src, batch.tgt, batch.ts)
        return total / max(n, 1)

    # ------------------------------------------------------------------ #
    # Hooks.
    # ------------------------------------------------------------------ #

    def set_time_scale(self, scale: float) -> None:
        """Set the alignment-loss time scale. Call after loading the dataset:
        a sensible default is (t_max_train − t_min_train) / max_walk_len so
        a typical one-step Δt maps to ~1 and the temporal-decay weight
        retains non-trivial mass at deep walk positions.

        ALSO re-initialises TimeEncoder's ω_i with the geometric schedule
        scaled to this `time_scale` — the encoder is now dataset-aware at
        init. Subsequent gradient updates will move ω_i; this is just the
        starting point.
        """
        self._time_scale = scale
        if self.time_encoder is not None:
            with torch.no_grad():
                k = self.time_encoder.k
                i = torch.arange(k, dtype=torch.float32, device=self.time_encoder.omegas.device)
                init_omegas = (1.0 / max(float(scale), 1.0)) * (1000.0 ** (-i / max(k - 1, 1)))
                self.time_encoder.omegas.copy_(init_omegas)
