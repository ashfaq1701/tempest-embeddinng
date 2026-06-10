"""Strict-causal training + eval loop.

Single Trainer class. Per-batch ordering:

  TRAINING:
    1. walks_bwd = walk_gen.walks_for_nodes_embedding_backward(seeds_tgt)
       ← pre-ingest state. Embedding alignment is backward-only;
       forward embedding alignment was ablated and dropped 2026-06-07
       on wiki (backward-only beats forward+backward by +0.009 test,
       outside the noise band).
    2. L_align = alignment_loss(walks_bwd, ...)      ← InfoNCE scalar
    3. neg = neg_sampler.sample(batch)               ← stateless uniform
       neg_tgt: [B, K_train]
    4. candidates_v = [pos_v | neg_tgt]              ← [B, 1+K_train]
       u-side walks = walk_gen.walks_for_nodes_link_pred_*(unique src)
       (backward walks — graphs are treated as undirected)
       logits = link_head(E[v].detach(), walks=walks)
       E is detached — L_link trains only the link head; L_align is
       the sole gradient path into E. (u enters only through its walks,
       whose seed slot is node u.)
       L_link = CE(logits / tau_link, target=zeros(B))
    5. L_total = L_align + L_link
    6. optimizer.zero_grad(set_to_none=True); L_total.backward(); optimizer.step()
    7. walk_gen.add_edges(B_src, B_tgt, B_ts, B_ef)  ← post-scoring, last

  EVAL (within torch.no_grad()):
    1. neg_dst_list = tgb_neg_sampler.sample(batch)  ← per-positive negs
    2. u-side walks sampled in the same direction as training.
    3. Score per-query: candidates_v = [pos_v | neg_v_padded_to_max_K]
       → logits [B, 1+max_K] via link_head.
    4. evaluator.score_to_metric(logits[i,0], logits[i,1:1+K_i]) per
       positive (TGB MRR).
    5. walk_gen.add_edges(batch)                     ← Tempest state
                                                       carries forward.
    NOTE: model parameters frozen at eval; walks ARE sampled because
    the head is walk-mediated.

Epoch boundary:
  - walk_gen.reset()
  - Model parameters and optimiser state are NOT reset.

Early stop:
  - Snapshot best-val model + head state_dicts.
  - Restore best-val weights at end of training.
  - Optimiser state not snapshotted (not needed after training stops).
  - The reported `best_val_mrr` / `best_test_mrr` are the snapshot
    values at the best-val epoch (test scored only on new-best
    epochs to save eval time).
"""

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import geoopt
import numpy as np
import torch
import torch.nn.functional as F
from prodigyopt import Prodigy
from torch.optim.lr_scheduler import LambdaLR

from .data import Batch
from .evaluator import Evaluator
from .losses import alignment_loss
from .link_pred_head import DeepSphereSimpleHead
from .link_pred_walks_features import make_head_inputs
from .model import EmbeddingTable
from .negatives import UniformNegativeSampler
from .utils import make_lr_lambda
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    # Dataset-derived (passed in by train.py).
    # Note: t_min / T_train are no longer needed by the loss under the
    # stationary-recency formulation — see `tempest_walks/data_stats.py`
    # for the TrainStats bundle that carries them (plus inter-arrival
    # statistics) for display + recency_scale derivation.
    num_nodes: int
    dst_pool: np.ndarray

    # Model.
    d_emb: int = 128

    # Loss-formulation.
    tau_align: float = 0.5      # InfoNCE alignment temperature
    tau_link: float = 1.0       # Softmax-CE link-prediction temperature
    K_train: int = 100          # Per-query training negatives. The link
                                # head sees [B, 1+K_train] candidates per
                                # query; positive at column 0.
    alignment_chunk_size: int = 8192
                                # Slices the unique-pool dimension V when
                                # computing the InfoNCE partition. Each
                                # chunk's forward is checkpointed, so
                                # backward peak memory is bounded by
                                # O(NK · chunk_size) rather than O(NK · V).
                                # When V ≤ chunk_size the loop runs once
                                # and the behaviour reduces to the dense
                                # path. 8192 fits wiki/coin in one chunk
                                # and bounds review's pathological pools.

    # Convex-combination stationary recency weight (replaces the old
    # additive 1/K_hop + t̃^β formulation). γ ∈ [0, 1] mixes the hop
    # profile and a recency profile with a FROZEN time constant.
    # `recency_scale` is the time constant in raw timestamp units
    # (data-driven from the train split's mean inter-arrival). It used
    # to be wrapped in softplus and treated as a learnable scalar
    # parameter, but it consistently collapsed toward zero under longer
    # runs — degrading the recency feature without improving val MRR —
    # so it's now plumbed through as a constant.
    gamma_recency: float = 0.4
    recency_scale: float = 1.0  # Plumbed in by train.py from TrainStats.

    # Walks. Two sides.
    #   embedding_* — drive the geometry of E via the alignment loss
    #                  (BACKWARD walks from each batch tgt only;
    #                  forward embedding alignment was ablated and
    #                  dropped 2026-06-07).
    #   link_pred_* — drive the walk-mediated link head. The head's
    #                  direction: BACKWARD walks (graphs undirected).
    embedding_num_walks_per_node: int = 10
    embedding_max_walk_len: int = 20
    embedding_backward_walk_bias:  str = "ExponentialWeight"
    embedding_backward_start_bias: str = "ExponentialWeight"
    link_pred_num_walks_per_node: int = 3
    link_pred_max_walk_len: int = 20
    link_pred_forward_walk_bias:  str = "ExponentialWeight"
    link_pred_forward_start_bias: str = "Uniform"
    link_pred_backward_walk_bias:  str = "ExponentialWeight"
    link_pred_backward_start_bias: str = "ExponentialWeight"
    max_time_capacity: int = -1     # Tempest sliding-window eviction
                                    # in raw timestamp units; -1 = unbounded.
    # Per-node train-edge incidence count, plumbed in by train.py.
    # Drives the always-on inverse-degree seed weighting in the
    # alignment loss (per-row weight = 1/log1p(deg(seed))).
    train_deg: Optional[np.ndarray] = None  # [num_nodes] int64

    # The link head (DeepSphereSimpleHead) scores each candidate against u's
    # recency-pooled walk history by a tied ReZero deep on-sphere map + bounded
    # cosine. The walk direction is BACKWARD (graphs are treated as
    # undirected; forward-embedding alignment was ablated and dropped).
    # E is detached on all link-head lookups (E[u], E[v], E_walks);
    # L_link trains only the link head, L_align is the sole gradient
    # path into E. A controllable L_link → E leak (convex-combo grad
    # mix) was tried and rejected on tgbl-wiki (4-cell α∈{0.2, 0.5, 1.0}
    # × {both, backward} sweep 2026-06-06: every α>0 cell trailed α=0
    # on test by 0.008–0.029).
    # Dataset-derived (plumbed by train.py from TrainStats). Only T_full is
    # consumed by the head (elapsed normaliser); the others are kept for
    # display / data-stats compatibility.
    t_min: int = 0
    T_train: float = 1.0
    t_max_full: int = 1     # max timestamp across train + val + test
    T_full: float = 1.0     # span (t_max_full - t_min); the head's elapsed
                            # channel divides by this so the gap
                            # is bounded [0, 1] at train and eval.

    # Optimisation. Two optimisers over disjoint params:
    #   head — Prodigy (parameter-free; `lr` is a multiplier left at 1.0,
    #          the schedule rides on top). weight_decay / lr_min below are
    #          the HEAD knobs.
    #   E    — RiemannianAdam on the unit sphere (emb_* knobs). Prodigy's
    #          auto-scale is gone for E, so emb_lr is a REAL peak LR.
    # warmup_fraction / warmup_steps_cap / decay_horizon_epochs set the
    # shared warmup+cosine SHAPE for both schedules.
    lr_min: float = 1e-5        # head Prodigy-multiplier cosine floor
                                # (~peak/1000; matches contrastive-SSL
                                # cosine-to-near-0 norm).
    warmup_fraction: float = 0.05
    warmup_steps_cap: int = 500
    decay_horizon_epochs: int = 50  # cosine reaches the floor at this
                                    # epoch count. SEPARATE from
                                    # num_epochs — short runs stay
                                    # near peak; full decay is hit
                                    # only at num_epochs = horizon.
    weight_decay: float = 1e-4  # head Prodigy weight decay (decoupled).
    emb_lr: float = 1e-3        # E peak LR (RiemannianAdam, real LR — no
                                # Prodigy auto-scale). Re-tune with tau_align.
    emb_lr_min: float = 1e-5    # E cosine floor.
    emb_weight_decay: float = 0.0  # MUST be 0: ‖E‖≡1 makes the ‖E‖² penalty
                                   # a constant (zero gradient); WD here is
                                   # dead arithmetic and can interact with
                                   # the retraction.
    num_epochs: int = 50
    early_stop_patience: int = 0

    # System.
    seed: int = 42
    use_gpu: bool = False
    use_gpu_tempest: bool = False    # independent from use_gpu


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
        )

        # Model. The alignment loss operates directly on E rows (no
        # projection head). When we want to fuse node / edge features
        # we will add a fusion path that both losses share, not a
        # learned MLP only seen by L_align.
        self.embedding_table = EmbeddingTable(
            num_nodes=config.num_nodes,
            d_emb=config.d_emb,
        ).to(self.device)
        # Walk-mediated link-pred head (see link_pred_head.py): a tied ReZero
        # deep on-sphere map + bounded cosine. The head consumes ONE direction
        # of walks; graphs are undirected so the head consumes BACKWARD
        # walks (u's most-recent predecessors).
        self.link_head = DeepSphereSimpleHead(d=int(config.d_emb)).to(self.device)
        # Frozen recency time constant. Plain Python float — no
        # tensor, no autograd. alignment_loss broadcasts it against
        # the per-batch gap tensor at the use site.
        self.recency_scale = float(config.recency_scale)

        # T_full (span over train+val+test) bounds the head's elapsed
        # normaliser at both train and eval; read by make_head_inputs.
        self.T_full = float(config.T_full)

        # Frozen training-degree per node, used by inverse-degree seed
        # weighting in alignment_loss. Plain int64 CPU tensor; the
        # call site indexes it by per-batch seeds_np and moves the
        # tiny slice to GPU on demand. Zero entries for nodes that
        # never appear in train edges (handled by log1p+division so
        # they map to a finite weight).
        if config.train_deg is not None:
            self.train_deg = torch.from_numpy(
                np.asarray(config.train_deg, dtype=np.int64)
            )
        else:
            self.train_deg = torch.zeros(config.num_nodes, dtype=torch.int64)

        # Walk sampler. Graphs are treated as undirected: both the embedding
        # side and the link-pred head draw BACKWARD walks (the head consumes
        # u's most-recent predecessors).
        emb_K = max(1, int(config.embedding_num_walks_per_node))
        link_K_per_dir = max(1, int(config.link_pred_num_walks_per_node))
        self.walk_gen = WalkGenerator(
            use_gpu=config.use_gpu_tempest,
            embedding_backward_walk_bias=config.embedding_backward_walk_bias,
            embedding_backward_start_bias=config.embedding_backward_start_bias,
            embedding_num_walks_per_node=emb_K,
            embedding_max_walk_len=config.embedding_max_walk_len,
            link_pred_forward_walk_bias=config.link_pred_forward_walk_bias,
            link_pred_forward_start_bias=config.link_pred_forward_start_bias,
            link_pred_backward_walk_bias=config.link_pred_backward_walk_bias,
            link_pred_backward_start_bias=config.link_pred_backward_start_bias,
            link_pred_num_walks_per_node=link_K_per_dir,
            link_pred_max_walk_len=config.link_pred_max_walk_len,
            max_time_capacity=config.max_time_capacity,
        )

        # Negative sampler (training). UniformNegativeSampler is the
        # only training-side sampler: HistoricalNegativeSampler was
        # removed because on recurrence-dominated datasets it pushes E
        # away from the eval-signal direction (now that detach is off
        # on the link path, the historical-negative gradient leaks into
        # E rather than just the link head).
        self.neg_sampler_train = UniformNegativeSampler(
            num_neg_per_pos=config.K_train,
            dst_pool=config.dst_pool,
            seed=config.seed,
        )

        # Single optimiser. E is trained by L_align only (directly —
        # no projection heads); L_link sees E.detach() so only the link
        # head's parameters receive the link-loss gradient.
        # recency_scale is frozen.
        # Two optimisers over DISJOINT parameter sets, one backward feeds
        # both (autograd fills .grad for every leaf with a gradient path;
        # each optimiser steps only its own params).
        #
        # Head — Prodigy (unchanged form): hyperparameter-free, `lr` is a
        # multiplier left at 1.0, the cosine LambdaLR rides on top.
        # decouple=True → AdamW-style WD; safeguard_warmup=True keeps d_k
        # from over-shooting before warmup. NOTE: Prodigy's global step-size
        # d_k is now estimated from HEAD gradients alone (E is no longer in
        # this group), so the head's effective LR differs from the old
        # combined optimiser — watched, not a no-op.
        self.opt_head = Prodigy(
            list(self.link_head.parameters()),
            lr=1.0,
            weight_decay=config.weight_decay,
            decouple=True,
            safeguard_warmup=True,
        )
        # Embedding — Riemannian Adam on the unit sphere. Reads `.manifold`
        # off the ManifoldParameter and does egrad→rgrad, retraction, and
        # transported momentum internally. weight_decay MUST be 0 (‖E‖≡1).
        # stabilize=10 periodically re-projects to correct float drift.
        self.opt_emb = geoopt.optim.RiemannianAdam(
            list(self.embedding_table.parameters()),
            lr=config.emb_lr,
            weight_decay=0.0,
            stabilize=10,
        )

        # LR schedules are set up at the start of train() once we know
        # batches_per_epoch (which depends on the train_batches_factory).
        self.sched_head: Optional[LambdaLR] = None
        self.sched_emb: Optional[LambdaLR] = None
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
        # Shared warmup+cosine SHAPE (same batches_per_epoch / horizon for
        # both optimisers); each gets its own peak and floor.
        decay_steps = self.config.decay_horizon_epochs * max(batches_per_epoch, 1)
        warmup_steps = min(
            int(self.config.warmup_fraction * decay_steps),
            self.config.warmup_steps_cap,
        )
        warmup_steps = max(warmup_steps, 1)

        def _build(optimizer, peak_lr, floor_lr):
            for pg in optimizer.param_groups:
                pg["lr"] = peak_lr
                pg["initial_lr"] = peak_lr
            ratio = floor_lr / peak_lr if peak_lr > 0 else 0.0
            return LambdaLR(optimizer, lr_lambda=make_lr_lambda(
                warmup_steps, decay_steps, ratio))

        # Head — Prodigy treats lr as a multiplier on its discovered d_k,
        # so peak=1.0 and config.lr_min is the multiplier floor.
        self.sched_head = _build(self.opt_head, 1.0, float(self.config.lr_min))
        # E — RiemannianAdam uses a REAL LR, so the schedule directly
        # modulates the step size between emb_lr and emb_lr_min.
        self.sched_emb = _build(
            self.opt_emb, float(self.config.emb_lr), float(self.config.emb_lr_min))
        self._global_step = 0
        print(
            f"  LR schedule (warmup+cosine): head peak=1.00 (Prodigy mult) "
            f"min={float(self.config.lr_min):.2e}; "
            f"E peak={float(self.config.emb_lr):.2e} min={float(self.config.emb_lr_min):.2e}; "
            f"warmup={warmup_steps} steps, decay_horizon={decay_steps} steps "
            f"({self.config.decay_horizon_epochs} epochs × {batches_per_epoch} batches)"
        )

    def _sample_u_walks_for_head(
        self,
        u_ids_np,                # np.ndarray [B] source ids
        t_query_np,              # np.ndarray [B] query timestamps
    ):
        """Sample u-side BACKWARD walks (graphs are undirected; the head
        consumes u's most-recent predecessors) and package them into the
        per-position feature dict the head consumes. Returns a single dict.
        """
        sampler = self.walk_gen.walks_for_nodes_link_pred_backward

        # Sample once per UNIQUE u, then scatter back to row order.
        uniq, inv = np.unique(u_ids_np, return_inverse=True)
        wd = sampler(uniq)
        # wd.nodes: [N_u*K, L]; rows [i*K:(i+1)*K) belong to uniq[i].
        K = wd.K
        L = wd.nodes.shape[1]
        nodes_full = wd.nodes.view(len(uniq), K, L)
        ts_full    = wd.timestamps.view(len(uniq), K, L)
        lens_full  = wd.lens.view(len(uniq), K)
        nodes_per_row = [nodes_full[idx] for idx in inv]
        ts_per_row    = [ts_full[idx]    for idx in inv]
        lens_per_row  = [lens_full[idx]  for idx in inv]

        t_query_t = torch.from_numpy(t_query_np).long()
        return make_head_inputs(
            walks_nodes_per_u=nodes_per_row,
            walks_ts_per_u=ts_per_row,
            walks_lens_per_u=lens_per_row,
            t_query_per_u=t_query_t,
            T_full=self.T_full,
            embedding_table=self.embedding_table,
            device=self.device,
        )

    def _head_logits(self, e_v: torch.Tensor, walks: dict) -> torch.Tensor:
        """Flatten the per-(W,L) walk dict to the head's [B, L, d] contract and
        score. The head pools over all positions, so W and L collapse."""
        B, _, d = e_v.shape
        E_w = walks["E_walks"].reshape(B, -1, d)
        elapsed = walks["elapsed"].reshape(B, -1)
        mask = walks["mask"].reshape(B, -1)
        return self.link_head(e_v, E_w, elapsed, mask)

    def _score_query(
        self,
        candidates_v: torch.Tensor,
        u_ids_np,                # numpy [B]  query sources, for walk sampling
        t_query_np,              # numpy [B] query timestamps
    ) -> torch.Tensor:
        """Per-query scoring for eval (no_grad context).

        candidates_v: [B, 1+K_eval]     — column 0 = positive dst,
                                          columns 1..K = TGB negatives

        Returns: [B, 1+K_eval] logits.

        The head consumes u-side walks (sampled per the configured
        direction); no symmetrisation (the head is inherently u→v
        directional — backward symmetrisation would require sampling
        walks from each candidate v, prohibitive at K_eval=999).
        """
        e_v = self.embedding_table(candidates_v).detach()
        walks = self._sample_u_walks_for_head(u_ids_np, t_query_np)
        return self._head_logits(e_v, walks)

    # ──────────────────────────────────────────────────────────────────
    # Per-batch training step
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device

        # Step 1: walks from PRE-INGEST state.
        # Embedding supervision: BACKWARD walks from each unique target
        # (forward embedding alignment was ablated and dropped — see
        # the module docstring).
        seeds_tgt_np = np.unique(batch.tgt)
        walks_bwd = self.walk_gen.walks_for_nodes_embedding_backward(seeds_tgt_np)

        # Inverse-degree seed weighting (always on). The phase-6
        # both-active hard cohort has median seed degree 1; the
        # popular end has degree 100+. Uniform per-row averaging
        # starves rare seeds. Weighting each walk row by
        # 1/log1p(deg(seed_of_row)) restores parity.
        seed_weights_bwd = 1.0 / torch.log1p(
            self.train_deg[torch.from_numpy(seeds_tgt_np.astype(np.int64))]
            .to(self.device).float()
        )

        # Step 2: InfoNCE contrastive alignment over batched walks.
        # The softmax denominator over all batch contexts is the
        # anti-collapse mechanism (replaces Wang-Isola uniformity).
        # Per-position weights are a convex combination of hop and
        # *stationary* recency profiles.
        l_align = alignment_loss(
            embedding_table=self.embedding_table,
            walks=walks_bwd,
            recency_scale=self.recency_scale,
            gamma_recency=self.config.gamma_recency,
            tau_align=self.config.tau_align,
            chunk_size=self.config.alignment_chunk_size,
            direction="backward",
            seed_weights=seed_weights_bwd,
        )

        # Step 3: per-query negatives. Sampler returns [B, K_train]
        # grouped output; no flattening — the link head consumes the
        # whole candidate matrix directly. The sampler's neg_src is
        # ignored (the head scores candidates against u's walks, not a
        # paired source embedding).
        _, neg_tgt = self.neg_sampler_train.sample(batch)
        B = len(batch.src)

        # Step 4: build [B, 1+K] candidate matrix with positive at
        # column 0, then score the whole batch in one forward and
        # apply softmax CE with target=0 per row (Bruch et al. 2019
        # — upper-bounds 1 − MRR).
        pos_v_t = torch.from_numpy(batch.tgt).long().to(device)      # [B]
        neg_v_t = torch.from_numpy(
            np.ascontiguousarray(neg_tgt, dtype=np.int64),
        ).to(device)                                                 # [B, K]

        candidates_v = torch.cat(
            [pos_v_t.unsqueeze(1), neg_v_t], dim=1,
        )                                                            # [B, 1+K]

        # Detach on the link path: L_link trains only the link head;
        # E is updated exclusively by L_align (walk-supervised). Without
        # detach, the softmax-CE over 100 random negatives drives E to
        # memorise training (u, v⁺) pair geometry — visible as link loss
        # collapsing far below the uniform baseline log(1+K_train) ≈ 4.62
        # while val MRR drops monotonically from epoch 1. On
        # high-surprise datasets (review surprise=0.987) this anti-ranks
        # val positives because memorised seen-pair geometry has no
        # bearing on unseen pairs.
        e_v = self.embedding_table(candidates_v).detach()            # [B, 1+K, d]
        # Walk-mediated head: sample u-side walks (one direction, set by
        # undirected) at the batch query times and pass the walk
        # feature dict. The head is inherently u→v directional; no
        # symmetrisation (cf. _score_query docstring).
        walks = self._sample_u_walks_for_head(
            batch.src.astype(np.int64), batch.ts.astype(np.int64),
        )
        logits = self._head_logits(e_v, walks)                       # [B, 1+K]

        target = torch.zeros(B, dtype=torch.long, device=device)
        l_link = F.cross_entropy(logits / self.config.tau_link, target)

        # Step 5 + 6: total loss + ONE backward + step both optimisers.
        # Autograd populates .grad for every leaf with a gradient path; each
        # optimiser steps only its own (disjoint) params. E receives grad
        # from L_align only (E is detached on the link path).
        l_total = l_align + l_link
        self.opt_head.zero_grad(set_to_none=True)
        self.opt_emb.zero_grad(set_to_none=True)
        l_total.backward()
        self.opt_head.step()
        self.opt_emb.step()
        if self.sched_head is not None:
            self.sched_head.step()
            self.sched_emb.step()
            self._global_step += 1

        # Step 7: ingest into Tempest (LAST).
        self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)

        return {
            "align": float(l_align.detach()),
            "link": float(l_link.detach()),
            "total": float(l_total.detach()),
            "lr": float(self.opt_emb.param_groups[0]["lr"]),
        }

    # ──────────────────────────────────────────────────────────────────
    # Eval — strict-causal, no_grad
    # ──────────────────────────────────────────────────────────────────

    def _eval(
        self,
        evaluator: Evaluator,
        batches: Iterable[Batch],
    ) -> float:
        """Streaming eval. Walks NOT sampled. Tempest state advances
        via post-scoring add_edges."""
        self.embedding_table.eval()
        self.link_head.eval()

        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in batches:
                B = len(batch.src)
                if B == 0:
                    # Still advance Tempest state with the FULL batch (strict-causal).
                    self.walk_gen.add_edges(
                        batch.src, batch.tgt, batch.ts, batch.edge_feat,
                    )
                    continue

                # TGB-supplied per-positive negative destinations.
                # neg_tgt_list[i] is a 1D array of negative dsts for
                # positive i; counts may differ across i (boundaries
                # of TGB's negative sets), so we pad to max_K and
                # slice per-row downstream.
                _, neg_tgt_list = evaluator.sample_negatives(batch)
                counts = [int(arr.shape[0]) for arr in neg_tgt_list]
                max_K = max(counts) if counts else 0

                # candidates_v: [B, 1 + max_K] int64.
                # Column 0 = positive dst; columns 1..1+K_i = negatives;
                # padded columns repeat the positive (safe lookup; the
                # scores for padded columns are never read because the
                # per-row slice uses K_i).
                pos_v_np = batch.tgt.astype(np.int64)
                cand_v_np = np.tile(pos_v_np[:, None], (1, 1 + max_K))
                for i in range(B):
                    K_i = counts[i]
                    if K_i > 0:
                        cand_v_np[i, 1 : 1 + K_i] = (
                            neg_tgt_list[i].astype(np.int64)
                        )

                u_ids_np = batch.src.astype(np.int64)
                t_query_np = batch.ts.astype(np.int64)
                candidates_v = torch.from_numpy(cand_v_np).to(self.device)
                logits = self._score_query(
                    candidates_v, u_ids_np, t_query_np,
                ).cpu().numpy()

                for i in range(B):
                    K_i = counts[i]
                    m = evaluator.score_to_metric(
                        float(logits[i, 0]),
                        logits[i, 1 : 1 + K_i],
                    )
                    total += m
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
        return {
            "embedding_table": self._cpu_state_dict(self.embedding_table),
            "link_head":       self._cpu_state_dict(self.link_head),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.embedding_table.load_state_dict(snap["embedding_table"])
        self.link_head.load_state_dict(snap["link_head"])

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
            # Epoch boundary: walks reset only (the uniform sampler is
            # stateless; nothing else carries over).
            self.walk_gen.reset()

            self.embedding_table.train()
            self.link_head.train()

            t0 = time.time()
            sums = {"align": 0.0, "link": 0.0, "total": 0.0}
            n_batches = 0
            for batch in train_batches_factory():
                metrics = self._train_step(batch)
                for k in sums:
                    sums[k] += metrics[k]
                n_batches += 1
            train_dt = time.time() - t0

            scale_now = self.recency_scale
            line = (
                f"epoch {ep}/{n_epochs}  "
                f"align={sums['align']/max(n_batches,1):.4f}  "
                f"link={sums['link']/max(n_batches,1):.4f}  "
                f"scale={scale_now:.1f}  "
                f"train {train_dt:.1f}s"
            )

            if val_evaluator is not None and val_batches_factory is not None:
                t1 = time.time()
                val_metric = self._eval(val_evaluator, val_batches_factory())
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
                            test_evaluator, test_batches_factory(),
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

        return {
            "stopped_at_epoch": best_epoch if best_snap is not None else n_epochs,
            "best_val_mrr": best_val,
            "best_test_mrr": best_test,
            "per_epoch_val_mrr": per_epoch_val,
            "per_epoch_test_mrr": per_epoch_test,
        }
