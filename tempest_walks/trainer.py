"""Strict-causal training + eval loop.

Single Trainer class. Per-batch ordering:

  TRAINING:
    1. walks = walk_gen.walks_for_nodes(seeds)       ← pre-ingest state
       seeds = unique(B_tgt)               if is_directed
       seeds = unique(B_src ∪ B_tgt)       if undirected
    2. L_align = alignment_loss(walks, tau_align)    ← InfoNCE scalar
    3. neg = neg_sampler.sample(batch)               ← pre-observe state
       neg_tgt: [B, K_train]
    4. candidates_v = [pos_v | neg_tgt]              ← [B, 1+K_train]
       logits = link_head(E[u].detach(), E[v].detach())  ← [B, 1+K_train]
                (undirected: 0.5 × (link_head(u,v) + link_head(v,u)))
       L_link = CE(logits / tau_link, target=zeros(B))
    5. L_total = L_align + L_link
    6. optimizer.zero_grad(set_to_none=True); L_total.backward(); optimizer.step()
    7. neg_sampler.observe(B_src, B_tgt)             ← post-scoring
    8. walk_gen.add_edges(B_src, B_tgt, B_ts, B_ef)  ← post-scoring, last

  EVAL (within torch.no_grad()):
    1. neg_dst_list = tgb_neg_sampler.sample(batch)  ← per-positive negs
    2. Score per-query: candidates_v = [pos_v | neg_v_padded_to_max_K]
       → logits [B, 1+max_K] via link_head (undirected symmetrise
       matches the training rule).
    3. evaluator.score_to_metric(logits[i,0], logits[i,1:1+K_i]) per
       positive (TGB MRR).
    4. walk_gen.add_edges(batch)                     ← Tempest state
                                                       carries forward.
    NOTE: reservoir not updated, walks not sampled at eval.
          Model parameters frozen.

Epoch boundary:
  - walk_gen.reset()
  - neg_sampler.reset() (if Historical)
  - Model parameters and optimiser state are NOT reset.

Early stop:
  - Snapshot best-val model + projection + head state_dicts.
  - Restore best-val weights at end of training.
  - Optimiser state not snapshotted (not needed after training stops).
  - The reported `best_val_mrr` / `best_test_mrr` are the snapshot
    values at the best-val epoch (test scored only on new-best
    epochs to save eval time).
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from prodigyopt import Prodigy
from torch.optim.lr_scheduler import LambdaLR

from .data import Batch
from .evaluator import Evaluator
from .losses import alignment_loss
from .model import EmbeddingTable, LinkHead, ProjectionHead
from .negatives import (
    HistoricalNegativeSampler,
    NegativeSampler,
    UniformNegativeSampler,
)
from .utils import make_lr_lambda
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    # Dataset-derived (passed in by train.py).
    num_nodes: int
    is_directed: bool
    dst_pool: np.ndarray
    t_min: int                          # min timestamp of the train split
    T_train: float                      # train-split span (t_max - t_min)
    d_node_feat: Optional[int] = None

    # Model.
    d_emb: int = 128
    d_proj: int = 128

    # Loss-formulation.
    tau_align: float = 0.5      # InfoNCE alignment temperature
    tau_link: float = 1.0       # Softmax-CE link-prediction temperature
    beta_time: float = 1.0      # hop/time weight exponent
    K_train: int = 100          # Per-query training negatives. The link
                                # head sees [B, 1+K_train] candidates per
                                # query; positive at column 0.

    # Walks.
    num_walks_per_node: int = 5
    max_walk_len: int = 20
    walk_bias: str = "ExponentialWeight"
    start_bias: str = "Uniform"
    max_time_capacity: int = -1     # Tempest sliding-window eviction
                                    # in raw timestamp units; -1 = unbounded.

    # Negatives (training).
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

    # System.
    seed: int = 42
    use_gpu: bool = False
    use_gpu_tempest: bool = False    # independent from use_gpu


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
        self.link_head = LinkHead(d_emb=config.d_emb).to(self.device)

        # Walk sampler.
        self.walk_gen = WalkGenerator(
            is_directed=config.is_directed,
            use_gpu=config.use_gpu_tempest,
            walk_bias=config.walk_bias,
            start_bias=config.start_bias,
            max_walk_len=config.max_walk_len,
            num_walks_per_node=config.num_walks_per_node,
            max_time_capacity=config.max_time_capacity,
        )

        # Negative samplers (training). The sampler's per-positive
        # negative count is K_train under the new per-query ranking
        # protocol — sampler's internal `num_neg_per_pos` kwarg
        # name stays the same (sampler API is unchanged).
        if config.hist_neg_ratio > 0:
            self.neg_sampler_train: NegativeSampler = HistoricalNegativeSampler(
                num_nodes=config.num_nodes,
                num_neg_per_pos=config.K_train,
                hist_ratio=config.hist_neg_ratio,
                reservoir_size=config.reservoir_size,
                dst_pool=config.dst_pool,
                seed=config.seed,
            )
        else:
            self.neg_sampler_train = UniformNegativeSampler(
                num_neg_per_pos=config.K_train,
                dst_pool=config.dst_pool,
                seed=config.seed,
            )

        # Single optimiser over all trainable parameters. The decoupling
        # between E-training (InfoNCE alignment) and link-head training
        # (per-query softmax CE) comes from the .detach() at the link
        # call site, NOT from separate optimisers.
        params = (
            list(self.embedding_table.parameters())
            + list(self.p_target.parameters())
            + list(self.p_context.parameters())
            + list(self.link_head.parameters())
        )
        # Prodigy: hyperparameter-free adaptive optimiser. Internally
        # estimates the optimal step-size; the `lr` arg is a multiplier
        # on the discovered scale and should be left at 1.0. The
        # cosine warmup+decay LambdaLR below still modulates the
        # effective step (schedule shape applies on top of Prodigy's
        # discovered d_k). decouple=True gives AdamW-style weight decay.
        # safeguard_warmup=True keeps d_k from over-shooting before the
        # warmup phase has produced meaningful gradient statistics.
        self.optimizer = Prodigy(
            params,
            lr=1.0,
            weight_decay=config.weight_decay,
            decouple=True,
            safeguard_warmup=True,
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
        # Prodigy treats `lr` as a multiplier on its internally
        # discovered step-size d_k. The cosine LambdaLR rides on top
        # of that multiplier — peak=1.0 leaves Prodigy free to scale
        # by its discovery, while the schedule shape (warmup +
        # cosine to a small ratio) anneals the multiplier near end
        # of training. lr_min_ratio is computed against peak=1.0,
        # so config.lr_min is interpreted directly as the floor.
        peak_lr = 1.0
        lr_min = float(self.config.lr_min)

        decay_steps = self.config.decay_horizon_epochs * max(batches_per_epoch, 1)
        warmup_steps = min(
            int(self.config.warmup_fraction * decay_steps),
            self.config.warmup_steps_cap,
        )
        warmup_steps = max(warmup_steps, 1)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = peak_lr
            param_group["initial_lr"] = peak_lr

        lr_min_ratio = lr_min / peak_lr if peak_lr > 0 else 0.0
        lr_lambda = make_lr_lambda(warmup_steps, decay_steps, lr_min_ratio)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self._global_step = 0
        print(
            f"  LR schedule (Prodigy multiplier): peak={peak_lr:.2e}, "
            f"min={lr_min:.2e}, warmup={warmup_steps} steps, "
            f"decay_horizon={decay_steps} steps "
            f"({self.config.decay_horizon_epochs} epochs × "
            f"{batches_per_epoch} batches)"
        )

    def _score_query(
        self,
        u_ids: torch.Tensor,
        candidates_v: torch.Tensor,
    ) -> torch.Tensor:
        """Per-query scoring for eval (no_grad context).

        u_ids:        [B]               — query source nodes
        candidates_v: [B, 1+K_eval]     — column 0 = positive dst,
                                          columns 1..K = TGB negatives
        Returns:       [B, 1+K_eval]    — logits per candidate.

        Memory: the link head's forward materialises tensors of shape
        [B, 1+K_eval, d_emb]. Budget eval-batch-size accordingly. With
        TGB's K values (wiki=999, review=100, coin=20, comment=20),
        comfortable eval_batch_size values are ~25-50 / ~200-500 /
        ~2000+ / ~2000+ at d_emb=128 on an 8 GB GPU.

        Symmetrisation matches the training rule: undirected datasets
        average link_head(E[u], E[v]) with link_head(E[v], E[u]) so
        train and eval score by the exact same formula; directed
        datasets do one call.
        """
        candidates_u = u_ids.unsqueeze(1).expand_as(candidates_v)
        e_u = self.embedding_table(candidates_u)
        e_v = self.embedding_table(candidates_v)
        if self.config.is_directed:
            return self.link_head(e_u, e_v)
        return 0.5 * (
            self.link_head(e_u, e_v) + self.link_head(e_v, e_u)
        )

    # ──────────────────────────────────────────────────────────────────
    # Per-batch training step
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device

        # Step 1: walks from PRE-INGEST state.
        # Seed selection depends on directedness:
        #  - undirected: seeds = unique(src ∪ tgt). Both endpoints are
        #    symmetric roles; walking from either captures the local
        #    neighbourhood equally.
        #  - directed:   seeds = unique(tgt). TGB ranks candidate dsts
        #    given a fixed src, so the dst side is what gets scored;
        #    walking from the dst follows its incoming-edge history,
        #    which is the predictive signal for "which dst did src hit".
        if self.config.is_directed:
            seeds_np = np.unique(batch.tgt)
        else:
            seeds_np = np.unique(np.concatenate([batch.src, batch.tgt]))
        walks = self.walk_gen.walks_for_nodes(seeds_np)

        # Step 2: InfoNCE contrastive alignment over batched walks.
        # The softmax denominator over all batch contexts is the
        # anti-collapse mechanism (replaces Wang-Isola uniformity).
        # Time weight is FIXED per edge via (t_min, T_train) from the
        # train split — no t_now reference, no drift across batches.
        l_align = alignment_loss(
            embedding_table=self.embedding_table,
            p_target=self.p_target,
            p_context=self.p_context,
            walks=walks,
            t_min=self.config.t_min,
            T_train=self.config.T_train,
            beta=self.config.beta_time,
            tau_align=self.config.tau_align,
            node_feat=self.node_feat,
        )

        # Step 3: per-query negatives. Sampler returns [B, K_train]
        # grouped output; no flattening — the link head consumes the
        # whole candidate matrix directly. The sampler's neg_src is
        # ignored (we re-derive candidates_u from batch.src).
        _, neg_tgt = self.neg_sampler_train.sample(batch)
        B = len(batch.src)
        K = neg_tgt.shape[1]

        # Step 4: build [B, 1+K] candidate matrix with positive at
        # column 0, then score the whole batch in one forward and
        # apply softmax CE with target=0 per row (Bruch et al. 2019
        # — upper-bounds 1 − MRR).
        src_t = torch.from_numpy(batch.src).long().to(device)        # [B]
        pos_v_t = torch.from_numpy(batch.tgt).long().to(device)      # [B]
        neg_v_t = torch.from_numpy(
            np.ascontiguousarray(neg_tgt, dtype=np.int64),
        ).to(device)                                                 # [B, K]

        candidates_v = torch.cat(
            [pos_v_t.unsqueeze(1), neg_v_t], dim=1,
        )                                                            # [B, 1+K]
        candidates_u = src_t.unsqueeze(1).expand(B, 1 + K)           # [B, 1+K]

        e_u = self.embedding_table(candidates_u).detach()            # [B, 1+K, d]
        e_v = self.embedding_table(candidates_v).detach()            # [B, 1+K, d]
        if self.config.is_directed:
            logits = self.link_head(e_u, e_v)                        # [B, 1+K]
        else:
            logits = 0.5 * (
                self.link_head(e_u, e_v) + self.link_head(e_v, e_u)
            )                                                        # [B, 1+K]

        target = torch.zeros(B, dtype=torch.long, device=device)
        l_link = F.cross_entropy(logits / self.config.tau_link, target)

        # Step 5 + 6: total loss + single backward + step.
        l_total = l_align + l_link
        self.optimizer.zero_grad(set_to_none=True)
        l_total.backward()
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
            "link": float(l_link.detach()),
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
    ) -> float:
        """Streaming eval. Walks NOT sampled; reservoir NOT updated.
        Tempest state advances via post-scoring add_edges."""
        self.embedding_table.eval()
        self.p_target.eval()
        self.p_context.eval()
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

                u_ids = torch.from_numpy(
                    batch.src.astype(np.int64),
                ).to(self.device)                                  # [B]
                candidates_v = torch.from_numpy(cand_v_np).to(self.device)
                logits = self._score_query(u_ids, candidates_v).cpu().numpy()

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
            "p_target":        self._cpu_state_dict(self.p_target),
            "p_context":       self._cpu_state_dict(self.p_context),
            "link_head":       self._cpu_state_dict(self.link_head),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.embedding_table.load_state_dict(snap["embedding_table"])
        self.p_target.load_state_dict(snap["p_target"])
        self.p_context.load_state_dict(snap["p_context"])
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
            # Epoch boundary: walks + reservoir reset only.
            self.walk_gen.reset()
            if isinstance(self.neg_sampler_train, HistoricalNegativeSampler):
                self.neg_sampler_train.reset()

            self.embedding_table.train()
            self.p_target.train()
            self.p_context.train()
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

            line = (
                f"epoch {ep}/{n_epochs}  "
                f"align={sums['align']/max(n_batches,1):.4f}  "
                f"link={sums['link']/max(n_batches,1):.4f}  "
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
