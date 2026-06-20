"""Strict-causal training + eval loop — link-supervised symmetric walk-encoder.

Per-batch ordering (training):
  1. neg = neg_sampler.sample(batch)                 — [B, K_train] uniform negs
  2. candidates = [pos | negs]                       — [B, 1+K_train]
  3. logits = score(src, candidates)                 — SYMMETRIC: sample walks for the
       unique sources AND the unique candidates, dedup each to a per-node CSR, score.
  4. L = cross_entropy(logits / tau_link, target=0)  — Bruch 2019, upper-bounds 1-MRR
  5. one backward + single optimizer step
  6. walk_gen.add_edges(batch)                        — post-scoring, LAST

E (on the unit sphere) and the head are trained together by L (no alignment, no detach)
by a single RiemannianAdam.

SYMMETRIC CSR TOKEN PREP — the source side (u → μ) and the candidate side (v → connectors)
go through the shared `walk_token_csr` module: walk for the unique seeds, flatten the (K,L) walk
axes, and DEDUPLICATE per seed into a compact CSR
    (node_ids [G,U], node_mask [G,U], ages [G,U,kmax], age_mask [G,U,kmax])
— distinct neighbour nodes per seed, each carrying its OCCURRENCE AGES (all of them, so the
recency mean stays exact) and, implicitly, its COUNT = age_mask.sum(-1). Both sides emit the
SAME layout; they differ only in which seeds (sources vs candidates) and which walk params.
The source CSR is computed per unique source [Ms,…] and gathered to [B,…]; the candidate CSR
is computed per unique candidate [Mv,…] and scattered to [B,C,…] via v_inv (the dedup is
exact — the per-node CSR is query-independent given a single pre-ingest snapshot and the
shift-invariant recency weighting). IDs are passed to the head, which gathers embeddings from
the shared table for both sides. Strict-causal: walks + stores reflect the pre-ingest state.
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import geoopt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .data import Batch
from .evaluator import Evaluator
from .link_pred_head import GeometricPointHead
from .model import EmbeddingTable
from .negatives import UniformNegativeSampler
from .pair_store import NodeLastSeenStore, PairRecencyStore
from .utils import make_lr_lambda
from .walk_token_csr import gather_csr, walk_csr
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    # Dataset-derived.
    num_nodes: int
    dst_pool: np.ndarray

    # Frozen train-split span. Sets the log-spaced init of the head's exp-decay rates:
    # the co-reachability decay ρ = exp(log_rate_coreach), init −log(t_train), and the
    # ExpDecayBasis rates (1/t_train … 1). Init only — never a per-step scaler.
    t_train: float = 1.0

    # Model.
    d_emb: int = 128

    # Link loss / head.
    tau_link: float = 1.0       # softmax-CE temperature
    K_train: int = 100          # per-query training negatives ([B, 1+K_train])

    # Exact pairwise (u,v) recurrence channel, added as one logit term. Flagged; the
    # in-geometry count term aims to make it droppable.
    use_pair_features: bool = False

    # Walks (BACKWARD only, undirected). Decoupled QUERY-side (source u → μ) and
    # CANDIDATE-side (v → connectors), both sampled from the SAME Tempest graph via
    # per-call overrides. Both sides now go through the SAME CSR dedup phase.
    num_walks_per_node_query_side: int = 5
    max_walk_len_query_side: int = 5
    walk_bias_query_side: str = "ExponentialWeight"
    start_bias_query_side: str = "ExponentialWeight"
    num_walks_per_node_candidate_side: int = 5
    max_walk_len_candidate_side: int = 5
    walk_bias_candidate_side: str = "Linear"
    start_bias_candidate_side: str = "Linear"
    max_time_capacity: int = -1   # Tempest sliding-window eviction; -1 = unbounded

    # Optimisation.
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    warmup_steps_cap: int = 500
    decay_horizon_epochs: int = 50

    # Run control.
    num_epochs: int = 50
    early_stop_patience: int = 0

    # System.
    seed: int = 42
    use_gpu: bool = False
    use_gpu_tempest: bool = False


class Trainer:
    def __init__(self, config: TrainerConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
        )
        self.embedding_table = EmbeddingTable(
            num_nodes=config.num_nodes, d_emb=config.d_emb,
        ).to(self.device)
        self.link_head = GeometricPointHead(
            d_emb=int(config.d_emb),
            use_pair_features=config.use_pair_features,
            t_train=float(config.t_train),
        ).to(self.device)

        self.pair_store = (
            PairRecencyStore(num_nodes=config.num_nodes)
            if config.use_pair_features else None
        )
        self.node_last = NodeLastSeenStore()

        # One generator, configured QUERY-side; candidate-side walks reuse it via
        # per-call overrides (different length / count / biases, same graph).
        self.walk_gen = WalkGenerator(
            use_gpu=config.use_gpu_tempest,
            walk_bias=config.walk_bias_query_side,
            start_bias=config.start_bias_query_side,
            num_walks_per_node=config.num_walks_per_node_query_side,
            max_walk_len=config.max_walk_len_query_side,
            max_time_capacity=config.max_time_capacity,
        )
        self.neg_sampler_train = UniformNegativeSampler(
            num_neg_per_pos=config.K_train, dst_pool=config.dst_pool, seed=config.seed,
        )

        params = list(self.embedding_table.parameters()) + list(self.link_head.parameters())
        self.opt = geoopt.optim.RiemannianAdam(
            params, lr=config.lr, weight_decay=config.weight_decay, stabilize=10)

        self.sched: Optional[LambdaLR] = None
        self._global_step = 0

    # ──────────────────────────────────────────────────────────────────
    # LR schedule (warmup + cosine)
    # ──────────────────────────────────────────────────────────────────

    def _setup_lr_scheduler(self, batches_per_epoch: int) -> None:
        decay_steps = self.config.decay_horizon_epochs * max(batches_per_epoch, 1)
        warmup_steps = max(1, min(
            int(self.config.warmup_fraction * decay_steps), self.config.warmup_steps_cap))
        peak = float(self.config.lr)
        floor = float(self.config.lr_min)
        for pg in self.opt.param_groups:
            pg["lr"] = peak
            pg["initial_lr"] = peak
        ratio = floor / peak if peak > 0 else 0.0
        self.sched = LambdaLR(
            self.opt, lr_lambda=make_lr_lambda(warmup_steps, decay_steps, ratio))
        self._global_step = 0
        print(
            f"  LR schedule (warmup+cosine): peak={peak:.2e} floor={floor:.2e}; "
            f"warmup={warmup_steps} decay_horizon={decay_steps} "
            f"({self.config.decay_horizon_epochs}ep x {batches_per_epoch} batches)"
        )

    # ──────────────────────────────────────────────────────────────────
    # Scoring — shared by train + eval
    # ──────────────────────────────────────────────────────────────────

    def _score(self, src_t: torch.Tensor, cand_t: torch.Tensor,
               t_query_t: torch.Tensor) -> torch.Tensor:
        """src_t [B] long, cand_t [B, C] long, t_query_t [B] long -> logits [B, C].

        SYMMETRIC: build a per-node CSR for the unique sources (→ μ tokens) and a per-node
        CSR for the unique candidates (→ connectors), via the SAME `walk_csr`. Gather the
        source CSR to [B,…] and scatter the candidate CSR to [B,C,…]. IDs go to the head;
        the head gathers embeddings from the shared table for both sides."""
        device = self.device
        B, C = cand_t.shape

        # --- SOURCE side: unique sources → CSR → gather to [B] ---
        uniq_s, u_pos = torch.unique(src_t, return_inverse=True)        # [Ms], [B]
        first_src = torch.argmax(
            (src_t.view(-1, 1) == uniq_s.view(1, -1)).to(torch.int64), dim=0)
        csr_s = walk_csr(
            self.walk_gen, self.device, uniq_s, t_query_t[first_src],
            max_walk_len=self.config.max_walk_len_query_side,
            num_walks_per_node=self.config.num_walks_per_node_query_side,
            start_bias=self.config.start_bias_query_side,
            walk_bias=self.config.walk_bias_query_side)
        src_ids, src_nmask, src_ages, src_amask = gather_csr(csr_s, u_pos, (B,))

        # --- CANDIDATE side: unique candidates → CSR → scatter to [B,C] ---
        uniq_v, v_inv = torch.unique(cand_t.reshape(-1), return_inverse=True)  # [Mv], [B*C]
        # Each unique candidate is aged against the query time of (one of) the rows that
        # names it; ages enter only the (recency) reductions and the per-seed snapshot is
        # shared, so any naming row's t_query is consistent for the dedup.
        first_row = torch.argmax(
            (cand_t.reshape(-1).view(-1, 1) == uniq_v.view(1, -1)).to(torch.int64), dim=0)
        tq_per_v = t_query_t.view(B, 1).expand(B, C).reshape(-1)[first_row]    # [Mv]
        csr_v = walk_csr(
            self.walk_gen, self.device, uniq_v, tq_per_v,
            max_walk_len=self.config.max_walk_len_candidate_side,
            num_walks_per_node=self.config.num_walks_per_node_candidate_side,
            start_bias=self.config.start_bias_candidate_side,
            walk_bias=self.config.walk_bias_candidate_side)
        cand_ids, cand_nmask, cand_ages, cand_amask = gather_csr(csr_v, v_inv, (B, C))

        E_u = self.embedding_table(src_t)                              # [B, d]
        E_v = self.embedding_table(cand_t)                             # [B, C, d]

        # Candidate staleness + (flagged) pair channel — unchanged stores.
        staleness_dt = self.node_last.query(cand_t, t_query_t)         # [B, C]
        pair_dt = pair_count_log = None
        if self.pair_store is not None:
            pair_dt, pair_count_log = self.pair_store.query(src_t, cand_t, t_query_t)

        return self.link_head(
            # source CSR (μ side)
            E_u, src_ids, src_nmask, src_ages, src_amask,
            # candidate CSR (identity + connectors side)
            E_v, cand_ids, cand_nmask, cand_ages, cand_amask,
            # shared table + additive channels
            self.embedding_table.E.weight, t_query_t, staleness_dt,
            pair_dt=pair_dt, pair_count_log=pair_count_log)

    # ──────────────────────────────────────────────────────────────────
    # Per-batch training step
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device
        B = len(batch.src)

        _, neg_tgt = self.neg_sampler_train.sample(batch)              # [B, K_train]
        src_t = torch.from_numpy(batch.src.astype(np.int64)).to(device)
        cand_np = np.concatenate(
            [batch.tgt.astype(np.int64)[:, None],
             np.ascontiguousarray(neg_tgt, dtype=np.int64)], axis=1)   # [B, 1+K]
        cand_t = torch.from_numpy(cand_np).to(device)
        t_query_t = torch.from_numpy(batch.ts.astype(np.int64)).to(device)

        logits = self._score(src_t, cand_t, t_query_t)                 # [B, 1+K]
        target = torch.zeros(B, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits / self.config.tau_link, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        if self.sched is not None:
            self.sched.step()
            self._global_step += 1

        # Strict-causal: ingest into Tempest LAST.
        self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)
        if self.pair_store is not None:
            self.pair_store.update(batch.src, batch.tgt, batch.ts)
        self.node_last.update(batch.src, batch.tgt, batch.ts)

        return {
            "link": float(loss.detach()),
            "lr": float(self.opt.param_groups[0]["lr"]),
        }

    # ──────────────────────────────────────────────────────────────────
    # Eval — strict-causal, no_grad
    # ──────────────────────────────────────────────────────────────────

    def _eval(self, evaluator: Evaluator, batches: Iterable[Batch],
              recorder: Any = None) -> float:
        self.embedding_table.eval()
        self.link_head.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in batches:
                B = len(batch.src)
                if recorder is not None:
                    recorder.before_batch(batch)
                if B == 0:
                    if recorder is not None:
                        recorder.after_batch(batch)
                    self.walk_gen.add_edges(
                        batch.src, batch.tgt, batch.ts, batch.edge_feat)
                    if self.pair_store is not None:
                        self.pair_store.update(batch.src, batch.tgt, batch.ts)
                    self.node_last.update(batch.src, batch.tgt, batch.ts)
                    continue

                _, neg_tgt_list = evaluator.sample_negatives(batch)
                counts = [int(arr.shape[0]) for arr in neg_tgt_list]
                max_K = max(counts) if counts else 0

                pos_v_np = batch.tgt.astype(np.int64)
                cand_v_np = np.tile(pos_v_np[:, None], (1, 1 + max_K))
                for i in range(B):
                    if counts[i] > 0:
                        cand_v_np[i, 1:1 + counts[i]] = neg_tgt_list[i].astype(np.int64)

                src_t = torch.from_numpy(batch.src.astype(np.int64)).to(self.device)
                cand_t = torch.from_numpy(cand_v_np).to(self.device)
                t_query_t = torch.from_numpy(batch.ts.astype(np.int64)).to(self.device)
                logits = self._score(src_t, cand_t, t_query_t).cpu().numpy()

                for i in range(B):
                    rr = evaluator.score_to_metric(
                        float(logits[i, 0]), logits[i, 1:1 + counts[i]])
                    total += rr
                    if recorder is not None:
                        recorder.on_positive(batch, i, rr)
                n += B

                if recorder is not None:
                    recorder.after_batch(batch)
                self.walk_gen.add_edges(
                    batch.src, batch.tgt, batch.ts, batch.edge_feat)
                if self.pair_store is not None:
                    self.pair_store.update(batch.src, batch.tgt, batch.ts)
                self.node_last.update(batch.src, batch.tgt, batch.ts)
        return total / max(n, 1)

    # ──────────────────────────────────────────────────────────────────
    # Snapshot / restore (early-stop)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "embedding_table": self._cpu_state_dict(self.embedding_table),
            "link_head": self._cpu_state_dict(self.link_head),
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

        batches_per_epoch = sum(1 for _ in train_batches_factory())
        self._setup_lr_scheduler(batches_per_epoch)

        best_val, best_test, best_epoch = -1.0, -1.0, -1
        best_snap: Optional[Dict[str, Any]] = None
        no_improve = 0
        per_epoch_val: List[float] = []
        per_epoch_test: List[float] = []

        for ep in range(1, n_epochs + 1):
            self.walk_gen.reset()
            if self.pair_store is not None:
                self.pair_store.reset()
            self.node_last.reset()
            self.embedding_table.train()
            self.link_head.train()

            t0 = time.time()
            link_sum, n_batches = 0.0, 0
            for batch in train_batches_factory():
                m = self._train_step(batch)
                link_sum += m["link"]
                n_batches += 1
            train_dt = time.time() - t0

            line = (
                f"epoch {ep}/{n_epochs}  "
                f"link={link_sum / max(n_batches, 1):.4f}  "
                f"lr={self.opt.param_groups[0]['lr']:.2e}  "
                f"train {train_dt:.1f}s"
            )

            if val_evaluator is not None and val_batches_factory is not None:
                t1 = time.time()
                val_metric = self._eval(val_evaluator, val_batches_factory())
                eval_dt = time.time() - t1
                per_epoch_val.append(val_metric)

                if val_metric > best_val:
                    best_val, best_epoch = val_metric, ep
                    best_snap = self._snapshot()
                    no_improve = 0
                    if test_evaluator is not None and test_batches_factory is not None:
                        best_test = self._eval(test_evaluator, test_batches_factory())
                        per_epoch_test.append(best_test)
                        line += f"  val {val_metric:.4f}  test {best_test:.4f} (new best)"
                    else:
                        line += f"  val {val_metric:.4f} (new best)"
                else:
                    no_improve += 1
                    line += f"  val {val_metric:.4f}  patience {no_improve}/{patience}"
                line += f"  eval {eval_dt:.1f}s"
            print(line)

            if patience > 0 and no_improve >= patience:
                break

        if best_snap is not None:
            self._restore(best_snap)
            print(
                f"  restored best weights from epoch {best_epoch} "
                f"(val {best_val:.4f}, test {best_test:.4f})")

        return {
            "stopped_at_epoch": best_epoch if best_snap is not None else n_epochs,
            "best_val_mrr": best_val,
            "best_test_mrr": best_test,
            "per_epoch_val_mrr": per_epoch_val,
            "per_epoch_test_mrr": per_epoch_test,
        }
