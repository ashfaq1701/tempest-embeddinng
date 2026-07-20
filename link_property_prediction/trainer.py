"""Per-query-causal training + eval loop — link-supervised geometric walk head (REACH).

Causality is enforced PER QUERY by Tempest's cutoff, not by ingestion order. The FULL graph
(train + val + test) is ingested into Tempest ONCE up front (`ingest_full_graph`, a single
`add_edges` call); there is no per-epoch reset and no per-batch ingestion. Per-batch ordering
(training):
  1. neg = neg_sampler.sample(batch)                 — [B, K_train] uniform negs
  2. candidates = [pos | negs]                       — [B, 1+K_train]
  3. logits = score(src, candidates)                 — for each query (u_i, t_i) sample K
       backward walks with cutoff = t_i (→ μ_u), pack to tokens, score with
       LinkPredHead (identity + velocity-line). Candidate side samples no walks (static E[v]).
  4. L = cross_entropy(logits, target=0)             — Bruch 2019, upper-bounds 1-MRR
  5. one backward + single optimizer step

Why one full graph is valid (and == TPNet): a walk for (u, t) with cutoff = t traverses only
edges with t_edge < t (EXCLUSIVE), so the target edge at t — and any simultaneous/future edge —
is never seen. Because the TGB splits are chronological (train < val < test), a TRAIN query at
time t sees only edges before t: every val/test edge is later and the cutoff excludes it, so
training is causally identical to having ingested train-only. VAL sees train + earlier val; TEST
sees everything before t. This is exactly TPNet's prebuilt-time-index queried strictly-before-t
per edge. The analysis-only stores (stratify) have no cutoff, so they are seeded explicitly over
the causal-past splits.

E (on the unit sphere) and the head are trained together by L (no alignment, no detach)
by a single RiemannianAdam.

TOKEN PREP — the source side (u → μ_u) goes through `walk_tokens.build_query_walk_tokens`:
walks are generated PER QUERY (no dedup — each row's (node, t) needs its own cutoff) and returned
in the RAW per-walk WalkTokens layout ([Q, K, L] nodes / nodes_mask / node-aligned timestamps,
seeds + cutoffs). This RAW layout is the SHARED walk contract for every head; LinkPredHead
flattens it to a [Q, K*L] token bag and masks padding + the seed node u via
`walk_tokens.flatten_tokens`, then builds μ_u with a per-row softmax (ages =
cutoffs − t_edge) and scores identity + velocity against E[v].
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import geoopt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .data import Batch, SplitData
from .evaluator import Evaluator
from .model import LinkPredHead
from .negatives import UniformNegativeSampler
from .probes import CommunityProbe
from .utils import make_lr_lambda
from .walk_tokens import build_query_walk_tokens
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    # Dataset-derived.
    num_nodes: int
    dst_pool: np.ndarray

    # Frozen train-split span. Sets the log-spaced init of the μ recency λ (init ≈ 10/t_train).
    # Init only — never a per-step scaler.
    t_train: float = 1.0

    # Model.
    d_emb: int = 128
    d_ef: int = 0             # per-edge-feature dim (0 = dataset has no edge features); fed into the
                             # NeighborhoodProjection attention keys. Set from the loaded dataset.

    # NeighborhoodProjection (attention pooling of the source's walk-token offsets -> mu_u).
    t2v_dim: int = 16         # Time2Vec output dim (16 ties dim100 on wiki: 0.8287/0.8040 vs 0.8289/0.8046)

    # Link loss / head.
    K_train: int = 100          # per-query training negatives ([B, 1+K_train])

    # Walks (BACKWARD only, undirected) for the source side (u → μ_u); the one-sided head samples
    # walks only for the source, each candidate v enters through its static embedding E[v].
    num_walks_per_node: int = 10
    max_walk_len: int = 5
    walk_bias: str = "ExponentialWeight"
    start_bias: str = "ExponentialWeight"
    t2nv_p: float = 4.0    # node2vec return param (used only when a bias is TemporalNode2Vec)
    t2nv_q: float = 0.25   # node2vec in-out param; low q/p = most diverse backward walks

    # Optimisation — cosine decay to lr_min over decay_horizon_epochs (no warmup). ONE param group:
    # RiemannianAdam applies the Riemannian update to E (a geoopt.ManifoldParameter) and standard Adam
    # to the Euclidean params, all under the same LR. weight_decay is LOAD-BEARING on the sphere head:
    # an A/B showed wd 1e-4 reached ~0.828/0.803 while no-wd capped ~0.825/0.797 (the test-gap>val-gap
    # signature of a lost regulariser).
    lr: float = 1e-3
    lr_min: float = 1e-7
    weight_decay: float = 1e-4

    # Run control.
    num_epochs: int = 25
    # LR-decay horizon in epochs — SEPARATE from num_epochs, ONE value shared by BOTH LR groups. The
    # per-group cosine reaches each group's lr_min at decay_horizon_epochs, so num_epochs < horizon
    # keeps LR near peak (freedom to run any epoch count without rescaling the schedule). NOTE: the
    # ball head's cliff-free "freeze-at-peak" was tuned with horizon == num_epochs == 25 (fast decay);
    # default 50 with num_epochs 25 leaves LR ~half-decayed at stop — set horizon 25 to reproduce it.
    decay_horizon_epochs: int = 30
    early_stop_patience: int = 10

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
        # Single module owning the sphere node embeddings AND the velocity head.
        self.model = LinkPredHead(
            num_nodes=config.num_nodes,
            d_emb=int(config.d_emb),
            t2v_dim=int(config.t2v_dim),
            d_ef=int(config.d_ef),
        ).to(self.device)

        # One generator, configured QUERY-side; only the source side samples walks.
        self.walk_gen = WalkGenerator(
            use_gpu=config.use_gpu_tempest,
            walk_bias=config.walk_bias,
            start_bias=config.start_bias,
            num_walks_per_node=config.num_walks_per_node,
            max_walk_len=config.max_walk_len,
            temporal_node2vec_p=config.t2nv_p,
            temporal_node2vec_q=config.t2nv_q,
        )
        self.neg_sampler_train = UniformNegativeSampler(
            num_neg_per_pos=config.K_train, dst_pool=config.dst_pool, seed=config.seed,
        )

        # ONE param group: RiemannianAdam applies the Riemannian update to E (a geoopt.ManifoldParameter)
        # and standard Adam to the Euclidean params, all under one LR + cosine floor.
        self.opt = geoopt.optim.RiemannianAdam(
            [{"params": list(self.model.parameters()),
              "lr": float(config.lr), "weight_decay": float(config.weight_decay)}],
            stabilize=10,
        )

        self.sched: Optional[LambdaLR] = None
        self._global_step = 0

    # ──────────────────────────────────────────────────────────────────
    # LR schedule (cosine)
    # ──────────────────────────────────────────────────────────────────

    def _setup_lr_scheduler(self, batches_per_epoch: int) -> None:
        # Cosine decay to lr_min over decay_horizon_epochs (no warmup); one lambda for the single group.
        decay_steps = self.config.decay_horizon_epochs * max(batches_per_epoch, 1)
        peak, floor = float(self.config.lr), float(self.config.lr_min)
        pg = self.opt.param_groups[0]
        pg["lr"] = peak
        pg["initial_lr"] = peak
        ratio = floor / peak if peak > 0 else 0.0
        self.sched = LambdaLR(self.opt, lr_lambda=make_lr_lambda(decay_steps, ratio))
        self._global_step = 0
        print(
            f"  LR schedule (cosine): {peak:.1e}->{floor:.1e}; "
            f"decay_horizon={decay_steps} ({self.config.decay_horizon_epochs}ep x {batches_per_epoch} batches)"
        )

    # ──────────────────────────────────────────────────────────────────
    # Full-graph ingestion (once, up front)
    # ──────────────────────────────────────────────────────────────────

    def ingest_full_graph(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray,
                          edge_feat: Optional[np.ndarray] = None) -> None:
        """Ingest the ENTIRE graph (all splits, concatenated) into Tempest in ONE add_edges call.
        The per-query cutoff (t_edge < t_query, EXCLUSIVE) then enforces causality: a train query at
        t sees only edges before t — every val/test edge is chronologically later (TGB splits are
        causal: train < val < test), so the cutoff excludes it; val sees train + earlier val; test
        sees everything before t. Call once before train()/eval; there is no per-epoch reset and no
        per-batch ingestion. Capacity is unbounded — the whole timeline must stay resident."""
        self.walk_gen.add_edges(src, tgt, ts, edge_feat)
        print(f"  Ingested full graph into Tempest: {len(src):,} edges "
              f"(once; per-query cutoff enforces causality)")

    # ──────────────────────────────────────────────────────────────────
    # Scoring — shared by train + eval
    # ──────────────────────────────────────────────────────────────────

    def _score(self, src_t: torch.Tensor, cand_t: torch.Tensor,
               t_query_t: torch.Tensor) -> torch.Tensor:
        """src_t [B] long, cand_t [B, C] long, t_query_t [B] long -> logits [B, C].

        ONE-SIDED per-query walks: the SOURCE side samples K backward walks for each query (u_i, t_i)
        with cutoff = t_i, so every token has t_edge < t_i (strict causal past of that query). The
        candidate side samples NO walks — each v enters only through its static embedding E[v]. Strict
        causality comes from the per-query cutoff, NOT from ingestion order, so the batch may already
        be in Tempest."""
        device = self.device

        # SOURCE side: per-query (u_i, t_i) → K cutoff=t_i backward walks → raw [B,K,L] token bag.
        src_tokens = build_query_walk_tokens(
            self.walk_gen, device, src_t, t_query_t,
            max_walk_len=self.config.max_walk_len,
            num_walks_per_node=self.config.num_walks_per_node,
            start_bias=self.config.start_bias,
            walk_bias=self.config.walk_bias)

        return self.model(src_tokens, cand_t)   # cand_t = candidate node ids; head owns E

    # ──────────────────────────────────────────────────────────────────
    # Per-batch training step
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device
        B = len(batch.src)

        # No ingestion here: the full graph is already in Tempest (ingest_full_graph, once).
        # Each query (u, t) walks with cutoff = t (EXCLUSIVE), so it only traverses edges with
        # t_edge < t — every val/test edge is chronologically later and is never seen.
        _, neg_tgt = self.neg_sampler_train.sample(batch)              # [B, K_train]
        src_t = torch.from_numpy(batch.src.astype(np.int64)).to(device)
        cand_np = np.concatenate(
            [batch.tgt.astype(np.int64)[:, None],
             np.ascontiguousarray(neg_tgt, dtype=np.int64)], axis=1)   # [B, 1+K]
        cand_t = torch.from_numpy(cand_np).to(device)
        t_query_t = torch.from_numpy(batch.ts.astype(np.int64)).to(device)

        logits = self._score(src_t, cand_t, t_query_t)                 # [B, 1+K]
        target = torch.zeros(B, dtype=torch.long, device=device)
        loss = F.cross_entropy(logits, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        if self.sched is not None:
            self.sched.step()
            self._global_step += 1

        return {
            "link": float(loss.detach()),
            "lr": float(self.opt.param_groups[0]["lr"]),
        }

    # ──────────────────────────────────────────────────────────────────
    # Eval — strict-causal, no_grad
    # ──────────────────────────────────────────────────────────────────

    def _eval(self, evaluator: Evaluator, batches: Iterable[Batch],
              recorder: Any = None) -> float:
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in batches:
                B = len(batch.src)
                if recorder is not None:
                    recorder.before_batch(batch)

                # No ingestion: the full graph (incl. val/test) is already in Tempest. The
                # per-query cutoff keeps every walk causal (t_edge < t_query), so future eval
                # edges in the index never leak.
                if B == 0:
                    if recorder is not None:
                        recorder.after_batch(batch)
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
        return total / max(n, 1)

    # ──────────────────────────────────────────────────────────────────
    # Snapshot / restore (early-stop)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "model": self._cpu_state_dict(self.model),
        }

    def _restore(self, snap: Dict[str, Any]) -> None:
        self.model.load_state_dict(snap["model"])

    # ──────────────────────────────────────────────────────────────────
    # Train loop
    # ──────────────────────────────────────────────────────────────────

    def train(
        self,
        train_batches_factory,
        full_graph: SplitData,
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory=None,
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory=None,
    ) -> Dict[str, Any]:
        # Ingest the FULL graph (train + val + test) into Tempest ONCE, up front. Per-query cutoffs
        # then keep every walk causal (TGB splits are chronological: train < val < test), so there is
        # no per-epoch reset and no per-batch ingestion.
        self.ingest_full_graph(
            full_graph.sources, full_graph.destinations,
            full_graph.timestamps, full_graph.edge_feat)

        n_epochs = self.config.num_epochs
        patience = self.config.early_stop_patience

        # One pass over the train batches: count them AND collect the full edge set (for the
        # community probe's fixed Louvain graph — built once).
        src_all, dst_all, batches_per_epoch = [], [], 0
        for b in train_batches_factory():
            src_all.append(np.asarray(b.src))
            dst_all.append(np.asarray(b.tgt))
            batches_per_epoch += 1
        self._setup_lr_scheduler(batches_per_epoch)
        self.comm_probe = CommunityProbe(
            np.concatenate(src_all), np.concatenate(dst_all), self.config.num_nodes)
        print(f"  CommunityProbe: {self.comm_probe.n_comms} Louvain communities "
              f"(Q={self.comm_probe.q:.3f}); random-neighbour null purity={self.comm_probe.null:.3f}")

        best_val, best_test, best_epoch = -1.0, -1.0, -1
        best_snap: Optional[Dict[str, Any]] = None
        no_improve = 0
        per_epoch_val: List[float] = []
        per_epoch_test: List[float] = []

        for ep in range(1, n_epochs + 1):
            self.model.train()

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
                f"lr={self.opt.param_groups[0]['lr']:.1e}  "
                f"train {train_dt:.1f}s"
            )
            cp = self.comm_probe.measure(self.model.E.weight.detach())     # community-formation probe
            line += f"  commP={cp:.3f}(x{cp / max(self.comm_probe.null, 1e-9):.1f})"

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
