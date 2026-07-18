"""Per-query-causal training + eval loop — link-supervised geometric walk head (REACH).

Causality is now enforced PER QUERY by Tempest's cutoff, not by ingestion order. Per-batch
ordering (training):
  1. walk_gen.add_edges(batch)                        — ingest the batch's true edges FIRST
  2. neg = neg_sampler.sample(batch)                 — [B, K_train] uniform negs
  3. candidates = [pos | negs]                       — [B, 1+K_train]
  4. logits = score(src, candidates)                 — for each query (u_i, t_i) sample K
       backward walks with cutoff = t_i (→ μ_u), pack to tokens, score with
       LinkPredHead (identity + velocity-line). Candidate side samples no walks (static E[v]).
  5. L = cross_entropy(logits, target=0)             — Bruch 2019, upper-bounds 1-MRR
  6. one backward + single optimizer step

Why ingest-first is valid (and == TPNet): a walk for (u, t) with cutoff = t traverses only
edges with t_edge < t (EXCLUSIVE), so the target edge at t — and any simultaneous/future
batch edge — is never seen, while same-batch-earlier edges (t' < t) legitimately are. This is
exactly TPNet's prebuilt-time-index queried strictly-before-t per edge, and it removes the
batch-size coupling of the old "walk the pre-ingest snapshot" scheme (which hid same-batch-
earlier edges). The stores have no per-query cutoff, so they keep a pre-batch snapshot
(updated last) — their strictly-causal form.

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

from .data import Batch
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
    proj_dim: int = 128       # attention (query/key) dim d_a
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
    max_time_capacity: int = -1   # Tempest sliding-window eviction; -1 = unbounded

    # Optimisation — cosine decay to lr_min over num_epochs (no warmup). TWO LR GROUPS: the manifold
    # embedding E and all other (Euclidean) params — decay independently. NOTE: the SPHERE head (this
    # branch's model.py) wants lr_manifold 1e-3; the Poincaré variant (feature/poincare-geodesic-rand)
    # uses 1e-4 — the one intended cross-branch LR diff.
    lr_manifold: float = 1e-3
    lr_min_manifold: float = 1e-7
    lr_model: float = 1e-3
    lr_min_model: float = 1e-7
    # Per-group weight decay (RiemannianAdam group["weight_decay"]). LOAD-BEARING on the sphere head:
    # an A/B showed wd 1e-4 reached ~0.828/0.803 while no-wd capped ~0.825/0.797 (the test-gap>val-gap
    # signature of a lost regulariser). Master-only vs the Poincaré branch (which runs no weight decay).
    weight_decay_manifold: float = 1e-4
    weight_decay_model: float = 1e-4

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
            proj_dim=int(config.proj_dim),
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
            max_time_capacity=config.max_time_capacity,
            temporal_node2vec_p=config.t2nv_p,
            temporal_node2vec_q=config.t2nv_q,
        )
        self.neg_sampler_train = UniformNegativeSampler(
            num_neg_per_pos=config.K_train, dst_pool=config.dst_pool, seed=config.seed,
        )

        # Two LR groups: the ball manifold params (E, a geoopt.ManifoldParameter) vs all other
        # Euclidean params. Each group gets its own peak LR + cosine floor (see _setup_lr_scheduler).
        manifold_params, model_params = [], []
        for p in self.model.parameters():
            (manifold_params if isinstance(p, geoopt.ManifoldParameter) else model_params).append(p)
        groups, self._group_lr = [], []   # self._group_lr: (peak, floor) per group, in opt order
        if manifold_params:
            groups.append({"params": manifold_params, "lr": config.lr_manifold,
                           "weight_decay": float(config.weight_decay_manifold)})
            self._group_lr.append((float(config.lr_manifold), float(config.lr_min_manifold)))
        if model_params:
            groups.append({"params": model_params, "lr": config.lr_model,
                           "weight_decay": float(config.weight_decay_model)})
            self._group_lr.append((float(config.lr_model), float(config.lr_min_model)))
        self.opt = geoopt.optim.RiemannianAdam(groups, stabilize=10)

        self.sched: Optional[LambdaLR] = None
        self._global_step = 0

    # ──────────────────────────────────────────────────────────────────
    # LR schedule (cosine)
    # ──────────────────────────────────────────────────────────────────

    def _setup_lr_scheduler(self, batches_per_epoch: int) -> None:
        # Independent cosine decay per param group (each to its own floor) over the whole run
        # (horizon = num_epochs); no warmup. LambdaLR takes one lambda per group.
        decay_steps = self.config.decay_horizon_epochs * max(batches_per_epoch, 1)
        lambdas = []
        for pg, (peak, floor) in zip(self.opt.param_groups, self._group_lr):
            pg["lr"] = peak
            pg["initial_lr"] = peak
            ratio = floor / peak if peak > 0 else 0.0
            lambdas.append(make_lr_lambda(decay_steps, ratio))
        self.sched = LambdaLR(self.opt, lr_lambda=lambdas)
        self._global_step = 0
        groups = " ".join(f"{peak:.1e}->{floor:.1e}" for peak, floor in self._group_lr)
        print(
            f"  LR schedule (cosine, per-group [manifold model]): {groups}; "
            f"decay_horizon={decay_steps} ({self.config.decay_horizon_epochs}ep x {batches_per_epoch} batches)"
        )

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

        # Ingest the batch's true edges into Tempest BEFORE scoring. This is causally
        # valid — and exactly TPNet's protocol (prebuilt time-sorted index queried
        # strictly-before-t per edge) — because every per-query walk is bounded by
        # cutoff = t_query (EXCLUSIVE): the walk for (u, t) only traverses edges with
        # t_edge < t, so the target edge at t (and any simultaneous/future batch edge) is
        # never visible, while same-batch-earlier edges (t' < t) legitimately are. The
        # stores below have NO cutoff mechanism, so they stay strictly causal the only way
        # they can — a pre-batch snapshot, updated AFTER scoring.
        self.walk_gen.add_edges(batch.src, batch.tgt, batch.ts, batch.edge_feat)

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

                # Ingest into Tempest BEFORE scoring — the per-query cutoff keeps every
                # walk causal (t_edge < t_query), so eval edges in the index never leak.
                self.walk_gen.add_edges(
                    batch.src, batch.tgt, batch.ts, batch.edge_feat)

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
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory=None,
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory=None,
    ) -> Dict[str, Any]:
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
            self.walk_gen.reset()
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
                f"lr={'/'.join('%.1e' % pg['lr'] for pg in self.opt.param_groups)}  "
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
