"""Strict-causal training + eval loop — link-supervised source-side walk-encoder.

Per-batch ordering (training):
  1. neg = neg_sampler.sample(batch)                 — [B, K_train] uniform negs
  2. candidates = [pos | negs]                       — [B, 1+K_train]
  3. logits = score(src, candidates)                 — sample K walks for the SOURCES
       only, GRU-encode to h[u], score by the chord  -scale*‖ĥ[u] - E[v]‖
  4. L = cross_entropy(logits / tau_link, target=0)  — Bruch 2019, upper-bounds 1-MRR
  5. one backward + single optimizer step
  6. walk_gen.add_edges(batch)                        — post-scoring, LAST

E (on the unit sphere) and the GRU are trained together by L (no alignment, no
detach) by a single RiemannianAdam: E gets the manifold update, the Euclidean
GRU/scale get the ordinary update.

Eval (no_grad): TGB per-positive negatives, score each [pos | ~999 negs] via the
same path, official Evaluator MRR; advance Tempest with the full batch after.
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
from .link_pred_head import DualMuHead
from .model import EmbeddingTable
from .negatives import UniformNegativeSampler
from .pair_store import NodeLastSeenStore, PairRecencyStore
from .utils import make_lr_lambda
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    # Dataset-derived.
    num_nodes: int
    dst_pool: np.ndarray

    # Frozen train-split span. Sets the log-spaced init of the head's exp-decay rates:
    # the cross-channel decay ρ_cross = exp(log_rate_cross), init −log(t_train) (so the
    # raw-age logsumexp stays O(1) at init), and the ExpDecayBasis rates (1/t_train … 1)
    # for the rec / pair channels. Init only — never a per-step scaler.
    t_train: float = 1.0

    # Model.
    d_emb: int = 128

    # Link loss / head.
    tau_link: float = 1.0       # softmax-CE temperature
    K_train: int = 100          # per-query training negatives ([B, 1+K_train])

    # Exact pairwise (u,v) recurrence + history from the streaming store, added as
    # one logit term. Off by default => baseline byte-identical.
    use_pair_features: bool = False

    # Walks (BACKWARD only, undirected). The QUERY side feeds μ_u (source u's
    # walk-neighbours) and the CANDIDATE side feeds μ_v (candidate v's walk-neighbours)
    # — SYMMETRIC: each μ is a recency-weighted mean over its node's walk-neighbours
    # (all context nodes, seed + padding excluded). Both sampled from the SAME Tempest
    # graph via per-call overrides (no second store, no extra ingest). For the dual-μ
    # head the candidate side should MIRROR the query side (default to matching args).
    num_walks_per_node_query_side: int = 5
    max_walk_len_query_side: int = 20
    walk_bias_query_side: str = "ExponentialWeight"
    start_bias_query_side: str = "ExponentialWeight"
    num_walks_per_node_candidate_side: int = 5
    max_walk_len_candidate_side: int = 20
    walk_bias_candidate_side: str = "ExponentialWeight"
    start_bias_candidate_side: str = "ExponentialWeight"
    max_time_capacity: int = -1   # Tempest sliding-window eviction; -1 = unbounded

    # Optimisation. Single RiemannianAdam over {E, GRU, scale}: E (the lone
    # ManifoldParameter) gets the sphere update, GRU/scale the Euclidean one.
    lr: float = 1e-3            # peak LR
    lr_min: float = 1e-5        # cosine floor
    weight_decay: float = 1e-4  # applies to GRU/scale; a no-op on the sphere E
    warmup_fraction: float = 0.05
    warmup_steps_cap: int = 500
    decay_horizon_epochs: int = 50

    # Run control.
    num_epochs: int = 50
    early_stop_patience: int = 0   # 0 = no early stopping

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
        self.link_head = DualMuHead(
            d_emb=int(config.d_emb),
            use_pair_features=config.use_pair_features,
            t_train=float(config.t_train),
        ).to(self.device)

        # Streaming pairwise-interaction store feeding the pair features. Lifecycle
        # mirrors walk_gen: reset() per epoch, update() after scoring, query() at
        # scoring time.
        self.pair_store = (
            PairRecencyStore(num_nodes=config.num_nodes)
            if config.use_pair_features else None
        )

        # Per-node last-seen store: supplies the candidate recency term without
        # sampling candidate walks (source-side-only head). Always on.
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

        # Single RiemannianAdam over every trainable parameter. E is a
        # ManifoldParameter (sphere update); the GRU/scale are Euclidean
        # (ordinary update). stabilize=10 periodically re-projects E.
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

        SOURCE-SIDE ONLY: sample K walks for the unique SOURCES only and expose each
        source's CONTEXT walk-neighbours as a flat token set {w_i} (seed + padding
        excluded). Each candidate v attends over those tokens and pulls a
        candidate-specific sphere log-map residual (the attention channel), while
        E[u] enters once as a direct chord(E[u], E[v]) base channel. Candidate recency
        (t_query - t_last[v]) comes from the per-node last-seen store (no candidate
        walks). Strict-causal: walks + store reflect the pre-ingest state."""
        device = self.device
        B, C = cand_t.shape
        # ONE per-batch recency reference (int64) shared by BOTH the source μ_u and the
        # candidate μ_v token ages: the latest query time in the batch. Walks are causal
        # (edge times ≤ their query time ≤ this max), so ref − t_edge ≥ 0. The μ softmax is
        # shift-invariant, so a per-batch-shared ref gives bit-identical μ to per-query
        # t_query — while letting us subtract in INT64 before the float cast (float32-exact).
        batch_t_max = t_query_t.max()                            # int64 scalar

        # Walks for the unique SOURCES only.
        uniq_s, u_pos = torch.unique(src_t, return_inverse=True)  # uniq_s [Ms], u_pos [B]
        wd = self.walk_gen.walks_for_nodes(uniq_s.cpu().numpy())
        Ms, K, L = uniq_s.shape[0], wd.K, wd.nodes.shape[1]
        nodes = wd.nodes.to(device).view(Ms, K, L)               # [Ms, K, L]
        ts = wd.timestamps.to(device).to(torch.int64).view(Ms, K, L)   # [Ms,K,L] raw edge times int64
        lens = wd.lens.to(device).view(Ms, K)                    # [Ms, K]
        # Context positions hold real edge times (the seed slot lens-1 is the
        # INT64_MAX sentinel; padding is -1). is_ctx is BOTH the time-stat mask AND
        # the token mask — it excludes the seed (= u itself), so u never enters the
        # attention pool. u's identity lives only in the base chord channel.
        is_ctx = torch.arange(L, device=device).view(1, 1, L) < (lens - 1).unsqueeze(-1)

        # CUMULATIVE PATH age per token: a token at hop-distance k from the seed carries
        # Σ_{i=1..k}(batch_t_max − t_edge_i) = the reverse-cumsum of edge ages toward the seed,
        # so the recency weight e^{−λ·age} = ∏ edge decays (TPNet's walk score) and deeper
        # tokens decay by their whole path — not just their connecting edge. edge_age = 0 at the
        # seed slot (INT64_MAX sentinel) and padding; where() discards them (no int64 underflow).
        edge_age = torch.where(is_ctx, batch_t_max - ts, torch.zeros_like(ts))     # [Ms,K,L]
        cum_age = torch.flip(torch.cumsum(torch.flip(edge_age, dims=[2]), dim=2), dims=[2])

        # Flatten the (K, L) walk axes into one per-source token set [Ms, n]. The head consumes
        # a FLAT token set + mask (μ = recency-weighted mean over all of u's walk-neighbours);
        # tok_mask = is_ctx keeps the seed (= u) and padding out (zero recency weight).
        n = K * L
        tok_emb_all = self.embedding_table(nodes.clamp_min(0)).view(Ms, n, -1)  # [Ms,n,d]
        tok_age_all = cum_age.view(Ms, n)                      # [Ms, n] cumulative path age int64
        tok_mask_all = is_ctx.view(Ms, n)                      # [Ms, n] bool

        tok_emb = tok_emb_all[u_pos]                           # [B, n, d]
        tok_age = tok_age_all[u_pos].clamp_min(0).float()     # [B, n] cumulative path age
        tok_mask = tok_mask_all[u_pos]                         # [B, n]

        E_u = self.embedding_table(src_t)                       # [B, d]   base point E[u]

        # Candidate v's own recency Δt (per-node last-seen store) and, when the pair
        # channel is on, the (u,v) last-interaction Δt — both RAW, both fed to the head's
        # ExpDecayBasis. Never-seen (u,v) → Δt=∞ ⇒ φ=0 (handled in PairRecencyStore).
        # These are PER-QUERY (not inside a softmax) so they stay [B, C].
        rec_v_dt = self.node_last.query(cand_t, t_query_t)       # [B, C] raw Δt
        pair_dt = pair_count_log = None
        if self.pair_store is not None:
            pair_dt, pair_count_log = self.pair_store.query(     # [B, C] raw Δt_uv, log1p(count)
                src_t, cand_t, t_query_t)

        # Candidate-side walk-neighbours for μ_v, PER UNIQUE candidate node (no [B,C,M,d]
        # blow-up): the head computes P_v once per unique v and indexes via v_inv.
        uniq_v, v_inv, cand_ids, cand_age, cand_mask = self._candidate_walk_tokens(
            cand_t, batch_t_max)

        return self.link_head(
            tok_emb, tok_age, tok_mask, E_u, rec_v_dt,
            uniq_v, v_inv, cand_ids, cand_age, cand_mask,
            self.embedding_table.E.weight,
            pair_dt=pair_dt, pair_count_log=pair_count_log)

    def _candidate_walk_tokens(self, cand_t: torch.Tensor, batch_t_max: torch.Tensor):
        """v's walk-neighbour tokens for μ_v — returned PER UNIQUE candidate node.

        For every UNIQUE candidate v, sample CANDIDATE-side BACKWARD walks on the SAME
        Tempest graph (per-call overrides — no second store). ALL context nodes of those
        walks (seed v and padding excluded) become v's walk-neighbour tokens — symmetric
        to how the query side exposes u's walk-neighbours for μ_u.

        μ_v is a PER-NODE quantity: it depends only on v's walks + E[v], not on the query.
        EXACT iff the whole batch is scored against ONE pre-ingest snapshot (no intra-batch
        graph update) — true here: both _train_step and _eval call add_edges AFTER scoring
        the full batch. Then v's walks are identical across all queries having v as a
        candidate, so the head computes P_v once per unique v and indexes via v_inv.

        The recency weight is shift-invariant: softmax(−λ·(t_query−t_edge)) is unchanged by
        any per-token-shared offset, so the PER-BATCH reference `batch_t_max` (the latest
        query time in the batch, passed in by the caller and shared with the source μ_u)
        replaces the per-query t_query. We subtract it in INT64 before the float cast so the
        resulting age stays small and float32-exact even on large-timestamp datasets, then
        weight by −λ·age — bit-identical to the per-query form and query-independent (so μ_v
        stays per-node). If eval ever ingests intra-batch, revisit.

        -> uniq_v   [Mv]      unique candidate node ids,
           v_inv    [B*C]     scatter index back to the [B, C] candidate grid,
           cand_ids [Mv, M]   walk-neighbour node ids (−1 at padding; head gathers E),
           cand_age [Mv, M]   per-batch-reference age ≥ 0 (0 at masked slots; → −λ·age),
           cand_mask[Mv, M]   True at a real walk-neighbour. M = K_cand · L_cand."""
        device = self.device
        uniq_v, v_inv = torch.unique(cand_t.reshape(-1), return_inverse=True)  # [Mv],[B*C]
        wc = self.walk_gen.walks_for_nodes(
            uniq_v.cpu().numpy(),
            max_walk_len=self.config.max_walk_len_candidate_side,
            num_walks_per_node=self.config.num_walks_per_node_candidate_side,
            start_bias=self.config.start_bias_candidate_side,
            walk_bias=self.config.walk_bias_candidate_side)
        Mv, K, L = uniq_v.shape[0], wc.K, wc.nodes.shape[1]
        nodes = wc.nodes.to(device).view(Mv, K, L)              # [Mv, K, L]
        ts_i = wc.timestamps.to(device).to(torch.int64).view(Mv, K, L)   # raw edge times (int64)
        lens = wc.lens.to(device).view(Mv, K)
        # Context = real walk-neighbours: exclude the seed v (slot lens-1) and padding.
        is_ctx = torch.arange(L, device=device).view(1, 1, L) < (lens - 1).unsqueeze(-1)
        # CUMULATIVE PATH age (matches the source side): each hop-k neighbour carries
        # Σ_{i=1..k}(batch_t_max − t_edge_i) = reverse-cumsum of edge ages toward the seed, so
        # e^{−λ·age} = ∏ edge decays (TPNet's walk score). edge_age = 0 at seed/padding; the
        # shared per-batch batch_t_max keeps μ_v query-independent (per-node dedup intact).
        edge_age = torch.where(is_ctx, batch_t_max - ts_i, torch.zeros_like(ts_i))   # [Mv,K,L]
        cum_age = torch.flip(torch.cumsum(torch.flip(edge_age, dims=[2]), dim=2), dims=[2])
        n = K * L
        cand_ids = nodes.view(Mv, n)                           # [Mv, n] node ids (−1 pad)
        cand_mask = is_ctx.view(Mv, n)                         # [Mv, n] bool
        cand_age = cum_age.view(Mv, n).clamp_min(0).float()   # [Mv, n] cumulative path age
        return uniq_v, v_inv, cand_ids, cand_age, cand_mask

    # ──────────────────────────────────────────────────────────────────
    # Per-batch training step
    # ──────────────────────────────────────────────────────────────────

    def _train_step(self, batch: Batch) -> Dict[str, float]:
        device = self.device
        B = len(batch.src)

        _, neg_tgt = self.neg_sampler_train.sample(batch)        # [B, K_train]
        src_t = torch.from_numpy(batch.src.astype(np.int64)).to(device)
        cand_np = np.concatenate(
            [batch.tgt.astype(np.int64)[:, None],
             np.ascontiguousarray(neg_tgt, dtype=np.int64)], axis=1)   # [B, 1+K]
        cand_t = torch.from_numpy(cand_np).to(device)
        t_query_t = torch.from_numpy(batch.ts.astype(np.int64)).to(device)

        logits = self._score(src_t, cand_t, t_query_t)           # [B, 1+K]
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

                # candidates_v [B, 1+max_K]; col 0 = positive, padded cols repeat
                # the positive (never read — per-row slice uses K_i).
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
