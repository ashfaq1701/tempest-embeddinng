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
go through ONE shared routine `_walk_csr`: walk for the unique seeds, flatten the (K,L) walk
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    num_walks_per_node_candidate_side: int = 10
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


# CSR bundle — the symmetric per-seed deduplicated walk-neighbourhood.
#   node_ids  [G, U]        distinct neighbour node ids per seed (−1 at padded node slots)
#   node_mask [G, U]        True at a real distinct node
#   ages      [G, U, kmax]  each distinct node's OCCURRENCE ages (raw; 0 at padded slots)
#   age_mask  [G, U, kmax]  True at a real occurrence (count_node = age_mask.sum(-1))
WalkCSR = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


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
    # Symmetric CSR token preparation — ONE routine for both sides
    # ──────────────────────────────────────────────────────────────────

    def _walk_csr(self, seeds_unique: torch.Tensor, t_query_per_seed: torch.Tensor,
                  *, max_walk_len: Optional[int] = None,
                  num_walks_per_node: Optional[int] = None,
                  start_bias: Optional[str] = None,
                  walk_bias: Optional[str] = None) -> WalkCSR:
        """Walk for `seeds_unique`, then DEDUPLICATE each seed's walk-neighbours into a
        compact per-node CSR. Symmetric: the source and candidate sides call this with the
        same contract, differing only in seeds + walk params.

           seeds_unique      [G] long      unique seed node ids (sources OR candidates)
           t_query_per_seed  [G] long      the query time to age each seed's tokens against
                                           (sources: per-source t_query; candidates: see note)
           -> WalkCSR (node_ids [G,U], node_mask [G,U], ages [G,U,kmax], age_mask [G,U,kmax])

        Dedup is per seed-row: occurrences of the same neighbour node collapse to ONE node
        slot carrying ALL its occurrence ages (recency mean stays exact) and its count
        (= #occurrences = age_mask.sum(-1)). U = max distinct neighbours over rows, kmax =
        max occurrences of any one node in a row; both padded. Seed (= the node itself) and
        walk padding are excluded before dedup. Strict-causal: pre-ingest graph snapshot."""
        device = self.device
        G = int(seeds_unique.shape[0])
        wd = self.walk_gen.walks_for_nodes(
            seeds_unique.cpu().numpy(),
            max_walk_len=max_walk_len,
            num_walks_per_node=num_walks_per_node,
            start_bias=start_bias,
            walk_bias=walk_bias)
        K, L = wd.K, wd.nodes.shape[1]
        nodes = wd.nodes.to(device).view(G, K, L)               # [G, K, L]
        ts = wd.timestamps.to(device).view(G, K, L)             # [G, K, L] int64 edge times
        lens = wd.lens.to(device).view(G, K)                    # [G, K]
        # Context = real walk-neighbours: exclude the seed slot (lens-1) and padding.
        is_ctx = torch.arange(L, device=device).view(1, 1, L) < (lens - 1).unsqueeze(-1)

        n = K * L
        flat_ids = nodes.reshape(G, n)                          # [G, n] node ids (−1 pad)
        flat_ts = ts.reshape(G, n)                              # [G, n] raw edge times
        flat_mask = is_ctx.reshape(G, n)                        # [G, n] True at real token
        # Per-token RAW age = t_query − t_edge (≥0), masked. clamp_min neutralises the seed
        # sentinel; the mask zeroes padding's bogus value — finite everywhere.
        flat_age = ((t_query_per_seed.view(G, 1).to(torch.int64) - flat_ts).clamp_min(0)
                    .to(torch.float32)) * flat_mask.to(torch.float32)       # [G, n]

        return self._dedup_to_csr(flat_ids, flat_age, flat_mask)

    @staticmethod
    def _dedup_to_csr(flat_ids: torch.Tensor, flat_age: torch.Tensor,
                      flat_mask: torch.Tensor) -> WalkCSR:
        """Collapse a flat per-seed token set [G, n] (with repeated nodes) into the compact
        CSR (node_ids [G,U], node_mask [G,U], ages [G,U,kmax], age_mask [G,U,kmax]) by
        grouping occurrences of the same node id within each row. Vectorised, dense-padded.

        U = max distinct nodes over rows; kmax = max occurrences of any one node in a row.
        All occurrence ages are kept (recency stays exact); count = age_mask.sum(-1)."""
        device = flat_ids.device
        G, n = flat_ids.shape

        # Sort each row by node id so equal ids are contiguous; invalids (mask False) pushed
        # to the end via a large sentinel key. Stable so occurrence order within a node holds.
        BIG = torch.iinfo(torch.int64).max
        sort_key = torch.where(flat_mask, flat_ids, torch.full_like(flat_ids, BIG))
        order = torch.argsort(sort_key, dim=1, stable=True)              # [G, n]
        ids_s = torch.gather(flat_ids, 1, order)                        # [G, n] sorted ids
        age_s = torch.gather(flat_age, 1, order)                        # [G, n]
        msk_s = torch.gather(flat_mask, 1, order)                       # [G, n]

        # "new distinct node" boundary within a row: first valid slot, or id changes.
        prev_id = torch.cat([torch.full((G, 1), -2, device=device, dtype=ids_s.dtype),
                             ids_s[:, :-1]], dim=1)
        prev_msk = torch.cat([torch.zeros((G, 1), device=device, dtype=torch.bool),
                              msk_s[:, :-1]], dim=1)
        is_new = msk_s & (~prev_msk | (ids_s != prev_id))               # [G, n] start of a node-run
        # distinct-node index per valid slot (cumsum of run-starts − 1); invalids → −1.
        node_idx = torch.cumsum(is_new.to(torch.int64), dim=1) - 1      # [G, n]
        node_idx = torch.where(msk_s, node_idx, torch.full_like(node_idx, -1))
        U = int(node_idx.max().item()) + 1 if msk_s.any() else 1

        # occurrence index WITHIN each distinct node: position minus the run-start position.
        ar = torch.arange(n, device=device).view(1, n).expand(G, n)     # [G, n] col positions
        run_start_pos = torch.where(is_new, ar, torch.zeros_like(ar))
        # cummax of run_start_pos gives, at each slot, the start position of its current run.
        run_start = torch.cummax(run_start_pos, dim=1).values            # [G, n]
        occ_idx = torch.where(msk_s, ar - run_start, torch.zeros_like(ar))   # [G, n]
        kmax = int(occ_idx[msk_s].max().item()) + 1 if msk_s.any() else 1

        # Scatter sorted (id, age) into [G, U, kmax] by (node_idx, occ_idx).
        node_ids = torch.full((G, U), -1, device=device, dtype=flat_ids.dtype)
        node_mask = torch.zeros((G, U), device=device, dtype=torch.bool)
        ages = torch.zeros((G, U, kmax), device=device, dtype=flat_age.dtype)
        age_mask = torch.zeros((G, U, kmax), device=device, dtype=torch.bool)

        valid = msk_s
        rows = torch.arange(G, device=device).view(G, 1).expand(G, n)[valid]
        u_at = node_idx[valid]
        k_at = occ_idx[valid]
        node_ids[rows, u_at] = ids_s[valid]
        node_mask[rows, u_at] = True
        ages[rows, u_at, k_at] = age_s[valid]
        age_mask[rows, u_at, k_at] = True
        return node_ids, node_mask, ages, age_mask

    @staticmethod
    def _gather_csr(csr: WalkCSR, index: torch.Tensor, out_shape) -> WalkCSR:
        """Index a per-unique-seed CSR [G,…] onto the batch grid via `index` (long [P]),
        reshaping the leading axis to `out_shape` (e.g. [B] for sources, [B,C] for
        candidates). The dedup is exact: each seed's CSR is query-independent, so gathering
        replicates it to every cell naming that seed (scatter-add adjoint on backward)."""
        node_ids, node_mask, ages, age_mask = csr
        U = node_ids.shape[-1]; kmax = ages.shape[-1]
        ni = node_ids[index].view(*out_shape, U)
        nm = node_mask[index].view(*out_shape, U)
        ag = ages[index].view(*out_shape, U, kmax)
        am = age_mask[index].view(*out_shape, U, kmax)
        return ni, nm, ag, am

    # ──────────────────────────────────────────────────────────────────
    # Scoring — shared by train + eval
    # ──────────────────────────────────────────────────────────────────

    def _score(self, src_t: torch.Tensor, cand_t: torch.Tensor,
               t_query_t: torch.Tensor) -> torch.Tensor:
        """src_t [B] long, cand_t [B, C] long, t_query_t [B] long -> logits [B, C].

        SYMMETRIC: build a per-node CSR for the unique sources (→ μ tokens) and a per-node
        CSR for the unique candidates (→ connectors), via the SAME `_walk_csr`. Gather the
        source CSR to [B,…] and scatter the candidate CSR to [B,C,…]. IDs go to the head;
        the head gathers embeddings from the shared table for both sides."""
        device = self.device
        B, C = cand_t.shape

        # --- SOURCE side: unique sources → CSR → gather to [B] ---
        uniq_s, u_pos = torch.unique(src_t, return_inverse=True)        # [Ms], [B]
        first_src = torch.argmax(
            (src_t.view(-1, 1) == uniq_s.view(1, -1)).to(torch.int64), dim=0)
        csr_s = self._walk_csr(
            uniq_s, t_query_t[first_src],
            max_walk_len=self.config.max_walk_len_query_side,
            num_walks_per_node=self.config.num_walks_per_node_query_side,
            start_bias=self.config.start_bias_query_side,
            walk_bias=self.config.walk_bias_query_side)
        src_ids, src_nmask, src_ages, src_amask = self._gather_csr(csr_s, u_pos, (B,))

        # --- CANDIDATE side: unique candidates → CSR → scatter to [B,C] ---
        uniq_v, v_inv = torch.unique(cand_t.reshape(-1), return_inverse=True)  # [Mv], [B*C]
        # Each unique candidate is aged against the query time of (one of) the rows that
        # names it; ages enter only the (recency) reductions and the per-seed snapshot is
        # shared, so any naming row's t_query is consistent for the dedup.
        first_row = torch.argmax(
            (cand_t.reshape(-1).view(-1, 1) == uniq_v.view(1, -1)).to(torch.int64), dim=0)
        tq_per_v = t_query_t.view(B, 1).expand(B, C).reshape(-1)[first_row]    # [Mv]
        csr_v = self._walk_csr(
            uniq_v, tq_per_v,
            max_walk_len=self.config.max_walk_len_candidate_side,
            num_walks_per_node=self.config.num_walks_per_node_candidate_side,
            start_bias=self.config.start_bias_candidate_side,
            walk_bias=self.config.walk_bias_candidate_side)
        cand_ids, cand_nmask, cand_ages, cand_amask = self._gather_csr(csr_v, v_inv, (B, C))

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
