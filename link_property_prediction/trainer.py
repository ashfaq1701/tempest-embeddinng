"""Per-query-causal training + eval loop — link-supervised MONOTONE metric head over walk bags.

Causality is enforced PER QUERY by Tempest's cutoff, not by ingestion order. The FULL graph
(train + val + test) is ingested into Tempest ONCE up front (`ingest_full_graph`, a single
`add_edges` call); there is no per-epoch reset and no per-batch ingestion. Per-batch ordering
(training):
  1. neg = neg_sampler.sample(batch)                 — [B, K_train] uniform negs
  2. candidates = [pos | negs]                       — [B, 1+K_train]
  3. logits = score(src, candidates)                 — TWO-SIDED: K backward walks are sampled for
       the source u AND for every candidate v, each bounded by the query's own cutoff t_i; both
       token bags go to LinkPredHead, which returns the negated monotone distance aggregate.
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

E (in the Poincare ball) and the head's channel weights are trained TOGETHER by the link CE, with
NO detach and no auxiliary loss. That is safe here precisely because the score is monotone in the
geodesic distances (every channel weight is a softplus, hence >= 0): the pull direction into E is
fixed by construction rather than learned, so there is no readout the head can reshape instead of
moving E. Intra-bag / cross-bag loss rows are the NEXT step and are not present yet.

TOKEN PREP — both sides go through `walk_tokens.build_query_walk_tokens`: walks are generated PER
QUERY (no dedup — each row's (node, t) needs its own cutoff) and returned ALREADY FLATTENED into a
[Q, T] WalkTokens bag (T = K*L). The head reads the flat fields directly, drops every slot whose
node is the bag's own seed, and softmaxes the fixed -(log1p(age) + log1p(hop-1)) recency/hop prior
over what remains.
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import geoopt
import numpy as np
import torch
import torch.nn.functional as F

from .data import Batch, SplitData
from .evaluator import Evaluator
from .model import LinkPredHead
from .negatives import MixedNegativeSampler
from .probes import CommunityProbe
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

    # data_stats mean-field per-node inter-event time T_train*N/(2E) — the characteristic AGE scale a
    # walk sees; used to scale-normalise the pooling recency weight (log1p(age / mean_node_inter_arrival)).
    mean_node_inter_arrival: float = 1.0

    # Model. The monotone weighted-mean head has NO tunable hyperparameters and NO head parameters at all —
    # the score is a fixed geometric aggregate of distances; E is the only trained tensor.
    d_emb: int = 128

    # Link loss / head.
    K_train: int = 10           # per-query training negatives ([B, 1+K_train]); 10 keeps the candidate
                                # bag small enough to fit bs 1000 on review
    hist_neg_ratio: float = 0.0  # fraction of training negatives drawn from the source's PAST destinations
                                 # (per-source reservoir), rest random — mirrors TGB eval's hist/rnd mix.
                                 # 0 = pure uniform (no reservoir). NOT for wiki (trains against eval signal);
                                 # for low-recurrence sets like review. See negatives.MixedNegativeSampler.
    reservoir_size: int = 256    # per-source historical reservoir depth M (only used when hist_neg_ratio>0);
                                 # size ~ typical per-source history — under-fill dilutes the hist fraction.

    # Walks (BACKWARD only, undirected). TWO-SIDED: the source u AND every candidate v are walked, each
    # bounded by the query's own cutoff t_i; both bags flow to the head.
    num_walks_per_node: int = 5
    max_walk_len: int = 5
    walk_bias: str = "ExponentialWeight"
    start_bias: str = "ExponentialWeight"
    t2nv_p: float = 4.0    # node2vec return param (used only when a bias is TemporalNode2Vec)
    t2nv_q: float = 0.25   # node2vec in-out param; low q/p = most diverse backward walks

    # Optimisation — CONSTANT lr (no schedule), ONE param group for everything. RiemannianAdam applies
    # the Riemannian update to E (a geoopt.ManifoldParameter) and standard Adam to the Euclidean head
    # params within the single group, so E and the head train at the same lr. NO weight decay: a wiki A/B
    # (with the boundary prior removed) showed no-wd lets E spread and beats wd 1e-4 on link MRR.
    lr: float = 1e-4

    # Run control.
    num_epochs: int = 25
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
        # Single module owning the Poincare-ball node embeddings AND the monotone metric score. There is
        # no scorer MLP and no feature channel: the score is -(d_id + weighted-mean bag distance over the
        # two walk bags), so node / edge features have nowhere to enter yet (they return later as a
        # modulation of the per-slot pooling weights, which preserves monotonicity).
        self.model = LinkPredHead(
            num_nodes=config.num_nodes,
            d_emb=int(config.d_emb),
            mean_node_inter_arrival=float(config.mean_node_inter_arrival),
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
        # Training negatives: hist_neg_ratio=0 -> pure uniform (no reservoir allocated);
        # >0 -> a hist/rnd mix from a per-source causal reservoir (fed post-scoring via observe()).
        self.neg_sampler_train = MixedNegativeSampler(
            num_neg_per_pos=config.K_train,
            hist_ratio=config.hist_neg_ratio,
            dst_pool=config.dst_pool,
            num_nodes=config.num_nodes,
            reservoir_size=config.reservoir_size,
            seed=config.seed,
        )

        # ONE param group at a single lr: RiemannianAdam gives E (the geoopt.ManifoldParameter) the
        # Riemannian update and the head's Euclidean params standard Adam, all under the same lr.
        self.opt = geoopt.optim.RiemannianAdam(
            self.model.parameters(), lr=float(config.lr), stabilize=10,
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
               t_query_t: torch.Tensor):
        """src_t [B] long, cand_t [B, C] long, t_query_t [B] long -> logits [B, C]. The head consumes the
        two walk bags internally; only the logits are returned.

        TWO-SIDED per-query walks: the SOURCE side samples K backward walks for each query (u_i, t_i)
        with cutoff = t_i; the CANDIDATE side samples K backward walks for every candidate v_ij with the
        SAME cutoff t_i (so both sides are causal as of the query time). Both bags flow to the head.
        Cost: the candidate side is C walk queries per positive, i.e. ~C× the source walks. This is
        already the dominant per-batch cost, and it scales with K_train, NOT with two-sidedness."""
        device = self.device

        # SOURCE side: per-query (u_i, t_i) → K cutoff=t_i backward walks → raw [B,K,L] token bag.
        src_tokens = build_query_walk_tokens(
            self.walk_gen, device, src_t, t_query_t,
            max_walk_len=self.config.max_walk_len,
            num_walks_per_node=self.config.num_walks_per_node,
            start_bias=self.config.start_bias,
            walk_bias=self.config.walk_bias)

        # CANDIDATE side: walk every candidate v with its query's cutoff t_i. Flatten [B,C] → [B*C]
        # query-major; each candidate inherits its query's cutoff so its walk is causal.
        b, c = cand_t.shape
        cand_seeds = cand_t.reshape(-1)                                  # [B*C]
        cand_cutoffs = t_query_t.unsqueeze(1).expand(b, c).reshape(-1)   # [B*C]
        cand_tokens = build_query_walk_tokens(
            self.walk_gen, device, cand_seeds, cand_cutoffs,
            max_walk_len=self.config.max_walk_len,
            num_walks_per_node=self.config.num_walks_per_node,
            start_bias=self.config.start_bias,
            walk_bias=self.config.walk_bias)

        return self.model(src_tokens, cand_tokens)

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

        logits = self._score(src_t, cand_t, t_query_t)                            # [B, 1+K]
        target = torch.zeros(B, dtype=torch.long, device=device)
        link_loss = F.cross_entropy(logits, target)

        # ALIGNMENT half of the Wang-Isola (ICML'20) decomposition: the mean geodesic distance between the
        # POSITIVE pair's embeddings. The head's score is monotone in the distances, so alignment/uniformity
        # is now measurable on the ACTUAL scoring geometry rather than on a proxy — falling alignment with
        # falling uniformity is the collapse signature, and it is the reason this is logged from day one.
        with torch.no_grad():
            e = self.model.E.weight
            align = self.model.geom.pairwise_dist(
                F.embedding(src_t, e).unsqueeze(-2),
                F.embedding(cand_t[:, 0], e).unsqueeze(-2)).mean()

        # loss = link CE ONLY. E is trained end-to-end by the link CE through the monotone head (no detach).
        # No boundary penalty: an inward hyperbolic spring was tried and removed — on spread init it strangled
        # E's radius and capped MRR well below the unpenalised basin, and every boundary control (spring,
        # projection cap, learnable curvature) failed to stop the fast-lr collapse anyway. The radius is left
        # to the optimiser; |E|mean / |E|max are still logged as the geometry watch.
        loss = link_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        # Feed this batch's positives into the (optional) historical reservoir AFTER scoring —
        # strict-causal: at score time the reservoir held only strictly-earlier batches' edges.
        # No-op when hist_neg_ratio == 0. Relies on chronological train-batch order (which the
        # per-query cutoff=t protocol already assumes).
        self.neg_sampler_train.observe(batch.src, batch.tgt)

        return {
            "link": float(link_loss.detach()),
            "align": float(align),
            "lr": float(self.opt.param_groups[0]["lr"]),
        }

    # ──────────────────────────────────────────────────────────────────
    # Geometry probe — the collapse / blow-up watch
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _geometry_probe(self, n_sample: int = 1024) -> Dict[str, float]:
        """UNIFORMITY half of the Wang-Isola decomposition plus the boundary watch.

          unif     = log mean exp(-d_H) over random OFF-DIAGONAL pairs of E. RISES toward 0 as E
                     contracts — read it against the per-epoch `align`: both falling together is healthy
                     (alignment without collapse); both rising is the over-clustering collapse.
          max_norm = max ||E||. The uniformity pressure in this design pushes norms outward and the
                     conformal factor blows up at 1; if this runs, CLAMP the ball radius rather than
                     adding a penalty term (the old boundary prior over-compressed E).
        """
        e = self.model.E.weight.detach()
        n = e.shape[0]
        idx = torch.randperm(n, device=e.device)[:min(n_sample, n)]
        x = e[idx]
        d = self.model.geom.pairwise_dist(x, x)                     # [n, n]
        off = ~torch.eye(x.shape[0], dtype=torch.bool, device=e.device)
        unif = torch.logsumexp(-d[off], dim=0) - torch.log(off.sum().to(d.dtype))
        norms = e.norm(dim=-1)
        return {"unif": float(unif), "max_norm": float(norms.max()), "mean_norm": float(norms.mean())}

    # ──────────────────────────────────────────────────────────────────
    # Eval — strict-causal, no_grad
    # ──────────────────────────────────────────────────────────────────

    def _eval(self, evaluator: Evaluator, batches: Iterable[Batch],
              recorder: Any = None) -> float:
        self.model.eval()
        # Rewind any fixed-negative cursor (TGB-Seq) so every pass scores against
        # the same negatives in split order; a no-op for content-addressed
        # samplers (TGB). Must precede the first sample_negatives call.
        evaluator.reset()
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
                logits = self._score(src_t, cand_t, t_query_t)
                logits = logits.cpu().numpy()

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
            self.neg_sampler_train.reset()   # drop last epoch's reservoir (re-derives identically next pass;
                                             # without it, epoch-1's whole pass would pollute epoch-2's early
                                             # batches with future edges — strict-causal violation). No-op at ratio 0.

            t0 = time.time()
            link_sum, align_sum, n_batches = 0.0, 0.0, 0
            for batch in train_batches_factory():
                m = self._train_step(batch)
                link_sum += m["link"]
                align_sum += m["align"]
                n_batches += 1
            train_dt = time.time() - t0

            line = (
                f"epoch {ep}/{n_epochs}  "
                f"link={link_sum / max(n_batches, 1):.4f}  "
                f"lr={self.opt.param_groups[0]['lr']:.0e}  "
                f"train {train_dt:.1f}s"
            )
            cp = self.comm_probe.measure(self.model.E.weight.detach())     # community-formation probe
            line += f"  commP={cp:.3f}(x{cp / max(self.comm_probe.null, 1e-9):.1f})"

            # Geometry watch: alignment (positive-pair distance) vs uniformity (spread), plus the boundary
            # radius. The head has no learnable channel mix — the score is a fixed weighted-mean aggregate.
            g = self._geometry_probe()
            line += (f"  align={align_sum / max(n_batches, 1):.3f}  unif={g['unif']:.3f}"
                     f"  |E|mean={g['mean_norm']:.3f}  |E|max={g['max_norm']:.3f}")

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
