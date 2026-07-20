"""TGB Evaluator wrapper — architecture-agnostic.

Wraps `tgb.linkproppred.evaluate.Evaluator` and a `NegativeSampler` so a
training loop can request per-positive negatives, score them with its
own model, and feed the resulting (pos, neg-array) pair into TGB's
official MRR scorer.

The new design supplies its own scoring loop; this module exists only
to (a) load TGB's per-positive negatives via the injected sampler, and
(b) call `tgb_eval.eval(...)` per positive for leaderboard-identical
metrics. No knowledge of the model architecture lives here.
"""

import numpy as np

from .data import Batch
from .negatives import NegativeSampler


class Evaluator:
    """Thin wrapper around TGB's `Evaluator.eval(...)` plus a NegativeSampler.

    Usage (from a training loop):

        evaluator = Evaluator(
            neg_sampler=TGBNegativeSampler(loaded.dataset, split_mode="val"),
            tgb_dataset_name=loaded.name,
            eval_metric=loaded.eval_metric,
        )
        for batch in val_batches:
            neg_src_list, neg_tgt_list = evaluator.sample_negatives(batch)
            # score(u, v) is the caller's responsibility — model-specific.
            pos_scores = score(batch.src, batch.tgt, batch.ts)
            neg_scores = [score(np.full_like(nt, s), nt, np.full_like(nt, t))
                          for s, nt, t in zip(batch.src, neg_tgt_list, batch.ts)]
            for pos, neg in zip(pos_scores, neg_scores):
                m = evaluator.score_to_metric(pos, neg)
                ...
    """

    def __init__(
        self,
        neg_sampler: NegativeSampler,
        tgb_dataset_name: str,
        eval_metric: str,
    ):
        from tgb.linkproppred.evaluate import Evaluator as TGBEvaluator

        self.neg_sampler = neg_sampler
        self.eval_metric = eval_metric
        self.tgb_eval = TGBEvaluator(name=tgb_dataset_name)

    def sample_negatives(self, batch: Batch):
        """Forwards to the injected sampler. Most useful for eval-time
        samplers that return per-positive variable-K negative arrays
        (TGB's pre-generated lists)."""
        return self.neg_sampler.sample(batch)

    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        """Run TGB's official scorer on a single positive plus its
        per-positive negative scores. Returns the metric value
        (`self.eval_metric` from the dataset)."""
        pos = np.asarray([pos_score], dtype=np.float64)
        neg = np.asarray(neg_scores, dtype=np.float64)
        res = self.tgb_eval.eval({
            "y_pred_pos": pos,
            "y_pred_neg": neg,
            "eval_metric": [self.eval_metric],
        })
        return float(res[self.eval_metric])
