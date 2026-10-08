"""Train the node classifier on top of a frozen encoder (DyGLib's node-classification protocol).

Adam over the classifier's parameters ONLY; BCE-with-logits on one label per interaction
(source side); chronological batches; val ROC-AUC over the whole split (all batches
concatenated, as DyGLib computes it) drives early stopping; test AUC is read at the
best-val classifier state.
"""
import copy
import time
from typing import Dict, Iterator, List, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from link_property_prediction.data import SplitData

from .classifier import NodeClassifier
from .encoder import FrozenEncoder


class ClassifierResult(NamedTuple):
    best_epoch: int
    best_val_auc: float
    test_auc: float
    per_epoch_val_auc: List[float]


def _label_batches(labels: np.ndarray, batch_size: int, device) -> Iterator[torch.Tensor]:
    """Label chunks aligned with `create_batches`' consecutive `batch_size` slices."""
    for start in range(0, len(labels), batch_size):
        yield torch.from_numpy(labels[start:start + batch_size]).float().to(device)


@torch.no_grad()
def evaluate_auc(encoder: FrozenEncoder, classifier: NodeClassifier, split: SplitData,
                 labels: np.ndarray, batch_size: int) -> float:
    classifier.eval()
    scores = [classifier(z_geo, z_ef).cpu()
              for z_geo, z_ef in encoder.encode(split, batch_size)]
    return float(roc_auc_score(labels, torch.cat(scores).numpy()))


def fit_classifier(encoder: FrozenEncoder, classifier: NodeClassifier,
                   splits: Dict[str, SplitData], labels: Dict[str, np.ndarray], *,
                   batch_size: int, lr: float, num_epochs: int, patience: int) -> ClassifierResult:
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    best_val, best_epoch, best_state = -1.0, -1, None
    per_epoch_val: List[float] = []
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        t0 = time.time()
        loss_sum, n_batches = 0.0, 0
        batches = zip(encoder.encode(splits["train"], batch_size),
                      _label_batches(labels["train"], batch_size, encoder.device))
        for (z_geo, z_ef), y in batches:
            if len(y) < 2:              # BatchNorm cannot train on a single row
                continue
            loss = F.binary_cross_entropy_with_logits(classifier(z_geo, z_ef), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        train_dt = time.time() - t0

        val_auc = evaluate_auc(encoder, classifier, splits["val"], labels["val"], batch_size)
        per_epoch_val.append(val_auc)
        line = (f"epoch {epoch}/{num_epochs}  bce={loss_sum / max(n_batches, 1):.4f}  "
                f"train {train_dt:.1f}s  val_auc {val_auc:.4f}")
        if val_auc > best_val:
            best_val, best_epoch = val_auc, epoch
            best_state = copy.deepcopy(classifier.state_dict())
            no_improve = 0
            line += " (new best)"
        else:
            no_improve += 1
            line += f"  patience {no_improve}/{patience}"
        print(line, flush=True)
        if no_improve >= patience:
            break

    classifier.load_state_dict(best_state)
    test_auc = evaluate_auc(encoder, classifier, splits["test"], labels["test"], batch_size)
    return ClassifierResult(best_epoch=best_epoch, best_val_auc=best_val,
                            test_auc=test_auc, per_epoch_val_auc=per_epoch_val)
