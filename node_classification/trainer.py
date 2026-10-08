"""Train the node classifier on frozen-encoder features (DyGLib's node-classification protocol).

Adam over the classifier's parameters ONLY; BCE-with-logits on one label per interaction
(source side); val ROC-AUC over the whole split drives early stopping; test AUC is read at
the best-val classifier state.

Training batches are shuffled. The features are already causal (each row was encoded with the
exclusive cutoff of its own interaction), so the order rows are visited in carries no leak;
in time order the rare positives arrive in clumps, which skews both the gradient and
BatchNorm's batch statistics.
"""
import copy
import time
from typing import Dict, List, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .model import NodeClassifier


class ClassifierResult(NamedTuple):
    best_epoch: int
    best_val_auc: float
    test_auc: float
    per_epoch_val_auc: List[float]


@torch.no_grad()
def evaluate_auc(classifier: NodeClassifier, x: torch.Tensor, y: np.ndarray,
                 batch_size: int = 4096) -> float:
    classifier.eval()
    scores = torch.cat([classifier(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    return float(roc_auc_score(y, scores.cpu().numpy()))


def fit_classifier(classifier: NodeClassifier, features: Dict[str, torch.Tensor],
                   labels: Dict[str, np.ndarray], *, batch_size: int, lr: float,
                   num_epochs: int, patience: int, seed: int) -> ClassifierResult:
    device = features["train"].device
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    x_train = features["train"]
    y_train = torch.from_numpy(labels["train"]).float().to(device)
    order = torch.Generator(device="cpu").manual_seed(seed)

    best_val, best_epoch, best_state = -1.0, -1, None
    per_epoch_val: List[float] = []
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        t0 = time.time()
        perm = torch.randperm(len(x_train), generator=order).to(device)
        loss_sum, n_batches = 0.0, 0
        for start in range(0, len(perm), batch_size):
            rows = perm[start:start + batch_size]
            if len(rows) < 2:           # BatchNorm cannot train on a single row
                continue
            loss = F.binary_cross_entropy_with_logits(classifier(x_train[rows]), y_train[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        train_dt = time.time() - t0

        val_auc = evaluate_auc(classifier, features["val"], labels["val"])
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
    test_auc = evaluate_auc(classifier, features["test"], labels["test"])
    return ClassifierResult(best_epoch=best_epoch, best_val_auc=best_val,
                            test_auc=test_auc, per_epoch_val_auc=per_epoch_val)
