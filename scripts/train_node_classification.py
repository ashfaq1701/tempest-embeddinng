"""Frozen-encoder dynamic node classification (DyGLib protocol).

Loads a Run-1 link-prediction checkpoint (`train_link_property_prediction.py
--data-suite dyglib --save-checkpoint ...`), freezes it, and trains only a small
classifier on [log_O(p_u), mean walk edge feature] per interaction, source side.
"""

import argparse
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from link_property_prediction.data import concat_splits
from link_property_prediction.dyglib_eval import load_dyglib
from link_property_prediction.utils import seed_all
from node_classification.classifier import NodeClassifier
from node_classification.encoder import FrozenEncoder
from node_classification.train import fit_classifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen-encoder node classification")
    p.add_argument("--dataset", required=True, type=str,
                   help="DyGLib dataset with dynamic labels: wikipedia or reddit.")
    p.add_argument("--data-root", default="datasets", type=str)
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Run-1 link-prediction checkpoint (state_dict + args).")
    p.add_argument("--batch-size", default=200, type=int)
    p.add_argument("--num-epochs", default=100, type=int)
    p.add_argument("--early-stop-patience", default=20, type=int)
    p.add_argument("--lr", default=1e-4, type=float, help="DyGLib's node-classification lr.")
    p.add_argument("--seed", default=42, type=int,
                   help="Classifier init and walk seed; pair seed k with the seed-k checkpoint.")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--use-gpu-tempest", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    loaded = load_dyglib(args.dataset, args.data_root)
    labels = loaded.dataset.labels
    splits = {"train": loaded.train, "val": loaded.val, "test": loaded.test}
    for name, split in splits.items():
        n_pos = int(labels[name].sum())
        print(f"  {name:5s} interactions {len(split.sources):>8,}  positives {n_pos:>5,}")

    encoder = FrozenEncoder.from_checkpoint(
        args.checkpoint, concat_splits(loaded.train, loaded.val, loaded.test),
        device=device, use_gpu_tempest=args.use_gpu_tempest, seed=args.seed)
    classifier = NodeClassifier(d_geo=encoder.d_geo, d_ef=encoder.d_ef).to(device)
    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  classifier params: {n_params:,}  (encoder frozen)")

    hash_before = encoder.state_hash()
    result = fit_classifier(
        encoder, classifier, splits, labels,
        batch_size=args.batch_size, lr=args.lr,
        num_epochs=args.num_epochs, patience=args.early_stop_patience)
    if encoder.state_hash() != hash_before:
        raise RuntimeError("encoder weights changed during classifier training")

    print("\n=== Final results ===")
    print(f"  dataset:        {args.dataset}")
    print(f"  seed:           {args.seed}")
    print(f"  best_epoch:     {result.best_epoch}")
    print(f"  best_val_auc:   {result.best_val_auc:.4f}")
    print(f"  test_auc:       {result.test_auc:.4f}")


if __name__ == "__main__":
    main()
