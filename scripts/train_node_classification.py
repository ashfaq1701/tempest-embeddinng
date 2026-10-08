import argparse
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from link_property_prediction.data import concat_splits
from link_property_prediction.dyglib_eval import load_dyglib
from link_property_prediction.utils import seed_all
from node_classification.encoder import FrozenEncoder
from node_classification.model import NodeClassifier
from node_classification.trainer import fit_classifier


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
    p.add_argument("--lr", default=1e-3, type=float,
                   help="Classifier lr. DyGLib uses 1e-4; 1e-3 won on validation here.")
    p.add_argument("--num-walks-per-node", default=20, type=int,
                   help="Walks per node at classification time (the checkpoint trained with 5).")
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

    stream = concat_splits(loaded.train, loaded.val, loaded.test)
    encoder = FrozenEncoder.from_checkpoint(
        args.checkpoint, stream, device=device, use_gpu_tempest=args.use_gpu_tempest,
        seed=args.seed, num_walks_per_node=args.num_walks_per_node)
    hash_before = encoder.state_hash()

    features = {}
    for name, split in splits.items():
        features[name] = encoder.encode(split)
    print(f"  checkpoint: {args.checkpoint}  (walks per node {encoder.walk_args['num_walks_per_node']})")

    classifier = NodeClassifier(n_in=features["train"].shape[1]).to(device)
    n_params = sum(p.numel() for p in classifier.parameters())
    print(f"  classifier params: {n_params:,}  (encoder frozen)")

    result = fit_classifier(
        classifier, features, {k: np.asarray(v) for k, v in labels.items()},
        batch_size=args.batch_size, lr=args.lr,
        num_epochs=args.num_epochs, patience=args.early_stop_patience, seed=args.seed)
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
