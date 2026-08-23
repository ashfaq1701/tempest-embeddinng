"""Capture and visualise training dynamics as a video of the embedding organising on the ball.

Runs ``scripts/train_link_property_prediction.py`` while snapshotting the E table every N batches
(via ``--snapshot-dir`` / ``--snapshot-every-n-batches``), then stitches the snapshots into an MP4:
each frame is the embedding on the Poincaré disk (radial = depth ``||E||``, angle = a FIXED PCA basis
so the motion is smooth, colour = node degree), labelled with ``batch / epoch / validation MRR`` --
the MRR being read back from the training log by regex.

Example (a quick 5-epoch GoogleLocal test):
    python scripts/animate_training_dynamics.py \
        --dataset GoogleLocal --is-bipartite \
        --embedding-snapshot-path /tmp/gl_snaps --d-emb 16 --patience 3 \
        --snapshot-per-n-batches 50 --n-epochs 5 \
        --output-video-file-path gl_dynamics.mp4
"""
import argparse
import math
import os
import pathlib
import re
import subprocess
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from plots.ball_frame_plot import build_frame_context, ball_frame_plot

# The train script's default batch size; this orchestrator does not override it, so batches-per-epoch
# is exact as train_edges / _BATCH_SIZE (rounded up).
_BATCH_SIZE = 1000


def run_training(args: argparse.Namespace, snapshot_dir: str) -> str:
    """Run the training script with snapshotting enabled; tee its output to <snapshot_dir>/training.log."""
    log_path = os.path.join(snapshot_dir, "training.log")
    cmd = [
        sys.executable, str(_PROJECT_ROOT / "scripts" / "train_link_property_prediction.py"),
        "--data-suite", "tgb-seq", "--dataset", args.dataset,
        "--d-emb", str(args.d_emb),
        "--num-epochs", str(args.n_epochs),
        "--early-stop-patience", str(args.patience),
        "--snapshot-dir", snapshot_dir,
        "--snapshot-every-n-batches", str(args.snapshot_per_n_batches),
        "--seed", str(args.seed),
        "--use-gpu", "--use-gpu-tempest",
    ]
    if args.is_bipartite:
        cmd.append("--is-bipartite")

    print("Running training:\n  " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"Training failed (exit {proc.returncode}); see {log_path}")
    return log_path


def parse_log(log_path: str):
    """Extract per-epoch validation MRR ({epoch: val}) and the train-edge count from the training log."""
    val_by_epoch, train_edges = {}, None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"train edges:\s*([\d,]+)", line)
            if m:
                train_edges = int(m.group(1).replace(",", ""))
            if line.startswith("epoch"):
                ep = re.search(r"^epoch (\d+)/", line)
                val = re.search(r"\bval ([0-9.]+)", line)
                if ep and val:
                    val_by_epoch[int(ep.group(1))] = float(val.group(1))
    if train_edges is None:
        sys.exit("Could not find 'train edges: N' in the training log.")
    return val_by_epoch, train_edges


def node_degrees(csv_path: str, num_nodes: int) -> np.ndarray:
    """Undirected degree per node id from a ``src,dst`` edge CSV."""
    edges = pd.read_csv(csv_path, usecols=["src", "dst"])
    deg = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(deg, edges["src"].to_numpy(), 1)
    np.add.at(deg, edges["dst"].to_numpy(), 1)
    return deg


def render_video(snapshot_dir: str, output_path: str, val_by_epoch: dict, batches_per_epoch: int,
                 degree: np.ndarray, fps: int, n_sample: int, seed: int) -> None:
    """Render every snapshot (sorted by batch) into an MP4 with a batch/epoch/MRR label."""
    snaps = sorted(pathlib.Path(snapshot_dir).glob("E_batch*.npy"))
    if not snaps:
        sys.exit(f"No snapshots (E_batch*.npy) found in {snapshot_dir}.")
    batch_of = [int(re.search(r"E_batch(\d+)", p.name).group(1)) for p in snaps]

    ctx = build_frame_context(str(snaps[-1]), degree, n_sample=n_sample, seed=seed)

    print(f"Rendering {len(snaps)} frames -> {output_path}", flush=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    try:
        for i, (path, batch) in enumerate(zip(snaps, batch_of)):
            completed_epoch = batch // batches_per_epoch     # last epoch with a measured val MRR
            mrr = val_by_epoch.get(completed_epoch)
            writer.append_data(
                ball_frame_plot(ctx, str(path), mrr, batch / batches_per_epoch, batch))
            if i % 100 == 0:
                print(f"  frame {i}/{len(snaps)}", flush=True)
    finally:
        writer.close()
    print(f"saved {output_path}  ({len(snaps)} frames, {len(snaps) / fps:.1f}s at {fps}fps)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="tgb-seq dataset name (e.g. GoogleLocal).")
    ap.add_argument("--embedding-snapshot-path", required=True,
                    help="Directory for the E snapshots (created if it does not exist).")
    ap.add_argument("--output-video-file-path", required=True, help="Output .mp4 path.")
    ap.add_argument("--d-emb", type=int, default=16, help="Embedding dimension.")
    ap.add_argument("--patience", type=int, default=3, help="Early-stop patience (epochs).")
    ap.add_argument("--snapshot-per-n-batches", type=int, default=100, help="Snapshot cadence (batches).")
    ap.add_argument("--n-epochs", type=int, default=50, help="Max training epochs.")
    ap.add_argument("--is-bipartite", action="store_true",
                    help="Set for bipartite datasets (GoogleLocal, ML-20M, Yelp, Taobao).")
    ap.add_argument("--tgb-root", default="datasets",
                    help="Root holding <dataset>/<dataset>.csv (for node degrees).")
    ap.add_argument("--fps", type=int, default=30, help="Video frame rate.")
    ap.add_argument("--n-sample", type=int, default=6000, help="Nodes drawn per frame.")
    ap.add_argument("--seed", type=int, default=42, help="Training + sampling seed.")
    args = ap.parse_args()

    snapshot_dir = args.embedding_snapshot_path
    os.makedirs(snapshot_dir, exist_ok=True)

    log_path = run_training(args, snapshot_dir)
    val_by_epoch, train_edges = parse_log(log_path)
    batches_per_epoch = math.ceil(train_edges / _BATCH_SIZE)

    num_nodes = int(np.load(sorted(pathlib.Path(snapshot_dir).glob("E_batch*.npy"))[0]).shape[0])
    csv_path = os.path.join(args.tgb_root, args.dataset, f"{args.dataset}.csv")
    degree = node_degrees(csv_path, num_nodes)

    render_video(snapshot_dir, args.output_video_file_path, val_by_epoch, batches_per_epoch,
                 degree, fps=args.fps, n_sample=args.n_sample, seed=args.seed)


if __name__ == "__main__":
    main()
