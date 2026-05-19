"""Parse run logs and emit a clean comparison table.

Reads logs matching a glob pattern and extracts:
  - dataset, seed, primary loss, λ_normbrake, λ_link, head_mode,
    n_layers, link_dropout, embedding_dropout
  - best epoch, best val MRR, best test MRR
  - per-epoch val MRR trajectory (and detects cliffs)
  - per-epoch col-norm and L_normbrake (if logged)

Usage:
    python3 scripts/summarize_runs.py 'runs/sec481_*.log'
"""

import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_log(path: str) -> Optional[Dict]:
    text = Path(path).read_text(errors="replace")
    out: Dict = {"path": Path(path).name}
    fname = Path(path).name

    # Derive config from filename if not visible in log body.
    # Recognised patterns:
    #   lossmining_v2_cell<N>_<loss>[_nb]_seed<S>_*.log
    #   sec481_<dataset>_<TAG>_seed<S>_*.log  where TAG encodes loss + λ_link
    #   sec482_<dataset>_<loss>_<TAG>_seed<S>_*.log
    m = re.search(r"cell\d+_([a-z]+)(_nb)?_seed(\d+)", fname)
    if m:
        out["primary_loss"] = m.group(1)
        out["lambda_normbrake"] = 0.1 if m.group(2) else 0.0
        out["seed"] = int(m.group(3))
    m = re.search(r"sec481_[a-z0-9-]+_([AITS])_jl([\d.]+)_seed(\d+)", fname)
    if m:
        out["primary_loss"] = {"A": "alignment", "I": "infonce", "T": "triplet", "S": "sgns"}[m.group(1)]
        out["lambda_link"] = float(m.group(2))
        out["seed"] = int(m.group(3))
        if m.group(1) == "S":
            out["lambda_normbrake"] = 0.1
    m = re.search(r"sec482_[a-z0-9-]+_([a-z]+)_A_", fname)
    if m:
        out["primary_loss"] = m.group(1)

    # Header
    m = re.search(r"Loading TGB dataset:\s*(\S+)", text)
    if m:
        out["dataset"] = m.group(1)
    # CLI args (peek the wrapper line if present)
    m = re.search(r"--primary-loss\s+(\S+)", text)
    if m:
        out["primary_loss"] = m.group(1)
    m = re.search(r"--seed\s+(\d+)", text)
    if m:
        out["seed"] = int(m.group(1))
    m = re.search(r"--lambda-normbrake\s+(\S+)", text)
    if m:
        out["lambda_normbrake"] = float(m.group(1))
    m = re.search(r"--lambda-link\s+(\S+)", text)
    if m:
        out["lambda_link"] = float(m.group(1))
    m = re.search(r"--link-mlp-n-layers\s+(\d+)", text)
    if m:
        out["n_layers"] = int(m.group(1))
    m = re.search(r"--link-mlp-dropout\s+(\S+)", text)
    if m:
        out["link_dropout"] = float(m.group(1))
    m = re.search(r"--embedding-dropout\s+(\S+)", text)
    if m:
        out["embedding_dropout"] = float(m.group(1))

    # Summary
    m = re.search(r"best_val_mrr\s*:\s*([\d.]+)", text)
    if m:
        out["best_val"] = float(m.group(1))
    m = re.search(r"best_test_mrr\s*:\s*([\d.]+)", text)
    if m:
        out["best_test"] = float(m.group(1))
    m = re.search(r"=== Summary \(best epoch (\d+)\)", text)
    if m:
        out["best_epoch"] = int(m.group(1))
    m = re.search(r"stopped_at_epoch\s*:\s*(\d+)", text)
    if m:
        out["stopped_at"] = int(m.group(1))
    # Per-epoch trajectories
    m = re.search(r"per_epoch_val_mrr\s*:\s*([\d.,\s]+)", text)
    if m:
        out["per_ep_val"] = [float(x) for x in m.group(1).split(",")]
    m = re.search(r"per_epoch_col_norm\s*:\s*([\d.,\s]+)", text)
    if m:
        out["per_ep_col_norm"] = [float(x) for x in m.group(1).split(",")]
    m = re.search(r"per_epoch_L_normbrake\s*:\s*([\d.,\s]+)", text)
    if m:
        out["per_ep_nb"] = [float(x) for x in m.group(1).split(",")]

    # Cliff detection: did val_mrr ever drop below (peak - 0.01) in the recorded trajectory?
    if "per_ep_val" in out and out["per_ep_val"]:
        peak = max(out["per_ep_val"])
        last_quarter_min = min(out["per_ep_val"][-max(3, len(out["per_ep_val"])//4):])
        out["cliff_drop"] = peak - last_quarter_min
        # Smoothness scores — the user's goal is smooth loss decrease + smooth
        # val MRR increase. Quantify with:
        #   val_monotonicity: fraction of consecutive epoch-pairs where val_mrr
        #                     does NOT drop by > 0.005 (anti-noise threshold).
        #   peak_position:    epoch / total_epochs where val_mrr peaked.
        #                     <0.3 = peaks early (bad — under-trained-look);
        #                     >0.7 = peaks late (good — still climbing).
        v = out["per_ep_val"]
        n_pairs = max(len(v) - 1, 1)
        n_smooth = sum(1 for i in range(len(v) - 1) if v[i+1] >= v[i] - 0.005)
        out["val_smoothness"] = n_smooth / n_pairs
        out["peak_position"] = (v.index(peak) + 1) / len(v) if v else 0.0
    return out


def format_row(r: Dict) -> str:
    primary = r.get("primary_loss", "?")[:7]
    nb = r.get("lambda_normbrake", "?")
    jl = r.get("lambda_link", 0.0)
    nl = r.get("n_layers", 3)
    ldr = r.get("link_dropout", 0.0)
    edr = r.get("embedding_dropout", 0.0)
    seed = r.get("seed", "?")
    best_ep = r.get("best_epoch", "?")
    stopped = r.get("stopped_at", "?")
    val = f"{r.get('best_val', float('nan')):.4f}"
    test = f"{r.get('best_test', float('nan')):.4f}"
    cliff = f"{r.get('cliff_drop', 0.0):.3f}"
    smooth = f"{r.get('val_smoothness', 0.0):.2f}"
    peak_pos = f"{r.get('peak_position', 0.0):.2f}"
    return (
        f"  {primary:<7}  nb={nb}  λL={jl}  nL={nl}  drL={ldr}  drE={edr}  "
        f"seed={seed}  best ep {best_ep:>2}/stop {stopped:>2}  "
        f"val {val}  test {test}  cliff {cliff}  smooth {smooth}  peak@{peak_pos}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pattern")
    p.add_argument("--sort-by", default="best_test",
                   choices=["best_test", "best_val", "primary_loss", "seed", "cliff_drop"])
    args = p.parse_args()
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"no files match: {args.pattern}")
        return
    rows = []
    for f in files:
        try:
            r = parse_log(f)
            if r and "best_test" in r:
                rows.append(r)
        except Exception as e:
            print(f"  ERROR parsing {f}: {e}")
    if not rows:
        print("no successfully parsed runs")
        return
    rev = args.sort_by in ("best_test", "best_val", "cliff_drop")
    rows.sort(key=lambda r: r.get(args.sort_by, 0.0), reverse=rev)
    print(f"=== {len(rows)} runs, sorted by {args.sort_by} ===")
    for r in rows:
        print(format_row(r))
    # Also print per-epoch val trajectory for the top-3
    print()
    print("=== top-3 val trajectories ===")
    for r in rows[:3]:
        tag = f"{r.get('primary_loss','?')[:7]} nb={r.get('lambda_normbrake','?')} λL={r.get('lambda_link',0.0)} seed={r.get('seed','?')}"
        traj = r.get("per_ep_val", [])
        print(f"  {tag}: " + ", ".join(f"{v:.4f}" for v in traj))


if __name__ == "__main__":
    main()
