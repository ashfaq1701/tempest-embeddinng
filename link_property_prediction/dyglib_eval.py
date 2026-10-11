"""DyGLib datasets (Wikipedia, Reddit, MOOC, ...) — download, native load, negatives.

The data is the Zenodo record DyGLib's README points to (Poursafaei et al., "Towards Better
Evaluation for Dynamic Link Prediction"), one zip per dataset. Each zip already carries the
processed `ml_<name>.csv` / `ml_<name>.npy` that DyGLib's own `preprocess_data.py --check`
compares against, so they are used as shipped: only those two files are extracted.

Conventions taken from DyGLib (`utils/DataLoader.py`), verified against its source:
  - ids are 1-indexed; on bipartite sets items are offset after users. Row 0 of the edge
    feature array is a zero pad, so edge `idx` reads `ml_<name>.npy[idx]`.
  - the split is by timestamp quantile: train `ts <= q(0.70)`, val `(q(0.70), q(0.85)]`,
    test `> q(0.85)`. This is DyGLib's node-classification split; its link-prediction split
    additionally holds out 10% of nodes, which we do NOT do -- one split serves both runs.
  - `label` is per interaction, about the source user (e.g. banned after this edit).

Negatives follow OUR convention, not DyGLib's 1:1 random AP/AUC: val and test each get fixed
`[N, k_eval]` uniform negatives over the full dataset's candidates (CRAFT's eval
splits), scored by MRR.
"""
import hashlib
import os
import shutil
import subprocess
import zipfile
from typing import Dict, NamedTuple

import numpy as np

from .data import Loaded, SplitData
from .evaluator import DataSuite, Evaluator
from .tgb_seq_eval import TGBSeqEvaluator, build_eval_negatives

ZENODO_RECORD = "https://zenodo.org/api/records/7213796"
VAL_QUANTILE, TEST_QUANTILE = 0.70, 0.85


def _paths(name: str, root: str):
    data_dir = os.path.join(root, name)
    return data_dir, os.path.join(data_dir, f"ml_{name}.csv"), os.path.join(data_dir, f"ml_{name}.npy")


def _fetch(name: str, root: str) -> None:
    """Download `<name>.zip` from the Zenodo record if its processed files are missing,
    verify the record's md5, and extract only `ml_<name>.csv` and `ml_<name>.npy`."""
    import requests

    data_dir, csv_path, npy_path = _paths(name, root)
    if os.path.exists(csv_path) and os.path.exists(npy_path):
        return

    record = requests.get(ZENODO_RECORD, timeout=60)
    record.raise_for_status()
    files = {f["key"]: f for f in record.json()["files"]}
    entry = files.get(f"{name}.zip")
    if entry is None:
        available = sorted(k[:-4] for k in files if k.endswith(".zip"))
        raise ValueError(f"DyGLib dataset {name!r} is not in the Zenodo record; "
                         f"available: {available}")

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{name}.zip.part")
    print(f"  [dyglib] downloading {name}.zip ({entry['size'] / 1e6:.0f} MB) from Zenodo ...")
    md5 = hashlib.md5()
    with requests.get(entry["links"]["self"], stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                md5.update(chunk)

    expected = entry["checksum"].split(":", 1)[1]
    if md5.hexdigest() != expected:
        os.remove(zip_path)
        raise IOError(f"DyGLib {name}.zip failed its md5 check "
                      f"(got {md5.hexdigest()}, record says {expected}); removed, retry.")

    wanted = {f"{name}/ml_{name}.csv": csv_path, f"{name}/ml_{name}.npy": npy_path}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member, dest in wanted.items():
                with zf.open(member) as packed, open(dest, "wb") as out:
                    shutil.copyfileobj(packed, out)
    except NotImplementedError:
        # reddit.zip is Deflate64, which Python's zipfile cannot decompress; Info-ZIP can.
        subprocess.run(["unzip", "-o", "-j", zip_path, *wanted, "-d", data_dir], check=True,
                       stdout=subprocess.DEVNULL)
    os.remove(zip_path)
    print(f"  [dyglib] extracted ml_{name}.csv / .npy into {data_dir}")


def _integral_time_scale(name: str, ts: np.ndarray) -> int:
    """Smallest power of ten that makes every timestamp integral, so the int64 cast keeps every
    gap exact (Wikipedia ships whole seconds -> 1; Reddit ships milliseconds -> 1000)."""
    for k in range(7):
        scaled = ts * 10 ** k
        if np.abs(scaled - np.round(scaled)).max() < 1e-3:
            return 10 ** k
    raise ValueError(f"{name}: timestamps are not integral at any scale up to 1e6")


class DyGLibDataset(NamedTuple):
    """`Loaded.dataset` for this suite: what DyGLib carries beyond the shared splits."""
    labels: Dict[str, np.ndarray]       # "train" / "val" / "test" -> [E_split] float32
    time_scale: int                     # int64 timestamps = shipped float ts * time_scale


def load_dyglib(name: str, root: str = "datasets") -> Loaded:
    """Native DyGLib load -> suite-agnostic `Loaded`, downloading on first use. The
    per-interaction labels ride on `Loaded.dataset` (a `DyGLibDataset`)."""
    import pandas as pd

    _fetch(name, root)
    _, csv_path, npy_path = _paths(name, root)
    df = pd.read_csv(csv_path)
    edge_feat_table = np.load(npy_path)

    ts_float = df.ts.to_numpy(dtype=np.float64)
    time_scale = _integral_time_scale(name, ts_float)
    if not np.all(np.diff(ts_float) >= 0):
        raise ValueError(f"{name}: edges are not in time order")

    src = df.u.to_numpy(dtype=np.int64)
    dst = df.i.to_numpy(dtype=np.int64)
    ts = np.round(ts_float * time_scale).astype(np.int64)
    edge_feat = edge_feat_table[df.idx.to_numpy()].astype(np.float32)
    labels = df.label.to_numpy(dtype=np.float32)

    # DyGLib cuts on quantiles of the float timestamps; same arithmetic, same boundaries.
    val_time, test_time = np.quantile(ts_float, [VAL_QUANTILE, TEST_QUANTILE])
    masks = {
        "train": ts_float <= val_time,
        "val": (ts_float > val_time) & (ts_float <= test_time),
        "test": ts_float > test_time,
    }

    def _split(mask: np.ndarray) -> SplitData:
        return SplitData(sources=src[mask], destinations=dst[mask],
                         timestamps=ts[mask], edge_feat=edge_feat[mask])

    return Loaded(
        train=_split(masks["train"]),
        val=_split(masks["val"]),
        test=_split(masks["test"]),
        dataset=DyGLibDataset(labels={k: labels[m] for k, m in masks.items()},
                              time_scale=time_scale),
        name=name,
        max_node_count=int(max(src.max(), dst.max())) + 1,
    )


class DyGLibSuite(DataSuite):
    """DyGLib adapter for link prediction. Val and test both use our fixed uniform
    negatives at `k_eval` (seeded, test one seed apart from val), scored by MRR."""

    def _load(self) -> Loaded:
        return load_dyglib(self.name, self.root)

    def make_evaluator(self, split_mode: str) -> Evaluator:
        loaded = self.load()
        if split_mode == "val":
            split, seed = loaded.val, self.seed
        elif split_mode == "test":
            split, seed = loaded.test, self.seed + 1
        else:
            raise ValueError(f"split_mode must be 'val' or 'test', got {split_mode!r}")
        neg = build_eval_negatives(split, self.eval_negative_pool, self.k_eval, seed,
                                   tag=f"[dyglib] {split_mode}")
        return TGBSeqEvaluator(neg_dst=neg)
