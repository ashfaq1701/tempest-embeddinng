"""Ensure all TGB linkproppred datasets are downloaded + preprocessed.

TGB stores data under <tgb package>/datasets/<name_with_underscores>/ and
auto-downloads when `download=True` (the default). The eval negative samples
(val_ns / test_ns .pkl) ship inside the same zip. We download+preprocess only
the datasets whose files are missing; cached ones are skipped (so we don't
re-preprocess the multi-GB coin/comment).
"""
import os
from tgb.utils.info import PROJ_DIR
from tgb.linkproppred.dataset import LinkPropPredDataset

NAMES = ["tgbl-wiki", "tgbl-subreddit", "tgbl-lastfm", "tgbl-review",
         "tgbl-coin", "tgbl-comment", "tgbl-flight"]
base = os.path.join(PROJ_DIR, "datasets")

for n in NAMES:
    d = os.path.join(base, n.replace("-", "_"))
    files = os.listdir(d) if os.path.isdir(d) else []
    has_edge = any("edgelist" in f and f.endswith(".csv") for f in files)
    has_ns = any("val_ns" in f for f in files)
    if has_edge and has_ns:
        print(f"SKIP     {n}: already present", flush=True)
        continue
    print(f"DOWNLOAD {n} ...", flush=True)
    try:
        ds = LinkPropPredDataset(name=n, root="datasets", preprocess=True,
                                 download=True)
        print(f"OK       {n}: {ds.full_data['sources'].shape[0]} edges", flush=True)
    except Exception as e:
        print(f"ERR      {n}: {type(e).__name__}: {e}", flush=True)
print("DOWNLOAD DONE", flush=True)
