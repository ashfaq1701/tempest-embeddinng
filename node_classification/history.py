"""Exact per-interaction history of the source user, from the event stream itself.

For a query (u, t), over u's edges strictly before t (the same exclusive cutoff the walks use):

    log1p(n)                         number of prior edges
    log1p(t - t_last)                time since the last one (log1p(1e7) if none)
    log1p(t - t_first)               account age at t (0 if none)
    log1p(mean gap), log1p(last gap) between consecutive prior edges (0 if fewer than 2)

Only the source column of the stream is used ("u's edges" = edges where u is the source), so
on a bipartite set these are the user's own edits.
"""
import numpy as np

from link_property_prediction.data import SplitData

N_FEATURES = 5
NO_HISTORY_LOG_GAP = float(np.log1p(1e7))


def history_features(stream: SplitData, queries: SplitData) -> np.ndarray:
    """[N_queries, 5] float32, causal: only stream edges with t_edge < t_query count."""
    src_all = stream.sources.astype(np.int64)
    ts_all = stream.timestamps.astype(np.int64)
    span = int(ts_all.max()) + 2                                   # key = src * span + t, collision-free
    order = np.argsort(src_all * span + ts_all, kind="stable")
    keys = src_all[order] * span + ts_all[order]
    times = ts_all[order]

    q_src = queries.sources.astype(np.int64)
    q_ts = queries.timestamps.astype(np.int64)
    lo = np.searchsorted(keys, q_src * span, side="left")          # u's first edge
    hi = np.searchsorted(keys, q_src * span + q_ts, side="left")   # past u's last edge before t
    n = hi - lo

    has_one, has_two = n >= 1, n >= 2
    last = times[np.where(has_one, hi - 1, 0)]
    first = times[np.where(has_one, lo, 0)]
    prev = times[np.where(has_two, hi - 2, 0)]

    out = np.zeros((len(q_src), N_FEATURES), dtype=np.float32)
    out[:, 0] = np.log1p(n)
    out[:, 1] = np.where(has_one, np.log1p(q_ts - last), NO_HISTORY_LOG_GAP)
    out[:, 2] = np.where(has_one, np.log1p(q_ts - first), 0.0)
    mean_gap = (last - first) / np.maximum(n - 1, 1)
    out[:, 3] = np.where(has_two, np.log1p(mean_gap), 0.0)
    out[:, 4] = np.where(has_two, np.log1p(last - prev), 0.0)
    return out
