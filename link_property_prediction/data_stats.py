"""Data-derived constants computed once at load."""

import numpy as np


def full_span(timestamps: np.ndarray) -> float:
    """Span of the whole edge list, t_max - t_min over EVERY split.

    This is the TimeEncoder's log-age normaliser. The train span is the wrong choice: ages are
    cutoff - t_edge and the cutoffs run to the end of test, so val/test ages exceed it and would
    push the encoder's u past 1.
    """
    ts = np.asarray(timestamps).astype(np.int64)
    return float(ts.max() - ts.min())
