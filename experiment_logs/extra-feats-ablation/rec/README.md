# Recency arm — not yet run

Branch `feature/duv-with-cand-rec`, commit `deaedbf1`.
`score = mix([-d_H, log1p(cand_recency)])`, `Linear(2, 1, bias=False)`, init `[1, 0]`.

Recency is an **age** (`cutoff - t_last(v)`, `cutoff + 1` when cold), so larger means
staler and `w[1]` is expected to go **negative** — opposite to popularity, which went
positive on YouTube.

Measured at init, `d(CE)/dw = [6.3e-05, 1.5e+00]`: the recency weight carries ~24,000x the
gradient of the distance weight, so it moves immediately. Inert in *value* is not inert in
*gradient*. Watch `w[1]` at epoch 1.

Logs land here when the arm runs.
