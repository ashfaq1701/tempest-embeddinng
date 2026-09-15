# Combined arm — not yet run

Branch `feature/duv-with-cand-pop-and-rec`, commit `afb785cb`.
`score = mix([-d_H, log1p(cand_popularity), log1p(cand_recency)])`,
`Linear(3, 1, bias=False)`, init `[1, 0, 0]`.

This is the arm that says whether the two channels **compose or overlap**. The precedent to
beat is not the baseline but the better of the two solo arms — `rad` and `dev` in the pooler
ablation were each individually real and still landed *below* `rad` alone when combined,
because the overlap was paid for in delayed escape.

Expect **opposite signs**: `w[1]` positive (popularity, a count) and `w[2]` negative
(recency, an age). A same-sign result is worth a second look.

Measured at init, `d(CE)/dw = [-2.5e-04, 7.6e-01, 1.5e+00]` — recency carries twice
popularity's gradient, so it should move first.

Logs land here when the arm runs.
