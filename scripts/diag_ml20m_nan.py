#!/usr/bin/env python
"""Locate the FIRST non-finite value in the ML-20M collapse, with projx disabled.

Answers one question: does the gradient go non-finite first (a forward/backward
problem) or does the parameter go non-finite from a finite gradient (an optimiser
step overflow)? Everything else follows from that.

Diagnostic only -- monkeypatches, writes nothing to the repo.
"""
import sys, torch
sys.path.insert(0, "/its/home/ms2420/tempest-embeddinng")

from link_property_prediction import lorentz as L
from link_property_prediction import trainer as T

# 1. disable the radius cap -- reproduce the ORIGINAL failing behaviour
L.LorentzManifold.projx = lambda self, x: x
print("### projx DISABLED (identity) -- reproducing the pre-fix behaviour", flush=True)

# 2. forensic wrapper around the train step
_orig = T.Trainer._train_step
STATE = {"n": 0, "done": False, "prev": None}

def _fwd_probe(self, batch):
    """Re-run the forward on this batch and report which tensor is first non-finite."""
    import numpy as np
    from link_property_prediction.walk_tokens import build_query_walk_tokens
    E = self.model.E.weight
    print(f"    E non-finite: {int((~torch.isfinite(E)).sum())} / {E.numel()}", flush=True)

def step(self, batch):
    STATE["n"] += 1
    n = STATE["n"]
    E = self.model.E.weight
    e_before = bool(torch.isfinite(E).all())
    r_before = float(self.model.geom.dist0(E.detach()).max()) if e_before else float("nan")
    amb_before = float(E.detach().abs().max()) if e_before else float("nan")

    out = _orig(self, batch)

    g = E.grad
    g_finite = bool(torch.isfinite(g).all()) if g is not None else True
    e_after = bool(torch.isfinite(E).all())

    if not STATE["done"] and (not g_finite or not e_after):
        STATE["done"] = True
        print("\n" + "="*78, flush=True)
        print(f"FIRST NON-FINITE at train step {n}", flush=True)
        print("="*78, flush=True)
        print(f"  BEFORE this step: E finite={e_before}  r_max={r_before:.4f}  |x'|max={amb_before:.1f}", flush=True)
        print(f"  loss on this step: {out['link']}", flush=True)
        print(f"  GRADIENT finite after backward : {g_finite}", flush=True)
        print(f"  PARAMETER finite after step    : {e_after}", flush=True)
        print("", flush=True)
        if not g_finite:
            bad = (~torch.isfinite(g)).any(dim=-1).nonzero().flatten()
            print(f"  -> GRADIENT went non-finite FIRST: {len(bad)} rows", flush=True)
            print(f"     => the failure is in the FORWARD/BACKWARD path, not the optimiser step.", flush=True)
            rr = self.model.geom.dist0(E.detach())
            print(f"     radii of the affected rows: {[round(float(rr[i]),3) for i in bad[:10]]}", flush=True)
            print(f"     |grad| on finite rows: max={float(g[torch.isfinite(g)].abs().max()):.3e}", flush=True)
        else:
            bad = (~torch.isfinite(E)).any(dim=-1).nonzero().flatten()
            print(f"  -> PARAMETER went non-finite with a FINITE gradient: {len(bad)} rows", flush=True)
            print(f"     => the failure is in the OPTIMISER STEP (egrad2rgrad / retr / Adam).", flush=True)
            print(f"     |grad|max was {float(g.abs().max()):.3e}", flush=True)
        print("="*78 + "\n", flush=True)
        raise SystemExit(0)
    return out

T.Trainer._train_step = step

sys.argv = ["train", "--data-suite", "tgb-seq", "--dataset", "ML-20M",
            "--d-emb", "64", "--k-train", "5", "--num-walks-per-node", "5",
            "--lr", "1e-3", "--seed", "5", "--early-stop-patience", "5",
            "--is-bipartite", "--use-gpu", "--use-gpu-tempest"]
from scripts.train_link_property_prediction import main
main()
