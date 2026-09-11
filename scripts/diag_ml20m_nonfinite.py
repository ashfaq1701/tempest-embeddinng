#!/usr/bin/env python
"""Pinpoint the first non-finite value in the ML-20M seed-5 run.

EVIDENCE ONLY. Fixes nothing, proposes nothing. Wraps every LorentzManifold
method and every stage of the head, records a GPU-side finiteness flag per op,
and syncs ONCE per batch. On the first non-finite output it dumps the op, its
inputs with actual numbers, the offending rows, and the nodes in those rows,
then stops.

Why deferred flags: ML-20M is 14,000,091 train edges at batch_size 1000 =
~14,001 batches/epoch. A .item() per op would be ~400k syncs/epoch. One sync
per batch is ~14k, which is noise against a 235s epoch.

Traps this harness is built to avoid (each has produced a wrong diagnosis):
  * torch.where evaluates both arms. We never use autograd anomaly mode; we
    check REAL outputs, so an untaken-arm NaN cannot be reported.
  * _safe_sqrt propagates NaN deliberately. We record ops in call order and
    report the FIRST flagged one, then print finiteness of its inputs, so a
    downstream reporter cannot be mistaken for the origin.
  * forward vs optimiser path are different bugs. Every record carries PHASE.
  * r_max from the epoch probe is once-per-epoch. We flag isfinite(E.weight)
    per batch instead.

Usage: python scripts/diag_ml20m_nonfinite.py [--seed 5] [--start-epoch N]
"""
import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from link_property_prediction import model as model_mod          # noqa: E402
from link_property_prediction import trainer as trainer_mod      # noqa: E402
from link_property_prediction.lorentz import LorentzManifold     # noqa: E402

# ----------------------------------------------------------------------------
# recorder
# ----------------------------------------------------------------------------
PHASE = "init"
BATCH_RECS: List[Dict[str, Any]] = []
CTX: Dict[str, Any] = {}
# Recording is armed ONLY inside a training step. Eval runs ~365 batches with
# K_eval=100 negatives, i.e. candidate tensors of ~646MB each; recording there
# accumulated references and OOM'd the 47GiB card during epoch 1's eval. Eval is
# also not under investigation -- the crash is in the train forward.
STATE = {"epoch": 1, "batch": 0, "fired": False, "active": False, "max_step": 0.0,
         "max_step_where": None, "outdir": None}
MANIFOLD_METHODS = ["_x0", "_gap", "inner", "egrad2rgrad", "expmap", "retr",
                    "transp", "dist", "dist0", "midpoint", "projx", "proju", "logmap"]


def _flag(t: torch.Tensor) -> Optional[torch.Tensor]:
    """GPU-side 'has non-finite' flag. No sync."""
    if not torch.is_tensor(t) or not t.is_floating_point():
        return None
    return (~torch.isfinite(t)).any()


def _rec(name: str, out: Any, inputs: Dict[str, Any]) -> None:
    if STATE["fired"] or not STATE["active"]:
        return
    outs = out if isinstance(out, (tuple, list)) else (out,)
    flags = [f for f in (_flag(o) for o in outs) if f is not None]
    if not flags:
        return
    BATCH_RECS.append({"name": name, "phase": PHASE, "flag": flags[0] if len(flags) == 1
                       else torch.stack(flags).any(),
                       "out": out, "inputs": inputs})


def _num(t: Any) -> Dict[str, Any]:
    """Summary with ACTUAL NUMBERS, not adjectives."""
    if not torch.is_tensor(t):
        return {"repr": repr(t)[:200]}
    d: Dict[str, Any] = {"shape": tuple(t.shape), "dtype": str(t.dtype)}
    if not t.is_floating_point():
        if t.numel():
            d["min"], d["max"] = int(t.min()), int(t.max())
        return d
    fin = torch.isfinite(t)
    d["numel"] = int(t.numel())
    d["n_nan"] = int(torch.isnan(t).sum())
    d["n_posinf"] = int((t == float("inf")).sum())
    d["n_neginf"] = int((t == float("-inf")).sum())
    if bool(fin.any()):
        f = t[fin]
        d["finite_min"] = float(f.min())
        d["finite_max"] = float(f.max())
        d["finite_absmax"] = float(f.abs().max())
    return d


def _bad_rows(t: torch.Tensor) -> List[int]:
    if not torch.is_tensor(t) or not t.is_floating_point():
        return []
    bad = ~torch.isfinite(t)
    while bad.dim() > 1:
        bad = bad.any(dim=-1)
    return torch.nonzero(bad).flatten()[:20].tolist()


# ----------------------------------------------------------------------------
# the dump
# ----------------------------------------------------------------------------
def dump_and_stop(rec: Dict[str, Any], all_recs: List[Dict[str, Any]],
                  first_idx: int, extra: Dict[str, Any]) -> None:
    STATE["fired"] = True
    out = STATE["outdir"]
    os.makedirs(out, exist_ok=True)
    geom: LorentzManifold = CTX.get("geom")
    E = CTX.get("E")

    L: List[str] = []
    def p(s: str = "") -> None:
        L.append(s)
        print(s, flush=True)

    o = rec["out"] if torch.is_tensor(rec["out"]) else rec["out"][0]
    kind = []
    if int(torch.isnan(o).sum()):
        kind.append("NaN")
    if int(torch.isinf(o).sum()):
        kind.append("inf")

    p("=" * 78)
    p("FIRST NON-FINITE VALUE")
    p("=" * 78)
    p(f"epoch            : {STATE['epoch']}")
    p(f"batch (in epoch) : {STATE['batch']}")
    p(f"op               : {rec['name']}")
    p(f"phase            : {rec['phase']}   (forward = loss path, optim = inside opt.step())")
    p(f"kind             : {'+'.join(kind) or 'none?'}")
    p(f"op index in batch: {first_idx} of {len(all_recs)} recorded ops")
    p("")
    p("--- OUTPUT ---")
    p(json.dumps(_num(o), indent=2))
    rows = _bad_rows(o)
    p(f"non-finite rows (first 20): {rows}")
    p("")
    p("--- INPUTS (is the origin here, or upstream?) ---")
    upstream = []
    for k, v in rec["inputs"].items():
        s = _num(v)
        p(f"[{k}] {json.dumps(s)}")
        if torch.is_tensor(v) and v.is_floating_point() and (s.get("n_nan", 0) or
                                                             s.get("n_posinf", 0) or
                                                             s.get("n_neginf", 0)):
            upstream.append(k)
    p("")
    if upstream:
        p(f"*** INPUTS {upstream} ARE ALREADY NON-FINITE -- this op REPORTS, it did not ORIGINATE.")
        p("*** The origin is upstream. Recorded op order for this batch follows.")
    else:
        p("*** ALL INPUTS FINITE -- this op ORIGINATED the non-finite value.")
    p("")
    p("--- recorded op order this batch (first 60) ---")
    for i, r in enumerate(all_recs[:60]):
        mark = " <<< FIRST NON-FINITE" if i == first_idx else ""
        p(f"  {i:3d} [{r['phase']:7s}] {r['name']}{mark}")
    p("")
    p("--- batch context ---")
    for k, v in extra.items():
        p(f"{k}: {v}")
    p("")
    mi = CTX.get("midpoint_internals")
    if mi is not None:
        p("--- midpoint internals at this batch (ACTUAL NUMBERS) ---")
        p(json.dumps(mi, indent=2))
        p("")
        d = mi.get("degeneracy", {})
        if d.get("rows_degenerate", 0):
            p(f"*** {d['rows_degenerate']} token rows are PAST the float32 wall "
              f"(|x'| > {d['wall_norm']}, r > {d['wall_r']}): k + |x'|^2 == |x'|^2, so x0 == |x'| "
              f"exactly and the point is numerically ON THE LIGHT CONE, not the hyperboloid.")
        else:
            p(f"*** NO token row is past the float32 wall (r > {d.get('wall_r')}). "
              f"The r=9.011 degeneracy is NOT implicated at this batch.")
        p("")
    if E is not None:
        p("--- float32 wall over the whole table ---")
        p(json.dumps(_degeneracy(E.weight.detach()), indent=2))
        p("")

    # nodes implicated
    if E is not None and geom is not None:
        with torch.no_grad():
            r_all = geom.dist0(E.weight.detach())
            finite_r = r_all[torch.isfinite(r_all)]
            p("--- E.weight table state ---")
            p(f"isfinite(E.weight).all() : {bool(torch.isfinite(E.weight).all())}")
            p(f"rows with non-finite     : {int((~torch.isfinite(E.weight)).any(dim=-1).sum())}")
            if finite_r.numel():
                p(f"dist0 finite max         : {float(finite_r.max()):.4f}")
                p(f"dist0 finite mean        : {float(finite_r.mean()):.4f}")
                topv, topi = torch.topk(torch.nan_to_num(r_all, nan=-1.0), k=min(15, r_all.numel()))
                p("")
                p("--- top-15 nodes by radius (node_id, dist0, ||x'||, exp_avg_sq_mean) ---")
                opt = CTX.get("opt")
                st = opt.state.get(E.weight, {}) if opt is not None else {}
                eas = st.get("exp_avg_sq")
                ea = st.get("exp_avg")
                for v, i in zip(topv.tolist(), topi.tolist()):
                    xn = float(E.weight[i].detach().norm())
                    q = float(eas[i].mean()) if eas is not None and eas.dim() > 1 else (
                        float(eas[i]) if eas is not None else float("nan"))
                    m = float(ea[i].norm()) if ea is not None and ea.dim() > 1 else float("nan")
                    p(f"  node {i:7d}  dist0={v:10.4f}  ||x'||={xn:14.4e}  "
                      f"exp_avg_sq_mean={q:12.4e}  ||exp_avg||={m:12.4e}")

    # token-level dump for the implicated rows
    for side in ("src", "cand"):
        tk = CTX.get(f"tokens_{side}")
        if tk is None:
            continue
        p("")
        p(f"--- {side} token bag (rows implicated: {rows[:5]}) ---")
        try:
            for r in rows[:5]:
                rr = r if r < tk["nodes"].shape[0] else None
                if rr is None:
                    continue
                nodes = tk["nodes"][rr].tolist()
                mask = tk["mask"][rr].tolist()
                w = tk["w"][rr].tolist() if tk.get("w") is not None else None
                p(f"  row {rr}: valid={sum(mask)}/{len(mask)}  seed={tk['seed'][rr] if tk.get('seed') is not None else '?'}")
                p(f"    nodes  : {nodes[:24]}")
                if w is not None:
                    p(f"    weights: {[round(x, 5) for x in w[:24]]}")
                    p(f"    weight  sum={sum(w):.6f}  nan={any(x != x for x in w)}")
                if geom is not None and E is not None:
                    rads = geom.dist0(E.weight.detach()[torch.tensor(nodes[:24], device=E.weight.device)])
                    p(f"    radii  : {[round(float(z), 3) for z in rads.tolist()]}")
        except Exception as e:  # dump must never mask the finding
            p(f"  (token dump failed: {type(e).__name__}: {e})")

    p("")
    p(f"max geodesic step length seen this run: {STATE['max_step']:.6e} at {STATE['max_step_where']}")
    p("=" * 78)

    with open(os.path.join(out, "FIRST_NONFINITE.txt"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    try:
        torch.save({"E": E.weight.detach().cpu() if E is not None else None,
                    "epoch": STATE["epoch"], "batch": STATE["batch"],
                    "op": rec["name"], "phase": rec["phase"]},
                   os.path.join(out, "crash_state.pt"))
    except Exception as e:
        p(f"(crash_state save failed: {e})")
    print(f"\n[diag] wrote {out}/FIRST_NONFINITE.txt", flush=True)
    raise SystemExit(17)


# ----------------------------------------------------------------------------
# install wrappers
# ----------------------------------------------------------------------------
# float32 degeneracy wall: k + |x'|^2 == |x'|^2 once |x'|^2 > 2^24, i.e. |x'| > 4096,
# i.e. r > asinh(4096) = 9.0109. Past it x0 == |x'| exactly: the point sits on the light
# cone numerically, not the hyperboloid. The prior collapse run named this wall; these
# probes test it directly rather than assuming it.
F32_WALL_NORM = 4096.0
F32_WALL_R = math.asinh(F32_WALL_NORM)


def _degeneracy(x: torch.Tensor) -> Dict[str, Any]:
    """Does k + |x'|^2 == |x'|^2 in this dtype, per row?"""
    with torch.no_grad():
        n2 = (x * x).sum(-1)
        deg = (1.0 + n2) == n2
        return {"rows_degenerate": int(deg.sum()), "n2_max": float(n2.max()),
                "norm_max": float(n2.max().sqrt()),
                "wall_norm": F32_WALL_NORM, "wall_r": round(F32_WALL_R, 4)}


def install_midpoint_probe() -> None:
    """midpoint is the prime suspect: neg = s0^2 - |s|^2 cancels catastrophically at
    radius. Record the actual intermediates so the dump carries numbers, not adjectives."""
    orig = LorentzManifold.midpoint

    def midpoint(self, x, w):
        if not STATE["fired"] and STATE["active"]:
            with torch.no_grad():
                wu = w.unsqueeze(-1)
                x0 = self._x0(x)
                s0 = (wu * x0).sum(dim=-2)
                s = (wu * x).sum(dim=-2)
                neg_raw = (s0 * s0 - (s * s).sum(-1, keepdim=True)) * self._inv_k
                wsum = w.sum(-1, keepdim=True)
                CTX["midpoint_internals"] = {
                    "s0_absmax": float(s0.abs().max()),
                    "s0sq_absmax": float((s0 * s0).abs().max()),
                    "neg_raw_min": float(neg_raw.min()),
                    "neg_raw_max": float(neg_raw.max()),
                    "neg_raw_n_nonfinite": int((~torch.isfinite(neg_raw)).sum()),
                    "neg_raw_n_le0": int((neg_raw <= 0).sum()),
                    "floor_would_fire_rows": int((neg_raw < wsum * wsum).sum()),
                    "wsum_min": float(wsum.min()), "wsum_max": float(wsum.max()),
                    "x0_absmax": float(x0.abs().max()),
                    "degeneracy": _degeneracy(x.reshape(-1, x.shape[-1])),
                }
            _rec("midpoint.s0", s0, {"x0": x0, "w": w})
            _rec("midpoint.s", s, {"x": x, "w": w})
            _rec("midpoint.neg_raw", neg_raw, {"s0": s0, "s": s})
        return orig(self, x, w)

    LorentzManifold.midpoint = midpoint


def install_manifold_wrappers() -> None:
    for mname in MANIFOLD_METHODS:
        if mname == "midpoint":
            continue          # handled by install_midpoint_probe, with internals
        if not hasattr(LorentzManifold, mname):
            continue
        orig = getattr(LorentzManifold, mname)

        def make(mname: str, orig):
            def wrapped(self, *a, **kw):
                out = orig(self, *a, **kw)
                if not STATE["fired"] and STATE["active"]:
                    names = ("x", "u", "v", "w")
                    ins = {names[i] if i < len(names) else f"arg{i}": v
                           for i, v in enumerate(a)}
                    _rec(f"LorentzManifold.{mname}", out, ins)
                    # expmap during optim IS the parameter update: record step length
                    if mname == "expmap" and PHASE == "optim" and len(a) >= 2:
                        try:
                            q = self.inner(a[0], a[1], keepdim=True).clamp_min(0.0)
                            m = float(q.max().sqrt())
                            if m > STATE["max_step"]:
                                STATE["max_step"] = m
                                STATE["max_step_where"] = (STATE["epoch"], STATE["batch"])
                        except Exception:
                            pass
                return out
            return wrapped
        setattr(LorentzManifold, mname, make(mname, orig))


def install_head_wrappers() -> None:
    BagWeights = model_mod.BagWeights
    LinkPredHead = model_mod.LinkPredHead
    bw_orig = BagWeights.forward
    pool_orig = LinkPredHead.pool
    fwd_orig = LinkPredHead.forward

    def bw_forward(self, geom, tokens, x, valid):
        age = torch.log1p(tokens.ages.clamp_min(0).to(x.dtype)).unsqueeze(-1)
        _rec("BagWeights.age", age, {"ages": tokens.ages})
        pos = tokens.positions.unsqueeze(-1).to(x.dtype)
        _rec("BagWeights.pos", pos, {"positions": tokens.positions})
        rad = geom.dist0(x.detach()).unsqueeze(-1)
        _rec("BagWeights.rad", rad, {"x": x})
        feat = torch.cat([age, pos, rad], dim=-1).to(x.dtype)
        _rec("BagWeights.feat", feat, {"age": age, "pos": pos, "rad": rad})
        logits = self.net(feat).squeeze(-1)
        _rec("BagWeights.logits", logits, {"feat": feat})
        filled = logits.masked_fill(~valid, float("-inf"))
        w = torch.softmax(filled, dim=-1)
        _rec("BagWeights.softmax", w, {"logits": logits, "valid_count": valid.sum(-1)})
        CTX["last_w"] = w
        # a row with no valid token gives all -inf -> NaN weights
        CTX["rows_no_valid"] = int((valid.sum(-1) == 0).sum())
        return w

    def pool(self, tokens, emb):
        nodes = tokens.nodes.clamp_min(0).clone()
        valid = tokens.mask.clone()
        cold = ~valid.any(dim=-1)
        n_cold = int(cold.sum())
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True
        x = torch.nn.functional.embedding(nodes, emb)
        _rec("LinkPredHead.pool.gather", x, {"emb": emb})
        w = self.bag_weights(self.geom, tokens, x, valid)
        side = CTX.get("side", "?")
        CTX[f"tokens_{side}"] = {"nodes": nodes.detach(), "mask": valid.detach(),
                                 "w": w.detach(), "seed": tokens.seeds.detach(),
                                 "n_cold": n_cold}
        out = self.geom.midpoint(x, w)
        _rec("LinkPredHead.pool.midpoint", out, {"x": x, "w": w, "wsum": w.sum(-1)})
        return out

    def forward(self, src_tokens, cand_tokens):
        emb = self.E.weight
        _rec("LinkPredHead.E.weight", emb, {})
        CTX["side"] = "src"
        p_u = pool(self, src_tokens, emb)
        CTX["side"] = "cand"
        p_v = pool(self, cand_tokens, emb)
        b, d = p_u.shape
        c = p_v.shape[0] // b
        p_v = p_v.view(b, c, d)
        geo = self.geom.dist(p_u.unsqueeze(1), p_v)
        _rec("LinkPredHead.geo", geo, {"p_u": p_u, "p_v": p_v})
        logits = self.geo_temp * (-geo)
        _rec("LinkPredHead.logits", logits, {"geo": geo, "geo_temp": self.geo_temp.detach()})
        return logits

    BagWeights.forward = bw_forward
    LinkPredHead.pool = pool
    LinkPredHead.forward = forward


# ----------------------------------------------------------------------------
# per-batch driver
# ----------------------------------------------------------------------------
def install_trainer_wrapper(start_epoch: int) -> None:
    Trainer = trainer_mod.Trainer
    step_orig = Trainer._train_step
    train_orig = Trainer.train

    def _sync_and_check(self, stage_extra: Dict[str, Any]) -> None:
        if STATE["fired"] or not BATCH_RECS:
            return
        flags = torch.stack([r["flag"] for r in BATCH_RECS])
        if not bool(flags.any()):          # the single sync per batch
            return
        idx = int(torch.nonzero(flags).flatten()[0])
        dump_and_stop(BATCH_RECS[idx], BATCH_RECS, idx, stage_extra)

    def _train_step(self, batch):
        global PHASE
        if STATE["fired"]:
            return step_orig(self, batch)
        BATCH_RECS.clear()
        STATE["active"] = True          # disarmed again before we return, so eval records nothing
        CTX["geom"] = self.model.geom
        CTX["E"] = self.model.E
        CTX["opt"] = self.opt
        STATE["batch"] += 1

        # entering-state flag: was the table already poisoned before this batch?
        pre = (~torch.isfinite(self.model.E.weight)).any()

        PHASE = "forward"
        device = self.device
        B = len(batch.src)
        neg_tgt = self.neg_sampler_train.sample(batch)
        src_t = torch.from_numpy(batch.src.astype(np.int64)).to(device)
        cand_np = np.concatenate(
            [batch.tgt.astype(np.int64)[:, None],
             np.ascontiguousarray(neg_tgt, dtype=np.int64)], axis=1)
        cand_t = torch.from_numpy(cand_np).to(device)
        t_query_t = torch.from_numpy(batch.ts.astype(np.int64)).to(device)

        logits = self._score(src_t, cand_t, t_query_t)
        target = torch.zeros(B, dtype=torch.long, device=device)
        link_loss = torch.nn.functional.cross_entropy(logits, target)
        _rec("cross_entropy", link_loss, {"logits": logits})

        extra = {"E_nonfinite_ON_ENTRY": None, "rows_no_valid": CTX.get("rows_no_valid"),
                 "geo_temp": float(self.model.geo_temp.detach()),
                 "src_cold": CTX.get("tokens_src", {}).get("n_cold"),
                 "cand_cold": CTX.get("tokens_cand", {}).get("n_cold")}
        _sync_and_check(self, {**extra, "E_nonfinite_ON_ENTRY": bool(pre)})

        PHASE = "backward"
        self.opt.zero_grad(set_to_none=True)
        link_loss.backward()
        for nm, prm in (("E.weight", self.model.E.weight),
                        ("geo_temp", self.model.geo_temp)):
            if prm.grad is not None:
                _rec(f"grad.{nm}", prm.grad, {"param": prm.detach()})
        for nm, prm in self.model.bag_weights.named_parameters():
            if prm.grad is not None:
                _rec(f"grad.pooler.{nm}", prm.grad, {"param": prm.detach()})
        _sync_and_check(self, {**extra, "E_nonfinite_ON_ENTRY": bool(pre),
                               "note": "backward phase"})

        PHASE = "optim"
        self.opt.step()
        st = self.opt.state.get(self.model.E.weight, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                _rec(f"adam.E.{key}", st[key], {})
        stg = self.opt.state.get(self.model.geo_temp, {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in stg:
                _rec(f"adam.geo_temp.{key}", stg[key], {})
        _rec("E.weight.after_step", self.model.E.weight, {})
        _sync_and_check(self, {**extra, "E_nonfinite_ON_ENTRY": bool(pre),
                               "note": "optimiser phase",
                               "max_step_this_run": STATE["max_step"]})
        PHASE = "idle"
        out = {"link": float(link_loss.detach()),
               "lr": float(self.opt.param_groups[0]["lr"])}
        # Drop every tensor reference: holding graph nodes across batches keeps
        # activations alive and was what exhausted the card during eval.
        STATE["active"] = False
        BATCH_RECS.clear()
        for k in ("tokens_src", "tokens_cand", "last_w"):
            CTX.pop(k, None)
        return out

    Trainer._train_step = _train_step

    # epoch bookkeeping + checkpoint, via the geometry probe (once per epoch, AFTER
    # that epoch's batches). STATE["epoch"] is the epoch currently training, 1-based.
    probe_orig = Trainer._geometry_probe

    def _geometry_probe(self):
        g = probe_orig(self)
        out = STATE["outdir"]
        os.makedirs(out, exist_ok=True)
        with torch.no_grad():
            W = self.model.E.weight.detach()
            r = self.model.geom.dist0(W)
            fin = torch.isfinite(r)
            deg = _degeneracy(W)
            print(f"[diag] end-of-epoch {STATE['epoch']}: "
                  f"E finite={bool(torch.isfinite(W).all())} "
                  f"r_max_finite={float(r[fin].max()) if bool(fin.any()) else float('nan'):.4f} "
                  f"n_nonfinite_rows={int((~torch.isfinite(W)).any(dim=-1).sum())} "
                  f"nodes_past_f32_wall(r>{F32_WALL_R:.3f})={deg['rows_degenerate']} "
                  f"|x'|max={deg['norm_max']:.4e} "
                  f"geo_temp={float(self.model.geo_temp):.4f} "
                  f"max_geo_step={STATE['max_step']:.4e}", flush=True)
        ep_done = STATE["epoch"]
        STATE["epoch"] += 1
        STATE["batch"] = 0
        if ep_done >= 7:   # keep the last clean epochs only
            ck = os.path.join(out, f"ck_ep{ep_done}.pt")
            torch.save({
                "epoch": ep_done,
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng": np.random.get_state(),
            }, ck)
            print(f"[diag] checkpoint -> {ck}", flush=True)
            old = os.path.join(out, f"ck_ep{STATE['epoch'] - 2}.pt")
            if os.path.exists(old):
                os.remove(old)
        return g

    Trainer._geometry_probe = _geometry_probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--dataset", default="ML-20M")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--num-epochs", type=int, default=16)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny truncated run to verify the harness plumbing only")
    a = ap.parse_args()

    STATE["outdir"] = a.outdir or (
        f"/its/home/ms2420/tempest-embeddinng/logs/diag_nonfinite/{a.dataset}_seed{a.seed}")
    os.makedirs(STATE["outdir"], exist_ok=True)

    install_manifold_wrappers()
    install_head_wrappers()
    install_trainer_wrapper(0)

    argv = [
        "train_link_property_prediction.py",
        "--data-suite", "tgb-seq", "--dataset", a.dataset,
        "--d-emb", "64", "--k-train", "5", "--num-walks-per-node", "5",
        "--lr", "1e-3", "--seed", str(a.seed),
        "--early-stop-patience", "99",           # never stop early; we want the crash
        "--num-epochs", str(a.num_epochs),
        "--use-gpu", "--use-gpu-tempest",
    ]
    if a.dataset in ("ML-20M", "Taobao", "Yelp", "GoogleLocal"):
        argv.append("--is-bipartite")
    if a.smoke:
        argv += ["--max-train-edges", "40000", "--max-eval-edges", "5000"]
        argv[argv.index("--num-epochs") + 1] = "2"
    sys.argv = argv
    print(f"[diag] argv: {' '.join(argv)}", flush=True)
    print(f"[diag] outdir: {STATE['outdir']}", flush=True)

    import importlib
    entry = importlib.import_module("scripts.train_link_property_prediction") \
        if os.path.exists("scripts/__init__.py") else None
    if entry is None:
        spec = importlib.util.spec_from_file_location(
            "entry", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "train_link_property_prediction.py"))
        entry = importlib.util.module_from_spec(spec)
        sys.modules["entry"] = entry
        spec.loader.exec_module(entry)
    t0 = time.time()
    try:
        entry.main()
    except SystemExit as e:
        if e.code == 17:
            print(f"[diag] STOPPED on first non-finite after {time.time() - t0:.0f}s", flush=True)
            raise
        raise
    print("[diag] run completed with NO non-finite value detected", flush=True)


if __name__ == "__main__":
    main()
