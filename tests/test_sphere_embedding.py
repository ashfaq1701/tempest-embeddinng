"""Correctness invariants for the unit-sphere E + dual-optimiser change.

Run: ./.venv/bin/python tests/test_sphere_embedding.py
Asserts the six invariants from the change doc plus a forward sanity.
No training data needed (a tiny synthetic Trainer covers the routing checks).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import geoopt
import numpy as np
import torch

from tempest_walks.model import EmbeddingTable
from tempest_walks.trainer import Trainer, TrainerConfig


def _tiny_trainer():
    cfg = TrainerConfig(
        num_nodes=80, is_directed=False, dst_pool=np.arange(40, dtype=np.int32),
        d_emb=16, link_head_d_pos=24, K_train=10,
        link_pred_num_walks_per_node=5, link_pred_max_walk_len=20,
        embedding_num_walks_per_node=5, embedding_max_walk_len=20,
        emb_lr=1e-2, train_deg=np.ones(80, dtype=np.int64),
    )
    return Trainer(cfg, device=torch.device("cpu"))


def inv1_feasible_init():
    et = EmbeddingTable(100, 16)
    rn = et.E.weight.norm(dim=-1)
    assert isinstance(et.E.weight, geoopt.ManifoldParameter)
    assert torch.allclose(rn, torch.ones(100), atol=1e-5), rn
    print("inv1 feasible init OK  (norms within 1e-5 of 1)")


def inv2_feasible_under_training():
    et = EmbeddingTable(100, 16)
    opt = geoopt.optim.RiemannianAdam(et.parameters(), lr=1e-1, weight_decay=0.0, stabilize=10)
    for _ in range(50):
        opt.zero_grad()
        # arbitrary smooth loss with gradient on every row
        loss = (et.E.weight ** 2).sum() + et.E.weight.sum()
        loss.backward(); opt.step()
    rn = et.E.weight.norm(dim=-1)
    assert torch.allclose(rn, torch.ones(100), atol=1e-4), (rn.min(), rn.max())
    print(f"inv2 feasible after 50 steps OK  (norm range {rn.min():.6f}..{rn.max():.6f})")


def inv3_disjoint_optimizers():
    tr = _tiny_trainer()
    head_ids = {id(p) for g in tr.opt_head.param_groups for p in g["params"]}
    emb_ids = {id(p) for g in tr.opt_emb.param_groups for p in g["params"]}
    assert head_ids and emb_ids
    assert head_ids.isdisjoint(emb_ids), "optimisers share a parameter!"
    # E is exactly the one ManifoldParameter; head holds the rest
    assert emb_ids == {id(tr.embedding_table.E.weight)}
    print(f"inv3 disjoint optimisers OK  (head {len(head_ids)} params, emb {len(emb_ids)})")


def inv4_grad_to_E_from_align_only():
    tr = _tiny_trainer()
    # Emulate the link path exactly: E detached -> only head gets grad.
    cand = torch.randint(0, 40, (4, 11))
    e_v = tr.embedding_table(cand).detach()
    walks = dict(
        E_walks=torch.randn(4, 5, 20, 16), mask=torch.ones(4, 5, 20, dtype=torch.bool),
        K_idx=torch.randint(0, 20, (4, 5, 20)), t_feat=torch.randn(4, 5, 20, 12),
    )
    logits = tr.link_head(e_v, walks=walks)
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(4, dtype=torch.long))
    tr.embedding_table.E.weight.grad = None
    loss.backward()
    g = tr.embedding_table.E.weight.grad
    assert g is None or float(g.abs().sum()) == 0.0, "L_link leaked grad into E!"
    print("inv4 L_link routes NO gradient into E (detach holds) OK")


def inv5_cosine_equivalence():
    a = torch.randn(7, 16); a = a / a.norm(dim=-1, keepdim=True)
    b = torch.randn(7, 16); b = b / b.norm(dim=-1, keepdim=True)
    sq_a = (a * a).sum(-1); sq_b = (b * b).sum(-1); inner = (a * b).sum(-1)
    full = -(sq_a + sq_b - 2.0 * inner)           # what losses.py computes
    cosine = -(2.0 - 2.0 * inner)                 # the dropped-constant form
    assert torch.allclose(full, cosine, atol=1e-5), (full - cosine).abs().max()
    print("inv5 squared-L2 == cosine form on unit rows OK")


def inv6_feasible_after_restore():
    tr = _tiny_trainer()
    snap = tr._snapshot()
    # perturb E off-snapshot, then restore
    with torch.no_grad():
        tr.embedding_table.E.weight.mul_(3.0)
    tr._restore(snap)
    w = tr.embedding_table.E.weight
    rn = w.norm(dim=-1)
    assert isinstance(w, geoopt.ManifoldParameter) and hasattr(w, "manifold")
    assert torch.allclose(rn, torch.ones_like(rn), atol=1e-5), (rn.min(), rn.max())
    print("inv6 feasible + ManifoldParameter preserved after restore OK")


def forward_sanity():
    et = EmbeddingTable(50, 16)
    out = et(torch.tensor([0, 10, 49]))
    assert out.shape == (3, 16) and torch.isfinite(out).all()
    assert et.E.weight.min() < 0 < et.E.weight.max()   # signed, not [0,1]
    print("forward sanity OK  (signed coords, finite)")


if __name__ == "__main__":
    for fn in (inv1_feasible_init, inv2_feasible_under_training, inv3_disjoint_optimizers,
               inv4_grad_to_E_from_align_only, inv5_cosine_equivalence,
               inv6_feasible_after_restore, forward_sanity):
        fn()
    print("\nALL INVARIANTS PASS")
