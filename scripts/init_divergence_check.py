"""10-minute init-divergence sanity check before Phase S.

The anchor validation produced bit-tight reproduction across seeds
{42, 7, 13} — per-epoch align/uniform/link agree to 0.001. Two readings:
 (A/C) the loss surface has one strong attractor at 2 epochs on this
       dataset; Component 0's signal dominates and pulls everything to
       the same local solution before init differences compound.
 (B)   seed plumbing is broken — np.random.seed / torch.manual_seed are
       called but some init path uses a different (unseeded) RNG.

This script disambiguates by dumping E_target / E_context / link MLP
first-Linear weight RIGHT AFTER `Trainer.__init__` (before any forward
pass) for each seed. If the dumps differ across seeds, init varies
correctly and the trajectory agreement is genuine (path A/C). If they
are identical, seed plumbing is broken (path B) and must be fixed
before Phase S — multi-seed validation in §4.3 becomes meaningless
otherwise.

Usage:
    python3 -m scripts.init_divergence_check
"""

import numpy as np
import torch

from tempest_walks.config import Config
from tempest_walks.data import load_tgb
from tempest_walks.trainer import Trainer


SEEDS = (42, 7, 13)


def _seed(seed: int, use_gpu: bool) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_init(loaded, seed: int, use_gpu: bool) -> dict:
    _seed(seed, use_gpu)
    edge_feat_dim = (
        int(loaded.train.edge_feat.shape[1])
        if loaded.train.edge_feat is not None
        else 0
    )
    train_dst_pool = np.unique(loaded.train.destinations)
    config = Config(
        tgb_name=loaded.name,
        max_node_count=loaded.max_node_count,
        is_directed=loaded.is_directed,
        num_epochs=2,
        seed=seed,
        use_gpu=use_gpu,
    )
    trainer = Trainer(
        config=config,
        train_dst_pool=train_dst_pool,
        node_feat=loaded.node_feat,
        edge_feat_dim=edge_feat_dim,
    )
    # ---- DUMP init weights BEFORE any forward pass ----
    et = trainer.embedding_store.E_target.weight.detach().cpu().numpy()
    ec = trainer.embedding_store.E_context.weight.detach().cpu().numpy()
    # link_predictor.net is nn.Sequential(Linear, ..., ...); first Linear at [0].
    lw = trainer.link_predictor.net[0].weight.detach().cpu().numpy()
    # Time encoder init (if used) — geometric schedule deterministic from
    # time_scale, but `set_time_scale` isn't called yet so omegas are at
    # their default init from k.
    te_omegas = None
    if trainer.time_encoder is not None:
        te_omegas = trainer.time_encoder.omegas.detach().cpu().numpy().copy()

    # ---- Also probe the negative sampler's first few RNG draws ----
    # HistoricalNegativeSampler / UniformNegativeSampler is seeded inside
    # Trainer; pull 5 integers from its internal rng (without advancing
    # the actual training state — we'll re-seed it by re-running the
    # whole __init__ at training time, but here we just want a witness
    # that the seed reached the rng).
    rng_witness = None
    if hasattr(trainer.neg_sampler_train, "rng"):
        rng_witness = trainer.neg_sampler_train.rng.integers(0, 1_000_000, size=5).tolist()

    return {
        "seed": seed,
        "E_target_0_3": et[0:3, 0:3].copy(),
        "E_context_0_3": ec[0:3, 0:3].copy(),
        "link_mlp_W0_0_3": lw[0, 0:3].copy(),
        "time_encoder_omegas": te_omegas,
        "neg_sampler_rng_first_5": rng_witness,
    }


def _fmt(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=6, suppress_small=False)


def _all_equal(arrs):
    base = arrs[0]
    return all(np.array_equal(a, base) for a in arrs[1:])


def main():
    print("Loading TGB dataset: tgbl-wiki")
    loaded = load_tgb("tgbl-wiki", root="datasets")
    print(f"  N={loaded.max_node_count}  is_directed={loaded.is_directed}")

    snapshots = []
    for seed in SEEDS:
        print(f"\n========== seed {seed}: dumping init weights ==========")
        snap = _capture_init(loaded, seed, use_gpu=True)
        snapshots.append(snap)
        print(f"  E_target[0:3, 0:3] =\n{_fmt(snap['E_target_0_3'])}")
        print(f"  E_context[0:3, 0:3] =\n{_fmt(snap['E_context_0_3'])}")
        print(f"  link_mlp.net[0].weight[0, 0:3] = {_fmt(snap['link_mlp_W0_0_3'])}")
        if snap["time_encoder_omegas"] is not None:
            print(f"  time_encoder.omegas (deterministic from k) = {_fmt(snap['time_encoder_omegas'][:5])}")
        if snap["neg_sampler_rng_first_5"] is not None:
            print(f"  neg_sampler.rng → first 5 ints = {snap['neg_sampler_rng_first_5']}")

    # ---- Verdict ----
    print("\n========== VERDICT ==========")
    et_list = [s["E_target_0_3"] for s in snapshots]
    ec_list = [s["E_context_0_3"] for s in snapshots]
    lw_list = [s["link_mlp_W0_0_3"] for s in snapshots]
    rng_list = [s["neg_sampler_rng_first_5"] for s in snapshots]

    et_equal = _all_equal(et_list)
    ec_equal = _all_equal(ec_list)
    lw_equal = _all_equal(lw_list)
    rng_equal = rng_list[0] == rng_list[1] == rng_list[2] if rng_list[0] is not None else None

    print(f"  E_target init identical across seeds: {et_equal}")
    print(f"  E_context init identical across seeds: {ec_equal}")
    print(f"  link_mlp first-Linear weight identical: {lw_equal}")
    if rng_equal is not None:
        print(f"  neg_sampler.rng first-5 draws identical: {rng_equal}")

    plumbing_broken = et_equal or ec_equal or lw_equal or (rng_equal is True)
    if plumbing_broken:
        print("\n  ❌ SEED PLUMBING BROKEN — at least one init path is identical")
        print("     across seeds. Fix before Phase S; multi-seed validation")
        print("     becomes meaningless otherwise.")
    else:
        print("\n  ✅ Init varies across seeds. The bit-tight trajectory")
        print("     reproduction in anchor validation is genuine: the loss")
        print("     surface has one strong attractor at 2 epochs on wiki,")
        print("     Component 0's signal dominates and pulls all inits")
        print("     into the same local solution. Proceed with Phase S.")


if __name__ == "__main__":
    main()
