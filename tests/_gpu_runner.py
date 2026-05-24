"""Ad-hoc GPU runner for test_chunked_infonce — same test cases,
but the model lives on CUDA. Used once to verify the chunked
InfoNCE math holds bit-for-bit on the actual training device.

This file is intentionally minimal and tied to the test module's
internals. Not part of the regular test suite.
"""

import sys
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

assert torch.cuda.is_available(), "GPU runner requires CUDA"

import tests.test_chunked_infonce as t


def _to_cuda(E, p_t, p_c):
    return E.cuda(), p_t.cuda(), p_c.cuda()


# Patch make_model and call_loss_and_grads so the CPU-built modules
# are moved to CUDA before use.
_orig_make_model = t.make_model
def _gpu_make_model(*args, **kwargs):
    return _to_cuda(*_orig_make_model(*args, **kwargs))
t.make_model = _gpu_make_model


# node_feat needs to be on CUDA too — patch test_chunk_vs_full_node_feat
_orig_node_feat = t.test_chunk_vs_full_node_feat
def _gpu_node_feat():
    # Re-implement with node_feat on cuda. Copy of the test fn body
    # minus device manipulation; cuda conversion handled via patched
    # make_model + explicit node_feat.cuda().
    print("Test 3: chunk vs full, node_feat path [GPU]")
    N, K, L = 60, 2, 14
    d_emb, d_proj, num_nodes, d_nf = 48, 48, 400, 8
    tau, beta, T_train = 0.5, 1.0, 1_000_000.0
    seed = 2

    walks = t.build_synthetic_walks(N, K, L, num_nodes, seed)
    NK = N * K
    g = torch.Generator().manual_seed(seed + 99)
    node_feat = torch.randn(num_nodes, d_nf, generator=g).cuda()

    E, p_t, p_c = t.make_model(num_nodes, d_emb, d_proj, seed, d_nf=d_nf)
    ref = t.call_loss_and_grads(
        E, p_t, p_c, walks, tau=tau, beta=beta,
        T_train=T_train, chunk_size=0, node_feat=node_feat,
    )
    print(f"  reference loss = {ref[0]:.10f}")

    failed = False
    for cs in [1, 13, NK]:
        E, p_t, p_c = t.make_model(num_nodes, d_emb, d_proj, seed, d_nf=d_nf)
        val = t.call_loss_and_grads(
            E, p_t, p_c, walks, tau=tau, beta=beta,
            T_train=T_train, chunk_size=cs, node_feat=node_feat,
        )
        if not t._assert_match(f"chunk={cs}", ref, val):
            failed = True
    return not failed
t.test_chunk_vs_full_node_feat = _gpu_node_feat


def main():
    print(f"Running on device: cuda  ({torch.cuda.get_device_name(0)})")
    results = [
        t.test_chunk_vs_full_K1(),
        t.test_chunk_vs_full_Kgt1(),
        t.test_chunk_vs_full_node_feat(),
        t.test_chunked_matches_naive_reference(),
    ]
    if not all(results):
        print("\nFAIL: at least one GPU test failed.")
        sys.exit(1)
    print(f"\nPASS: all chunked InfoNCE tests match reference on GPU within {t.TOL:.0e}.")


if __name__ == "__main__":
    main()
