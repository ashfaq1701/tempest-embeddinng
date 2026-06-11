"""Vitter-R uniformity test for HistoricalNegativeSampler.observe.

Under true Vitter R, after n observations each item should be in the
reservoir with probability M/n. Pre-Lesson-32 (always-replace when
full) gives an exponential recency bias instead. The test below uses
two complementary checks:

  (1) Per-item presence count across trials should match expected
      mean M/n_obs.
  (2) The empirical distribution of "which observation index is in
      the reservoir" should be roughly uniform across the n_obs
      observation indices — discriminator: under always-replace,
      recent observations dominate; under Vitter R, distribution is flat.

Run:
  /home/ms2420/CLionProjects/tempest-walk-embedding/.venv/bin/python \
      -m tests.test_vitter_r_uniformity
"""

import numpy as np

from tempest_walks.negatives import HistoricalNegativeSampler


def test_vitter_r_uniformity():
    M = 8
    n_observations = 1000
    n_trials = 200

    presence_counts = np.zeros(n_observations, dtype=np.int64)

    for trial in range(n_trials):
        sampler = HistoricalNegativeSampler(
            num_nodes=1,
            num_neg_per_pos=0,
            hist_ratio=0.0,
            reservoir_size=M,
            dst_pool=np.array([0], dtype=np.int32),
            seed=trial,
        )
        for i in range(n_observations):
            sampler.observe(
                np.array([0], dtype=np.int64),
                np.array([i], dtype=np.int32),
            )
        present = sampler.reservoir[0][sampler.reservoir[0] >= 0]
        presence_counts[present] += 1

    expected_per_item = n_trials * M / n_observations   # 1.6 with these settings

    # Check 1 — overall mean.
    mean_count = presence_counts.mean()
    assert abs(mean_count - expected_per_item) < 0.05 * expected_per_item, (
        f"mean presence {mean_count:.3f} vs expected {expected_per_item:.3f} "
        f"(>5% deviation)."
    )

    # Check 2 — empirical distribution over observation indices should be
    # roughly flat. Under always-replace, recent indices have much higher
    # mean than early ones. Discriminator: ratio of late-half mean to
    # early-half mean should be ≈ 1.0 under Vitter R; under always-replace
    # it's > 5 even at n=1000, M=8.
    half = n_observations // 2
    early_mean = presence_counts[:half].mean()
    late_mean = presence_counts[half:].mean()
    ratio = late_mean / max(early_mean, 1e-6)
    print(f"  M={M}, n_obs={n_observations}, n_trials={n_trials}")
    print(f"  expected per-item presence ≈ {expected_per_item:.3f}")
    print(f"  mean = {mean_count:.3f}, std = {presence_counts.std():.3f}")
    print(f"  early-half mean = {early_mean:.3f}, late-half mean = {late_mean:.3f}")
    print(f"  late/early ratio = {ratio:.3f} (expected ≈ 1.0; >2.0 ⇒ recency bias)")

    assert ratio < 1.5, (
        f"late/early presence ratio = {ratio:.2f}; expected ≈ 1.0 for true "
        f"Vitter R. The sampler is recency-biased."
    )

    print("PASS — reservoir contents are uniformly distributed over the "
          "source's observation history.")


if __name__ == "__main__":
    test_vitter_r_uniformity()
