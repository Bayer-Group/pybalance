import pandas as pd
import pytest

from pybalance.sim import rng


def test_truncated_norm_rct():
    data = rng.generate_random_feature_data_rct(size=1000)

    assert data.age.min() >= 18
    assert data.age.max() <= 75
    assert data.height.min() >= 125
    assert data.height.max() <= 195


def test_truncated_norm_rwd():
    data = rng.generate_random_feature_data_rwd(size=1000)

    assert data.age.min() >= 18
    assert data.age.max() <= 75
    assert data.height.min() >= 125
    assert data.height.max() <= 195


def test_demo_aggregate_targets_match_generator():
    for use_case in rng.DEMO_AGGREGATE_USE_CASES:
        generated = rng.generate_aggregate_target(use_case)
        loaded = rng.load_demo_aggregate_target(use_case)

        assert loaded.n == generated.n
        assert loaded.headers == generated.headers
        pd.testing.assert_frame_equal(loaded.to_frame(), generated.to_frame())


def test_generate_aggregate_target_use_cases():
    means = rng.generate_aggregate_target("means")
    assert all("std" not in stats for stats in means.numeric.values())

    means_std = rng.generate_aggregate_target("means_std")
    assert all({"mean", "std"} <= set(stats) for stats in means_std.numeric.values())

    quantiles = rng.generate_aggregate_target("quantiles")
    assert [q for q, _ in quantiles.numeric["age"]["quantile"]] == [0.25, 0.5, 0.75]
    assert [q for q, _ in quantiles.numeric["weight"]["quantile"]] == [0.5]
    assert all("mean" not in stats for stats in quantiles.numeric.values())

    with pytest.raises(ValueError):
        rng.generate_aggregate_target("nope")
