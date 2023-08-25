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
