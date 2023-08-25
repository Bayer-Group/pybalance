from pybalance.visualization.distributions import _debin_features


def test_debin():
    original_features = ["x", "y", "z"]
    effective_features = ["x_1", "x_123", "z_10", "z_0", "y_12"]
    input_output_column_mapping = {
        "x": ["x_1", "x_123"],
        "y": ["y_12"],
        "z": ["z_10", "z_0"],
    }
    indices = _debin_features(effective_features, input_output_column_mapping)
    assert indices["x"] == [0, 1]
    assert indices["y"] == [4]
    assert indices["z"] == [2, 3]
