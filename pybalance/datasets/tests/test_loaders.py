from pybalance.datasets import load_toy_data


def test_loader():
    m = load_toy_data()
    assert len(m) == 11000
