from pybalance.utils import split_target_pool
from pybalance.sim import load_paper_dataset, generate_toy_dataset


def test_loader():
    m = load_paper_dataset()
    target, pool = split_target_pool(m)
    assert len(pool) == 250000
    assert len(target) == 25000


def test_generator1():
    m = generate_toy_dataset(n_pool=12345, n_target=234)
    target, pool = split_target_pool(m)
    assert len(pool) == 12345
    assert len(target) == 234
