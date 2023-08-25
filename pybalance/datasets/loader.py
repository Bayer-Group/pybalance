import os
from pybalance.utils import MatchingData


def load_toy_data(n_pool=10000):
    """
    Load a toy data set. Two datasets are currently available, either with
    n_pool = 10,000 or n_pool = 250,000.
    """
    filepath = {
        10000: "pool10000-target1000-normal0-lognormal0-binary4.parquet",
        250000: "pool10000-target1000-normal0-lognormal0-binary4.parquet",
    }[n_pool]

    resource = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", filepath
    )

    return MatchingData(resource)
