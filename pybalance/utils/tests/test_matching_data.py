import pandas as pd
import numpy as np
import copy
import pytest

from pybalance.sim import generate_toy_dataset
from pybalance.utils import (
    MatchingData,
    MatchingHeaders,
    split_target_pool,
    infer_matching_headers,
)


def _make_fake_matching_data(
    N_pool: int, N_target: int, pool_name: str = "pool", target_name="target"
) -> pd.DataFrame:
    # data shape
    N_numeric_features = 20
    N_categoric_features = 8

    # numeric data
    pool_n = np.random.rand(N_pool, N_numeric_features)
    target_n = np.random.rand(N_target, N_numeric_features)

    # categoric data
    pool_c = np.random.choice([1, 2, 3], size=(N_pool, N_categoric_features - 2))
    target_c = np.random.choice([1, 2, 3], size=(N_target, N_categoric_features - 2))

    # these have a lot of categories, but should be identified as categoric
    # since they cannot be cast to float
    pool_c2 = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=(N_pool, 2))
    target_c2 = np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), size=(N_target, 2))

    # column names
    numeric_cols = [f"num_col_{d}" for d in range(N_numeric_features)]
    categoric_cols = [f"cat_col_{d}" for d in range(N_categoric_features)]

    # all data
    pool = np.hstack((pool_n, pool_c, pool_c2))
    target = np.hstack((target_n, target_c, target_c2))
    pool = pd.DataFrame.from_records(pool, columns=numeric_cols + categoric_cols)
    target = pd.DataFrame.from_records(target, columns=numeric_cols + categoric_cols)
    pool.loc[:, "population"] = pool_name
    target.loc[:, "population"] = target_name
    data = pd.concat([pool, target])
    data.loc[:, "patient_id"] = list(range(len(data)))

    return data


def test_split_target_pool():
    m = generate_toy_dataset()

    # test we can correctly infer the populations based on size
    target, pool = split_target_pool(m)
    assert len(target) == 1000
    assert len(pool) == 10000
    assert target[m.population_col].unique() == ["target"]
    assert pool[m.population_col].unique() == ["pool"]

    # test we can override the guessing behavior by explicitly giving the
    # poplation names
    target, pool = split_target_pool(m, target_name="pool", pool_name="target")
    assert len(target) == 10000
    assert len(pool) == 1000

    # test we can find the populations when extra ones are in there
    target, pool = split_target_pool(m, target_name="pool", pool_name="target")
    assert len(target) == 10000
    assert len(pool) == 1000


def test_split_target_pool2():
    N_pool, N_target = 1000, 100
    pool_name, target_name = "p", "t"
    data = _make_fake_matching_data(N_pool, N_target, pool_name, target_name)

    matching_data = MatchingData(data, population_col="population")
    target, pool = split_target_pool(matching_data)

    assert len(target) == N_target
    assert len(pool) == N_pool
    assert pool_name == pool["population"].unique()[0]
    assert target_name == target["population"].unique()[0]
    assert len(matching_data.data) == len(data)


def test_split_target_pool3():
    N_pool, N_target = 500, 250
    pool_name, target_name = "p", "t"
    data = _make_fake_matching_data(N_pool, N_target, pool_name, target_name)

    matching_data = MatchingData(data, population_col="population")

    # test we can find the right populations when more than two are present
    match = matching_data.get_population(pool_name).sample(N_target)
    matching_data.append(match, name="match")
    target, pool = split_target_pool(
        matching_data, target_name=target_name, pool_name=pool_name
    )
    assert target_name == target["population"].unique()[0]
    assert pool_name == pool["population"].unique()[0]


def test_split_target_pool4():
    # Test can split when two populations present and only one name passed
    m = generate_toy_dataset()
    target, pool = split_target_pool(m, target_name="pool")
    assert target[m.population_col].unique()[0] == "pool"


def test_get_population():
    m = generate_toy_dataset()

    # Check we can lookup populations by name
    pool = m.get_population("pool")
    assert len(pool) == 10000
    assert pool[m.population_col].unique() == ["pool"]

    # check that a KeyError is raised when population does not exists
    with pytest.raises(KeyError):
        m.get_population("abcdefg")


def test_describe():
    m = generate_toy_dataset()
    quantiles = [0, 0.25, 0.5, 0.75, 1]
    summary = m.describe(aggregations=[], quantiles=quantiles)
    target, pool = split_target_pool(m)
    for c in m.headers["numeric"]:
        assert np.allclose(
            summary["pool"][c].values, pool[c].quantile(quantiles), atol=1e-2
        )
        assert np.allclose(
            summary["target"][c].values, target[c].quantile(quantiles), atol=1e-2
        )


def test_infer_matching_headers():
    size = 1000
    np.random.seed(1234)
    data = pd.DataFrame.from_dict(
        {
            "patient_id": list(range(size)),
            "cat1": np.random.choice(list("abcdef"), size=size, replace=True),
            "cat2": np.random.choice(list(range(6)), size=size, replace=True),
            "cat3": np.random.choice(list(range(3)), size=size, replace=True),
            "num1": np.random.rand(size),
            "num2": np.random.randn(size),
        }
    )

    # cat1, cat2, cat3 should be inferred to be categoric and the rest are
    # numeric; patient_id is a special column that is ignored
    headers = infer_matching_headers(data, max_categories=10)
    assert set(headers.categoric) == set(["cat1", "cat2", "cat3"])
    assert set(headers.numeric) == set(["num1", "num2"])

    # cat2 should now be considered numeric, since it has too many categories
    # cat1 has too many categories, but can't be cast to numeric so it's still
    # categoric
    headers = infer_matching_headers(data, max_categories=3)
    print(headers)
    assert set(headers.categoric) == set(["cat1", "cat3"])
    assert set(headers.numeric) == set(["cat2", "num1", "num2"])


def test_matching_data_append():
    N_pool, N_target = 1000, 100
    pool_name, target_name = "p", "t"
    data = _make_fake_matching_data(N_pool, N_target, pool_name, target_name)
    matching_data = MatchingData(data, population_col="population")

    # test append
    match = matching_data.get_population(pool_name).sample(N_target)
    matching_data.append(match, name="match")
    assert set(matching_data.populations) == set(["match", pool_name, target_name])


def test_matching_data_append2():
    N_pool, N_target = 1000, 100
    pool_name, target_name = "p", "t"
    data = _make_fake_matching_data(N_pool, N_target, pool_name, target_name)

    # test append fails with missing columns
    new_data = copy.deepcopy(data)
    new_data.loc[:, "adfadsf"] = 1
    matching_data = MatchingData(new_data)
    with pytest.raises(ValueError):
        matching_data.append(data)


def test_matching_data_copy():
    # check there are no side effects when making changes on a copy

    m1 = generate_toy_dataset()
    m2 = m1.copy()

    pool1 = m1.get_population("pool")
    orig_value = pool1["age"][0]
    pool2 = m2.get_population("pool")
    pool2.loc[:, "age"] = -100

    new_value = pool1["age"][0]
    assert orig_value == new_value
    assert pool2["age"][0] == -100


def test_specified_headers():
    N = 100
    df = pd.DataFrame.from_dict(
        {
            "cat3": np.random.choice(list(range(3)), size=N, replace=True),
            "population": ["x"] * N,
        }
    )

    # Force cat3 to be considered numeric, even though it would be inferred as
    # categoric
    headers = MatchingHeaders(numeric=["cat3"], categoric=[])
    m = MatchingData(df, headers=headers)
    assert m.headers["categoric"] == []
    assert m.headers["numeric"] == ["cat3"]
