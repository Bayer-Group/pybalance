import pandas as pd
import numpy as np
from pybalance.utils import (
    BalanceCalculator,
    BatchedBalanceCaclulator,
    BetaBalance,
    GammaBalance,
    MatchingData,
    GammaPreprocessor,
    split_target_pool,
)
from pybalance.sim import generate_toy_dataset
from pybalance.utils.balance_calculators import (
    _get_batch_size,
    map_input_output_weights,
)
import pytest


def test_batcher():
    # Test that we get same result for balance with and without batching

    matching_data = generate_toy_dataset()

    # Without batching
    fc1 = BalanceCalculator(matching_data, "gamma", n_bins=20)

    # With batching
    max_batch_size_gb = 0.01
    fc2 = BatchedBalanceCaclulator(fc1, max_batch_size_gb=max_batch_size_gb)

    # Make sure n_candidate_populations is > batch size and non-integral
    # multiple.
    target, pool = split_target_pool(matching_data)
    target_population_size = len(target)
    n_features = len(matching_data.headers["all"])
    batch_size = _get_batch_size(
        target_population_size, n_features, max_batch_size_gb=max_batch_size_gb
    )
    candidate_populations = np.array(
        [
            pool.sample(n=target_population_size).index.values.tolist()
            for _ in range(int(2.16 * batch_size))
        ]
    )

    # Compare distances
    distance1 = fc1.distance(candidate_populations)
    distance2 = fc2.distance(candidate_populations)
    assert np.all(np.isclose(distance1, distance2, rtol=0.000001))


def test_beta1():
    np.random.seed(1234)

    N_pool, N_target = 100000, 99000
    pool = pd.DataFrame.from_dict(
        {"col1": np.random.randn(N_pool), "population": "pool"}
    )
    target = pd.DataFrame.from_dict(
        {"col1": 1 + np.random.randn(N_target), "population": "target"}
    )
    # beta = mean difference / sum(variances)**0.5
    #      = 1 / (1 + 1)**0.5
    matching_data = MatchingData(pd.concat([pool, target]))
    fc = BalanceCalculator(matching_data, "beta")
    beta = fc.distance(pool)
    assert pytest.approx(beta, rel=1e-2) == 1 / (1 + 1) ** 0.5

    target = pd.DataFrame.from_dict(
        {"col1": 1 + 0.5 * np.random.randn(N_target), "population": "target"}
    )
    # beta = mean difference / sum(variances)**0.5
    #      = 1 / (1 + 0.5**2)**0.5
    matching_data = MatchingData(pd.concat([pool, target]))
    fc = BetaBalance(matching_data)
    beta = fc.distance(pool)
    assert pytest.approx(beta, rel=1e-2) == 1 / (1 + 0.5**2) ** 0.5

    N_pool, N_target = 100000, 99000
    p_target, p_pool = 0.5, 0.1
    mu_target, mu_pool = 0, 1
    sigma_target, sigma_pool = 0.5, 1.5

    pool = pd.DataFrame.from_dict(
        {
            "col1": mu_pool + sigma_pool * np.random.randn(N_pool),
            "col2": np.random.choice(["a", "b"], size=N_pool, p=[p_pool, 1 - p_pool]),
            "population": "pool",
        }
    )

    target = pd.DataFrame.from_dict(
        {
            "col1": mu_target + sigma_target * np.random.randn(N_target),
            "col2": np.random.choice(
                ["a", "b"], size=N_target, p=[p_target, 1 - p_target]
            ),
            "population": "target",
        }
    )
    matching_data = MatchingData(pd.concat([pool, target]))
    fc = BetaBalance(matching_data)
    beta = fc.distance(pool)

    # beta = mean mean difference / sum(variances)**0.5
    beta_expected = (1 / 2) * (
        abs(mu_pool - mu_target) / (sigma_pool**2 + sigma_target**2) ** 0.5
        + abs(p_pool - p_target)
        / (p_pool * (1 - p_pool) + p_target * (1 - p_target)) ** 0.5
    )
    assert pytest.approx(beta, rel=1e-2) == beta_expected


def test_map_feature_weights():
    m = generate_toy_dataset()
    n_bins = 12
    pp = GammaPreprocessor(cumulative=True, n_bins=n_bins)
    pp.fit_transform(m)

    input_weights = {"age": 10, "weight": 1, "height": 2}
    output_weights = map_input_output_weights(pp, weights=input_weights)
    for c in m.headers["numeric"]:
        c_out = pp.get_feature_names_out(c)
        for _c in c_out:
            assert output_weights[_c] == input_weights[c]


def test_beta_is_standardized_mean_difference_numeric():
    # tests that our way of computing SMD matches with a more direct approach;
    # in this case, we simulate numeric variables
    n_features = 10
    target = np.random.randn(250, n_features)
    pool = np.random.uniform(n_features) + np.random.uniform(
        n_features
    ) * np.random.randn(2500, n_features)
    feature_data = np.vstack([target, pool])

    population = ["t"] * len(target) + ["p"] * len(pool)
    feature_cols = [str(i) for i in range(n_features)]
    df = pd.DataFrame.from_records(feature_data, columns=feature_cols)
    df.loc[:, "population"] = population
    m = MatchingData(df)

    fc = BetaBalance(m)
    for frac in [0.05, 0.5, 1]:
        pool = m.get_population("p").sample(frac=frac)
        smd_popmat = fc.distance(pool).item()

        pool = pool[feature_cols]
        smd_direct = (
            (
                (pool.mean(axis=0) - target.mean(axis=0))
                / np.sqrt(pool.std(axis=0) ** 2 + target.std(axis=0) ** 2)
            )
            .abs()
            .mean()
        )
        assert np.isclose(smd_popmat, smd_direct, atol=1e-3)


def test_beta_is_standardized_mean_difference_binary():
    # tests that our way of computing SMD matches with a more direct approach;
    # in this case, we simulate binary variables
    n_features = 10
    target = np.random.uniform(0, 1, (250, n_features))
    p_target = np.random.uniform(0, 1, n_features)
    target = (target < p_target).astype(int)

    pool = np.random.uniform(0, 1, (2500, n_features))
    p_pool = np.random.uniform(0, 1, n_features)
    pool = (pool < p_pool).astype(int)
    feature_data = np.vstack([target, pool])

    population = ["t"] * len(target) + ["p"] * len(pool)
    feature_cols = [str(i) for i in range(n_features)]
    df = pd.DataFrame.from_records(feature_data, columns=feature_cols)
    df.loc[:, "population"] = population
    m = MatchingData(df)

    fc = BetaBalance(m)
    for frac in [0.05, 0.5, 1]:
        pool = m.get_population("p").sample(frac=frac)
        smd_popmat = fc.distance(pool).item()

        pool = pool[feature_cols]
        smd_direct = (
            (
                (pool.mean(axis=0) - target.mean(axis=0))
                / np.sqrt(pool.std(axis=0) ** 2 + target.std(axis=0) ** 2)
            )
            .abs()
            .mean()
        )
        assert np.isclose(smd_popmat, smd_direct, atol=1e-3)


def test_beta_is_standardized_mean_difference():
    # tests that our way of computing SMD matches with a more direct approach;
    # this also tests alignment between get_feature_names_out and
    # per_feature_loss.
    m = generate_toy_dataset()
    fc = BetaBalance(m)
    per_feature_loss = fc.per_feature_loss(m.get_population("pool"))

    m_transformed = fc.preprocessor.transform(m)
    for j, feature in enumerate(fc.preprocessor.get_feature_names_out()):
        pm = m_transformed.get_population("pool")[feature].mean()
        ps = m_transformed.get_population("pool")[feature].std()
        tm = m_transformed.get_population("target")[feature].mean()
        ts = m_transformed.get_population("target")[feature].std()
        smd_direct = (pm - tm) / np.sqrt(ps**2 + ts**2)
        assert np.isclose(per_feature_loss[0][j], smd_direct, atol=1e-3)


def test_gamma_is_area_between_cdfs():
    m = generate_toy_dataset()
    m = MatchingData(m.data, headers={"numeric": m.headers["numeric"], "categoric": []})

    fc = GammaBalance(m, n_bins=10, standardize_difference=False)
    distance_popmat = fc.distance(m.get_population("pool"))

    # Comupte area between CDFs a bit more directly
    distance_direct = 0
    for j, feature in enumerate(m.headers["all"]):
        pool = m.get_population("pool")[feature]
        target = m.get_population("target")[feature]

        bins = fc.preprocessor.preprocessors[1].discretizer.bin_edges_[j]
        n_p, bins = np.histogram(pool, bins=bins, density=True)
        dx = bins[1] - bins[0]
        pool_cdf = np.cumsum(n_p) * dx

        n_t, bins = np.histogram(target, bins=bins, density=True)
        dx = bins[1:] - bins[0:-1]
        target_cdf = np.cumsum(n_t) * dx

        # *  1 / len(m.headers['all']) to mean it a mean
        distance_direct += (np.abs(pool_cdf - target_cdf)[:-1]).mean() / len(
            m.headers["all"]
        )

    assert np.isclose(distance_popmat, distance_direct, atol=1e-3)


def test_target_subsets():

    matching_data = generate_toy_dataset()
    target, pool = split_target_pool(matching_data)
    gamma = BalanceCalculator(matching_data, "gamma", n_bins=20)

    d1 = gamma.distance(pool)
    d2 = gamma.distance(pool, target)
    assert d1 == d2

    pool_subsets = np.array([
        np.random.choice(pool.reset_index().index.values, size=200, replace=False),
        np.random.choice(pool.reset_index().index.values, size=200, replace=False)
    ])
    target_subsets = np.array([
        np.random.choice(target.reset_index().index.values, size=100, replace=False),
        np.random.choice(target.reset_index().index.values, size=100, replace=False)
    ])
    distances = gamma.distance(pool_subsets, target_subsets)
    assert len(distances) == 2

    target_subsets = np.array([
        np.random.choice(target.reset_index().index.values, size=100, replace=False),
        np.random.choice(target.reset_index().index.values, size=100, replace=False),
        np.random.choice(target.reset_index().index.values, size=100, replace=False)
    ])
    with pytest.raises(ValueError):
        distances = gamma.distance(pool_subsets, target_subsets)