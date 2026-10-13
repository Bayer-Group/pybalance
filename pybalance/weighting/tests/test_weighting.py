import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from pybalance.sim import generate_toy_dataset
from pybalance.utils import (
    AggregateTarget,
    MatchingData,
    MatchingHeaders,
    split_target_pool,
)
from pybalance.weighting import (
    EntropyBalanceWeighter,
    MAICWeighter,
    PropensityScoreWeighter,
    propensity_score_weights,
    effective_sample_size,
    weighted_balance_table,
)


def _biased_pool_subsample(
    pool_df: pd.DataFrame, n: int, seed: int = 0
) -> pd.DataFrame:
    """
    Draw a covariate-shifted (but not adversarially so) subsample of the pool
    to use as a target population. Because the target is literally a sample
    of pool patients, the target's mean is guaranteed to lie within the
    convex hull of the pool's covariates, so exact moment-balancing weights
    are guaranteed to exist -- unlike, say, a target drawn from a genuinely
    different population, which may not be feasible to balance exactly (e.g.
    if it lies outside the pool's covariate support on some combination of
    features).
    """
    rng = np.random.default_rng(seed)
    bias_score = pool_df["age"].values + 5 * pool_df["gender"].values
    prob = np.exp(0.03 * (bias_score - bias_score.mean()))
    prob = prob / prob.sum()
    idx = rng.choice(len(pool_df), size=n, replace=False, p=prob)
    return pool_df.iloc[idx].reset_index(drop=True)


def _aggregate_target_from_frame(
    df: pd.DataFrame, headers: MatchingHeaders
) -> AggregateTarget:
    numeric = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in headers.numeric
    }
    categoric = {
        col: {k: float(v) for k, v in df[col].value_counts(normalize=True).items()}
        for col in headers.categoric
    }
    return AggregateTarget(
        n=len(df), numeric=numeric, categoric=categoric, headers=headers
    )


def test_effective_sample_size():
    assert effective_sample_size(np.ones(10)) == pytest.approx(10)
    # one patient carrying all the weight -> ESS collapses to ~1
    weights = np.zeros(10)
    weights[0] = 1.0
    assert effective_sample_size(weights) == pytest.approx(1)


def test_maic_patient_level_target_matches_mean():
    matching_data = generate_toy_dataset(n_pool=3000, n_target=300, seed=1)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(
        drop=True
    )
    target_df = _biased_pool_subsample(pool_df, n=300, seed=1)

    md = MatchingData(pool=pool_df, target=target_df, headers=matching_data.headers)
    weighter = MAICWeighter(md, verbose=False)
    matched = weighter.match()

    assert weighter.diagnostics["converged"]

    target, pool = split_target_pool(matched)
    w = pool["sample_weight"].values

    table = weighted_balance_table(weighter)
    residual = (table["weighted_pool"] - table["target"]).abs()
    assert (residual < 1e-3).all()
    assert w.sum() == pytest.approx(len(target), rel=1e-6)


def test_maic_aggregate_target_matches_mean():
    matching_data = generate_toy_dataset(n_pool=3000, n_target=300, seed=2)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(
        drop=True
    )
    target_df = _biased_pool_subsample(pool_df, n=300, seed=2)
    aggregate_target = _aggregate_target_from_frame(target_df, matching_data.headers)

    md = MatchingData(
        pool=pool_df, target=aggregate_target, headers=matching_data.headers
    )

    weighter = MAICWeighter(md, verbose=False)
    weighter.match()
    assert weighter.diagnostics["converged"]

    table = weighted_balance_table(weighter)
    residual = (table["weighted_pool"] - table["target"]).abs()
    assert (residual < 1e-3).all()
    assert weighter.weights.sum() == pytest.approx(aggregate_target.n, rel=1e-6)


def test_entropy_balance_weighter_matches_variance():
    matching_data = generate_toy_dataset(n_pool=3000, n_target=300, seed=3)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(
        drop=True
    )
    target_df = _biased_pool_subsample(pool_df, n=300, seed=3)

    md = MatchingData(pool=pool_df, target=target_df, headers=matching_data.headers)
    weighter = EntropyBalanceWeighter(md, match_variance=True, verbose=False)
    weighter.match()
    assert weighter.diagnostics["converged"]

    table = weighted_balance_table(weighter)
    residual = (table["weighted_pool"] - table["target"]).abs()
    assert (residual < 1e-2).all()
    # every numeric feature should have both a mean and a variance row
    assert set(table.loc[table["moment"] == "variance", "feature"]) == set(
        matching_data.headers.numeric
    )


def test_normalize_options_scale_weights_without_changing_balance():
    matching_data = generate_toy_dataset(n_pool=1000, n_target=100, seed=4)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(
        drop=True
    )
    target_df = _biased_pool_subsample(pool_df, n=100, seed=4)
    md = MatchingData(pool=pool_df, target=target_df, headers=matching_data.headers)

    n_target = len(target_df)
    n_pool = len(pool_df)

    w_target = MAICWeighter(md, normalize="target", verbose=False).match()
    w_pool = MAICWeighter(md, normalize="pool", verbose=False).match()

    pool_target = w_target.get_population(w_target.pool_name)["sample_weight"].values
    pool_pool = w_pool.get_population(w_pool.pool_name)["sample_weight"].values

    assert pool_target.sum() == pytest.approx(n_target, rel=1e-6)
    assert pool_pool.sum() == pytest.approx(n_pool, rel=1e-6)

    # Balance (relative weights) is unaffected by the overall scale.
    ratio = pool_target / pool_pool
    assert np.allclose(ratio, ratio[0], rtol=1e-6)


def test_weight_col_collision_raises():
    matching_data = generate_toy_dataset(n_pool=200, n_target=50, seed=5)
    with pytest.raises(ValueError):
        MAICWeighter(matching_data, weight_col="age", verbose=False)


def test_hard_covariate_shift_reports_diagnostics_without_crashing():
    """
    When the pool and target come from genuinely different distributions
    (rather than the target being literally a sample of the pool), jointly
    balancing many covariates exactly may not be achievable; the weighter
    should still return usable, finite weights and honestly report
    non-convergence / a degraded effective sample size rather than silently
    producing incorrect balance.
    """
    matching_data = generate_toy_dataset(n_pool=2000, n_target=200, seed=6)

    weighter = MAICWeighter(matching_data, max_iter=50, verbose=False)
    matched = weighter.match()

    pool = matched.get_population(matched.pool_name)
    w = pool["sample_weight"].values
    assert np.all(np.isfinite(w))
    assert np.all(w >= 0)
    assert weighter.effective_sample_size() <= len(pool)


def _toy_pool(n=1000, seed=7):
    m = generate_toy_dataset(n_pool=n, n_target=0, seed=seed)
    return m.get_population("pool").reset_index(drop=True)


def test_weighter_only_constrains_disclosed_statistics():
    pool = _toy_pool()
    headers = MatchingHeaders(numeric=["age", "height"], categoric=["country"])
    target = AggregateTarget(
        n=100, numeric={"age": {"mean": 60.0}}, categoric={"country": {1: 0.3}}
    )
    weighter = MAICWeighter(
        MatchingData(pool=pool, target=target, headers=headers), verbose=False
    )
    weighter.match()

    table = weighted_balance_table(weighter).set_index("feature")
    assert weighter.diagnostics["converged"]
    assert table.loc["age", "weighted_pool"] == pytest.approx(60.0, abs=1e-3)
    assert table.loc["country_1", "weighted_pool"] == pytest.approx(0.3, abs=1e-3)
    # undisclosed features have no target and are free to move, not pinned to the pool
    for feature in ("height", "country_2", "country_3"):
        assert np.isnan(table.loc[feature, "target"])
        assert (
            abs(
                table.loc[feature, "weighted_pool"]
                - table.loc[feature, "unweighted_pool"]
            )
            > 0.005
        )


def test_weighter_matches_quantiles_and_softly_limits_max():
    pool = _toy_pool()
    headers = MatchingHeaders(numeric=["age", "weight"], categoric=[])
    numeric = {"age": {"median": 53.0}, "weight": {"median": 82.0, "max": 100.0}}
    weighter = MAICWeighter(
        MatchingData(
            pool=pool, target=AggregateTarget(n=100, numeric=numeric), headers=headers
        ),
        verbose=False,
    )
    weighter.match()

    table = weighted_balance_table(weighter).set_index("feature")
    assert weighter.diagnostics["converged"]
    assert table.loc["age_q0.5_1.0", "weighted_pool"] == pytest.approx(0.5, abs=1e-3)
    assert table.loc["weight_q0.5_1.0", "weighted_pool"] == pytest.approx(0.5, abs=1e-3)
    # the max is soft: the weight above it shrinks a lot but need not reach 0
    above = (pool["weight"] > 100).values
    assert above.mean() > 0.1
    share = weighter.weights[above].sum() / weighter.weights.sum()
    assert 0 < share < 0.03
    assert weighter.diagnostics["max_soft_violation"] > 0


def test_weighter_rejects_variance_without_a_mean():
    pool = _toy_pool(300)
    headers = MatchingHeaders(numeric=["age"], categoric=[])
    data = MatchingData(
        pool=pool,
        target=AggregateTarget(n=50, numeric={"age": {"std": 5.0}}),
        headers=headers,
    )
    with pytest.raises(ValueError, match="no mean"):
        EntropyBalanceWeighter(data, match_variance=True)


def _shifted_md(n_pool=1500, n_target=200, seed=0):
    matching_data = generate_toy_dataset(n_pool=n_pool, n_target=n_target, seed=seed)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(
        drop=True
    )
    target_df = _biased_pool_subsample(pool_df, n=n_target, seed=seed)
    return MatchingData(pool=pool_df, target=target_df, headers=matching_data.headers)


def _weighter(md, **kwargs):
    kwargs = {"max_iter": 6, "seed": 0, "verbose": False, **kwargs}
    return PropensityScoreWeighter(md, **kwargs)


@pytest.mark.parametrize("method", ["att", "atc", "ate", "overlap", "matching"])
def test_propensity_score_weights_are_positive_and_shaped(method):
    ps_pool = np.array([0.1, 0.3, 0.5, 0.9])
    ps_target = np.array([0.2, 0.6, 0.8])
    w_pool, w_target = propensity_score_weights(ps_pool, ps_target, method)
    assert w_pool.shape == ps_pool.shape and w_target.shape == ps_target.shape
    assert np.all(w_pool > 0) and np.all(w_target > 0)


def test_propensity_score_weights_overlap_and_att_definitions():
    ps_pool, ps_target = np.array([0.2, 0.5]), np.array([0.4, 0.9])
    w_pool, w_target = propensity_score_weights(ps_pool, ps_target, "overlap")
    assert np.allclose(w_pool, ps_pool) and np.allclose(w_target, 1 - ps_target)

    w_pool, w_target = propensity_score_weights(ps_pool, ps_target, "att")
    assert np.allclose(w_pool, ps_pool / (1 - ps_pool))
    assert np.allclose(w_target, 1)

    with pytest.raises(ValueError):
        propensity_score_weights(ps_pool, ps_target, "nope")


def test_propensity_score_weighter_improves_balance_over_unweighted_pool():
    md = _shifted_md(seed=7)
    weighter = _weighter(md)
    matched = weighter.match()

    table = weighted_balance_table(weighter)
    unweighted_residual = (table["unweighted_pool"] - table["target"]).abs()
    weighted_residual = (table["weighted_pool"] - table["target"]).abs()
    assert weighted_residual.sum() < unweighted_residual.sum()
    assert weighter.effective_sample_size() <= len(md.get_population("pool"))

    # The reported score is the balance objective evaluated on the weights.
    assert weighter.best_score == pytest.approx(
        float(
            weighter.balance_calculator.distance(
                pool_weights=weighter.weights, target_weights=weighter.target_weights
            )
        )
    )
    assert weighter.weight_col in matched.get_population(matched.pool_name)


def test_propensity_score_weighter_search_improves_on_unweighted_pool_score():
    md = _shifted_md(seed=8)
    weighter = _weighter(md, method="overlap")
    weighter.match()

    unweighted = float(weighter.balance_calculator.distance())
    assert weighter.best_score < unweighted


def test_overlap_weights_are_bounded_and_target_is_weighted():
    md = _shifted_md(seed=12)
    weighter = _weighter(md, method="overlap")
    matched = weighter.match()

    assert 0 < weighter.weights.min() and weighter.weights.max() < 1
    assert 0 < weighter.target_weights.min() and weighter.target_weights.max() < 1
    assert np.allclose(
        matched.get_population(matched.target_name)[weighter.weight_col],
        weighter.target_weights,
    )


def test_propensity_score_weighter_rejects_bad_arguments():
    md = _shifted_md(n_pool=300, n_target=50, seed=13)
    with pytest.raises(ValueError):
        _weighter(md, method="nope")
    with pytest.raises(ValueError):
        _weighter(md, trim_quantiles=(0.9, 0.1))
    with pytest.raises(ValueError):
        _weighter(md, min_ess_fraction=2.0)

    matching_data = generate_toy_dataset(n_pool=500, n_target=100, seed=8)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(
        drop=True
    )
    target_df = matching_data.get_population(matching_data.target_name).reset_index(
        drop=True
    )
    aggregate_target = _aggregate_target_from_frame(target_df, matching_data.headers)
    agg_md = MatchingData(
        pool=pool_df, target=aggregate_target, headers=matching_data.headers
    )
    with pytest.raises(ValueError):
        _weighter(agg_md)


def test_propensity_score_weighter_trim_quantiles_caps_extreme_weights():
    md = _shifted_md(seed=9)
    untrimmed = _weighter(md, trim_quantiles=None)
    untrimmed.match()
    trimmed = _weighter(md, trim_quantiles=(0.05, 0.95))
    trimmed.match()

    assert trimmed.weights.max() <= untrimmed.weights.max()
    assert trimmed.diagnostics["n_trimmed"] > 0


def test_min_ess_fraction_rejects_spiky_candidates():
    md = _shifted_md(seed=14)
    # Only perfectly uniform weights have ESS equal to the population size, and
    # a weakly regularized model is not uniform.
    strict = _weighter(md, min_ess_fraction=1.0)
    strict.hyperparam_space = {
        LogisticRegression: {
            "C": [100.0],
            "l1_ratio": [0],
            "solver": ["saga"],
            "fit_intercept": [True],
            "max_iter": [500],
        }
    }
    with pytest.raises(ValueError, match="min_ess_fraction"):
        strict.match()

    weighter = _weighter(md, min_ess_fraction=0.3)
    weighter.match()
    n_pool, n_target = len(weighter.weights), len(weighter.target_weights)
    assert weighter.effective_sample_size() >= 0.3 * n_pool
    assert weighter.diagnostics["target_effective_sample_size"] >= 0.3 * n_target


def test_match_with_another_method_reuses_the_fit():
    md = _shifted_md(seed=15)
    weighter = _weighter(md, method="att")
    weighter.match()
    model, att_weights = weighter.best_model, weighter.get_weights().copy()

    matched = weighter.match(method="overlap")
    overlap_weights = matched.get_population("pool")[weighter.weight_col].values
    assert overlap_weights.max() < 1
    assert not np.allclose(att_weights, overlap_weights)
    assert np.allclose(overlap_weights, weighter.get_weights("overlap"))
    assert weighter.score("overlap") < weighter.score("att")

    # The fitted state is still that of the att fit.
    assert weighter.method == "att" and weighter.best_model is model
    assert np.allclose(weighter.get_weights(), att_weights)
    default = weighter.match().get_population("pool")[weighter.weight_col].values
    assert np.allclose(default, att_weights)

    with pytest.raises(ValueError):
        weighter.match(method="nope")


def test_get_best_match_before_match_raises():
    weighter = _weighter(_shifted_md(n_pool=300, n_target=50, seed=16))
    with pytest.raises(ValueError):
        weighter.get_best_match()


def test_plot_methods():
    md = _shifted_md(n_pool=800, n_target=120, seed=10)
    weighter = _weighter(md, method="overlap")
    weighter.match()

    g = weighter.plot_propensity_score_distributions()
    assert len(g.axes.flat) == 2
    g = weighter.plot_propensity_score_distributions(method="att")
    assert len(g.axes.flat) == 2

    g = weighter.plot_weight_distributions()
    assert len(g.axes.flat) == 2
    g = weighter.plot_weight_distributions(method="att")
    assert len(g.axes.flat) == 2

    with pytest.raises(ValueError):
        _weighter(md).plot_weight_distributions()
