import numpy as np
import pandas as pd
import pytest

from pybalance.sim import generate_toy_dataset
from pybalance.utils import AggregateTarget, MatchingData, MatchingHeaders, split_target_pool
from pybalance.weighting import (
    EntropyBalanceWeighter,
    MAICWeighter,
    IPTWWeighter,
    effective_sample_size,
    weighted_balance_table,
)


def _biased_pool_subsample(pool_df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
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


def _aggregate_target_from_frame(df: pd.DataFrame, headers: MatchingHeaders) -> AggregateTarget:
    numeric = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in headers.numeric
    }
    categoric = {
        col: {k: float(v) for k, v in df[col].value_counts(normalize=True).items()}
        for col in headers.categoric
    }
    return AggregateTarget(n=len(df), numeric=numeric, categoric=categoric, headers=headers)


def test_effective_sample_size():
    assert effective_sample_size(np.ones(10)) == pytest.approx(10)
    # one patient carrying all the weight -> ESS collapses to ~1
    weights = np.zeros(10)
    weights[0] = 1.0
    assert effective_sample_size(weights) == pytest.approx(1)


def test_maic_patient_level_target_matches_mean():
    matching_data = generate_toy_dataset(n_pool=3000, n_target=300, seed=1)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(drop=True)
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
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(drop=True)
    target_df = _biased_pool_subsample(pool_df, n=300, seed=2)
    aggregate_target = _aggregate_target_from_frame(target_df, matching_data.headers)

    md = MatchingData(pool=pool_df, target=aggregate_target, headers=matching_data.headers)

    weighter = MAICWeighter(md, verbose=False)
    weighter.match()
    assert weighter.diagnostics["converged"]

    table = weighted_balance_table(weighter)
    residual = (table["weighted_pool"] - table["target"]).abs()
    assert (residual < 1e-3).all()
    assert weighter.weights.sum() == pytest.approx(aggregate_target.n, rel=1e-6)


def test_entropy_balance_weighter_matches_variance():
    matching_data = generate_toy_dataset(n_pool=3000, n_target=300, seed=3)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(drop=True)
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
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(drop=True)
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


def test_iptw_improves_balance_over_unweighted_pool():
    matching_data = generate_toy_dataset(n_pool=3000, n_target=300, seed=7)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(drop=True)
    target_df = _biased_pool_subsample(pool_df, n=300, seed=7)

    md = MatchingData(pool=pool_df, target=target_df, headers=matching_data.headers)
    weighter = IPTWWeighter(md, verbose=False)
    weighter.match()

    table = weighted_balance_table(weighter)
    unweighted_residual = (table["unweighted_pool"] - table["target"]).abs()
    weighted_residual = (table["weighted_pool"] - table["target"]).abs()

    # IPTW doesn't solve for exact balance (unlike entropy balancing), but a
    # correctly-fit propensity model should substantially reduce imbalance.
    assert weighted_residual.sum() < unweighted_residual.sum()
    assert weighter.effective_sample_size() <= len(pool_df)


def test_iptw_rejects_aggregate_target():
    matching_data = generate_toy_dataset(n_pool=500, n_target=100, seed=8)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(drop=True)
    target_df = matching_data.get_population(matching_data.target_name).reset_index(drop=True)
    aggregate_target = _aggregate_target_from_frame(target_df, matching_data.headers)
    md = MatchingData(pool=pool_df, target=aggregate_target, headers=matching_data.headers)

    with pytest.raises(ValueError):
        IPTWWeighter(md, verbose=False)


def test_iptw_trim_quantiles_caps_extreme_weights():
    matching_data = generate_toy_dataset(n_pool=2000, n_target=200, seed=9)
    pool_df = matching_data.get_population(matching_data.pool_name).reset_index(drop=True)
    target_df = _biased_pool_subsample(pool_df, n=200, seed=9)
    md = MatchingData(pool=pool_df, target=target_df, headers=matching_data.headers)

    untrimmed = IPTWWeighter(md, verbose=False)
    untrimmed.match()

    trimmed = IPTWWeighter(md, trim_quantiles=(0.05, 0.95), verbose=False)
    trimmed.match()

    assert trimmed.weights.max() <= untrimmed.weights.max()
    assert trimmed.diagnostics["n_trimmed"] > 0

    with pytest.raises(ValueError):
        IPTWWeighter(md, trim_quantiles=(0.9, 0.1), verbose=False)

