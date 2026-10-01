import numpy as np
import pandas as pd
import pytest
import torch

from pybalance.sim import generate_toy_dataset
from pybalance.utils import (
    AggregateTarget,
    BalanceCalculator,
    MatchingData,
    MatchingHeaders,
    split_target_pool,
)
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


def _make_derived_pool(n=2000, seed=42):
    """Synthetic pool with continuous and binary columns for derived-feature tests."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.normal(65, 10, n),
        "psa": rng.lognormal(4, 1, n),
        "ecog": rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2]),
        "prior_chemo": rng.binomial(1, 0.6, n).astype(float),
    })


def test_maic_derived_median_balances_to_half():
    pool = _make_derived_pool()
    target = AggregateTarget(
        n=100,
        derived={
            "age": {"mode": "median", "value": 68},
            "psa": {"mode": "median", "value": 80},
        },
    )
    md = MatchingData(pool=pool, target=target)
    weighter = MAICWeighter(md, verbose=False)
    weighter.match()
    assert weighter.diagnostics["converged"]

    w = weighter.get_weights()
    age_binary = (pool["age"] > 68).astype(float).values
    psa_binary = (pool["psa"] > 80).astype(float).values
    assert np.average(age_binary, weights=w) == pytest.approx(0.5, abs=1e-3)
    assert np.average(psa_binary, weights=w) == pytest.approx(0.5, abs=1e-3)


def test_maic_derived_indicator_balances_to_rate():
    pool = _make_derived_pool()
    target = AggregateTarget(
        n=100,
        derived={
            "ecog": {"mode": "indicator", "op": "eq", "threshold": 0, "rate": 0.55},
        },
    )
    md = MatchingData(pool=pool, target=target)
    weighter = MAICWeighter(md, verbose=False)
    weighter.match()
    assert weighter.diagnostics["converged"]

    w = weighter.get_weights()
    ecog0 = (pool["ecog"] == 0).astype(float).values
    assert np.average(ecog0, weights=w) == pytest.approx(0.55, abs=1e-3)


def test_maic_derived_presence_balances_to_rate():
    pool = _make_derived_pool()
    target = AggregateTarget(
        n=100,
        derived={
            "prior_chemo": {"mode": "presence", "rate": 0.75},
        },
    )
    md = MatchingData(pool=pool, target=target)
    weighter = MAICWeighter(md, verbose=False)
    weighter.match()
    assert weighter.diagnostics["converged"]

    w = weighter.get_weights()
    assert np.average(pool["prior_chemo"].values, weights=w) == pytest.approx(0.75, abs=1e-3)


def test_maic_mixed_numeric_and_derived():
    pool = _make_derived_pool()
    target = AggregateTarget(
        n=100,
        numeric={"age": {"mean": 70.0}},
        derived={
            "ecog": {"mode": "indicator", "op": "eq", "threshold": 0, "rate": 0.50},
            "prior_chemo": {"mode": "presence", "rate": 0.70},
        },
    )
    md = MatchingData(pool=pool, target=target)
    weighter = MAICWeighter(md, verbose=False)
    weighter.match()
    assert weighter.diagnostics["converged"]

    w = weighter.get_weights()
    assert np.average(pool["age"].values, weights=w) == pytest.approx(70.0, abs=0.1)
    ecog0 = (pool["ecog"] == 0).astype(float).values
    assert np.average(ecog0, weights=w) == pytest.approx(0.50, abs=1e-3)
    assert np.average(pool["prior_chemo"].values, weights=w) == pytest.approx(0.70, abs=1e-3)


def test_derived_feature_validation():
    with pytest.raises(ValueError, match="unknown mode"):
        AggregateTarget(n=50, derived={"x": {"mode": "bogus"}})
    with pytest.raises(ValueError, match="requires 'value'"):
        AggregateTarget(n=50, derived={"x": {"mode": "median"}})
    with pytest.raises(ValueError, match="requires 'op'"):
        AggregateTarget(n=50, derived={"x": {"mode": "indicator", "threshold": 0, "rate": 0.5}})
    with pytest.raises(ValueError, match="unknown op"):
        AggregateTarget(
            n=50, derived={"x": {"mode": "indicator", "op": "ne", "threshold": 0, "rate": 0.5}}
        )
    with pytest.raises(ValueError, match="requires 'rate'"):
        AggregateTarget(n=50, derived={"x": {"mode": "presence"}})
    with pytest.raises(ValueError, match="unknown key"):
        AggregateTarget(n=50, derived={"x": {"mode": "median", "value": 1, "column": "y"}})
    for bad_rate in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            AggregateTarget(n=50, derived={"x": {"mode": "presence", "rate": bad_rate}})


def test_derived_feature_overlap_with_numeric_rejected():
    with pytest.raises(ValueError, match="both derived and numeric"):
        AggregateTarget(
            n=50,
            numeric={"age": {"mean": 70.0}},
            derived={"age": {"mode": "median", "value": 68}},
        )
    with pytest.raises(ValueError, match="both derived and numeric"):
        AggregateTarget(
            n=50,
            categoric={"ecog": {0: 0.5, 1: 0.5}},
            derived={"ecog": {"mode": "indicator", "op": "eq", "threshold": 0, "rate": 0.5}},
        )


@pytest.mark.parametrize(
    "spec",
    [
        {"mode": "median", "value": 1},
        {"mode": "indicator", "op": "eq", "threshold": 0, "rate": 0.5},
        {"mode": "presence", "rate": 0.5},
    ],
)
def test_derived_missing_values_raise(spec):
    pool = _make_derived_pool()
    col = "prior_chemo" if spec["mode"] == "presence" else "ecog"
    pool[col] = pool[col].astype(float)
    pool.loc[:9, col] = np.nan
    target = AggregateTarget(n=100, derived={col: spec})
    md = MatchingData(pool=pool, target=target)
    with pytest.raises(ValueError, match="10 missing value"):
        MAICWeighter(md, verbose=False)


def test_derived_presence_requires_binary_column():
    pool = _make_derived_pool()
    target = AggregateTarget(n=100, derived={"ecog": {"mode": "presence", "rate": 0.5}})
    md = MatchingData(pool=pool, target=target)
    with pytest.raises(ValueError, match="must be binary"):
        MAICWeighter(md, verbose=False)


def test_derived_match_variance_with_numeric_std():
    pool = _make_derived_pool()
    target = AggregateTarget(
        n=100,
        numeric={"age": {"mean": 66.0, "std": 9.0}},
        derived={"ecog": {"mode": "indicator", "op": "eq", "threshold": 0, "rate": 0.5}},
    )
    md = MatchingData(pool=pool, target=target)
    weighter = EntropyBalanceWeighter(md, match_variance=True, verbose=False)
    weighter.match()
    assert weighter.diagnostics["converged"]
    assert "age (variance)" in weighter.constraint_labels
    assert "ecog (variance)" not in weighter.constraint_labels

    w = weighter.get_weights()
    age = pool["age"].values
    mean = np.average(age, weights=w)
    assert mean == pytest.approx(66.0, abs=0.1)
    assert np.sqrt(np.average((age - mean) ** 2, weights=w)) == pytest.approx(9.0, abs=0.1)
    ecog0 = (pool["ecog"] == 0).astype(float).values
    assert np.average(ecog0, weights=w) == pytest.approx(0.5, abs=1e-3)


def test_balance_calculator_encodes_derived_features():
    """A BalanceCalculator built directly on derived-target data must compare the
    0/1 encoding with the disclosed rate, the same as a pre-encoded presence target."""
    pool = _make_derived_pool()
    derived_md = MatchingData(
        pool=pool,
        target=AggregateTarget(n=100, derived={"age": {"mode": "median", "value": 68}}),
    )
    encoded = pool.copy()
    encoded["age"] = (pool["age"] > 68).astype(float)
    presence_md = MatchingData(
        pool=encoded,
        target=AggregateTarget(n=100, derived={"age": {"mode": "presence", "rate": 0.5}}),
    )

    bc_derived = BalanceCalculator(derived_md, "beta")
    bc_presence = BalanceCalculator(presence_md, "beta")
    assert torch.allclose(bc_derived.pool, bc_presence.pool)
    raw_pool = derived_md.get_population(derived_md.pool_name)
    encoded_pool = presence_md.get_population(presence_md.pool_name)
    assert bc_derived.distance(raw_pool).item() == pytest.approx(
        bc_presence.distance(encoded_pool).item(), rel=1e-6
    )

    # The same holds for the MatchingData returned by match(), which keeps the
    # raw columns.
    matched = MAICWeighter(derived_md, verbose=False).match()
    bc_matched = BalanceCalculator(matched, "beta")
    assert torch.allclose(bc_matched.pool, bc_presence.pool)


def test_weighted_balance_table_reports_derived_rates():
    pool = _make_derived_pool()
    target = AggregateTarget(
        n=100,
        derived={
            "age": {"mode": "median", "value": 68},
            "ecog": {"mode": "indicator", "op": "eq", "threshold": 0, "rate": 0.55},
        },
    )
    weighter = MAICWeighter(MatchingData(pool=pool, target=target), verbose=False)
    weighter.match()
    table = weighted_balance_table(weighter).set_index("feature")
    assert table.loc["age", "target"] == pytest.approx(0.5)
    assert table.loc["ecog", "target"] == pytest.approx(0.55)
    assert table.loc["age", "unweighted_pool"] == pytest.approx((pool["age"] > 68).mean(), abs=1e-6)
    assert table.loc["age", "weighted_pool"] == pytest.approx(0.5, abs=1e-3)
    assert table.loc["ecog", "weighted_pool"] == pytest.approx(0.55, abs=1e-3)

