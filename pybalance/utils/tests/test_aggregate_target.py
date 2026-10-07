import math

import numpy as np
import pandas as pd
import pytest

from pybalance.sim import generate_toy_dataset
from pybalance.utils import (
    AggregateTarget,
    BetaBalance,
    MatchingData,
    MatchingHeaders,
    split_target_pool,
)
from pybalance.lp import (
    ConstraintSatisfactionMatcher,
    AggregateConstraintSatisfactionMatcher,
)


def _patient_level_to_aggregate(
    target_df: pd.DataFrame, headers: MatchingHeaders
) -> AggregateTarget:
    numeric = {}
    for col in headers.numeric:
        numeric[col] = {
            "mean": float(target_df[col].mean()),
            "std": float(target_df[col].std()),
        }
    categoric = {}
    for col in headers.categoric:
        rates = target_df[col].value_counts(normalize=True).to_dict()
        categoric[col] = {k: float(v) for k, v in rates.items()}
    return AggregateTarget(
        n=len(target_df),
        numeric=numeric,
        categoric=categoric,
        headers=MatchingHeaders(
            numeric=list(headers.numeric),
            categoric=list(headers.categoric),
        ),
    )


def test_aggregate_target_from_dict_validation():
    target = AggregateTarget.from_dict(
        {
            "n": 100,
            "numeric": {"age": {"mean": 65.0, "std": 10.0}},
            "categoric": {"sex": {"F": 0.4, "M": 0.6}},
        }
    )
    assert target.n == 100
    assert target.numeric["age"]["mean"] == 65.0
    assert set(target.categoric["sex"]) == {"F", "M"}

    # Partial disclosure is fine (rates need not sum to 1; unlisted categories
    # are left unconstrained) ...
    partial = AggregateTarget.from_dict(
        {"n": 10, "categoric": {"country": {"US": 0.6}}}
    )
    assert partial.categoric["country"] == {"US": 0.6}

    # ... but summing to more than 1 is impossible and still rejected.
    with pytest.raises(ValueError):
        AggregateTarget.from_dict({"n": 10, "categoric": {"sex": {"F": 0.7, "M": 0.6}}})


def test_matching_data_pool_and_target_frames():
    m = generate_toy_dataset(n_pool=200, n_target=50, seed=1)
    pool = m.get_population("pool").drop(columns=[m.population_col])
    target = m.get_population("target").drop(columns=[m.population_col])

    m2 = MatchingData(pool=pool, target=target, headers=m.headers)
    assert set(m2.populations) == {"pool", "target"}
    assert len(m2.get_population("pool")) == 200
    assert len(m2.get_population("target")) == 50
    assert not m2.has_aggregate_target


def test_matching_data_pool_and_aggregate_target():
    m = generate_toy_dataset(n_pool=200, n_target=50, seed=2)
    pool = m.get_population("pool").drop(columns=[m.population_col])
    target_df = m.get_population("target")
    agg = _patient_level_to_aggregate(target_df, m.headers)

    m_agg = MatchingData(pool=pool, target=agg, headers=m.headers)
    assert m_agg.has_aggregate_target
    assert set(m_agg.populations) == {"pool", "target"}
    assert len(m_agg.data) == 200
    assert len(m_agg.get_population("pool")) == 200

    with pytest.raises(KeyError, match="aggregate target"):
        m_agg.get_population("target")

    with pytest.raises(ValueError, match="aggregate target"):
        split_target_pool(m_agg)

    m_copy = m_agg.copy()
    assert m_copy.has_aggregate_target
    assert m_copy.aggregate_target.n == agg.n


def test_matching_data_rejects_mixed_constructors():
    m = generate_toy_dataset(n_pool=50, n_target=10, seed=3)
    with pytest.raises(ValueError, match="either `data` or `pool`"):
        MatchingData(
            data=m.data,
            pool=m.get_population("pool"),
            target=m.get_population("target"),
        )


def test_beta_balance_aggregate_matches_patient_moments():
    m = generate_toy_dataset(n_pool=500, n_target=80, seed=4)
    # Drop country: pool has a category with rate 0, so one-hot columns differ
    # when the preprocessor is fit on pool-only vs both populations.
    headers = MatchingHeaders(
        numeric=["age", "height", "weight"],
        categoric=[
            "gender",
            "haircolor",
            "binary_0",
            "binary_1",
            "binary_2",
            "binary_3",
        ],
    )
    pool = m.get_population("pool").drop(columns=[m.population_col])
    target_df = m.get_population("target")
    agg = _patient_level_to_aggregate(target_df, headers)

    m_patient = MatchingData(
        pool=pool,
        target=target_df.drop(columns=[m.population_col]),
        headers=headers,
    )
    m_agg = MatchingData(pool=pool, target=agg, headers=headers)

    beta_patient = BetaBalance(m_patient)
    beta_agg = BetaBalance(m_agg)

    assert np.allclose(
        beta_patient.target_mean.cpu().numpy(),
        beta_agg.target_mean.cpu().numpy(),
        atol=1e-5,
    )
    # Stds should be close for numeric; categoric uses Bernoulli approx for aggregate.
    # Compare distances on full pool — numeric-dominated agreement is enough.
    d_patient = float(beta_patient.distance(list(range(len(beta_patient.pool)))))
    d_agg = float(beta_agg.distance(list(range(len(beta_agg.pool)))))
    assert math.isclose(d_patient, d_agg, rel_tol=0.15, abs_tol=0.05)


def test_lp_matcher_with_aggregate_target():
    m = generate_toy_dataset(n_pool=300, n_target=40, seed=5)
    pool = m.get_population("pool").drop(columns=[m.population_col])
    target_df = m.get_population("target")
    # Keep the feature set small so the solver stays quick in CI.
    headers = MatchingHeaders(
        numeric=["age", "weight"],
        categoric=["gender"],
    )
    agg = _patient_level_to_aggregate(target_df, headers)
    m_agg = MatchingData(pool=pool, target=agg, headers=headers)

    matcher = AggregateConstraintSatisfactionMatcher(
        m_agg,
        pool_size=40,
        time_limit=30,
        num_workers=1,
        verbose=False,
    )
    match = matcher.match()
    assert match.has_aggregate_target
    assert len(match.get_population("pool")) == 40
    assert match.aggregate_target.n == 40


def test_matchers_reject_wrong_target_type():
    m = generate_toy_dataset(n_pool=300, n_target=40, seed=5)
    pool = m.get_population("pool").drop(columns=[m.population_col])
    target_df = m.get_population("target")
    headers = MatchingHeaders(numeric=["age", "weight"], categoric=["gender"])

    agg = _patient_level_to_aggregate(target_df, headers)
    m_agg = MatchingData(pool=pool, target=agg, headers=headers)
    with pytest.raises(ValueError, match="patient-level target"):
        ConstraintSatisfactionMatcher(m_agg)

    with pytest.raises(ValueError, match="aggregate"):
        AggregateConstraintSatisfactionMatcher(m)


def test_aggregate_target_csv_roundtrip(tmp_path):
    target = AggregateTarget(
        n=120,
        numeric={
            "age": {"median": 41.0},
            "weight": {"mean": 80.5, "std": 12.25, "quantile": [(0.2, 70.0), (0.9, 95.0)]},
        },
        categoric={"gender": {0.0: 0.4, 1.0: 0.6}, "country": {"US": 0.3, "DE": 0.1}},
    )
    path = tmp_path / "target.csv"
    target.to_csv(path)
    loaded = AggregateTarget.from_csv(path)

    assert loaded.n == 120
    assert loaded.numeric == target.numeric
    assert loaded.categoric == target.categoric
    assert loaded.headers == target.headers
    # to_csv() without a path returns the same text that would be written
    assert target.to_csv() == path.read_text()


@pytest.mark.parametrize(
    "body, message",
    [
        ("age,mean,,50\n", "no 'n' row"),
        (",n,,10\n,n,,10\n", "duplicate 'n'"),
        (",n,,10\nage,mode,,50\n", "unknown statistic"),
        (",n,,10\nage,mean,,50\nage,mean,,51\n", "duplicate 'mean'"),
        (",n,,10\nage,mean,,abc\n", "must be a number"),
        (",n,,10\nweight,quantile,,70\n", "quantile parameter"),
        (",n,,10\ngender,rate,,0.5\n", "requires a level"),
        (",n,,10\n,mean,,50\n", "requires a feature"),
    ],
)
def test_aggregate_target_from_csv_errors(tmp_path, body, message):
    path = tmp_path / "bad.csv"
    path.write_text("feature,statistic,parameter,value\n" + body)
    with pytest.raises(ValueError, match=message):
        AggregateTarget.from_csv(path)


def test_aggregate_target_from_csv_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("feature,value\nage,50\n")
    with pytest.raises(ValueError, match="missing required columns"):
        AggregateTarget.from_csv(path)


def test_aggregate_match_keeps_undisclosed_features_unconstrained():
    m = generate_toy_dataset(n_pool=300, n_target=0, seed=3)
    pool = m.get_population("pool")
    target = AggregateTarget(
        n=40, numeric={"age": {"mean": 60.0}}, categoric={"gender": {0: 0.5, 1: 0.5}}
    )
    headers = MatchingHeaders(numeric=["age", "height"], categoric=["gender", "country"])
    data = MatchingData(pool=pool, target=target, headers=headers)
    match = AggregateConstraintSatisfactionMatcher(
        data, time_limit=20, num_workers=1, verbose=False
    ).match()

    # undisclosed features stay on the match and are reported, with no target value
    assert match.headers == headers
    described = match.describe(quantiles=[])
    assert described.loc[("height", "mean"), "pool"] > 0
    assert math.isnan(described.loc[("height", "mean"), "target"])
    assert described.loc[("country",), "target"].isna().all()
    assert described.loc[("population size", "N"), "pool"] == 40
    assert described.loc[("population size", "N"), "target"] == 40
    # the disclosed mean is still matched
    assert abs(match.get_population("pool")["age"].mean() - 60.0) < 1.0


def test_aggregate_target_min_max_are_0th_and_100th_quantile(tmp_path):
    target = AggregateTarget(n=50, numeric={"age": {"median": 50.0, "min": 18.0, "max": 75.0}})
    assert target.numeric["age"]["quantile"] == [(0.0, 18.0), (0.5, 50.0), (1.0, 75.0)]
    assert AggregateTarget(
        n=50, numeric={"age": {"quantile": [(0.0, 18.0), (0.5, 50.0), (1.0, 75.0)]}}
    ).numeric == target.numeric

    path = tmp_path / "t.csv"
    target.to_csv(path)
    assert {"min", "median", "max"} <= set(pd.read_csv(path)["statistic"])
    assert AggregateTarget.from_csv(path).numeric == target.numeric

    with pytest.raises(ValueError):
        AggregateTarget(n=5, numeric={"age": {"quantile": [(1.5, 10.0)]}})


def test_aggregate_match_max_is_a_soft_limit():
    m = generate_toy_dataset(n_pool=400, n_target=0, seed=3)
    pool = m.get_population("pool")
    headers = MatchingHeaders(numeric=["weight"], categoric=["gender"])
    gender = {"gender": {0: 0.5, 1: 0.5}}

    def match(numeric):
        data = MatchingData(
            pool=pool, target=AggregateTarget(n=50, numeric=numeric, categoric=gender), headers=headers
        )
        return data, AggregateConstraintSatisfactionMatcher(
            data, time_limit=20, num_workers=1, verbose=False
        ).match()

    before, after = match({"weight": {"median": 85.0, "max": 100.0}})
    assert (before.get_population("pool")["weight"] > 100).mean() > 0.1
    assert (after.get_population("pool")["weight"] > 100).mean() < 0.05

    # a limit no pool patient violates has nothing to match and is skipped
    _, after = match({"weight": {"median": 85.0, "max": 500.0}})
    assert len(after.get_population("pool")) == 50


def test_aggregate_target_needs_only_one_statistic():
    assert AggregateTarget(n=5, numeric={"age": {"std": 3.0}}).numeric == {"age": {"std": 3.0}}
    with pytest.raises(ValueError):
        AggregateTarget(n=5, numeric={"age": {}})


def _aggregate_match(numeric, categoric, headers, **kwargs):
    m = generate_toy_dataset(n_pool=400, n_target=0, seed=3)
    data = MatchingData(
        pool=m.get_population("pool"),
        target=AggregateTarget(n=50, numeric=numeric, categoric=categoric),
        headers=headers,
    )
    return data, AggregateConstraintSatisfactionMatcher(
        data, time_limit=20, num_workers=1, verbose=False, **kwargs
    ).match()


def test_aggregate_match_std_without_a_mean():
    headers = MatchingHeaders(numeric=["age", "weight"], categoric=[])
    before, after = _aggregate_match({"age": {"std": 6.0}}, {}, headers)
    assert before.get_population("pool")["age"].std() > 10
    assert abs(after.get_population("pool")["age"].std(ddof=0) - 6.0) < 1.0

    # a std of 0 asks for the most homogeneous subset
    _, after = _aggregate_match({"weight": {"std": 0.0}}, {}, headers)
    assert after.get_population("pool")["weight"].std() < before.get_population("pool")["weight"].std() / 3


def test_aggregate_match_does_not_constrain_undisclosed_levels():
    headers = MatchingHeaders(numeric=[], categoric=["country"])
    # level 1 is the one a drop-first encoding would have discarded
    before, after = _aggregate_match({}, {"country": {1: 0.3}}, headers)
    rates = after.get_population("pool")["country"].value_counts(normalize=True)
    assert abs(rates[1] - 0.3) < 0.03
    # the other levels are free to absorb the rest, not pinned to their pool rates
    pool_rates = before.get_population("pool")["country"].value_counts(normalize=True)
    assert max(abs(rates.get(k, 0) - pool_rates[k]) for k in pool_rates.index if k != 1) > 0.03
