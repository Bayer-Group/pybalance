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
