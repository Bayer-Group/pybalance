import pytest

import numpy as np
import pandas as pd

from pybalance.lp import AggregateConstraintSatisfactionMatcher
from pybalance.utils import (
    AggregateTarget,
    MatchingData,
    MatchingHeaders,
    aggregate_target_constraints,
)
from pybalance.visualization import plot_aggregate_target_match
from pybalance.visualization.distributions import _debin_features


def _aggregate_matching_data():
    rng = np.random.default_rng(0)
    n = 800
    pool = pd.DataFrame(
        {
            "age": rng.normal(55, 10, n),
            "weight": rng.normal(80, 14, n),
            "country": rng.choice(["US", "DE", "FR"], n, p=[0.5, 0.3, 0.2]),
        }
    )
    target = AggregateTarget(
        n=80,
        numeric={
            "age": {"median": 50.0},
            "weight": {"mean": 80.0, "quantile": [(0.2, 70.0)]},
        },
        categoric={"country": {"US": 0.3}},
    )
    headers = MatchingHeaders(numeric=["age", "weight"], categoric=["country"])
    return MatchingData(pool=pool, target=target, headers=headers)


def test_aggregate_target_constraints_lists_disclosed_constraints_only():
    table = aggregate_target_constraints(_aggregate_matching_data())
    labels = set(zip(table["feature"], table["constraint"]))
    assert labels == {
        ("age", "P(x <= 50)"),
        ("weight", "mean"),
        ("weight", "P(x <= 70)"),
        ("country", "rate of US"),
    }
    # The median is the 0.5 quantile; undisclosed countries are not listed.
    assert table.set_index("constraint").loc["P(x <= 50)", "target"] == 0.5


def test_plot_aggregate_target_match_shows_pool_moving_to_target():
    before = _aggregate_matching_data()
    after = AggregateConstraintSatisfactionMatcher(
        before, time_limit=20, num_workers=1, verbose=False
    ).match()

    t_before = aggregate_target_constraints(before).set_index(["feature", "constraint"])
    t_after = aggregate_target_constraints(after).set_index(["feature", "constraint"])
    err_before = (t_before["pool"] - t_before["target"]).abs() / t_before["target"]
    err_after = (t_after["pool"] - t_after["target"]).abs() / t_after["target"]
    assert err_after.mean() < err_before.mean()

    fig = plot_aggregate_target_match(before, after)
    assert len(fig.axes) >= len(t_before)


def test_debin():
    effective_features = ["x_1", "x_123", "z_10", "z_0", "y_12"]
    input_output_column_mapping = {
        "x": ["x_1", "x_123"],
        "y": ["y_12"],
        "z": ["z_10", "z_0"],
    }
    indices = _debin_features(effective_features, input_output_column_mapping)
    assert indices["x"] == [0, 1]
    assert indices["y"] == [4]
    assert indices["z"] == [2, 3]


@pytest.mark.parametrize(
    "numeric, categoric, n_panels",
    [
        ({"age": {"mean": 55.0}}, {"country": {"US": 0.5, "DE": 0.3, "FR": 0.2}}, 4),
        (
            {"age": {"mean": 55.0, "std": 0.0}, "weight": {"mean": 80.0, "std": 9.0}},
            {},
            4,
        ),
        (
            {
                "age": {
                    "quantile": [(0.25, 48.0), (0.75, 62.0)],
                    "median": 55.0,
                    "min": 30.0,
                    "max": 80.0,
                },
                "weight": {"mean": 80.0, "max": 500.0},
            },
            {},
            7,
        ),
        ({}, {"country": {"US": 0.4}}, 1),
    ],
)
def test_plot_aggregate_target_match_handles_every_target_type(
    numeric, categoric, n_panels
):
    pool = _aggregate_matching_data().get_population("pool")
    target = AggregateTarget(n=80, numeric=numeric, categoric=categoric)
    before = MatchingData(pool=pool, target=target, headers=target.headers)
    after = MatchingData(
        pool=pool.sample(80, random_state=0), target=target, headers=target.headers
    )

    for quantiles_as in ("value", "proportion"):
        fig = plot_aggregate_target_match(before, after, quantiles_as=quantiles_as)
        assert sum(ax.get_visible() for ax in fig.axes) == n_panels
        # proportions never leave [0, 1] by more than the margin
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            if ax.get_title().split("\n")[1].startswith(("P(x", "rate of")):
                assert ax.get_ylim()[0] >= -0.05 and ax.get_ylim()[1] <= 1.05


def test_quantile_constraints_can_be_listed_by_value():
    data = _aggregate_matching_data()
    by_value = aggregate_target_constraints(data, quantiles_as="value")
    row = by_value.set_index("constraint").loc["median"]
    assert row["target"] == 50.0
    pool_age = data.get_population("pool")["age"]
    assert row["pool"] == pool_age.median()
    assert set(by_value["constraint"]) == {
        "median",
        "mean",
        "20th percentile",
        "rate of US",
    }


def test_plot_aggregate_target_match_accepts_weights():
    data = _aggregate_matching_data()
    pool = data.get_population("pool").copy()
    # weight the pool toward older patients
    pool["w"] = np.exp(0.05 * (pool["age"] - pool["age"].mean()))
    weighted = MatchingData(
        pool=pool, target=data.aggregate_target, headers=data.headers
    )

    plain = aggregate_target_constraints(weighted).set_index("constraint")
    by_weight = aggregate_target_constraints(weighted, weights="w").set_index(
        "constraint"
    )
    assert by_weight.loc["mean", "pool"] > plain.loc["mean", "pool"]
    assert by_weight.loc["P(x <= 50)", "pool"] < plain.loc["P(x <= 50)", "pool"]
    assert by_weight.loc["P(x <= 50)", "pool"] == pytest.approx(
        np.average(pool["age"] <= 50, weights=pool["w"])
    )

    # equal weights reproduce the unweighted statistics
    pool["w"] = 3.0
    uniform = MatchingData(
        pool=pool, target=data.aggregate_target, headers=data.headers
    )
    assert aggregate_target_constraints(uniform, weights="w")[
        "pool"
    ].tolist() == pytest.approx(
        aggregate_target_constraints(uniform)["pool"].tolist(), rel=1e-2
    )

    pool["w"] = np.exp(0.05 * (pool["age"] - pool["age"].mean()))
    fig = plot_aggregate_target_match(data, weighted, weights="w")
    assert any(ax.get_visible() for ax in fig.axes)
    with pytest.raises(ValueError, match="not found"):
        plot_aggregate_target_match(data, weighted, weights="nope")
