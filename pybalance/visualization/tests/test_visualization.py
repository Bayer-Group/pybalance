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
