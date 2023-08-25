"""
Some helpful functions for plotting distribution.
"""
import re
from collections import defaultdict
from typing import List, Optional
import itertools

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from pybalance import MatchingData, BaseBalanceCalculator, split_target_pool

import logging

logger = logging.getLogger(__name__)


def _get_default_hue_order(matching_data: MatchingData) -> List[str]:
    return sorted(matching_data.populations)


def _get_reference_population(matching_data: MatchingData) -> str:
    # Try to make a reasonable choice for what to use as a reference calculation
    # when computing differences.
    try:
        # If two populations defined, use target
        target, pool = split_target_pool(matching_data)
        reference_population = target[matching_data.population_col].unique()[0]
    except ValueError:
        try:
            # If something called 'target' exists, use it
            matching_data.get_population("target")
            reference_population = "target"
        except KeyError:
            # Else pick one of the smallest populations
            reference_population = (
                matching_data.counts()
                .reset_index()
                .sort_values([matching_data.population_col, "N"])
                .head(1)[matching_data.population_col]
                .values[0]
            )
    return reference_population


def _debin_features(effective_features, input_output_column_mapping):
    # Map original features to effective features
    indices = defaultdict(list)
    original_features = input_output_column_mapping.keys()
    for feature in original_features:
        for j, new_feature in enumerate(effective_features):
            if new_feature in input_output_column_mapping[feature]:
                indices[feature].append(j)
    if not len(sum(list(indices.values()), start=[])) == len(effective_features):
        raise ValueError("debinning not possible, try reruning with debin=False")
    return indices


def _plot_1d_marginals(matching_data, headers, col_wrap, height, **plot_params):
    # Set up figure of correct size and shape.
    ncols = col_wrap
    nrows = 1 + (len(headers) - 1) // ncols
    fig = plt.figure(figsize=(height * ncols, 3 * height / 4 * nrows))

    # PLOT!
    data = matching_data.data
    for j, col in enumerate(headers):
        ax = plt.subplot(nrows, ncols, j + 1)
        sns.histplot(data=data, x=col, **plot_params, ax=ax)
        ax.grid(True)

    # Align axes
    if plot_params["cumulative"]:
        ymax = 1
    else:
        ymax = max([ax.get_ylim()[1] for ax in fig.axes])
    [ax.set_ylim(0, ymax) for ax in fig.axes]

    return fig


def plot_categoric_features(
    matching_data: MatchingData,
    col_wrap: int = 2,
    height: float = 6,
    include_binary=True,
    **plot_params,
) -> plt.Figure:
    """
    Plot the one-dimensional marginal distributions for all categoric features
    and all treatment groups found in matching_data. Extra keyword arguments are
    passed to seaborn.histplot and override defaults.
    """
    # Set up default plotting params for categoric varaibles.
    default_params = {
        "hue": matching_data.population_col,
        "hue_order": _get_default_hue_order(matching_data),
        "stat": "probability",
        "cumulative": False,
        "common_norm": False,
        "discrete": True,
        "multiple": "dodge",
        "shrink": 0.8,
    }

    # overrides default settings (including hue_order, if supplied)
    default_params.update(plot_params)

    # Determine which covariates to plot.
    headers = matching_data.headers["categoric"]
    if not include_binary:
        headers = [c for c in headers if matching_data[c].nunique() > 2]

    # PLOT!
    fig = _plot_1d_marginals(matching_data, headers, col_wrap, height, **default_params)
    [
        fig.axes[j].set_xticks(matching_data[col].unique())
        for j, col in enumerate(headers)
    ]

    return fig


def plot_numeric_features(
    matching_data: MatchingData, col_wrap: int = 2, height: float = 6, **plot_params
) -> plt.Figure:
    """
    Plot the one-dimensional marginal distributions for all numerical features
    and all treatment groups found in matching_data. Extra keyword arguments are
    passed to seaborn.histplot and override defaults.
    """
    # Set up default plotting params for numeric varaibles.
    default_params = {
        "hue": matching_data.population_col,
        "hue_order": _get_default_hue_order(matching_data),
        "stat": "probability",
        "cumulative": True,
        "common_norm": False,
        "discrete": False,
        "element": "step",
        "bins": 500,
        "linewidth": 3,
        "fill": False,
    }

    # overrides default settings (including hue_order, if supplied)
    default_params.update(plot_params)

    # Determine which covariates to plot.
    headers = matching_data.headers["numeric"]

    # PLOT!
    fig = _plot_1d_marginals(matching_data, headers, col_wrap, height, **default_params)

    return fig


def plot_binary_features(
    matching_data: MatchingData,
    max_features: int = 25,
    include_only: Optional[List[str]] = None,
    orient_horizontal: bool = False,
    standardize_difference: bool = False,
    reference_population: Optional[str] = None,
    **plot_params,
) -> plt.Figure:
    """
    Plot all binary features for all treatment groups found in matching_data.
    Additional keyword arguments are passed to sns.barplot and override default.

    :param matching_data: MatchingData instance containing at least a pool and
        target population.

    :param max_features: Max number of features to show in plot, in case there
        are a lot of binary features. Features are sorted in descending order by
        the initial mismatch between pool and target. The top max_features will
        be shown.

    :param include_only: List of features to consider for plotting. Otherwise,
        all binary features are plotted.

    :param orient_horizontal: If True, orient features along the x-axis.
        Otherwise, features will be along the y-axis.

    :param standardize_difference: Whether to use the absolute standardized mean
        difference for the differences plot (otherwise plots absolute mean
        difference).

    :param reference_population: Name of population in matching_data against
        which other populations should be compared. If not supplied, will use
        the smaller population as the reference population.

    :param plot_params: Parameters passed on to seaborn routines.
    """
    if reference_population is None:
        reference_population = _get_reference_population(matching_data)

    default_params = {
        "hue": matching_data.population_col,
        "hue_order": _get_default_hue_order(matching_data),
    }
    default_params.update(plot_params)

    binary_cols = [
        c
        for c in matching_data.headers["categoric"]
        if matching_data.data[c].nunique() == 2
    ]
    binary_cols = [c for c in binary_cols if include_only is None or c in include_only]

    if len(binary_cols) == 0:
        logger.warning("No binary features found!!")
        return None

    data = matching_data.copy().data
    data.loc[:, binary_cols] = data[binary_cols].rank(method="dense") - 1

    # Frequencies
    frequencies = (
        data.groupby(matching_data.population_col)[binary_cols].mean().T.reset_index()
    )
    frequencies = pd.melt(
        frequencies, id_vars=["index"], value_vars=default_params["hue_order"]
    )

    # Differences
    target_values = frequencies[
        frequencies[matching_data.population_col] == reference_population
    ]
    frequencies = frequencies.merge(target_values, suffixes=["", "_target"], on="index")
    frequencies.loc[:, "difference"] = np.abs(
        frequencies["value"] - frequencies["value_target"]
    )
    if standardize_difference:
        variance = frequencies["value"] * (1 - frequencies["value"]) + frequencies[
            "value_target"
        ] * (1 - frequencies["value_target"])
        frequencies.loc[:, "difference"] = frequencies.loc[:, "difference"] / np.sqrt(
            variance
        )
        difference_label = "Standard Difference"
    else:
        difference_label = "Abs Difference"

    # Restrict to top features
    frequencies = frequencies.sort_values(
        [matching_data.population_col, "difference"], ascending=[False, False]
    )
    features = (
        frequencies[frequencies[matching_data.population_col] != reference_population][
            ["index"]
        ]
        .drop_duplicates()
        .head(max_features)["index"]
        .values
    )
    frequencies = frequencies[frequencies["index"].isin(features)]

    # Sort features by mismatch in the pool
    pool_frequencies = frequencies[
        frequencies[matching_data.population_col] != reference_population
    ]
    non_pool_frequencies = frequencies[
        frequencies[matching_data.population_col] == reference_population
    ]
    frequencies = pd.concat([pool_frequencies, non_pool_frequencies])
    plt.rc("legend", fontsize=14)

    if orient_horizontal:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(len(pool_frequencies) / 2, 16),
            gridspec_kw={"height_ratios": [1, 3]},
        )

        plt.subplot(2, 1, 1)
        sns.barplot(data=frequencies, y="difference", x="index", **default_params)
        xticks, labels = plt.xticks()
        plt.gca().set_xticks(xticks + 0.5, minor=False)
        plt.gca().set_xticks(xticks, minor=True)
        plt.gca().set_xticklabels([""] * len(labels), minor=True, rotation=90)
        plt.grid(True)
        plt.axhline(
            y=0.1,
            xmin=xticks.min(),
            xmax=xticks.max(),
            c="k",
            lw=2.5,
            zorder=10000,
            linestyle="--",
        )
        plt.ylim([0, 0.25])
        plt.ylabel("Abs Difference", fontsize=18)
        plt.gca().get_legend().remove()
        plt.xlabel("")

        plt.subplot(2, 1, 2)
        sns.barplot(data=frequencies, y="value", x="index", **default_params)
        xticks, labels = plt.xticks()
        plt.gca().set_xticks(xticks + 0.5, minor=False)
        plt.gca().set_xticks(xticks, minor=True)
        plt.gca().set_xticklabels(
            labels, minor=True, rotation=45, ha="right", fontsize=16
        )
        plt.grid(True)
        plt.ylabel("Frequency", fontsize=18)
        plt.xlabel("Feature")

    else:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(16, len(pool_frequencies) / 2),
            gridspec_kw={"width_ratios": [3, 1]},
        )

        plt.subplot(1, 2, 2)
        sns.barplot(data=frequencies, x="difference", y="index", **default_params)
        ticks, labels = plt.yticks()
        plt.gca().set_yticks(ticks + 0.5, minor=False)
        plt.gca().set_yticks(ticks, minor=True)
        plt.gca().set_yticklabels([""] * len(labels), minor=True)
        plt.grid(True)
        plt.axvline(
            x=0.1, ymin=ticks.min(), ymax=ticks.max(), c="k", lw=2.5, linestyle="--"
        )
        plt.xlim([0, 0.25])
        plt.xlabel(difference_label, fontsize=18)
        plt.gca().get_legend().remove()
        plt.ylabel("")

        plt.subplot(1, 2, 1)
        sns.barplot(data=frequencies, x="value", y="index", **default_params)
        ticks, labels = plt.yticks()
        plt.gca().set_yticks(ticks + 0.5, minor=False)
        plt.gca().set_yticks(ticks, minor=True)
        plt.gca().set_yticklabels(labels, minor=True, fontsize=16)
        plt.grid(True)
        plt.xlabel("Frequency", fontsize=18)
        plt.ylabel("Feature")

    plt.tight_layout()
    return fig


def plot_per_feature_loss(
    matching_data: MatchingData,
    balance_calculator: BaseBalanceCalculator,
    reference_population: Optional[str] = None,
    debin: bool = True,
    normalize: bool = False,
    **plot_params,
) -> plt.Figure:
    """
    Plot the mismatch as a function of feature.

    :param matching_data: Input data to plot.

    :param balance_calculator: Balance metric to use for calculating the per
        feature loss. Balance calculator must implement a 'per_feature_loss'
        method.

    :param reference_population: Name of population in matching_data against
        which other populations should be compared. If not supplied, will use
        the smaller population as the reference population.

    :param debin: If True, attempt to map effective features back into the real
        feature space. This is not always possible, e.g., features like
        age*height can't be mapped back to a single feature but features like
        country_US, country_Germany can. In the former case, routine will plot
        loss per effective feature; in the latter, loss per input feature.

    :param normalize: If True, divide loss by number of features such that the
        sum is the total loss. Otherwise, the plotted loss contributions must be
        averaged to obtain the total loss.

    :param plot_params: Parameters passed on to seaborn routines.
    """
    if reference_population is None:
        reference_population = _get_reference_population(matching_data)

    default_params = {
        "hue": matching_data.population_col,
        "hue_order": _get_default_hue_order(matching_data),
    }
    default_params.update(plot_params)

    # Map original features to effective features, if requested and possible
    effective_features = balance_calculator.preprocessor.output_headers["all"]
    input_output_column_mapping = dict((c, [c]) for c in effective_features)
    if debin:
        try:
            input_output_column_mapping = dict(
                (c, balance_calculator.preprocessor.get_feature_names_out(c))
                for c in matching_data.headers["all"]
            )
        except NotImplementedError:
            raise NotImplementedError(
                "Debinning not possible with given balance calculator."
            )

    indices = _debin_features(effective_features, input_output_column_mapping)

    # Compute total loss and per feature loss for all populations
    total_losses = {}
    records = []
    for population in default_params["hue_order"]:
        data = matching_data.get_population(population)
        total_losses[population] = balance_calculator.distance(data)

        per_effective_feature_loss = np.abs(
            balance_calculator.per_feature_loss(data).cpu().numpy()[0]
        )
        if normalize:
            per_effective_feature_loss /= len(
                effective_features
            )  # Normalize per feature

        # Aggregate loss per physical feature
        per_feature_loss = []
        for feature in indices.keys():
            per_feature_loss.append(per_effective_feature_loss[indices[feature]].sum())

        df = pd.DataFrame.from_dict(
            {
                "mismatch": per_feature_loss,
                "feature": list(indices.keys()),
                matching_data.population_col: [population] * len(indices.keys()),
            }
        )
        # Sort by descending mismatch wrt pool
        if population == reference_population:
            df = df.sort_values("mismatch", ascending=False)
        records.append(df)
    records = pd.concat(records)

    # Plot results
    fig = plt.figure(figsize=(len(indices.keys()) / 2, 6))
    ax = sns.barplot(data=records, x="feature", y="mismatch", **default_params)

    plt.grid(axis="y")
    handles, labels = fig.gca().get_legend_handles_labels()
    labels = [
        f"{g} ({balance_calculator.name} = {total_losses[g]:.3f})"
        for g in default_params["hue_order"]
    ]

    fig.gca().legend(handles, labels)
    plt.title(f"Contribution to {balance_calculator.name} by feature")

    plt.ylim(ymin=0)
    ymin, ymax = fig.gca().get_ylim()
    plt.vlines(plt.xticks()[0] + 0.5, ymin, ymax, linewidth=0.5, color="k")
    plt.vlines(plt.xticks()[0][0] - 0.5, ymin, ymax, linewidth=0.5, color="k")

    plt.xticks(rotation=90)
    xmin, xmax = plt.xticks()[0][0] - 0.5, plt.xticks()[0][-1] + 0.5
    plt.hlines(0.1, xmin, xmax, linewidth=2.5, color="k", linestyle="--")
    plt.xlim(xmin, xmax)

    return fig


def plot_joint_numeric_categoric_distributions(
    matching_data: MatchingData,
    include_only_numeric: Optional[List[str]] = None,
    include_only_categoric: Optional[List[str]] = None,
    **plot_params,
):
    """
    Plot 2D distributions of pairs of numeric and categoric features from
    matching_data. Choose subsets of features using include_only. Additional
    keyword arguments are passed to sns.JointGrid and override default.
    """
    default_params = {
        "hue": matching_data.population_col,
        "hue_order": _get_default_hue_order(matching_data),
    }
    default_params.update(plot_params)
    grids = []

    headers_numeric = include_only_numeric or matching_data.headers["numeric"]
    headers_categoric = include_only_categoric or matching_data.headers["categoric"]

    for x, y in itertools.product(headers_categoric, headers_numeric):
        g = sns.JointGrid(data=matching_data.data, x=x, y=y, **default_params)
        grids.append(g)

        g.plot_joint(sns.violinplot, s=25, split=True, saturation=0.9, dodge=True)
        g.ax_joint.grid(True)

        sns.histplot(
            data=matching_data.data,
            x=x,
            multiple="dodge",
            shrink=0.8,
            common_norm=False,
            discrete=True,
            stat="probability",
            ax=g.ax_marg_x,
            **default_params,
        )
        sns.histplot(
            data=matching_data.data,
            y=y,
            common_norm=False,
            stat="probability",
            ax=g.ax_marg_y,
            **default_params,
        )

        g.ax_marg_x.legend_.remove()
        g.ax_marg_y.legend_.remove()

    return grids


def plot_joint_numeric_distributions(
    matching_data: MatchingData,
    joint_kind: str = "kde",
    include_only: Optional[List[str]] = None,
    **plot_params,
):
    """
    Plot 2D distributions of pairs of numeric features from matching_data.
    joint_kind can be either kde or scatter. scatter is usually a bad choice
    for large datasets.  Choose subsets of features using include_only. Additional keyword arguments are passed to sns.JointGrid
    and override default.
    """
    default_params = {
        "hue": matching_data.population_col,
        "hue_order": _get_default_hue_order(matching_data),
    }
    default_params.update(plot_params)
    grids = []

    headers = include_only or matching_data.headers["numeric"]

    for x, y in itertools.combinations(headers, 2):
        g = sns.JointGrid(data=matching_data.data, x=x, y=y, **default_params)
        grids.append(g)

        if joint_kind == "kde":
            g.plot_joint(sns.kdeplot, levels=5)

        elif joint_kind == "scatter":
            # Give larger populations more alpha, so they don't overwhelm the plot
            counts = matching_data.counts()
            weights = (1 - counts / counts.sum()).reset_index()
            alpha = np.clip(
                matching_data.data.merge(weights, on=matching_data.population_col)[
                    "N"
                ].values,
                0.25,
                1,
            )
            g.plot_joint(sns.scatterplot, s=25, alpha=alpha)

        else:
            raise NotImplementedError(f"Unsupported joint_kind: {joint_kind}.")

        g.ax_joint.grid(True)
        g.plot_marginals(sns.histplot, bins=20, common_norm=False, stat="probability")

    return grids
