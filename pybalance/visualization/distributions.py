"""
Some helpful functions for plotting distribution.
"""

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
    # matching_data.populations includes the target name even for an
    # AggregateTarget (which has no rows); using it as a hue_order would give
    # seaborn a category with no data, i.e. a phantom, empty legend entry.
    return sorted(matching_data.data[matching_data.population_col].unique().tolist())


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


def _merge_legend(ax, new_handles, new_labels):
    """
    Add new_handles/new_labels to whatever legend is already on ax (e.g. the
    one seaborn's histplot creates for hue). Seaborn builds its legend from
    proxy handles passed directly to ax.legend(handles=..., labels=...)
    rather than from labeled artists, so a plain ax.legend() call here would
    not pick them up -- it would silently replace them with only new_handles.
    Placed above the axes (rather than seaborn's default "best" location)
    since a categoric probability plot often has bars filling the full
    y-range, leaving no in-axes corner free of data.
    """
    existing = ax.get_legend()
    if existing is not None:
        # matplotlib >=3.7 renamed Legend.legendHandles to legend_handles.
        if hasattr(existing, "legend_handles"):
            handles = list(existing.legend_handles)
        else:
            handles = list(existing.legendHandles)
        labels = [t.get_text() for t in existing.get_texts()]
    else:
        handles, labels = [], []
    ax.legend(
        handles=handles + new_handles,
        labels=labels + new_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )


def _overlay_aggregate_target_numeric(fig, matching_data, headers):
    """
    Draw the AggregateTarget's mean (and std, if disclosed) on each numeric
    subplot, since an aggregate target has no patient-level rows for
    histplot to draw in the first place.
    """
    target_name = matching_data.target_name
    for j, feature in enumerate(headers):
        stats = matching_data.aggregate_target.numeric.get(feature)
        if stats is None:
            continue
        ax = fig.axes[j]
        handles, labels = [], []
        if "mean" in stats:
            handles.append(
                ax.axvline(stats["mean"], color="k", linestyle="--", linewidth=2)
            )
            labels.append(f"{target_name} mean")
            if "std" in stats:
                handles.append(
                    ax.axvspan(
                        stats["mean"] - stats["std"],
                        stats["mean"] + stats["std"],
                        color="k",
                        alpha=0.12,
                    )
                )
                labels.append(f"{target_name} mean \u00b1 std")
        # A disclosed quantile (q, value) says P(raw <= value) = q: mark the cutpoint.
        for q, value in stats.get("quantile", []):
            handles.append(ax.axvline(value, color="k", linestyle=":", linewidth=2))
            label = {0.0: "min", 1.0: "max"}.get(q, f"q={q:g}")
            labels.append(f"{target_name} {label} ({value:g})")
        _merge_legend(ax, handles, labels)


def _overlay_aggregate_target_categoric(fig, matching_data, headers):
    """
    Draw the AggregateTarget's rate per level on each categoric subplot,
    since an aggregate target has no patient-level rows for histplot to draw
    in the first place.
    """
    target_name = matching_data.target_name
    for j, feature in enumerate(headers):
        rates = matching_data.aggregate_target.categoric.get(feature)
        if not rates:
            continue
        ax = fig.axes[j]
        levels = list(rates.keys())
        values = [rates[level] for level in levels]
        scatter = ax.scatter(levels, values, marker="D", color="k", s=60, zorder=5)
        _merge_legend(ax, [scatter], [f"{target_name} rate"])


def plot_categoric_features(
    matching_data: MatchingData,
    col_wrap: int = 2,
    height: float = 6,
    include_binary=True,
    include_only: Optional[List[str]] = None,
    **plot_params,
) -> plt.Figure:
    """
    Plot the one-dimensional marginal distributions for all categoric features
    and all treatment groups found in matching_data. Extra keyword arguments are
    passed to seaborn.histplot and override defaults.

    If matching_data has an AggregateTarget, it has no patient-level rows to
    plot a distribution for; instead, each feature's disclosed rate(s) are
    overlaid as diamond markers.

    :param matching_data: MatchingData instance containing at least one population.
    :param include_binary: Whether to include binary features in the plot.
    :param include_only: List of features to consider for plotting. Otherwise,
        all categoric features are plotted. If include_binary is False, binary
        features are excluded, even if present in include_only.
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
    if include_only is None:
        headers = matching_data.headers["categoric"]
    else:
        headers = include_only

    if not include_binary:
        headers = [c for c in headers if matching_data[c].nunique() > 2]

    # PLOT!
    fig = _plot_1d_marginals(matching_data, headers, col_wrap, height, **default_params)
    [
        fig.axes[j].set_xticks(matching_data[col].unique())
        for j, col in enumerate(headers)
    ]
    if matching_data.has_aggregate_target:
        _overlay_aggregate_target_categoric(fig, matching_data, headers)

    return fig


def plot_numeric_features(
    matching_data: MatchingData,
    col_wrap: int = 2,
    height: float = 6,
    include_only: Optional[List[str]] = None,
    **plot_params,
) -> plt.Figure:
    """
    Plot the one-dimensional marginal distributions for all numerical features
    and all treatment groups found in matching_data. Extra keyword arguments are
    passed to seaborn.histplot and override defaults.

    If matching_data has an AggregateTarget, it has no patient-level rows to
    plot a distribution for; instead, each feature's disclosed mean is
    overlaid as a dashed vertical line, and its mean +/- std (when disclosed)
    as a shaded band.

    :param matching_data: MatchingData instance containing at least one population.
    :param include_only: List of features to consider for plotting. Otherwise,
        all numeric features are plotted.
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
    if include_only is None:
        headers = matching_data.headers["numeric"]
    else:
        headers = include_only

    # PLOT!
    fig = _plot_1d_marginals(matching_data, headers, col_wrap, height, **default_params)
    if matching_data.has_aggregate_target:
        _overlay_aggregate_target_numeric(fig, matching_data, headers)

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
    # FIXME this should be renamed to something like plot_difference_binary_features
    if len(matching_data.populations) < 2:
        raise ValueError(
            "plot_binary_features() only implemented for MatchingData with >= 2 populations."
        )

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
        difference_label = "Std. Mean\nDifference"
    else:
        difference_label = "Abs. Mean\nDifference"

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
            figsize=(len(pool_frequencies) / 2, 8),
            gridspec_kw={"height_ratios": [1, 3]},
        )

        plt.subplot(2, 1, 1)
        sns.barplot(data=frequencies, y="difference", x="index", **default_params)
        ticks, labels = plt.xticks()
        ticks = np.array(ticks)
        plt.gca().set_xticks(ticks + 0.5, minor=False)
        plt.gca().set_xticklabels([""] * len(labels), minor=True)
        plt.gca().set_xticklabels([""] * len(labels), minor=False)
        plt.grid(True)
        plt.axhline(
            y=0.1,
            xmin=ticks.min(),
            xmax=ticks.max(),
            c="k",
            lw=2.5,
            zorder=10000,
            linestyle="--",
        )
        plt.ylim([0, 0.25])
        plt.ylabel(difference_label, fontsize=14)
        plt.gca().get_legend().remove()
        plt.xlabel("")

        plt.subplot(2, 1, 2)
        sns.barplot(data=frequencies, y="value", x="index", **default_params)
        ticks, labels = plt.xticks()
        ticks = np.array(ticks)
        plt.gca().set_xticks(ticks, minor=True)
        plt.gca().set_xticklabels(labels, minor=True)
        plt.gca().set_xticks(ticks + 0.5, minor=False)
        plt.gca().set_xticklabels([""] * len(labels), minor=False)
        plt.xticks(rotation=90, fontsize=12, ha="right", minor=True)
        plt.grid(True)
        plt.ylabel("Frequency", fontsize=14)
        plt.xlabel("Feature", fontsize=14)

    else:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(8, len(pool_frequencies) / 2),
            gridspec_kw={"width_ratios": [3, 1]},
        )

        plt.subplot(1, 2, 2)
        sns.barplot(data=frequencies, x="difference", y="index", **default_params)
        ticks, labels = plt.yticks()
        ticks = np.array(ticks)
        plt.gca().set_yticks(ticks + 0.5, minor=False)
        plt.gca().set_yticklabels([""] * len(labels), minor=True)
        plt.gca().set_yticklabels([""] * len(labels), minor=False)
        plt.grid(True)
        plt.axvline(
            x=0.1, ymin=ticks.min(), ymax=ticks.max(), c="k", lw=2.5, linestyle="--"
        )
        plt.xlim([0, 0.25])
        plt.xlabel(difference_label, fontsize=14)
        plt.gca().get_legend().remove()
        plt.ylabel("")

        plt.subplot(1, 2, 1)
        sns.barplot(data=frequencies, x="value", y="index", **default_params)
        ticks, labels = plt.yticks()
        ticks = np.array(ticks)
        plt.gca().set_yticks(ticks, minor=True)
        plt.gca().set_yticklabels(labels, minor=True)
        plt.gca().set_yticks(ticks + 0.5, minor=False)
        plt.gca().set_yticklabels([""] * len(labels), minor=False)
        plt.yticks(rotation=0, fontsize=12, minor=True)
        plt.grid(True)
        plt.xlabel("Frequency", fontsize=14)
        plt.ylabel("Feature", fontsize=14)

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
    figsize = default_params.pop("figsize", (len(indices.keys()) / 2, 6))
    fig = plt.figure(figsize=figsize)
    sns.barplot(data=records, x="feature", y="mismatch", ax=fig.gca(), **default_params)

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
    plt.vlines(np.array(plt.xticks()[0]) + 0.5, ymin, ymax, linewidth=0.5, color="k")
    plt.vlines(np.array(plt.xticks()[0]) - 0.5, ymin, ymax, linewidth=0.5, color="k")

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

        g.plot_joint(sns.violinplot, split=True, saturation=0.9, dodge=True)
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
            g.plot_joint(sns.scatterplot, s=25)

        else:
            raise NotImplementedError(f"Unsupported joint_kind: {joint_kind}.")

        g.ax_joint.grid(True)
        g.plot_marginals(sns.histplot, bins=20, common_norm=False, stat="probability")

    return grids


def _fractional_differences(
    matching_data: MatchingData, include_only: Optional[List[str]] = None
) -> pd.Series:
    """
    Fractional difference between pool and target for each *matched moment*:
    (pool - target) / target. Indexed by a (feature, moment) MultiIndex so
    that a feature's related rows (e.g. its mean and std) can be grouped
    together by the caller rather than scattered based on magnitude alone.

    A numeric feature contributes a "mean" row, plus a separate "std" row --
    but only when the target actually has/discloses one (an AggregateTarget
    may only disclose a mean; a patient-level target always has both). A
    binary categoric feature contributes one "rate" row, for its "1" level
    (the complementary rate is implied). A categoric feature with more than
    two levels contributes one row per level (moment = the level), since no
    single number can summarize a multi-category mismatch. Built entirely on
    top of MatchingData.describe(), so this shows a row for every moment that
    was actually available to constrain against, and works identically
    whether the target is patient-level or an AggregateTarget.

    Unlike a standardized mean difference, this makes no assumption about --
    and needs no knowledge of -- the target's variance, which an
    AggregateTarget frequently does not disclose.
    """
    described = matching_data.describe(normalize=True)
    if include_only:
        features = include_only
    elif matching_data.has_aggregate_target:
        # features the target does not disclose have nothing to compare against
        features = matching_data.aggregate_target.feature_names
    else:
        features = matching_data.headers.all

    def fractional_diff(pool_value, target_value):
        return (pool_value - target_value) / target_value if target_value else np.nan

    def format_level(level):
        # A level disclosed on an AggregateTarget but absent from the pool
        # (e.g. a category the pool structurally never contains) can come
        # back from describe() as a float (e.g. 0.0) even when sibling levels
        # are ints, due to how pandas enlarges a MultiIndex row-by-row; strip
        # the ".0" so labels stay consistent.
        if isinstance(level, float) and level.is_integer():
            return str(int(level))
        return str(level)

    diffs = {}
    for feature in features:
        rows = described.loc[feature]
        if feature in matching_data.headers.numeric:
            diffs[(feature, "mean")] = fractional_diff(
                rows.loc["mean", matching_data.pool_name],
                rows.loc["mean", matching_data.target_name],
            )
            target_std = rows.loc["std", matching_data.target_name]
            if pd.notna(target_std):
                diffs[(feature, "std")] = fractional_diff(
                    rows.loc["std", matching_data.pool_name], target_std
                )
        elif len(rows.index) <= 2:
            level = 1 if 1 in rows.index else rows.index[0]
            diffs[(feature, "rate")] = fractional_diff(
                rows.loc[level, matching_data.pool_name],
                rows.loc[level, matching_data.target_name],
            )
        else:
            for level in rows.index:
                diffs[(feature, format_level(level))] = fractional_diff(
                    rows.loc[level, matching_data.pool_name],
                    rows.loc[level, matching_data.target_name],
                )

    index = pd.MultiIndex.from_tuples(diffs.keys(), names=["feature", "moment"])
    return pd.Series(list(diffs.values()), index=index, name="fractional_difference")


def _format_fractional_difference_label(feature: str, moment: str) -> str:
    if moment in ("mean", "rate"):
        return feature
    if moment == "std":
        return f"{feature} (std)"
    return f"{feature}={moment}"


def plot_fractional_difference(
    before: MatchingData,
    after: MatchingData,
    include_only: Optional[List[str]] = None,
    clip: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot the fractional difference between pool and target -- (pool - target)
    / target -- for each *matched moment*, before vs. after matching: one dot
    per numeric feature's mean, plus a separate "<feature> (std)" dot for its
    standard deviation whenever the target has/discloses one; a binary
    categoric feature gets one dot (its rate), while a categoric feature with
    more than two levels gets one "<feature>=<level>" dot per level. A
    feature's rows are always kept adjacent (e.g. mean directly next to std),
    with feature groups ordered by their worst mismatch so poorly-balanced
    features are still easy to spot. Dashed reference lines at +/-10% mark
    the usual "good balance" threshold.

    Unlike a standardized mean difference, this requires no knowledge of the
    target's variance, so it is well defined even when the target only
    discloses a mean/proportion (the common case for a published Table 1).

    Works identically for a patient-level or aggregate target, since it is
    built entirely on top of MatchingData.describe().

    :param before: MatchingData with the (unmatched) pool and target, e.g. what
        was passed into ConstraintSatisfactionMatcher.
    :param after: MatchingData with the matched pool and target, e.g. the
        return value of ConstraintSatisfactionMatcher.match().
    :param include_only: Restrict to these features; otherwise uses all of
        before.headers.
    :param clip: If supplied, clip fractional differences to [-clip, clip]
        before plotting (e.g. clip=1 caps at +/-100%). A feature whose target
        value is close to zero can otherwise blow up the x-axis and squash
        every other feature's bar to look "balanced" by comparison. Off by
        default so the raw values are shown.
    :param ax: Existing axes to draw on. If not supplied, a new figure/axes is
        created, sized to fit the number of features.
    """
    diffs = pd.DataFrame(
        {
            "before": _fractional_differences(before, include_only),
            "after": _fractional_differences(after, include_only),
        }
    )

    # Group each feature's rows together (e.g. mean next to std, or a
    # multi-level categoric feature's levels next to each other) rather than
    # interleaving them; order the groups by their worst (largest-magnitude)
    # "before" mismatch so the least-balanced features are still easy to spot.
    group_order = (
        diffs["before"].abs().groupby(level="feature").max().sort_values().index
    )
    diffs = pd.concat(
        [
            diffs.xs(feature, level="feature", drop_level=False)
            for feature in group_order
        ]
    )

    if clip is not None:
        diffs = diffs.clip(lower=-clip, upper=clip)

    labels = [
        _format_fractional_difference_label(feature, moment)
        for feature, moment in diffs.index
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(diffs) + 1))
    else:
        fig = ax.figure

    y = np.arange(len(diffs))
    ax.hlines(y, diffs["after"], diffs["before"], color="grey", linewidth=1, zorder=1)
    ax.scatter(diffs["before"], y, label="before matching", zorder=2)
    ax.scatter(diffs["after"], y, label="after matching", zorder=2)
    ax.axvline(0, color="k", linewidth=1)
    ax.axvline(0.1, linestyle="--", color="k", linewidth=1)
    ax.axvline(-0.1, linestyle="--", color="k", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Fractional Difference, (pool - target) / target")
    ax.set_title("Covariate Balance")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_aggregate_target_match(
    before: MatchingData,
    after: MatchingData,
    include_only: Optional[List[str]] = None,
    col_wrap: int = 4,
    tolerance: float = 0.1,
    quantiles_as: str = "value",
    weights: Optional[str] = None,
) -> plt.Figure:
    """
    Show, for every constraint an AggregateTarget discloses, where the pool
    lands before vs. after matching relative to the disclosed target value.

    One small panel per constraint (a numeric feature's mean / std, a
    disclosed quantile, a disclosed categoric level's rate; a ``median`` /
    ``min`` / ``max`` is the 0.5 / 0 / 1 quantile). Each panel has its own
    y-axis in the constraint's natural units: the dashed line is the disclosed
    target value, the shaded band is +/- ``tolerance`` around it, and the two
    dots are the pool before and after matching. A successful match moves the
    "after" dot onto the dashed line. Categoric levels the target does not
    disclose are unconstrained and not shown. See
    ``pybalance.utils.aggregate_target_constraints`` for the underlying table.

    :param before: MatchingData with the unmatched pool and the AggregateTarget.
    :param after: MatchingData returned by the matcher (same AggregateTarget).
    :param include_only: Restrict to these features.
    :param col_wrap: Number of panels per row.
    :param tolerance: Half-width of the shaded band, as a fraction of the target value.
    :param quantiles_as: ``"value"`` (default) fixes the quantile and shows the
        pool's e.g. 75th percentile -- in the feature's units -- against the
        disclosed one. ``"proportion"`` fixes the disclosed cutpoint instead and
        shows the fraction of the pool at or below it against the disclosed
        quantile ``q``.
    :param weights: Name of the pool column in ``after`` holding patient weights
        (e.g. ``"sample_weight"`` from a Weighter): the "after" dots are then
        the *weighted* pool, compared with the unweighted ``before``. A
        disclosed ``min`` / ``max`` is plotted as the weighted fraction beyond
        it, since weights stay positive and the extreme value itself never
        moves.
    """
    from pybalance.utils.aggregate import aggregate_target_constraints

    weighted = weights is not None
    merged = aggregate_target_constraints(
        before, quantiles_as, limits_as_proportion=weighted
    ).merge(
        aggregate_target_constraints(
            after, quantiles_as, weights=weights, limits_as_proportion=weighted
        ),
        on=["feature", "constraint", "target"],
        suffixes=("_before", "_after"),
    )
    if include_only is not None:
        merged = merged[merged["feature"].isin(include_only)]
    if merged.empty:
        raise ValueError("No constraints to plot.")

    ncols = min(col_wrap, len(merged))
    nrows = 1 + (len(merged) - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows), squeeze=False
    )

    for ax, (_, row) in zip(axes.flat, merged.iterrows()):
        target, b, a = row["target"], row["pool_before"], row["pool_after"]
        # Keep the band and both dots in view with a margin, instead of
        # starting at zero, so small residual errors are visible.
        half = max(1.25 * max(abs(b - target), abs(a - target)), tolerance * abs(target) * 1.5, 1e-9)
        ax.axhspan(
            target - tolerance * abs(target),
            target + tolerance * abs(target),
            color="k",
            alpha=0.1,
        )
        ax.axhline(target, color="k", linestyle="--", linewidth=1.5)
        ax.scatter([0], [b], s=70, color="tab:orange", zorder=3)
        ax.scatter([1], [a], s=70, color="tab:green", zorder=3)
        for x, v in ((0, b), (1, a)):
            ax.annotate(
                f"{v:.3g}", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8
            )
        ax.set_xlim(-0.6, 1.6)
        if row["constraint"].startswith(("P(x", "rate of")):
            # proportions
            ax.set_ylim(max(target - half, -0.05), min(target + half, 1.05))
        else:
            ax.set_ylim(target - half, target + half)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["before", "after"])
        ax.set_title(f"{row['feature']}\n{row['constraint']}  (target {target:.3g})", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(merged) :]:
        ax.set_visible(False)

    fig.suptitle("Aggregate target constraints: pool vs. disclosed target")
    fig.tight_layout()
    return fig
