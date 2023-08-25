import pandas as pd
import argparse
import os
import boto3
import re
import uuid

from pybalance.utils import MatchingData
from pybalance.visualization import (
    plot_numeric_features,
    plot_binary_features,
    plot_categoric_features,
)
from pybalance.utils.matching_data import (
    _load_matching_data,
)

import logging

logger = logging.getLogger(__name__)


def _setup_matching_data(
    input_path, weights_path, dropna=True, on_missing_features="warn"
):
    data = _load_matching_data(input_path)

    if weights_path is not None:
        feature_list = pd.read_csv(weights_path, comment="#")
        weights = dict(zip(feature_list["feature"], feature_list["weight"]))
        # Assume missing features have 0 weight. FIXME don't hardcode name of
        # cols to ignore.
        missing_features = set(weights.keys()) - set(data.columns.tolist())
        if len(missing_features) > 0:
            if on_missing_features == "warn":
                logger.warning(
                    f"Matching features {''.join(missing_features)} not found! Skipping."
                )
            else:
                logger.error(
                    f"Matching features {''.join(missing_features)} not found! Cannot proceed."
                )
            for feature in missing_features:
                del weights[feature]

        data = data[["patient_id", "population"] + list(weights.keys())]
    else:
        weights = {}

    logger.info("Feature weights:")
    logger.info(weights)

    if dropna:
        logger.info("Dropping patients with missing values:")
        logger.info(f"{data.isna().sum().sort_values(ascending=False)}")
        data = data.dropna()
        logger.info(f"Remaining records: {len(data)}")

    return data, weights


def upload_file_s3(local_path, remote_path):
    # upload to s3
    output_path = remote_path[5:].strip("/")
    bucket = output_path.split("/")[0]
    path = output_path.split("/")[1:]

    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, "/".join(path))
    os.remove(local_path)


def save_mpl_fig(fig, output_path):
    logger.info(f"Saving plot to {output_path} ...")

    # Save locally
    local_path = f"{uuid.uuid4()}.png"
    fig.savefig(local_path, bbox_inches="tight")

    if output_path.startswith("s3://"):
        upload_file_s3(local_path, output_path)
    else:
        os.rename(local_path, output_path)


def save_pandas_table(df, output_path):
    # Save locally
    if output_path.endswith(".parquet"):
        local_path = f"{uuid.uuid4()}.parquet"
        df.to_parquet(local_path)
    else:
        local_path = f"{uuid.uuid4()}.csv"
        df.to_csv(local_path)

    if output_path.startswith("s3://"):
        upload_file_s3(local_path, output_path)
    else:
        os.rename(local_path, output_path)


def parse_command_line():
    parser = argparse.ArgumentParser(description="Match two patient populations.")

    parser.add_argument(
        "--features",
        action="store",
        default=None,
        help='Path to input matching data. Input must be csv or parquet and contain a column named "population_col" which indicates which population each row belongs two. All other columns (except patient_id) are treated as features unless --feature-list is specified.',
    )

    parser.add_argument(
        "--match",
        action="store",
        default=os.environ.get("POPMAP_INPUT"),
        help='Path to output matching data. Input must be csv or parquet and contain a column named "population_col" which indicates which population each row belongs two. All other columns (except patient_id) are treated as features unless --feature-list is specified.',
    )

    parser.add_argument(
        "--feature-list",
        type=str,
        action="store",
        default=os.environ.get("POPMAT_FEATURE_LIST"),
        help='A csv file with columns ["feature", "weight"], where "feature" contains the list of features to be used for matching and "weight" indicates the weight to be used for the balance calculation.',
    )

    args = parser.parse_args()

    logger.info("Parsed command line arguments:")
    for arg in vars(args):
        logger.info(f"    {arg} = {getattr(args, arg)}")

    logger.info(args)
    return args


def write_summary_tables(match, match_path):
    logger.info(f"Writing summary tables for match {match_path} ...")

    categoric_summary1 = match.describe_categoric(normalize=False)
    categoric_summary2 = match.describe_categoric()
    categoric_summary = categoric_summary1.merge(
        categoric_summary2, left_index=True, right_index=True, suffixes=["_cnt", "_pct"]
    )
    save_pandas_table(
        categoric_summary, re.sub(".csv|.parquet", "_categoric_summary.csv", match_path)
    )

    numeric_summary = match.describe_numeric()
    save_pandas_table(
        numeric_summary, re.sub(".csv|.parquet", "_numeric_summary.csv", match_path)
    )

    return categoric_summary, numeric_summary


def write_summary_plots(match, match_path, feature_weights=None):
    logger.info(f"Writing summary plots for match {match_path} ...")

    logger.info("Plotting 1d marginals ...")
    fig = plot_numeric_features(match)
    save_mpl_fig(fig, re.sub(".csv|.parquet", "_1d_marginals.png", match_path))

    logger.info("Plotting binary features ...")
    fig = plot_binary_features(match, max_features=50)
    save_mpl_fig(fig, re.sub(".csv|.parquet", "_binary_features.png", match_path))

    if feature_weights is not None:
        feature_weights = pd.read_csv(feature_weights, comment="#")[
            ["feature", "weight"]
        ].dropna()
        high_priority_features = feature_weights[
            feature_weights.weight > 1
        ].feature.tolist()
        if len(high_priority_features) > 0:
            logger.info("Plotting high weight binary features ...")
            fig = plot_binary_features(match, include_only=high_priority_features)
            save_mpl_fig(
                fig,
                re.sub(".csv|.parquet", "_high_weight_binary_features.png", match_path),
            )

    logger.info("Plotting categoric features ...")
    fig = plot_categoric_features(match)
    save_mpl_fig(fig, re.sub(".csv|.parquet", "_categoric_features.png", match_path))

    # logger.info(f"Plotting standardized mean difference ...")
    # beta = BetaBalance(match)
    # fig = plot_per_feature_loss(match, beta, debin=False)
    # save_mpl_fig(
    #     fig, re.sub(".csv|.parquet", "_standardized_mean_difference.png", match_path)
    # )


def main(args):
    data, weights = _setup_matching_data(args.match, args.feature_list)
    match = MatchingData(data=data, population_col="population")

    if args.features is not None:
        prematch_data, weights = _setup_matching_data(args.features, args.feature_list)
        prematch_data.loc[:, "population"] = (
            prematch_data["population"] + " (pre-match)"
        )
        match.append(prematch_data)

    write_summary_tables(match, args.match)
    write_summary_plots(match, args.match, args.feature_list)


if __name__ == "__main__":
    args = parse_command_line()
    main(args)
