from typing import Dict
import pandas as pd
import numpy as np
import argparse
import sys
import time
import copy
import os
import boto3

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pybalance.genetic import GeneticMatcher, get_global_defaults
from pybalance.cs.matcher import ConstraintSatisfactionMatcher
from pybalance.propensity import (
    PropensityScoreMatcher,
    plot_propensity_score_match_distributions,
    plot_propensity_score_match_pairs,
)
from pybalance.utils import (
    BaseBalanceCalculator,
    BalanceCalculator,
    MatchingData,
    BALANCE_CALCULATORS,
)
from pybalance.utils.matching_data import (
    _load_matching_data,
    split_target_pool,
)
from plot_match import save_mpl_fig


def _setup_matching_data(
    input_path, feature_list_path, dropna=True, on_missing_features="warn"
):
    data = _load_matching_data(input_path)

    if feature_list_path is not None:
        feature_list = pd.read_csv(feature_list_path, comment="#")
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


def duplicate_target(matching_data, n=1):
    target, pool = split_target_pool(matching_data)
    target = copy.copy(target)

    if n is not None and n != 1:
        assert int(n) == n and n > 1
        # Not defined for other parameters in this case!
        target = pd.concat([target] * int(n))

    if len(target) > len(pool):
        raise ValueError(f"Pool is not large enough for {n}:1 matching.")

    return MatchingData(
        data=pd.concat([target, pool]),
        headers=matching_data.headers,
        population_col=matching_data.population_col,
    )


def ga_train(matching_data, bc, args):
    params = get_global_defaults(n_candidate_populations=args.n_candidate_populations)
    override_default_params = {
        "n_generations": 100000,
        "seed": 123,
        "max_batch_size_gb": 2,
        "verbose": True,
        "time_limit": args.time_limit,
        "objective": args.objective,
        "n_iter_no_change": 10000,
        "initialization": {
            "benchmarks": {"propensity": "include"},
            "sampling": {"propensity": 1.0, "uniform": 1.0},
        },
    }
    params.update(override_default_params)

    logger.info("Genetic algorithm configuration parameters:")
    logger.info(params)

    matcher = GeneticMatcher(matching_data, **params)

    if args.seed is not None:
        if args.seed.endswith(".parquet"):
            seed = pd.read_parquet(args.seed)
        else:
            seed = pd.read_csv(args.seed)

        _, pool = split_target_pool(matching_data)
        # We need to get the indices of the seed patients in the pool. Note that
        # the GA does not use the actual index of the pool but it's own internal
        # index. We probably should FIXME this, but anyway that's why we
        # reset_index() here.
        pool = pool.reset_index()
        # FIXME should not hardcode column names
        args.seed = [
            pool[pool["patient_id"].isin(seed["patient_id"])].index.values.tolist()
        ]

    match = matcher.match(args.seed)

    return match


def cs_train(matching_data, bc, args):
    params = {
        "objective": args.objective,
        "time_limit": args.time_limit,
        "parallel_solve": True,
        "verbose": True,
        "match_size": args.match_size,
        "ps_hinting": args.ps_hinting,
    }

    if args.seed is not None:
        if args.seed.endswith(".parquet"):
            seed = pd.read_parquet(args.seed)
        else:
            seed = pd.read_csv(args.seed)

        _, pool = split_target_pool(matching_data)
        # We need to get the indices of the seed patients in the pool. Note that
        # the GA does not use the actual index of the pool but it's own internal
        # index. We probably should FIXME this, but anyway that's why we
        # reset_index() here.
        pool = pool.reset_index()
        # FIXME should not hardcode column names
        args.seed = [
            pool[pool["patient_id"].isin(seed["patient_id"])].index.values.tolist()
        ][0]

    matcher = ConstraintSatisfactionMatcher(matching_data, **params)
    match = matcher.match(args.seed)

    return match


def ps_train(matching_data, bc, args):
    params = {
        "objective": args.objective,
        "time_limit": args.time_limit,
        "max_iter": 1000,
        "method": args.method,
        "caliper": args.caliper,
    }
    matcher = PropensityScoreMatcher(matching_data, **params)
    match = matcher.match()

    fig = plot_propensity_score_match_distributions(matcher)
    save_mpl_fig(
        fig,
        args.output.replace(".csv", "").replace(".parquet", "")
        + "_ps_distributions.png",
    )
    fig = plot_propensity_score_match_pairs(matcher)
    save_mpl_fig(
        fig, args.output.replace(".csv", "").replace(".parquet", "") + "_ps_pairs.png"
    )

    return match


def parse_command_line():
    parser = argparse.ArgumentParser(description="Match two patient populations.")

    parser.add_argument(
        "--output",
        action="store",
        default="match.csv",
        help="Path to store output match population.",
    )

    parser.add_argument(
        "--input",
        action="store",
        default=None,
        required=True,
        help='Path to input matching data. Input must be csv or parquet and contain a column named "population" which indicates which population each row belongs two. All other columns (except "patient_id" if present) are treated as matching features unless --feature-list is specified.',
    )

    parser.add_argument(
        "--solver",
        action="store",
        default="PS",
        choices=["GA", "CS", "PS"],
        help="Solver to use for matching.",
    )

    default_methods = {"GA": None, "PS": "greedy", "CS": "SAT_solver"}
    valid_methods = {
        "GA": [None],
        "CS": ["SAT_solver"],
        "PS": ["greedy", "greedy_prio", "linear_sum_assignment"],
    }
    parser.add_argument(
        "--method",
        action="store",
        default=None,
        choices=sum(valid_methods.values(), []),
        help="Which method to use for chosen solver.",
    )

    parser.add_argument(
        "--objective",
        action="store",
        default="beta",
        choices=list(BALANCE_CALCULATORS.keys()),
        help="Objective function to optimize.",
    )

    parser.add_argument(
        "--n-bins",
        action="store",
        default=10,
        help="For objective functions that bin numeric variables, how many bins to use.",
    )

    parser.add_argument(
        "--seed",
        action="store",
        default=None,
        help="Path to seed population. Only implemented for GA/CS solvers.",
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        action="store",
        default=3600,
    )

    parser.add_argument("--caliper", type=float, action="store", default=None)

    parser.add_argument("--n-to-one-matching", type=int, action="store", default=None)

    parser.add_argument(
        "--n-candidate-populations",
        type=int,
        action="store",
        default=os.environ.get("POPMAT_N_CANDIDATE_POPULATIONS"),
    )

    parser.add_argument(
        "--match-size",
        type=float,
        action="store",
        default=os.environ.get("POPMAT_MATCH_SIZE"),
    )

    parser.add_argument(
        "--feature-list",
        type=str,
        action="store",
        default=os.environ.get("POPMAT_FEATURE_LIST"),
        help='A csv file with columns ["feature", "weight"], where "feature" contains the list of features to be used for matching and "weight" indicates the weight to be used for the balance calculation.',
    )

    parser.add_argument(
        "--on-missing-features",
        type=str,
        action="store",
        choices=["warn", "error"],
        default="warn",
    )

    parser.add_argument("--ps-hinting", action="store_true")

    args = parser.parse_args()

    if args.solver == "CS" and args.objective in ["max_beta", "max_gamma"]:
        raise ValueError(
            f"CS solver can only be used with linear objective functions, whereas chosen objective {args.objective} is nonlinear."
        )

    if args.caliper is not None and args.solver != "PS":
        raise ValueError("Caliper argument only meaningful for PS sovler.")

    if args.method is None:
        args.method = default_methods[args.solver]
        if args.method not in valid_methods[args.solver]:
            raise ValueError(f"Unknown method {args.method} for solver {args.sovler}")

    if args.seed is not None and args.solver not in ["CS", "GA"]:
        raise ValueError(f"Seed not valid for solver {args.solver}")

    if args.solver == "GA" and args.n_candidate_populations is None:
        logger.warn("N_CANDIDATE_POPULATIONS not specified. Using 2**10 ...")
        args.n_candidate_populations = 2**10

    logger.info("Parsed command line arguments:")
    for arg in vars(args):
        logger.info(f"    {arg} = {getattr(args, arg)}")

    return args


def save_match(match, output):
    if output.startswith("s3://"):
        local_path = output.split("/")[-1]
    else:
        local_path = output

    logger.info(f"Writing to {output}...")

    if output.endswith(".parquet"):
        match.to_parquet(local_path, index=False)
    else:
        match.to_csv(local_path, index=False)

    if output.startswith("s3://"):
        # parse s3 URI
        output_path = output[5:].strip("/")
        bucket = output_path.split("/")[0]
        path = output_path.split("/")[1:]

        # upload to s3
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, "/".join(path))


def get_balance_calculator(
    matching_data: MatchingData, objective: str, feature_weights: Dict[str, float]
) -> BaseBalanceCalculator:
    if "beta" in objective:
        # For these objectives, n_bins is not defined
        bc = BalanceCalculator(
            matching_data, objective, feature_weights=feature_weights
        )
    elif objective in ["gamma", "gamma_squared"]:
        bc = BalanceCalculator(
            matching_data, objective, feature_weights=feature_weights, n_bins=10
        )
    else:
        # For these objectives, weights are not defined.
        bc = BalanceCalculator(matching_data, objective, n_bins=10)
    return bc


SOLVERS = {"GA": ga_train, "CS": cs_train, "PS": ps_train}


def main(args):
    data, feature_weights = _setup_matching_data(
        args.input, args.feature_list, args.on_missing_features
    )
    solver = SOLVERS[args.solver]
    matching_data = MatchingData(data=data, population_col="population")
    matching_data = duplicate_target(matching_data, n=args.n_to_one_matching)

    bc = get_balance_calculator(matching_data, args.objective, feature_weights)

    match = solver(matching_data, bc, args)
    if match is None:
        raise RuntimeError("No solution found!!")

    save_match(match, args.output)


if __name__ == "__main__":
    args = parse_command_line()
    main(args)
