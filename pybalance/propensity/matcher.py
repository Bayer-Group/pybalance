import time
import copy
from typing import Optional, List, Union

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from scipy.stats import loguniform, randint

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import ParameterSampler
from sklearn import preprocessing

from pybalance.utils import (
    MatchingData,
    split_target_pool,
    BaseBalanceCalculator,
    BalanceCalculator,
)

import logging

logger = logging.getLogger(__name__)


def _check_fitted(matcher):
    if matcher.best_match is None:
        raise (ValueError, "Matcher has not been fitted!")


class PropensityScoreMatcher:
    """
    Use a propensity score model to match two populations. The Matcher searches
    randomly over hyperparameters for the propensity score model and selects the
    match that performs best according to the given optimization objective.

    :param matching_data: Data containing pool and target populations to be
        matched.

    :param objective: Matching objective to optimize in hyperparameter search.
        Can be a string referring to any balance calculator known to
        utils.balance_calculators.BalanceCalculator or an instance of
        BaseBalanceCalculator.

    :param caliper: If defined, restricts matches to those patients with
        propensity scores within the caliper of each other. Note that caliper
        matching may lead to a loss of patients in the target population if no
        patient in the pool exists within the specified caliper. Should be in
        (0, 1].

    :param max_iter: Maximum number of hyperparameters to try before returning
        the best match.

    :param time_limit: Restrict hyperparameter search based on time. No new
        model will be trained after time_limit seconds have passed since
        matching began.

    :param method: Method to use for propensity score matching. Can be either
        'greedy' or 'linear_sum_assignment'. The former method is locally
        optimal and globally sub-optimial; the latter globally optimial but far
        more compute intensive. For large problems, use greedy.

    :param verbose: Flag to indicate whether to print diagnositic information
        during training.
    """

    DEFAULT_HYPERPARAM_SPACE = {
        LogisticRegression: {
            "C": loguniform(a=0.01, b=100),
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
            "fit_intercept": [True, False],
            "max_iter": [500],
        },
        SGDClassifier: {
            "penalty": ["l1", "l2", "elasticnet"],
            "alpha": loguniform(a=0.02, b=20),
            "fit_intercept": [True, False],
            "early_stopping": [True, False],
            "loss": ["log_loss", "modified_huber"],
            "max_iter": [1500],
            "class_weight": [None, "balanced"],
        },
    }

    POOL = 0
    TARGET = 1

    def __init__(
        self,
        matching_data: MatchingData,
        objective: Union[str, BaseBalanceCalculator] = "beta",
        caliper: Optional[float] = None,
        max_iter: int = 50,
        time_limit: float = 60 * 5,
        method: str = "greedy",
        verbose: bool = True,
    ):
        self.caliper = caliper
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.method = method
        self.hyperparam_space = self.DEFAULT_HYPERPARAM_SPACE
        self.verbose = verbose

        self.matching_data = matching_data.copy()
        self.target, self.pool = split_target_pool(matching_data)
        if isinstance(objective, str):
            self.balance_calculator = BalanceCalculator(self.matching_data, objective)
            self.objective = objective
        else:
            self.balance_calculator = objective
            self.objective = self.balance_calculator.name
        self._train_preprocessors(self.matching_data)
        self._reset_best_match()

    def get_params(self):
        params = ["objective", "caliper", "max_iter", "time_limit", "method"]
        return dict((p, getattr(self, p)) for p in params)

    def _reset_best_match(self):
        self.best_match = None
        self.best_match_idx = None
        self.best_score = np.inf
        self.best_model = None
        self.best_ps = None
        self.solution_time = 0

    def _describe_best_match(self):
        _check_fitted(self)
        logger.info("Best propensity score match found:")
        logger.info(f'\tModel: {str(self.best_model).split("(")[0]}')
        for key, val in self.best_params.items():
            logger.info(f"\t* {key}: {val}")
        logger.info(f"\tScore ({self.balance_calculator.name}): {self.best_score:.4f}")
        logger.info(f"\tSolution time: {self.solution_time:.3f} min")

    def _update_best_match(
        self, clf, params, match, score, ps_pool, ps_target, solution_time
    ):
        self.best_score = score
        self.best_match = match
        self.best_model = clf
        self.best_params = params
        self.best_ps = {"pool": ps_pool, "target": ps_target}
        self.solution_time = solution_time

        if self.verbose:
            self._describe_best_match()

    def get_best_match(self):
        _check_fitted(self)
        if self.verbose:
            self._describe_best_match()

        return self.best_match

    def match(self) -> MatchingData:
        """
        Match populations passed during __init__(). Returns MatchingData
        instance containing the matched pool and target populations.
        """
        t0 = time.time()

        # Search over hyperparameter search space to find a propensity score
        # that optimizes the given objective
        X, y = self._preprocess_data_for_sklearn(self.matching_data)
        hyperparams = self._get_hyperparams(self.max_iter)
        for i, (model, params) in enumerate(hyperparams):
            if (time.time() - t0) > self.time_limit:
                logger.warning("Time limit exceeded. Stopping early.")
                break

            clf = model(**params)
            logger.info(
                f'Training model {str(clf).split("(")[0]} (iter {i + 1}/{self.max_iter}, {(time.time() - t0)/60:.3f} min) ...'
            )

            clf.fit(X, y)
            ps_pool, ps_target = self.get_propensity_score(clf, self.matching_data)

            pool_matches, target_matches = propensity_score_match(
                ps_pool, ps_target, method=self.method, caliper=self.caliper
            )
            pool = self.pool.iloc[pool_matches]
            target = self.target.iloc[target_matches]
            match = MatchingData(
                pd.concat([pool, target]),
                headers=self.matching_data.headers,
                population_col=self.matching_data.population_col,
            )

            score = self.balance_calculator.distance(pool)

            if score < self.best_score:
                solution_time = (time.time() - t0) / 60
                self._update_best_match(
                    clf, params, match, score, ps_pool, ps_target, solution_time
                )

        return self.get_best_match()

    def _train_preprocessors(self, matching_data):
        self.standard_scaler = preprocessing.StandardScaler()
        X = self.balance_calculator.preprocessor.transform(matching_data)
        self.standard_scaler.fit(X.data[X.headers["all"]])

    def _preprocess_data_for_sklearn(self, matching_data):
        """
        Preprocess matching data to prepare it for estimating a propensity score.
        """
        matching_data = self.balance_calculator.preprocessor.transform(matching_data)
        target, pool = split_target_pool(matching_data)

        # construct feature data in a single array
        X = pd.concat([pool, target])[matching_data.headers["all"]]
        X = self.standard_scaler.transform(X)

        # construct target variable
        y = np.array([self.POOL] * len(pool) + [self.TARGET] * len(target))

        return X, y

    def _get_hyperparams(self, n_iter):
        hyperparams = []
        n_models = len(self.hyperparam_space)
        for model, params in self.hyperparam_space.items():
            hyperparams.extend(
                [
                    (model, p)
                    for p in ParameterSampler(params, n_iter=int(n_iter / n_models))
                ]
            )

        np.random.shuffle(hyperparams)

        return hyperparams

    def get_propensity_score(
        self, clf: BaseEstimator = None, matching_data: MatchingData = None
    ) -> tuple:
        """
        Returns the estimated propensity scores for the pool and target
        populations in a MatchingData object as predicted by a given
        propensity score model. If no model is passed, the method uses the
        best_model computed during matching. If no data are passed, the
        method uses the MatchingData passed during matching.
        """
        if clf is None:
            _check_fitted(self)
            clf = self.best_model

        if matching_data is None:
            matching_data = self.matching_data

        X, y = self._preprocess_data_for_sklearn(matching_data)
        propensity_pool = clf.predict_proba(X[y == self.POOL, :])[:, 1]
        propensity_target = clf.predict_proba(X[y == self.TARGET, :])[:, 1]

        return propensity_pool, propensity_target


def propensity_score_match_greedy(ps_pool, ps_target, caliper=None, clip=True):
    # copy array, since we will be changing it and don't want to impact caller
    ps_pool = copy.copy(ps_pool)
    if caliper is not None and clip:
        ps_pool[ps_pool > 1 - caliper] = -np.inf

    pool_matches = []
    target_matches = []
    for i, ps_pat in enumerate(ps_target):
        # find closest patient in ps_pool based on propensity score
        scores = np.abs(ps_pool - ps_pat)
        if caliper is not None:
            if all(scores > caliper) or (clip and not caliper <= ps_pat <= 1 - caliper):
                continue
        match = np.argmin(scores)
        pool_matches.append(match)
        target_matches.append(i)

        # set the propensity score for the matched patient to negative infinity
        # to ensure the patient is never matched again
        ps_pool[match] = -np.inf

    return np.array(pool_matches), np.array(target_matches)


def propensity_score_match_greedy_prio(ps_pool, ps_target, caliper=None):
    """
    An alternative greedy approach in which we still do greedy matching but we
    sort patients by how "close" they are to the pool PS distribution. Patients
    that are furthest from the pool distribution are matched first.
    """
    pool_matches, target_matches = [], []
    ps_broad = np.abs(np.subtract.outer(ps_pool, ps_target))
    order = ps_broad.mean(axis=0).argsort()[::-1]
    pool_matches, target_matches = propensity_score_match(
        ps_pool, ps_target[order], method="greedy", caliper=caliper
    )
    target_matches = order[target_matches]
    return np.array(pool_matches), np.array(target_matches)


def propensity_score_match_linear_sum_assignment(ps_pool, ps_target):
    scores = np.ones((len(ps_pool), len(ps_target)), dtype=np.float32)
    scores = np.abs(scores * np.expand_dims(ps_pool, -1) - ps_target)
    pool_matches, target_matches = optimize.linear_sum_assignment(scores)
    return pool_matches, target_matches


def propensity_score_match(ps_pool, ps_target, method="greedy", caliper=None):
    """
    Return the set of indices to use for matching an RCT to RWD based on
    propensity scores. Inputs are propensity scores, output is index to real
    world data array.

    ps_pool: np.array
        propensity scores of the population to be matched
    ps_target: np.array
        propensity scores of the reference population (usually the RCT
        population)
    method: string
        method to use for matching, can be one of the following:
           greedy -- match each RCT patient to its nearest match by propensity
           greedy_prio -- greedy but use a heuristic to sort patients and match
           the hardest patients first
           score linear_sum_assignment -- return a
           globally optimum pairing that minimizes the absolute sum
                of differences between the RCT patients and the real world
                patients

    :return: Tuple of numpy arrays pool_matches, target_matches whose elements
        indicate the indices of matched patients in the pool and target
        populations respectively.
    """
    ps_pool, ps_target = copy.deepcopy(ps_pool), copy.deepcopy(ps_target)
    if method == "greedy":
        pool_matches, target_matches = propensity_score_match_greedy(
            ps_pool, ps_target, caliper=caliper
        )
    elif method == "greedy_prio":
        pool_matches, target_matches = propensity_score_match_greedy_prio(
            ps_pool, ps_target, caliper=caliper
        )

    elif method == "linear_sum_assignment":
        if caliper is not None:
            logger.error("Caliper is not supported for linear_sum_assignment!")
        pool_matches, target_matches = propensity_score_match_linear_sum_assignment(
            ps_pool, ps_target
        )

    else:
        raise NotImplementedError(f"Matching method {method} not currently supported.")

    return pool_matches, target_matches


def plot_propensity_score_match_distributions(matcher: PropensityScoreMatcher):
    """
    Plot histograms of the estimated propensity score for pool and target
    populations pre- and post-matching.

    :param matcher: Fitted PropensityScoreMatcher model.
    """
    _check_fitted(matcher)
    m = matcher.matching_data
    match = matcher.best_match
    target, pool = split_target_pool(match)
    target_name, pool_name = (
        target[match.population_col].unique()[0],
        pool[match.population_col].unique()[0],
    )

    ps_pool_match, ps_target_match = matcher.get_propensity_score(matching_data=match)
    ps_pool, ps_target = matcher.get_propensity_score(matching_data=m)

    data = pd.concat(
        [
            pd.DataFrame.from_dict(
                {
                    "propensity": ps_pool_match,
                    "matched": True,
                    "population": pool_name,
                }
            ),
            pd.DataFrame.from_dict(
                {
                    "propensity": ps_target_match,
                    "matched": True,
                    "population": target_name,
                }
            ),
            pd.DataFrame.from_dict(
                {"propensity": ps_pool, "matched": False, "population": pool_name}
            ),
            pd.DataFrame.from_dict(
                {"propensity": ps_target, "matched": False, "population": target_name}
            ),
        ]
    )

    g = sns.FacetGrid(data=data, col="matched", height=4, xlim=[0, 1])
    g.map_dataframe(
        sns.histplot,
        bins=np.linspace(0, 1, 25),
        x="propensity",
        hue="population",
        hue_order=[pool_name, target_name],
        alpha=0.5,
        common_norm=False,
        stat="probability",
    )
    [ax.grid(True) for axes in g.axes for ax in axes]

    legend_patches = [
        matplotlib.patches.Patch(color=sns.color_palette()[0], label=pool_name),
        matplotlib.patches.Patch(color=sns.color_palette()[1], label=target_name),
    ]
    matplotlib.pyplot.legend(handles=legend_patches)

    return g


def plot_propensity_score_match_pairs(matcher: PropensityScoreMatcher):
    """
    Plot scatterplot of pool-target pairs formed by propensity score matching.

    :param matcher: Fitted PropensityScoreMatcher model.
    """
    _check_fitted(matcher)
    ps_pool, ps_target = matcher.get_propensity_score()
    pool_matches, target_matches = propensity_score_match(
        ps_pool, ps_target, method=matcher.method, caliper=matcher.caliper
    )
    plt.figure()
    plt.scatter(
        [ps_target[i] for i in target_matches], [ps_pool[i] for i in pool_matches]
    )
    plt.grid()
    plt.xlabel("PS Target")
    plt.ylabel("PS Pool (Matched)")
    plt.plot([0, 1], [0, 1])
    return plt.gcf()
