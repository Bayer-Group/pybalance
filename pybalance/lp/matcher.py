from ortools.sat.python import cp_model
import multiprocessing
import time
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sklearn.preprocessing import MinMaxScaler

import logging

logger = logging.getLogger(__name__)

from pybalance.propensity import PropensityScoreMatcher
from pybalance.utils import (
    MatchingData,
    BaseBalanceCalculator,
    BalanceCalculator,
    split_target_pool,
)


def _check_fitted(matcher):
    if matcher.best_match is None:
        raise ValueError("Matcher has not been fitted!")


def compute_truncation_error(x: np.ndarray) -> float:
    return np.abs(x.astype(int) - x).sum() / np.abs(x).sum()


def _rescale_for_discretization(
    target: np.array, pool: np.array, tolerance: float = 0.01
):
    """
    Find a scale factor that allows one to convert datasets to integer datatypes
    with minimal loss. Input parameter tolerance specifies the max allowable
    truncation error as a fraction.

    tolerance has a somewhat surprising effect of requiring more memory to
    search. by using a lower tolerance, one can put better bounds on the
    optimization variables and I think this helps memory usage.
    """

    # Get initial truncation error
    truncation_error = max(
        compute_truncation_error(target), compute_truncation_error(pool)
    )

    # Ideally, no scaling is required at all, e.g., if everything is already
    # an integer. Start with an initial scaling of 1 and increase as needed
    # to find the right scale.
    scalefac = 1

    # Keep increasing the scale factor until the (worst) truncation error is
    # less than the specified tolerance.
    while truncation_error > tolerance:
        truncation_error = max(
            compute_truncation_error(target * scalefac),
            compute_truncation_error(pool * scalefac),
        )
        scalefac *= 1.2

    # We add an extra factor of 2 for good measure. Better safe than sorry
    scalefac *= 2
    logger.info(
        f"Scaling features by factor {scalefac:.2f} in order to use integer solver with <= {100 * truncation_error:.4f}% loss."
    )

    target = (scalefac * target).astype(int).tolist()
    pool = (scalefac * pool).astype(int).tolist()

    return target, pool


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, ind, deltas, matcher):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__ind = ind
        self.__deltas = deltas
        self.__solution_count = 0
        self.__start_time = time.time()
        self.solutions = []
        self.balance_scores = []
        self.matcher = matcher
        self.balance_calculator = matcher.balance_calculator
        if self.balance_calculator is not None:
            logger.info(
                f"Initial balance score: {self.balance_calculator.distance(list(range(len(self.balance_calculator.pool)))):.4f}"
            )

    def on_solution_callback(self):
        current_time = time.time()
        logger.info("=========================================")
        logger.info(
            "Solution %i, time = %0.2f m"
            % (self.__solution_count + 1, (current_time - self.__start_time) / 60)
        )
        logger.info(f"Objective:\t{self.ObjectiveValue()}")

        indices = []
        for i in range(len(self.__ind)):
            if self.Value(self.__ind[i]):
                indices.append(i)
        self.solutions.append(indices)

        if self.balance_calculator is not None:
            # ---- Get indices of patients ----#

            balance = self.balance_calculator.distance(indices)
            self.balance_scores.append(balance)
            logger.info(f"Balance:\t{balance:.4f}")

        # ---- Print Deltas per feature ----#
        logger.info(f"Patients:\t{sum([self.Value(x) for x in self.__ind])}")

        # ---- Print Deltas per feature ----#
        # Don't touch this code. You may think it is written by a crazy person,
        # but I promise ORTools will crash if you try to change it. I have no
        # idea why.
        headers = self.balance_calculator.preprocessor.output_headers["all"]
        j = 0
        for i in self.__deltas:
            feat = headers[j]
            logger.debug(f"Feature deltas: %12i {feat} " % self.Value(i))
            j += 1
        logger.info(" ")

        self.__solution_count += 1

    def solution_count(self):
        return self.__solution_count


class ConstraintSatisfactionMatcher(object):
    """
    Constructs the class patient selection class.

    :param matching_data: A MatchingData object describing the pool and target
        populations. See utils.matching_data.

    :param objective: Matching objective to optimize. Technically, you can pass
        any balance calculator, but this solver cannot handle non-linear
        objective functions. The solver uses the preprocessing from the balance
        calculator for setting up the problem; the balance calculator itself is
        used to report the balance of generated matches but not in actually
        finding solutions (since the CS solver needs a discretized objective
        function). The solver will optimize the absolute mean difference on the
        output features of the balance calculator's preprocessing.

    :param match_size: How large the match population should be. If passed as a
        float, interpreted to be a fraction of the target population.

    :param time_limit: Time limit to stop solving in seconds (def: 180 sec).

    :param num_workers: Number of workers to use to optimize objective. See
        https://github.com/google/or-tools/blob/stable/ortools/sat/sat_parameters.proto#L556
        for more detail.

    :param ps_hinting: Compute a propensity score match and use the result as a
        hint for the solver

    :param verbose: Verbose solving.
    """

    def __init__(
        self,
        matching_data: MatchingData,
        objective: Union[str, BaseBalanceCalculator] = "beta",
        match_size: int = None,
        time_limit: float = 180,
        num_workers: int = 4,
        ps_hinting: bool = False,
        verbose: bool = True,
    ):
        self.matching_data = matching_data.copy()
        self.orig_target, self.orig_pool = split_target_pool(self.matching_data)

        if isinstance(objective, str):
            self.balance_calculator = BalanceCalculator(self.matching_data, objective)
            self.objective = objective
        else:
            self.balance_calculator = objective
            self.objective = self.balance_calculator.name

        target, pool = (
            self.balance_calculator.target.cpu().numpy(),
            self.balance_calculator.pool.cpu().numpy(),
        )
        # We will have to rescale everything to use an integer solver. If
        # features are not normalized, then features with larger overall scale
        # end up being emphasized more in the objective function.
        scaler = MinMaxScaler()
        target = scaler.fit_transform(target)
        pool = scaler.transform(pool)
        self.n_target, self.n_pool = len(target), len(pool)
        self.n_features = len(target[0])
        self.weights = self.get_weights()
        self.target, self.pool = _rescale_for_discretization(
            target * self.weights, pool * self.weights, tolerance=0.01
        )

        # ===== Sanity Check ======#
        assert len(self.target[0]) == len(
            self.pool[0]
        ), "The number of features should be the same in both reference population and input!"
        assert (
            self.n_target < self.n_pool
        ), "Number of patients in treatment arm should be equal or less then control."

        self.match_size = match_size
        self.time_limit = time_limit
        self.num_workers = num_workers
        self.ps_hinting = ps_hinting
        self.verbose = verbose
        self._check_params()

        self.ref_features = [
            sum([t[f] for t in self.target]) for f in range(self.n_features)
        ]
        self._reset_best_match()

    def _check_params(self):
        # set match size
        if self.match_size is not None:
            if not self.match_size <= self.n_target:
                raise ValueError(
                    "Match size must not be greater than target population size."
                )
            if self.match_size <= 1:
                # Assume it represents a fraction of the target size
                self.match_size = int(self.match_size * self.n_target)
        else:
            self.match_size = self.n_target

    def get_params(self):
        params = [
            "objective",
            "match_size",
            "time_limit",
            "num_workers",
            "ps_hinting",
            "verbose",
        ]
        return dict((p, getattr(self, p)) for p in params)

    def _reset_best_match(self):
        self.best_match = None
        self.best_match_idx = None
        self.best_score = np.inf

    def get_weights(self):
        if hasattr(self.balance_calculator.preprocessor, "feature_weights"):
            weights = self.balance_calculator.preprocessor.feature_weights.cpu().numpy()
            min_nonzero_weight = min([w for w in weights if w > 0])
            weights = [int(10 * w / min_nonzero_weight) for w in weights]
        else:
            weights = [1] * self.n_features

        headers = self.balance_calculator.preprocessor.output_headers["all"]
        for feat, weight in zip(headers, weights):
            logger.debug(f"Feature weights: {weight:6d} {feat}")

        return weights

    def _get_limits(self, f, match_size=None):
        if match_size is None:
            match_size = self.n_target

        weight = self.weights[f]

        # sort feature values ascending
        pool_features = sorted([p[f] for p in self.pool])
        min_pool_feature = sum(pool_features[:match_size])
        max_pool_feature = sum(pool_features[-match_size:])

        target_feature_values = sorted([p[f] for p in self.target])
        min_target_feature = sum(target_feature_values[:match_size])
        max_target_feature = sum(target_feature_values[-match_size:])

        max_delta = weight * max(
            abs(max_pool_feature - min_target_feature),
            abs(max_target_feature - min_pool_feature),
        )

        if max_delta >= 2**50:
            logger.warning(f"Inferred max_delta = {max_delta}.")
            logger.warning(
                "Feature dynamic range may exceed integer dynamic range and lead to suboptimal solutions."
            )

        min_feature = min(min_target_feature, min_pool_feature)
        max_feature = max(max_target_feature, max_pool_feature)

        feat = self.balance_calculator.preprocessor.output_headers["all"][f]
        logger.debug(f"Feature limits:  [{min_feature}, {max_feature}] {feat}")

        return min_feature, max_feature, max_delta

    def match(self, hint: Optional[List[int]] = None) -> MatchingData:
        """
        Match populations passed during __init__(). Returns MatchingData
        instance containing the matched pool and target populations.

        :param hint: You can supply a "hint" as either (1) A list of indices to
            the pool. It will be assumed that the entire target is used, or (2)
            A list of two lists, the first list being the indices to the target,
            and the second being the indices to the pool, or (3) by omitting the
            hint altoghether and passing ps_hinting=True in __init__(). In case
            (3), a propensity score model will be estimated on the fly and used
            to create a match population as a hint to the solver.

            I admit the interface here is a bit confusing. We will clean this
            up in a later release.
        """
        time_limit = self.time_limit

        logger.info(
            f"Solving for match population with size = {self.match_size}. Will match on {self.n_features} dimensions ..."
        )

        # ========= Create Model===========#
        model = cp_model.CpModel()

        # ========= Variables===========#
        # --- Binary flag for inclusion of patients ---------#
        x = []  # pool
        y = []  # target
        for i in range(self.n_pool):
            x.append(model.NewBoolVar(f"x[{i}]"))
        for i in range(self.n_target):
            y.append(model.NewBoolVar(f"y[{i}]"))

        logger.info("Calculating bounds on feature variables ...")

        abs_deltas = []
        features = []
        ref_features = []
        deltas = []
        for i in range(self.n_features):
            min_feature, max_feature, max_delta = self._get_limits(i, self.match_size)
            abs_deltas.append(model.NewIntVar(0, max_delta, f"abs_deltas[{i}]"))
            features.append(model.NewIntVar(min_feature, max_feature, f"features[{i}]"))
            deltas.append(model.NewIntVar(-max_delta, max_delta, f"deltas[{i}]"))
            ref_features.append(
                model.NewIntVar(min_feature, max_feature, f"ref_features[{i}]")
            )

        logger.info("Applying constraints ...")

        # ========= Constraints===========#
        for j in range(self.n_features):
            # --- aggregate for feature j ---------#
            model.Add(
                sum(self.pool[i][j] * x[i] for i in range(self.n_pool)) == features[j]
            )
            model.Add(
                sum(self.target[i][j] * y[i] for i in range(self.n_target))
                == ref_features[j]
            )
            # --- Taxicab distance ---------#
            model.Add(self.weights[j] * (features[j] - ref_features[j]) == deltas[j])
            model.AddAbsEquality(abs_deltas[j], deltas[j])

        # --- Cardinality ---------#
        model.Add(sum(x[i] for i in range(self.n_pool)) == self.match_size)
        model.Add(sum(y[i] for i in range(self.n_target)) == self.match_size)

        # ========= Hint! ==============#
        if self.ps_hinting or hint is not None:
            matching_data = self.matching_data.copy()
            target, pool = split_target_pool(matching_data)
            target_name = target[matching_data.population_col].unique()[0]
            pool_name = pool[matching_data.population_col].unique()[0]

            if hint is None:
                logger.info("Training PS model as guide for solver ...")

                target.loc[:, "ix"] = list(range(len(target)))
                pool.loc[:, "ix"] = list(range(len(pool)))
                matching_data = MatchingData(
                    pd.concat([target, pool]), headers=self.matching_data.headers
                )
                ps_matcher = PropensityScoreMatcher(
                    matching_data, objective=self.objective
                )
                ps_match = ps_matcher.match()

                target_hint = (
                    ps_match.data[ps_match.data.population == target_name]
                    .ix.astype(int)
                    .tolist()
                )
                pool_hint = (
                    ps_match.data[ps_match.data.population == pool_name]
                    .ix.astype(int)
                    .tolist()
                )
            else:
                if len(hint) == 2:
                    target_hint, pool_hint = hint
                else:
                    target_hint = [1] * len(target)
                    pool_hint = hint

            # Kind of sanity check that the indices are aligned between what
            # comes out of the PS solver and what we will apply as hints
            obj = 0
            for j in range(self.n_features):
                feat = sum(self.pool[i][j] for i in pool_hint)
                ref = sum(self.target[i][j] for i in target_hint)
                delta = self.weights[j] * abs(feat - ref)
                obj += delta
            logger.info(f"Hint achieves objective value = {obj}.")

            logger.info("Applying hints ...")
            for i, var in enumerate(x):
                if i in pool_hint:
                    model.AddHint(var, 1)
                else:
                    model.AddHint(var, 0)

            for i, var in enumerate(y):
                if i in target_hint:
                    model.AddHint(var, 1)
                else:
                    model.AddHint(var, 0)

        # ========= Objective===========#
        model.Minimize(sum(abs_deltas))

        # ===== Creates a solver =======#
        solver = cp_model.CpSolver()
        if time_limit is not None:
            solver.parameters.max_time_in_seconds = time_limit

        solver.parameters.num_workers = min(
            self.num_workers, multiprocessing.cpu_count()
        )
        if solver.parameters.num_workers != 1:
            solver.parameters.log_search_progress = False
            solver.parameters.share_objective_bounds = True
            solver.parameters.share_level_zero_bounds = True

        logger.info("Solving with %d workers ..." % solver.parameters.num_workers)
        # ========= Solve===========#
        # ----- Append the important stuff for printing -----#

        if self.verbose:
            solution_printer = SolutionPrinter(x, abs_deltas, self)
            status = solver.Solve(model, solution_printer)
            logger.info("Status = %s" % solver.StatusName(status))
            logger.info(
                "Number of solutions found: %i" % solution_printer.solution_count()
            )
        else:
            status = solver.Solve(model)

        if solver.StatusName(status) in ["FEASIBLE", "OPTIMAL"]:
            pool_indices = [j for j, i in enumerate(x) if solver.Value(i)]
            pool = self.orig_pool.iloc[pool_indices]
            target_indices = [j for j, i in enumerate(y) if solver.Value(i)]
            target = self.orig_target.iloc[target_indices]
            match = MatchingData(
                data=pd.concat([target, pool]),
                headers=self.matching_data.headers,
                population_col=self.matching_data.population_col,
            )
        else:
            match = self.matching_data

        self.best_match = match

        return match

    def get_best_match(self):
        _check_fitted(self)
        return self.best_match
