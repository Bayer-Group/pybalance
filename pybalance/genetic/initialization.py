import numpy as np
from scipy import optimize
from scipy.spatial import distance
import torch

import logging

logger = logging.getLogger(__name__)

from pybalance.propensity import (
    PropensityScoreMatcher,
    propensity_score_match,
)
from pybalance.utils import split_target_pool


class GeneticMatcherInitializer:
    def __init__(self, matcher):
        self.matcher = matcher
        self.verbose = matcher.params["verbose"]
        self.params = matcher.params["initialization"]
        self.matcher_params = matcher.params
        self.balance_calculator = matcher.balance_calculator
        self._propensity = None

    def initialize(self, n_candidate_populations, seed=None):
        """
        Initialize genetic algorithm. In the genetic algorithm, we take a large
        number of candidate patient groups and evolve them under selection
        pressure from a balance function. The initial population is constructed
        here.
        """
        if self.verbose:
            logging.info("Optimizing balance with genetic algorithm ...")
            logging.info("Initial balance scores:")
            candidate_population = split_target_pool(
                self.balance_calculator.matching_data
            )[1]
            benchmark = self.balance_calculator.distance(candidate_population)
            logging.info(f"\t{self.balance_calculator.name}:\t{benchmark:3.3f}")
            logging.info("Initializing candidate populations ...")

        # initialize candidate populations
        if not seed:
            candidate_populations = []
        else:
            candidate_populations = seed

        # computing reference benchmarks and add to initial populations if requested by user
        benchmarks = self.compute_benchmark_populations()
        candidate_populations.extend(benchmarks)

        # randomly sample remaining candidate populations
        n_remaining = n_candidate_populations - len(candidate_populations)
        n_samples = self.split_samples_among_methods(
            self.params["sampling"], n_remaining
        )
        for sampling_method, N in n_samples.items():
            candidate_populations.extend(self.sample_patients(N, sampling_method))

        return candidate_populations

    def compute_benchmark_populations(self):
        benchmarks = []

        def choose_to_include(candidate_population, mode):
            if self.verbose:
                benchmark = self.balance_calculator.distance(candidate_population)
                logging.info(f"\t{self.balance_calculator.name}:\t{benchmark:3.3f}")

            if mode == "include":
                benchmarks.append(list(candidate_population))
                if self.verbose:
                    logging.info("\tIncluded in initial population.\n")
            elif self.verbose:
                logging.info("\tExcluded from initial population.\n")

            return

        for method, mode in self.params["benchmarks"].items():
            # possible modes are exclude, include or benchmark
            # exclude -- do no nothing
            # inlucde -- compute benchmark, add to initial population
            # benchmark -- compute benchmark, do not add to initial population
            if mode == "exclude":
                continue

            if self.verbose:
                logging.info(f"Computing {method.upper()} 1-1 matching method ...")

            if method == "propensity":
                # scipy appears to be modifying input arrays inplace - thus one MUST copy the arrays
                # here, since we are not done with propensity scores yet
                propensity_pool, propensity_target = (
                    self.propensity["pool"],
                    self.propensity["target"],
                )
                candidate_population, _ = propensity_score_match(
                    np.copy(propensity_pool),
                    np.copy(propensity_target),
                    method="greedy",
                )
                choose_to_include(candidate_population, mode)

            else:
                raise (
                    NotImplementedError(
                        f"Unknown initialization method method {method}."
                    )
                )

        return benchmarks

    def sample_patients(self, N, method="uniform"):
        candidate_populations = []
        if not N:
            # some of the downstream calculations (e.g. mahalanobis matrix) are
            # mildly expensive and so we should short circuit here if we aren't
            # actually needing any samples
            return candidate_populations

        # Get sampling probabilities for the patients from the pool
        idx = list(range(len(self.balance_calculator.pool)))
        if method == "uniform":
            init_probs = np.array([1] * len(idx))
        elif method == "propensity":
            init_probs = self.propensity["pool"]

        # this is a hacky form of regularization, don't allow the model to be
        # over confident about excluding patients
        init_probs = np.clip(init_probs, 1e-8, 1)
        p = init_probs / init_probs.sum()

        if self.verbose:
            logging.info(
                f"Sampling {N} candidate populations according to {method.upper()} distribution ...\n"
            )

        for _ in range(N):
            candidate_population = np.random.choice(
                idx,
                size=self.matcher_params["candidate_population_size"],
                p=p,
                replace=False,
            )
            candidate_populations.append(candidate_population)

        # p = torch.FloatTensor(p)
        # weights = p.repeat(N, 1)
        # candidate_populations = torch.multinomial(weights, num_samples=self.matcher_params['candidate_population_size']).cpu().numpy()

        return candidate_populations

    def split_samples_among_methods(self, sampling_methods, N):
        """
        Spread N samples among the chosen sampling_methods to approximately the
        correct proportions.
        """
        total = sum(sampling_methods.values())
        target = N
        for method in list(sampling_methods.keys())[:-1]:
            sampling_methods[method] = round(sampling_methods[method] * N / total)
            target -= sampling_methods[method]
        method = list(sampling_methods.keys())[-1]
        sampling_methods[method] = target
        return sampling_methods

    @property
    def propensity(self):
        if self._propensity is None:
            ps_matcher = PropensityScoreMatcher(
                matching_data=self.balance_calculator.matching_data,
                objective=self.balance_calculator,
            )
            ps_matcher.match()
            propensity_pool, propensity_target = ps_matcher.get_propensity_score()
            self._propensity = {"pool": propensity_pool, "target": propensity_target}
        return self._propensity
