from collections import deque
import numpy as np
import pandas as pd
from typing import Union, Optional
import time

import logging

logger = logging.getLogger(__name__)

from pybalance.utils.balance_calculators import (
    BalanceCalculator,
    BatchedBalanceCalculator,
)
from pybalance.utils import (
    MatchingData,
    split_target_pool,
    BaseBalanceCalculator,
)
from pybalance.genetic.initialization import GeneticMatcherInitializer
from pybalance.genetic.logger import BasicLogger
import torch


def _check_fitted(matcher):
    if matcher.best_match is None:
        raise ValueError("Matcher has not been fitted!")


def get_global_defaults(n_candidate_populations=5000):
    """
    Get a set of reasonable default values for evolutionary configuration. We
    break parameters into two groups: evolutionary, i.e., those that govern how
    the candidate populations are mixed, and initialization, i.e., those that
    govern the initial set of candidate populations.

    :param n_candidate_populations: Number of candidate populations to evolve.
    """
    #
    # Evolutionary params -- how to mix populations
    #
    config = {
        # Size of candidate population to match against reference data. If not
        # specified, will use the same size as the reference population.
        "candidate_population_size": None,
        # n_candidate_populations = number of candidate populations to simultaneously evolve.
        # n_keep_best = keep top N current best scoring candidate populations
        # n_voting_populations = form new candidate populations based on frequency of patient occurence
        # n_mutation = make individual patient swaps for top candidate populations
        "n_candidate_populations": n_candidate_populations,
        "n_keep_best": int(n_candidate_populations / 4),
        "n_voting_populations": int(n_candidate_populations / 4),
        "n_mutation": int(n_candidate_populations / 4),
        # n_generations = number of generations to evolve the candidate populations.
        # time_limit = time limit (in seconds), checked only at end of every iteration
        "n_generations": 1000,
        "n_iter_no_change": 100,
        "time_limit": None,
        "max_batch_size_gb": 2,
        "seed": 1234,
        "verbose": True,
        "log_every": 5,
    }

    #
    # Initialization params -- how to initialize the candidate populations
    #
    config["initialization"] = {
        "benchmarks": {"propensity": "include"},
        "sampling": {
            "propensity": 1.0,
            "uniform": 1.0,
        },
    }

    return config


class GeneticMatcher:
    """
    Match two populations using a genetic algorithm.

    :param matching_data: MatchingData to be matched. Must contain exactly two
        populations. The larger population will be matched to the smaller.

    :param objective: Matching objective to optimize. Can be a string
        referring to any balance calculator known to
        utils.balance_calculators.BalanceCalculator or an instance of
        BaseBalanceCalculator.

    :param time_limit: Time limit in seconds for matching. None means no limit.
        Defaults to 300 seconds.

    :param verbose: Whether to print progress and diagnostic information.

    :param candidate_population_size: Size of candidate population to match
        against reference data. If not specified, will use the same size as
        the reference population.

    :param n_candidate_populations: Number of candidate populations to
        simultaneously evolve. Defaults to 5000.

    :param n_keep_best: Keep top N current best scoring candidate populations.
        If None, defaults to n_candidate_populations / 4.

    :param n_voting_populations: Form new candidate populations based on
        frequency of patient occurrence. If None, defaults to
        n_candidate_populations / 4.

    :param n_mutation: Make individual patient swaps for top candidate
        populations. If None, defaults to n_candidate_populations / 4.

    :param n_generations: Number of generations to evolve the candidate
        populations. Defaults to 1000.

    :param n_iter_no_change: Stop if no improvement after this many iterations.
        Defaults to 100.

    :param max_batch_size_gb: Maximum batch size in GB for GPU operations.
        Defaults to 2.0.

    :param seed: Random seed for reproducibility. If None, results will vary
        between runs.

    :param log_every: Log progress every N generations. Defaults to 5.
    """

    def __init__(
        self,
        matching_data: MatchingData,
        objective: Union[str, BaseBalanceCalculator] = "beta",
        time_limit: Optional[float] = 300,
        verbose: bool = True,
        candidate_population_size: Optional[int] = None,
        n_candidate_populations: int = 5000,
        n_keep_best: Optional[int] = None,
        n_voting_populations: Optional[int] = None,
        n_mutation: Optional[int] = None,
        n_generations: int = 1000,
        n_iter_no_change: int = 100,
        max_batch_size_gb: float = 2.0,
        seed: Optional[int] = None,
        log_every: int = 5,
    ):
        self.matching_data = matching_data
        self.target, self.pool = split_target_pool(matching_data)

        if isinstance(objective, str):
            self.balance_calculator = BalanceCalculator(self.matching_data, objective)
            self.objective = objective
        else:
            self.balance_calculator = objective
            self.objective = self.balance_calculator.name

        # Build params dict from explicit parameters
        params = {
            "candidate_population_size": candidate_population_size,
            "n_candidate_populations": n_candidate_populations,
            "n_keep_best": n_keep_best,
            "n_voting_populations": n_voting_populations,
            "n_mutation": n_mutation,
            "n_generations": n_generations,
            "n_iter_no_change": n_iter_no_change,
            "time_limit": time_limit,
            "max_batch_size_gb": max_batch_size_gb,
            "seed": seed,
            "verbose": verbose,
            "log_every": log_every,
        }
        params = self._check_params(params)

        # Store all parameters as instance attributes
        self.candidate_population_size = params["candidate_population_size"]
        self.n_candidate_populations = params["n_candidate_populations"]
        self.n_keep_best = params["n_keep_best"]
        self.n_voting_populations = params["n_voting_populations"]
        self.n_mutation = params["n_mutation"]
        self.n_generations = params["n_generations"]
        self.n_iter_no_change = params["n_iter_no_change"]
        self.time_limit = params["time_limit"]
        self.max_batch_size_gb = params["max_batch_size_gb"]
        self.seed = params["seed"]
        self.verbose = params["verbose"]
        self.log_every = params["log_every"]
        self.initialization = params["initialization"]

        self.balance_calculator = BatchedBalanceCalculator(
            self.balance_calculator, self.max_batch_size_gb
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.logger = BasicLogger(self.log_every)
        logger.info(self.device)

        self._reset_best_match()

    def _check_params(self, config):
        # set default values for configuration parameters and check sanity
        # of resulting parameter set

        n_candidate_populations = config.setdefault("n_candidate_populations", 1024)
        standard_config = get_global_defaults(n_candidate_populations)

        # Only update with non-None values from config
        for key, value in config.items():
            if value is not None:
                standard_config[key] = value

        if standard_config["candidate_population_size"] is None:
            standard_config["candidate_population_size"] = len(self.target)
        if standard_config["n_keep_best"] >= standard_config["n_candidate_populations"]:
            raise ValueError(
                "GeneticMatcher cannot train if it keeps all candidate populations. \
                Either select more candidate populations or keep fewer from round to round."
            )
        if standard_config["n_keep_best"] <= 1:
            raise ValueError("n_keep_best must be > 1")

        return standard_config

    def get_params(self):
        """Return the matcher's configuration parameters as a dict."""
        return {
            "objective": self.objective,
            "candidate_population_size": self.candidate_population_size,
            "n_candidate_populations": self.n_candidate_populations,
            "n_keep_best": self.n_keep_best,
            "n_voting_populations": self.n_voting_populations,
            "n_mutation": self.n_mutation,
            "n_generations": self.n_generations,
            "n_iter_no_change": self.n_iter_no_change,
            "time_limit": self.time_limit,
            "max_batch_size_gb": self.max_batch_size_gb,
            "seed": self.seed,
            "verbose": self.verbose,
            "log_every": self.log_every,
        }

    def _reset_best_match(self):
        self.best_match = None
        self.best_match_idx = None
        self.best_score = np.inf
        self.balance = None
        self._recent_balance = deque([], maxlen=self.n_iter_no_change)
        self.generation = 0
        self.elapsed_time = 0

    def match(self) -> MatchingData:
        """
        Match populations passed during __init__(). Returns MatchingData
        instance containing the matched pool and target populations.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        t0 = time.time()
        self._init_first_generation()
        stop = self._check_stopping_conditions()
        while not stop:
            self._log()
            self._generate_offspring()
            self.elapsed_time = time.time() - t0
            stop = self._check_stopping_conditions()

        self._log(finalize=True)
        return self.get_best_match()

    def _init_first_generation(self):
        initializer = GeneticMatcherInitializer(self)
        candidate_populations = initializer.initialize(self.n_candidate_populations)
        self.candidate_populations = torch.from_numpy(
            np.array(candidate_populations)
        ).to(self.device)

    def _check_stopping_conditions(self):
        best_match = max(self.balance)
        self._recent_balance.append(best_match)
        if best_match == 0:
            logger.info("Optimal solution found! Stopping")
            return True
        if (len(self._recent_balance) == self._recent_balance.maxlen) and (
            best_match == self._recent_balance[0]
        ):
            logger.info(
                f"No improvement in last {self._recent_balance.maxlen} iterations. Stopping."
            )
            return True
        if self.generation == self.n_generations:
            logger.info("Reached maximum number of generations. Stopping.")
            return True
        if self.time_limit is not None and self.elapsed_time > self.time_limit:
            logger.info("Time limit exceeded. Stopping.")
            return True
        return False

    def _log(self, finalize=False):
        if self.logger is not None:
            self.logger.on_generation_end(self)
            if finalize:
                self.logger.on_matching_end(self)

    def _generate_offspring(self):
        """
        Create a new set of candidate_populations by preferentially mating
        fitter i.e. more similar to target candidate_populations. Higher balance
        is more likely to mate
        """
        self.generation += 1

        # keep the best N groups so as to not regress
        # Note that higher balance values are better, so we take the candidate
        # populations from the end of the list
        # FIXME n_keep_best MUST BE AT LEAST ONE OR IT WILL TAKE EVERYTHING!!!
        # balance = torch.from_numpy(self.balance).to(self.device)
        idxs_best_n_matches = torch.argsort(self.balance)[-self.n_keep_best :]
        offspring = self.candidate_populations[idxs_best_n_matches, :]

        # make individual patient swaps
        if self.n_mutation:
            offspring = torch.vstack(
                (offspring, self.generate_mutated_populations(N=self.n_mutation))
            )

        # add candidate populations formed from the entire pool of candidate populations
        # according to frequency of occurrence
        if self.n_voting_populations:
            offspring = torch.vstack(
                (offspring, self.generate_voting_populations(self.n_voting_populations))
            )

        # perform random mating
        n_remaining = self.n_candidate_populations - offspring.shape[0]
        if n_remaining:
            offspring = torch.vstack(
                (offspring, self.generate_mating_populations(N=n_remaining))
            )

        self.candidate_populations = offspring

    def generate_mating_populations(self, N=None, n_way_mating=2):
        # get the mating probabilities for all candidate populations
        # basically equivalent to rankdata which the cpu version uses
        # note 1.0 is needed to convert to float and avoid dropping the worst candpop.
        p = 1.0 + torch.argsort(self.balance)

        # randomly select groups of candidate populations to mix
        weights = p.repeat(N, 1)
        mating_populations = torch.multinomial(
            weights, num_samples=n_way_mating, replacement=False
        )
        mated_populations = self.candidate_populations[mating_populations].reshape(
            N, n_way_mating * self.candidate_population_size
        )

        # shuffle in the last (patient) dimension
        # permutes all cols together but that's ok since the rows are uncorrelated!
        perm = torch.randperm(2 * self.candidate_population_size, device=self.device)
        mated_populations = mated_populations[:, perm]
        mated_populations = torch.vstack(
            [
                torch.unique(t)[: self.candidate_population_size]
                for t in torch.unbind(mated_populations)
            ]
        )

        return mated_populations

    def generate_voting_populations(self, N=None):
        # count patient frequency within candidate populations. this unique
        # function doesn't seem to scale (in terms of peak memory usage) so well
        # when you have a lot of large candidate populations, so here we just
        # estimate the counts based on a sample of the candidate populations
        max_candidate_populations = 1024
        sample_populations = torch.randperm(len(self.candidate_populations))
        patients, counts = torch.unique(
            self.candidate_populations[
                sample_populations[:max_candidate_populations], :
            ],
            return_counts=True,
        )
        voting_probabilities = counts / counts.sum()

        candidate_populations = torch.empty(
            size=(0, self.candidate_population_size),
            device=self.device,
            dtype=self.candidate_populations.dtype,
        )
        while len(candidate_populations) < N:
            # FIXME calculate the max _N based on batch size. I can't quite
            # figure out what the memory requirements of the operation below are
            # so I don't know how to do that. For now, just hard code to a nice
            # power of two
            this_N = min(max_candidate_populations, N - len(candidate_populations))
            weights = voting_probabilities.repeat(this_N, 1)
            this_candidate_populations = torch.multinomial(
                weights, num_samples=self.candidate_population_size, replacement=False
            )
            candidate_populations = torch.vstack(
                [candidate_populations, this_candidate_populations]
            )

        return candidate_populations

    def generate_mutated_populations(self, N=None, n_swap=1):
        mutaters = torch.argsort(self.balance)[-N:]
        mutated_populations = self.candidate_populations[
            mutaters
        ]  # N x candidate_population_size
        torch.ones(
            n_swap,
        )

        random_pool_patients = torch.randperm(len(self.pool), device=self.device)[
            :N
        ].reshape(N, 1)
        mutated_populations = torch.hstack([random_pool_patients, mutated_populations])

        # would love to avoid this for loop but I don't see how. For some
        # reason, the shuffling is absolutely needed. Such is life. It's still
        # faster that numpy, even on a CPU.
        out = []
        for t in torch.unbind(mutated_populations):
            t = torch.unique(t)
            t = t[torch.randperm(len(t), device=self.device)][
                : self.candidate_population_size
            ]
            out.append(t)

        mutated_populations = torch.vstack(out)

        return mutated_populations

    def _calculate_balance(self):
        self.balance = self.balance_calculator.balance(self.candidate_populations)

    @property
    def candidate_populations(self):
        return self._candidate_populations

    @candidate_populations.setter
    def candidate_populations(self, value):
        self._candidate_populations = value
        # To avoid having the balance array get out of sync with the candidate
        # populations, always recompute balance whenever updating the candidate
        # populations
        self._calculate_balance()

    def get_best_match_idxs(
        self, balance_calculator: Optional[BaseBalanceCalculator] = None
    ):
        if balance_calculator is not None:
            balance = balance_calculator.balance(self.candidate_populations)
        else:
            balance = self.balance
        idx_best_match = balance.argmax()
        return self.candidate_populations[idx_best_match, :]

    def get_best_match(
        self, balance_calculator: Optional[BaseBalanceCalculator] = None
    ) -> MatchingData:
        # TODO: Consider removing the balance_calculator parameter for consistency with
        # other matchers, or add it to all matchers for flexibility. Currently only
        # GeneticMatcher supports re-evaluating with a different balance calculator.
        best_match_patient_idxs = (
            self.get_best_match_idxs(balance_calculator).cpu().numpy()
        )
        pool = self.pool.iloc[best_match_patient_idxs]
        target = self.target
        match = MatchingData(
            data=pd.concat([target, pool]),
            headers=self.matching_data.headers,
            population_col=self.matching_data.population_col,
        )

        return match
