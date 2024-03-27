from collections import deque
import numpy as np
import pandas as pd
from typing import Union
import time

import logging

logger = logging.getLogger(__name__)

from pybalance.utils.balance_calculators import (
    BalanceCalculator,
    BatchedBalanceCaclulator,
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

    :param objective: Matching objective to optimize in hyperparameter search.
        Can be a string referring to any balance calculator known to
        utils.balance_calculators.BalanceCalculator or an instance of
        BaseBalanceCalculator.

    :param params: Configuration params for the genetic matcher. See
        pybalance.genetic.get_global_defaults for a list of options.
    """

    def __init__(
        self,
        matching_data: MatchingData,
        objective: Union[str, BaseBalanceCalculator] = "beta",
        **params,
    ):
        self.matching_data = matching_data
        self.target, self.pool = split_target_pool(matching_data)

        if isinstance(objective, str):
            self.balance_calculator = BalanceCalculator(self.matching_data, objective)
            self.objective = objective
        else:
            self.balance_calculator = objective
            self.objective = self.balance_calculator.name

        params = self._check_params(params)
        self.params = {"objective": self.objective}
        self.set_params(**params)

        self.balance_calculator = BatchedBalanceCaclulator(
            self.balance_calculator, self.max_batch_size_gb
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.logger = BasicLogger(params.pop("log_every"))
        logger.info(self.device)

        self._reset_best_match()

    def _check_params(self, config):
        # set default values for configuration parameters and check sanity
        # of resulting parameter set

        n_candidate_populations = config.setdefault("n_candidate_populations", 1024)
        standard_config = get_global_defaults(n_candidate_populations)
        standard_config.update(config)

        if standard_config["candidate_population_size"] is None:
            standard_config["candidate_population_size"] = len(self.target)
        if standard_config["n_keep_best"] >= standard_config["n_candidate_populations"]:
            raise ValueError(
                "GeneticMatcher cannot train if it keeps all candidate populations. \
                Either select more candidate populations or keep fewer from round to round."
            )
        if standard_config["n_keep_best"] <= 1:
            raise ValueError("n_keep_best must be > 1")
        if standard_config["seed"] is not None:
            np.random.seed(standard_config["seed"])

        return standard_config

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        for kw, val in kwargs.items():
            setattr(self, kw, val)

    def get_params(self):
        return self.params

    def _reset_best_match(self):
        self.best_match = None
        self.best_match_idx = None
        self.best_score = np.inf
        self.balance = None
        self._recent_balance = deque([], maxlen=self.params["n_iter_no_change"])
        self.generation = 0
        self.elapsed_time = 0

    def match(self, seed=None):
        """
        Match populations passed during __init__(). Returns MatchingData
        instance containing the matched pool and target populations.
        """
        t0 = time.time()
        self._init_first_generation(seed)
        stop = self._check_stopping_conditions()
        while not stop:
            self._log()
            self._generate_offspring()
            self.elapsed_time = time.time() - t0
            stop = self._check_stopping_conditions()

        self._log(finalize=True)
        return self.get_best_match()

    def _init_first_generation(self, seed=None):
        initializer = GeneticMatcherInitializer(self)
        candidate_populations = initializer.initialize(
            self.n_candidate_populations, seed=seed
        )
        self.candidate_populations = torch.tensor(np.array(candidate_populations)).to(
            self.device
        )

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
        # balance = torch.tensor(self.balance).to(self.device)
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

    def get_best_match_idxs(self, balance_calculator=None):
        if balance_calculator is not None:
            balance = balance_calculator.balance(self.candidate_populations)
        else:
            balance = self.balance
        idx_best_match = balance.argmax()
        return self.candidate_populations[idx_best_match, :]

    def get_best_match(self, balance_calculator=None):
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
