import numpy as np
import time
import torch

import logging

logger = logging.getLogger(__name__)


class BasicLogger:
    """
    BasicLogger class for use with GeneticMatcher class. All loggers must
    implement at least two methods: on_generation_end, which is called at the
    end of each generation, and on_matching_end, which is called at the end of
    all iterations.  These methods must accept a GeneticMatcher instance as the
    only argument.
    """

    def __init__(self, log_every=10, working_directory="results/"):
        self.log_every = log_every
        self._t0 = time.time()
        self._set_directories(working_directory)

    def _set_directories(self, working_directory):
        self.working_directory = working_directory
        # self.history_directory = os.path.join(self.working_directory, 'history')
        # pathlib.Path(self.history_directory).mkdir(parents=True, exist_ok=True)

    def on_generation_end(self, matcher):
        verbose = matcher.params["verbose"]

        unique_patients = torch.unique(matcher.candidate_populations)
        if verbose and not (matcher.generation % self.log_every):
            logger.info(f"Generation {matcher.generation}")
            logger.info(f"\tremaining patients: {len(unique_patients)}")
            logger.info(f"\telapsed time: {(time.time() - self._t0)/60:.2f} min")

        balance_tensor = matcher.balance.cpu().numpy()
        idx_best_match = balance_tensor.argmax()
        best_score = np.abs(balance_tensor[idx_best_match])
        idx_worst_match = balance_tensor.argmin()
        worst_score = np.abs(balance_tensor[idx_worst_match])

        if verbose and not (matcher.generation % self.log_every):
            logger.info(
                f"\tbest {matcher.balance_calculator.name}: {best_score:3.5f} \tworst {matcher.balance_calculator.name}: {worst_score:3.5f}"
            )

    def on_matching_end(self, matcher):
        # self.make_figures(matcher)
        self.save_results_locally(matcher)

    def make_figures(self, matcher):
        # # initial distributions
        # fig = plot_1d_marginals(matcher.matching_data)
        # figname = os.path.join(self.working_directory, 'fig_initial_distributions.png')
        # fig.savefig(figname)

        # # final distributions
        # best_match = matcher.get_best_match()
        # matching_data = matcher.matching_data.append(best_match)
        # fig = plot_1d_marginals(best_match, matcher.target, matcher.headers)
        # figname = os.path.join(self.working_directory, 'fig_final_distributions.png')
        # fig.savefig(figname)

        # # metrics history / convergence
        # figname = os.path.join(self.working_directory, 'fig_distribution_of_gamma_and_beta.png')
        # # fig = plot_history_of_beta_and_gamma(matcher.history, matcher.params['balance_metrics'])
        # fig.savefig(figname)
        pass

    def save_results_locally(self, matcher):
        pass

    def save_results_remotely(self, matcher):
        pass
