from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pybalance.propensity.matcher import (
    PropensityModelSearch,
    _facet_population_histograms,
)
from pybalance.utils import BalanceCalculator, BaseBalanceCalculator, MatchingData
from pybalance.weighting.base import (
    BaseWeighter,
    _check_fitted,
    effective_sample_size,
)

logger = logging.getLogger(__name__)


def _trim(weights: np.ndarray, trim_quantiles: Tuple[float, float]):
    lo, hi = np.quantile(weights, trim_quantiles)
    return np.clip(weights, lo, hi), int(np.sum((weights < lo) | (weights > hi)))


def _att(ps_pool, ps_target):
    return ps_pool / (1 - ps_pool), np.ones_like(ps_target)


def _atc(ps_pool, ps_target):
    return np.ones_like(ps_pool), (1 - ps_target) / ps_target


def _ate(ps_pool, ps_target):
    return 1 / (1 - ps_pool), 1 / ps_target


def _overlap(ps_pool, ps_target):
    return ps_pool, 1 - ps_target


def _matching(ps_pool, ps_target):
    return (
        np.minimum(ps_pool, 1 - ps_pool) / (1 - ps_pool),
        np.minimum(ps_target, 1 - ps_target) / ps_target,
    )


PROPENSITY_WEIGHT_METHODS = {
    "att": _att,
    "atc": _atc,
    "ate": _ate,
    "overlap": _overlap,
    "matching": _matching,
}


def propensity_score_weights(
    ps_pool: np.ndarray, ps_target: np.ndarray, method: str = "att"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map propensity scores ``p = P(target | X)`` to weights for the pool and
    target patients. The weighting analogue of
    ``pybalance.propensity.propensity_score_match``.

    - "att": pool weighted by the odds ``p / (1 - p)``, target weight 1. The
      pool is reweighted onto the target (Rosenbaum & Rubin, 1983; Austin,
      2011).
    - "atc": the reverse: target reweighted onto the pool, pool weight 1.
    - "ate": inverse probability weights for both populations, which
      reweights each onto their union.
    - "overlap": pool weighted by ``p``, target by ``1 - p`` (Li, Morgan &
      Zaslavsky, 2018). Emphasizes patients who could belong to either
      population; weights are bounded in [0, 1] and, with an unregularized
      logistic model, covariate means are balanced exactly.
    - "matching": ``min(p, 1 - p)`` over the inverse probability weight (Li &
      Greiner, 2013), a smoother analogue of 1:1 matching.

    Propensity scores should be clipped away from 0 and 1 by the caller.

    :return: Tuple ``(pool_weights, target_weights)``.
    """
    if method not in PROPENSITY_WEIGHT_METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Must be one of "
            f"{', '.join(PROPENSITY_WEIGHT_METHODS)}."
        )
    return PROPENSITY_WEIGHT_METHODS[method](ps_pool, ps_target)


class PropensityScoreWeighter(PropensityModelSearch, BaseWeighter):
    """
    Weight patients using a propensity score model for pool vs. target
    membership. As in ``PropensityScoreMatcher``, the weighter searches
    randomly over the hyperparameters of the propensity score model and keeps
    the model whose weights perform best according to the given optimization
    objective -- here, the balance of the *weighted* populations. The
    propensity score is mapped to weights by ``method`` (see
    ``propensity_score_weights()``), which plays the role of the matching
    method in ``PropensityScoreMatcher``.

    Optimizing balance alone can favor spiky weights that rest on a handful
    of patients, so candidate models whose effective sample size falls below
    ``min_ess_fraction`` of either population are rejected.

    Requires a patient-level target (there is no pool-vs-target classification
    problem to fit against an ``AggregateTarget``).

    :param matching_data: MatchingData whose pool is to be weighted.

    :param objective: Balance objective to optimize in the hyperparameter
        search. Can be a string referring to any balance calculator known to
        utils.balance_calculators.BalanceCalculator or an instance of
        BaseBalanceCalculator.

    :param method: How to turn propensity scores into weights: "att", "atc",
        "ate", "overlap" or "matching".

    :param trim_quantiles: Optional ``(low, high)`` quantiles (e.g. ``(0.01,
        0.99)``) at which to clip the weights of each population before
        scoring. Trades a little bias for lower variance when extreme weights
        occur, e.g. with "att" or "ate".

    :param min_ess_fraction: Reject candidate models whose pool or target
        effective sample size is below this fraction of the population size.

    :param max_iter: Maximum number of hyperparameters to try before returning
        the best weights.

    :param time_limit: Restrict hyperparameter search based on time. No new
        model will be trained after time_limit seconds have passed (def: 300
        sec).

    :param weight_col: Name of the column used to store weights on the
        MatchingData returned by transform().

    :param verbose: Whether to log diagnostic information during training.

    :param seed: Seed for the hyperparameter search and the classifiers. Set
        it to make weighting reproducible; if None, results vary between runs.
    """

    def __init__(
        self,
        matching_data: MatchingData,
        objective: Union[str, BaseBalanceCalculator] = "beta",
        method: str = "att",
        trim_quantiles: Optional[Tuple[float, float]] = None,
        min_ess_fraction: float = 0.1,
        max_iter: int = 50,
        time_limit: Optional[float] = 60 * 5,
        weight_col: str = "sample_weight",
        verbose: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(matching_data, weight_col=weight_col, verbose=verbose)

        if self.matching_data.has_aggregate_target:
            raise ValueError(
                "PropensityScoreWeighter requires a patient-level target (it "
                "fits a pool-vs-target propensity model), so it cannot be used "
                "with an AggregateTarget. Use EntropyBalanceWeighter/MAICWeighter "
                "for aggregate (e.g. published Table 1) targets."
            )
        self._check_method(method)
        if trim_quantiles is not None:
            lo, hi = trim_quantiles
            if not (0 <= lo < hi <= 1):
                raise ValueError(
                    f"trim_quantiles must satisfy 0 <= low < high <= 1; got {trim_quantiles!r}."
                )
        if not 0 <= min_ess_fraction <= 1:
            raise ValueError("min_ess_fraction must be in [0, 1].")

        self.method = method
        self.trim_quantiles = trim_quantiles
        self.min_ess_fraction = min_ess_fraction
        self.max_iter = max_iter
        self.time_limit = time_limit
        self.seed = seed
        self.hyperparam_space = self.DEFAULT_HYPERPARAM_SPACE

        if isinstance(objective, str):
            self.balance_calculator = BalanceCalculator(self.matching_data, objective)
            self.objective = objective
        else:
            self.balance_calculator = objective
            self.objective = self.balance_calculator.name
        self.preprocessor = self.balance_calculator.preprocessor
        self._train_preprocessors(self.matching_data)

        self.best_model = None
        self.best_params = None
        self.best_ps = None
        self.best_score = np.inf
        self.solution_time = 0

    @staticmethod
    def _check_method(method):
        if method not in PROPENSITY_WEIGHT_METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Must be one of "
                f"{', '.join(PROPENSITY_WEIGHT_METHODS)}."
            )

    def get_params(self) -> Dict:
        """Return the weighter's configuration parameters as a dict."""
        params = [
            "objective",
            "method",
            "trim_quantiles",
            "min_ess_fraction",
            "max_iter",
            "time_limit",
            "seed",
        ]
        return dict((p, getattr(self, p)) for p in params)

    def _compute_weights(self, ps_pool, ps_target, method):
        # Clip away from 0/1 to avoid infinite weights from a (near-)perfectly
        # separable model.
        ps_pool = np.clip(ps_pool, 1e-4, 1 - 1e-4)
        ps_target = np.clip(ps_target, 1e-4, 1 - 1e-4)
        w_pool, w_target = propensity_score_weights(ps_pool, ps_target, method)

        n_trimmed = 0
        if self.trim_quantiles is not None:
            w_pool, n_pool = _trim(w_pool, self.trim_quantiles)
            w_target, n_target = _trim(w_target, self.trim_quantiles)
            n_trimmed = n_pool + n_target
        return w_pool, w_target, n_trimmed

    def _score(self, w_pool, w_target) -> float:
        return float(
            self.balance_calculator.distance(
                pool_weights=w_pool, target_weights=w_target
            )
        )

    def _set_weights(self, w_pool, w_target, n_trimmed, score):
        self.weights = w_pool
        self.target_weights = w_target
        self.best_score = score
        self.diagnostics = {
            "effective_sample_size": effective_sample_size(w_pool),
            "target_effective_sample_size": effective_sample_size(w_target),
            "n_trimmed": n_trimmed,
            "score": score,
        }

    def _describe_best_weights(self):
        logger.info(f"Best propensity score weights found ({self.method}):")
        logger.info(f'\tModel: {str(self.best_model).split("(")[0]}')
        for key, val in self.best_params.items():
            logger.info(f"\t* {key}: {val}")
        logger.info(f"\tScore ({self.balance_calculator.name}): {self.best_score:.4f}")
        logger.info(
            f"\tEffective sample size: {self.diagnostics['effective_sample_size']:.1f}"
            f" / {len(self.weights)} pool patients"
        )
        logger.info(f"\tSolution time: {self.solution_time:.1f} s")

    def fit(self) -> "PropensityScoreWeighter":
        """
        Search over propensity score model hyperparameters and keep the model
        whose weights, for the ``method`` given at construction, best satisfy
        the balance objective. Use transform() to apply weights.
        """
        t0 = time.time()

        X, y = self._preprocess_data_for_sklearn(self.matching_data)
        X_pool, X_target = X[y == self.POOL], X[y == self.TARGET]
        n_rejected = 0
        for i, (model, params) in enumerate(self._get_hyperparams(self.max_iter)):
            clf = model(**params, random_state=self.seed)
            logger.info(
                f'Training model {str(clf).split("(")[0]} (iter {i + 1}/{self.max_iter}, {time.time() - t0:.1f} s) ...'
            )

            clf.fit(X, y)
            ps_pool = clf.predict_proba(X_pool)[:, 1]
            ps_target = clf.predict_proba(X_target)[:, 1]
            w_pool, w_target, n_trimmed = self._compute_weights(
                ps_pool, ps_target, self.method
            )

            ess_pool = effective_sample_size(w_pool) / len(w_pool)
            ess_target = effective_sample_size(w_target) / len(w_target)
            if min(ess_pool, ess_target) < self.min_ess_fraction:
                n_rejected += 1
            else:
                score = self._score(w_pool, w_target)
                if score < self.best_score:
                    self.best_model = clf
                    self.best_params = params
                    self.best_ps = {"pool": ps_pool, "target": ps_target}
                    self.solution_time = time.time() - t0
                    self._set_weights(w_pool, w_target, n_trimmed, score)
                    if self.verbose:
                        self._describe_best_weights()

            # Put break at the end to ensure at least one iteration
            if self.time_limit is not None and (time.time() - t0) > self.time_limit:
                logger.warning("Time limit exceeded. Stopping early.")
                break

        if self.best_model is None:
            raise ValueError(
                f"All {n_rejected} candidate propensity models were rejected for "
                f"an effective sample size below min_ess_fraction="
                f"{self.min_ess_fraction}. Lower min_ess_fraction, trim weights "
                "with trim_quantiles, or check for non-overlapping populations."
            )

        return self

    def _weights_for(self, method: Optional[str]):
        _check_fitted(self)
        if method is None:
            method = self.method
        self._check_method(method)
        return self._compute_weights(
            self.best_ps["pool"], self.best_ps["target"], method
        )

    def get_weights(self, method: Optional[str] = None) -> np.ndarray:
        """
        Pool weights from the fitted model for ``method`` (default: the method
        the model was selected for). Does not change the fitted state.
        """
        return self._weights_for(method)[0]

    def get_target_weights(self, method: Optional[str] = None) -> np.ndarray:
        """Target weights from the fitted model for ``method``; see get_weights()."""
        return self._weights_for(method)[1]

    def score(self, method: Optional[str] = None) -> float:
        """
        Balance objective (lower is better) of the weighted populations for
        ``method``; see get_weights().
        """
        w_pool, w_target, _ = self._weights_for(method)
        return self._score(w_pool, w_target)

    def transform(self, method: Optional[str] = None) -> MatchingData:
        """
        Apply the fitted propensity model: return a MatchingData instance in
        which every pool and target patient is retained and carries a new
        column (see weight_col) holding their weight under ``method`` (default:
        the method the model was selected for). Passing another method reuses
        the model without searching again, but note the model was selected to
        optimize the balance of the method given at construction. Does not
        change the fitted state.
        """
        w_pool, w_target, _ = self._weights_for(method)
        return self._with_weights(w_pool, w_target)

    def plot_propensity_score_distributions(self, method: Optional[str] = None):
        """
        Plot histograms of the estimated propensity score for the pool and
        target populations, before vs. after weighting -- the weighting
        analogue of ``plot_propensity_score_distributions()`` for matchers.

        Unlike a Matcher, a Weighter never drops patients, so there is no matched
        subset to compare against; instead, the "before" panel shows every
        patient counted equally and the "after" panel shows the same propensity
        scores counted by their weight. A successful fit should show the
        "after" histograms move towards each other.

        :param method: Weighting method to plot (default: the method the model
            was selected for).
        """
        _check_fitted(self)
        md = self.matching_data
        pool_name, target_name = md.pool_name, md.target_name

        pool_weights = self.get_weights(method)
        target_weights = self.get_target_weights(method)

        data = pd.concat(
            [
                pd.DataFrame.from_dict(
                    {
                        "propensity": self.best_ps["pool"],
                        "weight": np.ones_like(self.best_ps["pool"]),
                        "weighted": False,
                        "population": pool_name,
                    }
                ),
                pd.DataFrame.from_dict(
                    {
                        "propensity": self.best_ps["target"],
                        "weight": np.ones_like(self.best_ps["target"]),
                        "weighted": False,
                        "population": target_name,
                    }
                ),
                pd.DataFrame.from_dict(
                    {
                        "propensity": self.best_ps["pool"],
                        "weight": pool_weights,
                        "weighted": True,
                        "population": pool_name,
                    }
                ),
                pd.DataFrame.from_dict(
                    {
                        "propensity": self.best_ps["target"],
                        "weight": target_weights,
                        "weighted": True,
                        "population": target_name,
                    }
                ),
            ]
        )

        return _facet_population_histograms(
            data,
            x="propensity",
            col="weighted",
            pool_name=pool_name,
            target_name=target_name,
            col_order=[False, True],
            binrange=(0, 1),
            xlim=[0, 1],
            weights="weight",
        )

    def plot_weight_distributions(self, method: Optional[str] = None):
        """
        Plot histograms of the fitted weights of the pool and target
        populations. The effective sample size of each population is shown in
        its panel title; a long right tail or a low effective sample size
        indicates that a few patients dominate.

        :param method: Weighting method to plot (default: the method the model
            was selected for).
        """
        _check_fitted(self)
        md = self.matching_data
        pool_name, target_name = md.pool_name, md.target_name
        weights = {
            pool_name: self.get_weights(method),
            target_name: self.get_target_weights(method),
        }
        max_weight = max(w.max() for w in weights.values())

        data = pd.concat(
            [
                pd.DataFrame.from_dict({"weight": w, "population": name})
                for name, w in weights.items()
            ]
        )

        g = _facet_population_histograms(
            data,
            x="weight",
            col="population",
            pool_name=pool_name,
            target_name=target_name,
            col_order=[pool_name, target_name],
            binrange=(0, max_weight),
            xlim=[0, 1.05 * max_weight],
        )
        for ax, (name, w) in zip(g.axes.flat, weights.items()):
            ax.set_title(f"{name} (ESS {effective_sample_size(w):.0f} / {len(w)})")

        return g
