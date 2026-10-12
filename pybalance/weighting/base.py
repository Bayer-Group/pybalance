from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from pybalance.utils import MatchingData


def _check_fitted(weighter):
    if weighter.weights is None:
        raise ValueError("Weighter has not been fitted!")


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Kish's effective sample size: ``(sum w)**2 / sum(w**2)``. Invariant to the
    overall scale of the weights. A large drop relative to ``len(weights)``
    indicates the reweighting is relying heavily on a small number of pool
    patients to match the target, and results should be interpreted
    cautiously (e.g. the target may lie outside the range of the pool's
    covariates).
    """
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.sum(weights**2))


class BaseWeighter:
    """
    Common interface for weighting methods. Unlike the Matcher classes
    (genetic/lp/propensity), a Weighter never drops pool patients; instead it
    assigns every pool patient a non-negative weight so that the *weighted*
    pool resembles the target on the moments of interest. This is the
    standard approach for indirect/external comparisons when excluding
    patients is undesirable or infeasible -- e.g. small pools, or a target
    known only through published aggregate statistics.

    :param matching_data: MatchingData whose pool is to be weighted. The
        target can be either patient-level or an ``AggregateTarget`` (e.g. a
        published Table 1).
    :param weight_col: Name of the column used to store weights on the
        MatchingData returned by transform(). Must not collide with an existing
        matching feature.
    :param verbose: Whether to log fitting diagnostics.
    """

    def __init__(
        self,
        matching_data: MatchingData,
        weight_col: str = "sample_weight",
        verbose: bool = True,
    ):
        if weight_col in matching_data.headers.all:
            raise ValueError(
                f"weight_col={weight_col!r} collides with an existing matching "
                "feature. Pass a different weight_col."
            )

        self.matching_data = matching_data.copy()
        self.weight_col = weight_col
        self.verbose = verbose
        self.weights: Optional[np.ndarray] = None
        # Only set by weighters that also reweight a patient-level target.
        self.target_weights: Optional[np.ndarray] = None
        self.diagnostics: Dict = {}

    def get_params(self) -> Dict:
        """Return the weighter's configuration parameters as a dict."""
        raise NotImplementedError

    def fit(self) -> "BaseWeighter":
        """
        Compute self.weights (and, if the weighter also reweights the target,
        self.target_weights) and self.diagnostics. Returns self.
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """
        Return the fitted per-patient pool weights (in pool row order).
        """
        _check_fitted(self)
        return self.weights

    def get_target_weights(self) -> Optional[np.ndarray]:
        """
        Return the fitted per-patient target weights, or None if the target is
        not reweighted (it then counts with weight 1).
        """
        _check_fitted(self)
        return self.target_weights

    def effective_sample_size(self) -> float:
        """
        Kish's effective sample size of the fitted pool weights. See
        ``effective_sample_size()``.
        """
        _check_fitted(self)
        return effective_sample_size(self.weights)

    def transform(self) -> MatchingData:
        """
        Apply the fitted weights: return a MatchingData instance in which every
        pool patient is retained and carries a new column (see weight_col)
        holding their weight. The target population (patient-level or
        aggregate) is passed through unchanged, with weight 1.0 assigned to any
        patient-level target rows unless the weighter also fits target weights.
        """
        _check_fitted(self)
        return self._with_weights(self.weights, self.target_weights)

    def match(self) -> MatchingData:
        """
        Fit (if not already fit) and transform, matching the public interface
        of the other matchers in pybalance.
        """
        if self.weights is None:
            self.fit()
        return self.transform()

    def _with_weights(self, pool_weights, target_weights) -> MatchingData:
        md = self.matching_data
        pool = md.get_population(md.pool_name).copy()
        pool[self.weight_col] = pool_weights

        if md.has_aggregate_target:
            return MatchingData(
                pool=pool,
                target=md.aggregate_target,
                headers=md.headers,
                population_col=md.population_col,
                pool_name=md.pool_name,
                target_name=md.target_name,
            )

        target = md.get_population(md.target_name).copy()
        target[self.weight_col] = 1.0 if target_weights is None else target_weights
        return MatchingData(
            pool=pool,
            target=target,
            headers=md.headers,
            population_col=md.population_col,
            pool_name=md.pool_name,
            target_name=md.target_name,
        )


def weighted_balance_table(
    weighter: BaseWeighter, method: Optional[str] = None
) -> pd.DataFrame:
    """
    Return a table comparing each balanced feature's weighted (and, for
    reference, unweighted) pool moment to the target moment -- a quick
    post-hoc check of how well the fitted weights balanced the pool. For
    EntropyBalanceWeighter/MAICWeighter, residuals should be ~0 for every
    constrained row; for PropensityScoreWeighter, which does not solve for exact
    balance, this is instead a diagnostic of how much balance improved
    relative to the unweighted pool.

    :param weighter: A fitted EntropyBalanceWeighter/MAICWeighter/PropensityScoreWeighter,
        i.e. one on which fit() or match() has already been called.
    :param method: For a PropensityScoreWeighter, the weighting method to
        tabulate (default: the method the model was selected for).
    """
    _check_fitted(weighter)
    kwargs = {} if method is None else {"method": method}
    w = weighter.get_weights(**kwargs)
    target_weights = weighter.get_target_weights(**kwargs)
    pool = weighter.balance_calculator.pool.cpu().numpy()
    target_mean = weighter.balance_calculator.target_mean.cpu().numpy().reshape(-1)
    if target_weights is not None:
        target_mean = np.average(
            weighter.balance_calculator.target.cpu().numpy(),
            weights=target_weights,
            axis=0,
        )
    target_std = weighter.balance_calculator.target_std.cpu().numpy().reshape(-1)
    out_features = weighter.preprocessor.output_headers["all"]

    variance_features = (
        set(weighter._numeric_features_with_disclosed_std())
        if getattr(weighter, "match_variance", False)
        else set()
    )

    constrained = weighter.balance_calculator.constrained
    disclosed = weighter.balance_calculator.disclosed

    rows = []
    for j, feature in enumerate(out_features):
        if disclosed[j] and not constrained[j]:
            continue  # a level implied by the others
        rows.append(
            {
                "feature": feature,
                "moment": "mean",
                # undisclosed: nothing to compare to, but still worth seeing move
                "target": target_mean[j] if disclosed[j] else np.nan,
                "unweighted_pool": pool[:, j].mean(),
                "weighted_pool": np.average(pool[:, j], weights=w),
            }
        )
        if feature in variance_features:
            rows.append(
                {
                    "feature": feature,
                    "moment": "variance",
                    "target": target_std[j] ** 2,
                    "unweighted_pool": pool[:, j].var(),
                    "weighted_pool": np.average(
                        (pool[:, j] - np.average(pool[:, j], weights=w)) ** 2,
                        weights=w,
                    ),
                }
            )

    return pd.DataFrame(rows)
