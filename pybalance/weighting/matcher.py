from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pybalance.utils import MatchingData, BalanceCalculator

import logging

logger = logging.getLogger(__name__)


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


def _softmax_weights(Z: np.ndarray, lam: np.ndarray) -> np.ndarray:
    linpred = Z @ lam
    linpred = linpred - linpred.max()
    w = np.exp(linpred)
    return w / w.sum()


def _solve_entropy_weights(
    Z: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-10,
    ridge: float = 1e-8,
    penalty: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Solve for weights ``w_i = softmax(Z @ lambda)_i`` (so ``sum_i w_i == 1`` by
    construction) by minimizing the (convex, smooth) dual of the
    maximum-entropy weighting problem, ``log(sum_i exp(Z_i @ lambda))``, via
    scipy's trust-region-Newton-CG solver using an analytic gradient and
    Hessian. At the optimum, ``sum_i w_i * Z_i == 0`` for every column of
    ``Z``, i.e. every (centered) moment constraint is exactly balanced in the
    weighted pool. This is the method-of-moments weighting scheme underlying
    MAIC (Signorovitch et al., 2010) and entropy balancing (Hainmueller,
    2012).

    The softmax parametrization (rather than the more commonly-quoted raw
    ``w_i = exp(z_i . lambda)``) is important for numerical robustness: the
    raw form leaves an unconstrained degree of freedom along which ``sum(w)``
    can be driven towards 0 (or diverge) while the *raw* gradient ``Z.T @ w``
    shrinks in lockstep, which can fool a naive convergence check into
    declaring success at a degenerate, non-solution point. Because the
    softmax always sums to 1, this failure mode is eliminated and the
    gradient (a genuine weighted mean) is a directly interpretable, scale-free
    convergence criterion. A trust-region solver (rather than plain damped
    Newton) is used because the Hessian -- a weighted covariance matrix of
    ``Z`` -- can be near-singular along directions with little effective
    curvature (e.g. highly imbalanced categoric rates), which otherwise
    produces enormous, unstable Newton steps.

    :param Z: (n_pool, n_constraints) matrix of centered, column-scaled moment
        constraints: ``Z[i, j]`` is patient ``i``'s (scaled) deviation from
        the target moment for constraint ``j``.
    :param max_iter: Maximum number of solver iterations.
    :param tol: Convergence tolerance on the max-norm of the constraint
        violation (weighted-mean residual).
    :param ridge: Ridge term added to the Hessian for numerical stability;
        important when constraints are collinear or near-collinear.
    :param penalty: Optional per-constraint soft-constraint strength. A
        positive entry adds ``penalty_j * lambda_j**2 / 2`` to the dual, which
        relaxes constraint ``j`` to a quadratic penalty on its residual
        instead of an exact equality. Needed for constraints sitting on the
        boundary of what positive weights can reach (e.g. "no patient above a
        max"), for which exact balance has no finite solution.
    :return: Tuple ``(weights, diagnostics)`` where diagnostics has keys
        'converged', 'n_iter', 'max_constraint_violation' (over the exact
        constraints) and, if ``penalty`` is used, 'max_soft_violation'.
    """
    n, k = Z.shape
    if k == 0:
        return np.ones(n) / n, {
            "converged": True,
            "n_iter": 0,
            "max_constraint_violation": 0.0,
        }

    penalty = np.zeros(k) if penalty is None else np.asarray(penalty, dtype=float)

    def _objective(lam):
        linpred = Z @ lam
        m = linpred.max()
        return m + np.log(np.exp(linpred - m).sum()) + 0.5 * np.sum(penalty * lam**2)

    def _grad(lam):
        return Z.T @ _softmax_weights(Z, lam) + penalty * lam

    def _hess(lam):
        w = _softmax_weights(Z, lam)
        zc = Z - (Z.T @ w)
        return (zc * w[:, None]).T @ zc + np.diag(penalty) + ridge * np.eye(k)

    res = minimize(
        _objective,
        x0=np.zeros(k),
        jac=_grad,
        hess=_hess,
        method="trust-ncg",
        options={"maxiter": max_iter, "gtol": tol},
    )

    w_final = _softmax_weights(Z, res.x)
    residual = np.abs(Z.T @ w_final)
    exact = penalty == 0
    max_violation = float(residual[exact].max()) if exact.any() else 0.0
    converged = bool(res.success) or max_violation < tol * 100

    diagnostics = {
        "converged": converged,
        "n_iter": int(res.nit),
        "max_constraint_violation": max_violation,
    }
    if (~exact).any():
        diagnostics["max_soft_violation"] = float(residual[~exact].max())
    return w_final, diagnostics


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
        MatchingData returned by match(). Must not collide with an existing
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
        self.diagnostics: Dict = {}

    def get_params(self) -> Dict:
        raise NotImplementedError

    def _fit(self) -> "BaseWeighter":
        """
        Compute self.weights/self.diagnostics. Subclasses implement this;
        end users should call match() instead (see below), which calls this
        internally and wraps the result in a MatchingData, matching the
        public interface of the other matchers in pybalance.
        """
        raise NotImplementedError

    def get_weights(self) -> np.ndarray:
        """
        Return the fitted per-patient pool weights (in pool row order).
        """
        _check_fitted(self)
        return self.weights

    def effective_sample_size(self) -> float:
        """
        Kish's effective sample size of the fitted weights. See
        ``effective_sample_size()``.
        """
        _check_fitted(self)
        return effective_sample_size(self.weights)

    def match(self) -> MatchingData:
        """
        Fit (if not already fit) and return a MatchingData instance in which
        every pool patient is retained and carries a new column (see
        weight_col) holding their fitted weight. The target population
        (patient-level or aggregate) is passed through unchanged, with weight
        1.0 assigned to any patient-level target rows.
        """
        if self.weights is None:
            self._fit()

        md = self.matching_data
        pool = md.get_population(md.pool_name).copy()
        pool[self.weight_col] = self.weights

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
        target[self.weight_col] = 1.0
        return MatchingData(
            pool=pool,
            target=target,
            headers=md.headers,
            population_col=md.population_col,
            pool_name=md.pool_name,
            target_name=md.target_name,
        )


class EntropyBalanceWeighter(BaseWeighter):
    """
    General maximum-entropy ("method of moments") weighting: solves for
    pool weights of the form ``w_i = exp(z_i . alpha)`` such that the weighted
    pool matches the target. For an ``AggregateTarget`` every disclosed
    statistic is its own constraint and nothing else is constrained: a
    numeric mean, the rate of each disclosed categoric level, and the rate
    above each disclosed median / quantile (the feature is dichotomized at the
    disclosed value). A feature or categoric level the target says nothing
    about is simply left free. Optionally also balances variance for numeric
    features whose target discloses (or, for a patient-level target, has) a
    standard deviation; for an ``AggregateTarget`` that needs the mean too,
    since the variance is taken around it.

    A disclosed ``min`` / ``max`` is a *soft* constraint, as in
    ``AggregateConstraintSatisfactionMatcher``: exactly zero weight on
    patients above a max cannot be reached by positive weights, so the
    fraction of weight there is only penalized (see ``limit_penalty``).

    This is the general form of Matching-Adjusted Indirect Comparison (MAIC;
    Signorovitch et al., 2010); see ``MAICWeighter`` for the classic
    mean-only formulation.

    :param matching_data: MatchingData whose pool is to be weighted.
    :param match_variance: If True, additionally constrain the weighted
        variance of numeric features. For an ``AggregateTarget``, only
        features that actually disclose a "std" are constrained (mirroring
        ``AggregateConstraintSatisfactionMatcher``); for a patient-level
        target, every numeric feature's variance is constrained. Categoric
        features are never variance-constrained: their variance is already
        determined by their (matched) rate.
    :param normalize: How to rescale the fitted weights for reporting: "target"
        (default) rescales so weights sum to the target population size,
        "pool" rescales so weights sum to the pool size, "none" leaves the raw
        dual-optimizer weights unscaled. This choice has no effect on balance
        (the moment constraints are scale-invariant) or on
        effective_sample_size(); it only affects the units the weights are
        reported in.
    :param max_iter: Maximum number of Newton iterations.
    :param tol: Convergence tolerance on the (max-norm) constraint violation.
    :param ridge: Ridge regularization added to the Newton step for numerical
        stability; increase this if fitting fails to converge due to
        collinear/near-collinear features.
    :param weight_col: Name of the column used to store weights on the
        MatchingData returned by match().
    :param verbose: Whether to log fitting diagnostics.
    """

    def __init__(
        self,
        matching_data: MatchingData,
        match_variance: bool = False,
        normalize: str = "target",
        max_iter: int = 200,
        tol: float = 1e-8,
        ridge: float = 1e-8,
        weight_col: str = "sample_weight",
        verbose: bool = True,
        limit_penalty: float = 1e-2,
    ):
        super().__init__(matching_data, weight_col=weight_col, verbose=verbose)

        if normalize not in ("target", "pool", "none"):
            raise ValueError(
                f"normalize must be one of 'target', 'pool', 'none'; got {normalize!r}."
            )

        self.match_variance = match_variance
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.limit_penalty = limit_penalty

        md = self.matching_data
        if md.has_aggregate_target and match_variance:
            no_mean = [
                f
                for f, stats in md.aggregate_target.numeric.items()
                if "std" in stats and "mean" not in stats
            ]
            if no_mean:
                raise ValueError(
                    f"Cannot balance the variance of {no_mean}: the target discloses "
                    "a std but no mean, so the variance is taken around the weighted "
                    "mean, which is not a moment constraint reweighting can solve. "
                    "Disclose the mean too, or use AggregateConstraintSatisfactionMatcher."
                )

        # Fitted preprocessor plus pool features and target mean/std in a single
        # consistent output feature space. An AggregateTarget needs the
        # calculator that dichotomizes quantile-disclosed features and says
        # which output columns the target actually discloses.
        objective = "aggregate_beta" if md.has_aggregate_target else "beta"
        self.balance_calculator = BalanceCalculator(md, objective)
        self.preprocessor = self.balance_calculator.preprocessor

    def get_params(self) -> Dict:
        return {
            "match_variance": self.match_variance,
            "normalize": self.normalize,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "ridge": self.ridge,
            "limit_penalty": self.limit_penalty,
        }

    def _limit_columns(self) -> set:
        """Output columns holding the indicator of a disclosed min / max."""
        md = self.matching_data
        if not md.has_aggregate_target:
            return set()
        out_features = self.preprocessor.output_headers["all"]
        columns = set()
        for feature, stats in md.aggregate_target.numeric.items():
            for q, _ in stats.get("quantile", []):
                if q in (0.0, 1.0):
                    prefix = f"{feature}_q{q}"
                    columns |= {c for c in out_features if c.startswith(prefix + "_")}
        return columns

    def _numeric_features_with_disclosed_std(self) -> List[str]:
        md = self.matching_data
        if md.has_aggregate_target:
            return [
                f
                for f in md.aggregate_target.headers.numeric
                if "std" in md.aggregate_target.numeric[f]
                and "mean" in md.aggregate_target.numeric[f]
            ]
        return list(md.headers.numeric)

    def _build_constraints(self) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Build the (n_pool, n_constraints) centered-and-scaled constraint
        matrix used to solve for weights, a human-readable label per
        constraint column (for diagnostics/reporting) and the per-constraint
        soft-constraint penalty (0 for an exact constraint).
        """
        pool = self.balance_calculator.pool.cpu().numpy()
        target_mean = self.balance_calculator.target_mean.cpu().numpy().reshape(-1)
        target_std = self.balance_calculator.target_std.cpu().numpy().reshape(-1)
        pool_std = pool.std(axis=0)

        out_features = self.preprocessor.output_headers["all"]
        n_pool, n_features = pool.shape

        constrained = self.balance_calculator.constrained
        limit_columns = self._limit_columns()

        columns = []
        labels = []
        penalty = []
        for j in range(n_features):
            if not constrained[j]:
                continue
            penalty.append(
                self.limit_penalty if out_features[j] in limit_columns else 0.0
            )
            scale = (
                target_std[j]
                if target_std[j] > 0
                else (pool_std[j] if pool_std[j] > 0 else 1.0)
            )
            columns.append((pool[:, j] - target_mean[j]) / scale)
            labels.append(f"{out_features[j]} (mean)")

        if self.match_variance:
            variance_features = set(self._numeric_features_with_disclosed_std())
            for j, feature in enumerate(out_features):
                if feature not in variance_features:
                    continue
                dev = (pool[:, j] - target_mean[j]) ** 2
                target_var = target_std[j] ** 2
                # Give the variance term its own scale so it neither dominates
                # nor is swamped by the mean terms purely due to units (same
                # concern handled in
                # AggregateConstraintSatisfactionMatcher._get_target_variance_targets).
                scale2 = target_var if target_var > 0 else max(dev.std(), 1.0)
                columns.append((dev - target_var) / scale2)
                labels.append(f"{feature} (variance)")
                penalty.append(0.0)

        Z = np.column_stack(columns) if columns else np.zeros((n_pool, 0))
        return Z, labels, np.array(penalty)

    def _fit(self) -> "EntropyBalanceWeighter":
        md = self.matching_data
        n_pool = len(md.get_population(md.pool_name))
        n_target = (
            md.aggregate_target.n
            if md.has_aggregate_target
            else len(md.get_population(md.target_name))
        )

        Z, labels, penalty = self._build_constraints()
        self.constraint_labels = labels

        weights, diagnostics = _solve_entropy_weights(
            Z, max_iter=self.max_iter, tol=self.tol, ridge=self.ridge, penalty=penalty
        )

        if self.normalize == "target":
            weights = weights * (n_target / weights.sum())
        elif self.normalize == "pool":
            weights = weights * (n_pool / weights.sum())

        self.weights = weights
        self.diagnostics = diagnostics
        self.diagnostics["effective_sample_size"] = effective_sample_size(weights)

        if not diagnostics["converged"]:
            logger.warning(
                f"{self.__class__.__name__} did not converge within "
                f"{self.max_iter} iterations (max constraint violation = "
                f"{diagnostics['max_constraint_violation']:.4g}). Weights may "
                "not exactly balance the requested moments; consider "
                "increasing max_iter/ridge, or check for unmatchable (e.g. "
                "near-extreme or non-overlapping) covariates."
            )
        elif self.verbose:
            logger.info(
                f"{self.__class__.__name__} converged in {diagnostics['n_iter']} "
                "iterations. Effective sample size: "
                f"{self.diagnostics['effective_sample_size']:.1f} / {n_pool} pool patients."
            )

        return self


class MAICWeighter(EntropyBalanceWeighter):
    """
    Matching-Adjusted Indirect Comparison (MAIC; Signorovitch et al., 2010,
    "Comparative effectiveness without head-to-head trials: a method for
    matching-adjusted indirect comparisons applied to psoriasis clinical
    trials"). Reweights patient-level ("IPD") pool data so that its weighted
    means match a target's aggregate statistics -- typically a comparator
    trial's published Table 1 -- using the method-of-moments / maximum-entropy
    weighting scheme of ``EntropyBalanceWeighter``, restricted to first
    moments only. This is the standard MAIC formulation and is appropriate
    whenever the comparator discloses only means (and category rates), not
    variances.

    Use ``EntropyBalanceWeighter(matching_data, match_variance=True)`` directly
    if the comparator additionally discloses standard deviations you also want
    to match.

    :param matching_data: MatchingData whose pool (IPD) is to be weighted to
        match the target's (typically aggregate, e.g. ``AggregateTarget``)
        moments.
    :param normalize: See ``EntropyBalanceWeighter``. Defaults to "target",
        i.e. weights sum to the target's sample size, matching common MAIC
        reporting conventions.
    :param max_iter: Maximum number of Newton iterations.
    :param tol: Convergence tolerance on the (max-norm) constraint violation.
    :param ridge: Ridge regularization added to the Newton step for numerical
        stability.
    :param weight_col: Name of the column used to store weights on the
        MatchingData returned by match().
    :param verbose: Whether to log fitting diagnostics.
    """

    def __init__(
        self,
        matching_data: MatchingData,
        normalize: str = "target",
        max_iter: int = 200,
        tol: float = 1e-8,
        ridge: float = 1e-8,
        weight_col: str = "sample_weight",
        verbose: bool = True,
        limit_penalty: float = 1e-2,
    ):
        super().__init__(
            matching_data,
            match_variance=False,
            normalize=normalize,
            max_iter=max_iter,
            tol=tol,
            ridge=ridge,
            weight_col=weight_col,
            verbose=verbose,
            limit_penalty=limit_penalty,
        )

    def get_params(self) -> Dict:
        params = super().get_params()
        del params["match_variance"]
        return params


class IPTWWeighter(BaseWeighter):
    """
    Inverse Probability of Treatment Weighting (IPTW; Rosenbaum & Rubin,
    1983; see Austin, 2011, "An Introduction to Propensity Score Methods for
    Reducing the Effects of Confounding in Observational Studies", for the
    ATT construction used here). Fits a propensity model ``p(X) = P(target |
    X)`` that classifies pool vs. target patients on their covariates, then
    reweights each pool patient by the odds ``p / (1 - p)``. The target
    population keeps weight 1, so the weighted pool is reweighted onto the
    target's covariate distribution -- i.e. this is the ATT estimand with the
    target playing the role of the fixed/reference ("treated") group, which
    matches the convention used by ``MAICWeighter``/``EntropyBalanceWeighter``
    and the package's typical use case of building an external comparator
    arm that represents a trial population.

    Unlike ``EntropyBalanceWeighter``, IPTW does not guarantee exact balance
    on any particular moment: it only guarantees balance asymptotically, and
    only if the propensity model is correctly specified. It also requires a
    patient-level target (there is no pool-vs-target classification problem
    to fit against a published aggregate Table 1). Its main practical
    advantages over entropy balancing are that it scales to many covariates
    without requiring the target's moments to be inside the pool's convex
    hull, and that it is the most widely recognized/reported method in the
    observational literature.

    :param matching_data: MatchingData whose pool is to be weighted. The
        target must be patient-level (not an ``AggregateTarget``).
    :param classifier: A fitted-or-unfitted sklearn-compatible classifier
        exposing ``predict_proba``, used to estimate ``P(target | X)``. It is
        cloned before fitting, so passing a pre-fitted instance does not
        reuse its fit. Defaults to ``LogisticRegression(max_iter=1000)``, the
        standard choice for propensity score estimation.
    :param trim_quantiles: Optional ``(low, high)`` quantiles (e.g. ``(0.01,
        0.99)``) at which to clip the fitted weights. Extreme weights (driven
        by pool patients whose covariates make them look almost certainly
        pool or almost certainly target) are the main practical failure mode
        of IPTW; trimming trades a little bias for a large reduction in
        variance. Left unset (``None``) by default so the raw weights are
        returned untouched.
    :param weight_col: Name of the column used to store weights on the
        MatchingData returned by match().
    :param verbose: Whether to log fitting diagnostics.
    """

    def __init__(
        self,
        matching_data: MatchingData,
        classifier: Optional[BaseEstimator] = None,
        trim_quantiles: Optional[Tuple[float, float]] = None,
        weight_col: str = "sample_weight",
        verbose: bool = True,
    ):
        super().__init__(matching_data, weight_col=weight_col, verbose=verbose)

        if self.matching_data.has_aggregate_target:
            raise ValueError(
                "IPTWWeighter requires a patient-level target (it fits a "
                "pool-vs-target propensity model), so it cannot be used with "
                "an AggregateTarget. Use EntropyBalanceWeighter/MAICWeighter "
                "for aggregate (e.g. published Table 1) targets."
            )

        if trim_quantiles is not None:
            lo, hi = trim_quantiles
            if not (0 <= lo < hi <= 1):
                raise ValueError(
                    f"trim_quantiles must satisfy 0 <= low < high <= 1; got {trim_quantiles!r}."
                )

        self.classifier = classifier
        self.trim_quantiles = trim_quantiles

        # Reuse BetaBalance purely to get a fitted preprocessor plus pool and
        # target feature tensors in a single consistent (one-hot categoric +
        # numeric passthrough) output space, as EntropyBalanceWeighter does.
        self.balance_calculator = BalanceCalculator(self.matching_data, "beta")
        self.preprocessor = self.balance_calculator.preprocessor

    def get_params(self) -> Dict:
        return {
            "classifier": self.classifier,
            "trim_quantiles": self.trim_quantiles,
        }

    def _fit(self) -> "IPTWWeighter":
        pool = self.balance_calculator.pool.cpu().numpy()
        target = self.balance_calculator.target.cpu().numpy()
        n_pool = len(pool)

        X = np.vstack([pool, target])
        y = np.concatenate([np.zeros(n_pool), np.ones(len(target))])

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        clf = (
            clone(self.classifier)
            if self.classifier is not None
            else LogisticRegression(max_iter=1000)
        )
        clf.fit(X, y)

        propensity_score = clf.predict_proba(X[:n_pool])[:, 1]
        # Clip away from 0/1 to avoid infinite weights from a
        # (near-)perfectly separable model.
        propensity_score = np.clip(propensity_score, 1e-4, 1 - 1e-4)
        weights = propensity_score / (1 - propensity_score)
        # Also kept (unclipped) so plot_iptw_propensity_distributions() can show
        # where the target itself sits on the fitted model.
        target_propensity_score = clf.predict_proba(X[n_pool:])[:, 1]

        n_trimmed = 0
        if self.trim_quantiles is not None:
            lo, hi = np.quantile(weights, self.trim_quantiles)
            n_trimmed = int(np.sum((weights < lo) | (weights > hi)))
            weights = np.clip(weights, lo, hi)

        self.propensity_model = clf
        self.propensity_score = propensity_score
        self.target_propensity_score = target_propensity_score
        self.weights = weights
        self.diagnostics = {
            "effective_sample_size": effective_sample_size(weights),
            "n_trimmed": n_trimmed,
        }

        if self.verbose:
            logger.info(
                f"{self.__class__.__name__} fit {str(clf).split('(')[0]}. "
                f"Effective sample size: {self.diagnostics['effective_sample_size']:.1f} "
                f"/ {n_pool} pool patients."
                + (f" Trimmed {n_trimmed} extreme weights." if n_trimmed else "")
            )

        return self


def weighted_balance_table(weighter: BaseWeighter) -> pd.DataFrame:
    """
    Return a table comparing each balanced feature's weighted (and, for
    reference, unweighted) pool moment to the target moment -- a quick
    post-hoc check of how well match() balanced the pool. For
    EntropyBalanceWeighter/MAICWeighter, residuals should be ~0 for every
    constrained row; for IPTWWeighter, which does not solve for exact
    balance, this is instead a diagnostic of how much balance improved
    relative to the unweighted pool.

    :param weighter: A fitted EntropyBalanceWeighter/MAICWeighter/IPTWWeighter,
        i.e. one on which match() has already been called.
    """
    _check_fitted(weighter)
    pool = weighter.balance_calculator.pool.cpu().numpy()
    target_mean = weighter.balance_calculator.target_mean.cpu().numpy().reshape(-1)
    target_std = weighter.balance_calculator.target_std.cpu().numpy().reshape(-1)
    out_features = weighter.preprocessor.output_headers["all"]
    w = weighter.weights

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


def _check_is_iptw(weighter: BaseWeighter) -> None:
    if not hasattr(weighter, "propensity_score"):
        raise TypeError(
            "plot_iptw_propensity_distributions() requires an IPTWWeighter "
            f"(got {type(weighter).__name__}), since only IPTWWeighter fits a "
            "propensity model -- MAICWeighter/EntropyBalanceWeighter solve "
            "directly for balancing weights without one."
        )


def plot_iptw_propensity_distributions(weighter: "IPTWWeighter"):
    """
    Plot histograms of the estimated propensity score for the pool and target
    populations, before vs. after IPTW weighting -- the weighting analogue of
    ``pybalance.propensity.plot_propensity_score_match_distributions``.

    Unlike a Matcher, a Weighter never drops patients, so there is no matched
    subset to compare against; instead, the "before" panel shows every pool
    patient counted equally and the "after" panel shows the same propensity
    scores counted by their fitted IPTW weight (the target always keeps
    weight 1). A successful fit should show the "after" pool histogram move
    towards the target's.

    :param weighter: A fitted ``IPTWWeighter``, i.e. one on which match() has
        already been called.
    """
    _check_fitted(weighter)
    _check_is_iptw(weighter)
    md = weighter.matching_data
    pool_name, target_name = md.pool_name, md.target_name

    data = pd.concat(
        [
            pd.DataFrame.from_dict(
                {
                    "propensity": weighter.propensity_score,
                    "weight": np.ones_like(weighter.propensity_score),
                    "weighted": False,
                    "population": pool_name,
                }
            ),
            pd.DataFrame.from_dict(
                {
                    "propensity": weighter.target_propensity_score,
                    "weight": np.ones_like(weighter.target_propensity_score),
                    "weighted": False,
                    "population": target_name,
                }
            ),
            pd.DataFrame.from_dict(
                {
                    "propensity": weighter.propensity_score,
                    "weight": weighter.weights,
                    "weighted": True,
                    "population": pool_name,
                }
            ),
            pd.DataFrame.from_dict(
                {
                    "propensity": weighter.target_propensity_score,
                    "weight": np.ones_like(weighter.target_propensity_score),
                    "weighted": True,
                    "population": target_name,
                }
            ),
        ]
    )

    g = sns.FacetGrid(
        data=data, col="weighted", col_order=[False, True], height=4, xlim=[0, 1]
    )
    g.map_dataframe(
        sns.histplot,
        bins=24,
        binrange=(0, 1),
        x="propensity",
        weights="weight",
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
    plt.legend(handles=legend_patches)

    return g
