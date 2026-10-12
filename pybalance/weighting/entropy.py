from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from pybalance.utils import BalanceCalculator, MatchingData
from pybalance.weighting.base import BaseWeighter, effective_sample_size

logger = logging.getLogger(__name__)


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
        MatchingData returned by transform().
    :param verbose: Whether to log fitting diagnostics.
    :param limit_penalty: Strength of the quadratic penalty on the weight
        placed beyond a disclosed min / max of an ``AggregateTarget`` (a soft
        constraint). Larger values enforce the limit more tightly.
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
        """Return the weighter's configuration parameters as a dict."""
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

    def fit(self) -> "EntropyBalanceWeighter":
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
