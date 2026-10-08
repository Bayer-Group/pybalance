from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import math

import numpy as np
import pandas as pd
import torch

from pybalance.utils.matching_data import AggregateTarget, MatchingData
from pybalance.utils.preprocess import (
    BaseMatchingPreprocessor,
    CategoricOneHotEncoder,
    ChainPreprocessor,
    CrossTermsPreprocessor,
    DecisionTreeEncoder,
    FixedNumericBinsEncoder,
    FloatEncoder,
    NumericBinsEncoder,
    StandardMatchingPreprocessor,
)


def _rate_lookup(rates: Dict[Any, float], category: Any) -> Optional[float]:
    if category in rates:
        return rates[category]
    # Tolerate int/float/"1" style mismatches between published tables and encoders.
    candidates = [category]
    try:
        candidates.append(float(category))
    except (TypeError, ValueError):
        pass
    try:
        as_float = float(category)
        if as_float.is_integer():
            candidates.append(int(as_float))
            candidates.append(str(int(as_float)))
        candidates.append(str(category))
        candidates.append(str(as_float))
    except (TypeError, ValueError):
        candidates.append(str(category))

    for key in candidates:
        if key in rates:
            return rates[key]
    # Category not disclosed for this feature: let the caller decide the
    # fallback (an undisclosed category is unconstrained, not necessarily 0%).
    return None


def _find_onehot_encoder(
    preprocessor: BaseMatchingPreprocessor,
) -> Optional[CategoricOneHotEncoder]:
    if isinstance(preprocessor, CategoricOneHotEncoder):
        return preprocessor
    if isinstance(preprocessor, ChainPreprocessor):
        for step in preprocessor.preprocessors:
            if isinstance(step, CategoricOneHotEncoder):
                return step
    return None


def _assert_preprocessor_supports_aggregates(
    preprocessor: BaseMatchingPreprocessor,
) -> None:
    unsupported = (NumericBinsEncoder, CrossTermsPreprocessor, DecisionTreeEncoder)
    if isinstance(preprocessor, ChainPreprocessor):
        steps = list(preprocessor.preprocessors)
    else:
        steps = [preprocessor]

    bad = [step.__class__.__name__ for step in steps if isinstance(step, unsupported)]
    if bad:
        raise ValueError(
            "Aggregate targets are currently supported only for beta-style "
            "objectives (means / one-hot category rates). Unsupported "
            f"preprocessor components: {', '.join(bad)}."
        )

    if isinstance(preprocessor, StandardMatchingPreprocessor):
        return

    if all(
        isinstance(
            step, (CategoricOneHotEncoder, FloatEncoder, FixedNumericBinsEncoder)
        )
        for step in steps
    ):
        return

    raise ValueError(
        "Aggregate targets require a StandardMatchingPreprocessor "
        f"(or equivalent). Got: {preprocessor.__class__.__name__}."
    )


def compute_aggregate_feature_moments(
    aggregate_target: AggregateTarget,
    preprocessor: BaseMatchingPreprocessor,
    pool_std: Optional[torch.Tensor] = None,
    pool_mean: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Map an AggregateTarget into the fitted preprocessor's output feature space.

    Returns ``(target_mean, target_std, constrained, disclosed)``. The first
    two have shape ``(1, n_features)``; the others are boolean vectors over the
    output columns. ``disclosed`` marks columns whose target mean is known (a
    numeric mean, the rate of a quantile indicator, or the rate of a categoric
    level). ``constrained`` is the subset of those that should actually be
    matched: it leaves out a level implied by the others (the "0" level of a
    quantile indicator, the first level of a fully disclosed categoric).
    Columns that are not ``disclosed`` have only placeholder entries in
    ``target_mean`` / ``target_std`` (the pool's own values when provided,
    otherwise 0) and callers must not match them.

    Disclosed categoric rates use ``sqrt(p * (1 - p))`` for std; a numeric
    feature without a disclosed std falls back to the pool std when provided.
    Quantile indicators are expected one-hot encoded without dropping a level;
    only the "1" level carries the constraint, since the "0" level is implied.
    If every level of a categoric feature is disclosed and no level is
    dropped, the first level is likewise implied and not constrained.
    """
    if not preprocessor.is_fitted:
        raise RuntimeError("Preprocessor must be fitted before mapping aggregates.")

    _assert_preprocessor_supports_aggregates(preprocessor)
    onehot = _find_onehot_encoder(preprocessor)

    output_features = preprocessor.output_headers["all"]
    feature_index = {name: i for i, name in enumerate(output_features)}

    pool_std_vec = None
    if pool_std is not None:
        pool_std_vec = pool_std.detach().cpu().numpy().reshape(-1)
        if len(pool_std_vec) != len(output_features):
            raise ValueError("pool_std length does not match preprocessor outputs.")

    pool_mean_vec = None
    if pool_mean is not None:
        pool_mean_vec = pool_mean.detach().cpu().numpy().reshape(-1)
        if len(pool_mean_vec) != len(output_features):
            raise ValueError("pool_mean length does not match preprocessor outputs.")

    means = (
        pool_mean_vec.astype(np.float32)
        if pool_mean_vec is not None
        else np.zeros(len(output_features), dtype=np.float32)
    )
    stds = (
        pool_std_vec.astype(np.float32)
        if pool_std_vec is not None
        else np.zeros(len(output_features), dtype=np.float32)
    )
    constrained = np.zeros(len(output_features), dtype=bool)
    disclosed = np.zeros(len(output_features), dtype=bool)

    for feature in aggregate_target.headers.numeric:
        stats = aggregate_target.numeric[feature]
        out_cols = preprocessor.get_feature_names_out(feature)
        if "mean" in stats or "std" in stats:
            # The raw column is needed for the mean and/or the variance.
            if feature not in out_cols:
                raise ValueError(
                    f"Numeric feature '{feature}' discloses a mean/std but the "
                    f"preprocessor dropped its raw column (got {out_cols}). Use "
                    "AggregateTargetBalanceCalculator."
                )
            idx = feature_index[feature]
            if "mean" in stats:
                means[idx] = stats["mean"]
                constrained[idx] = disclosed[idx] = True
            if "std" in stats:
                stds[idx] = stats["std"]
        for q, value in stats.get("quantile", []):
            # FixedNumericBinsEncoder names its output "{feature}_q{q}", which
            # CategoricOneHotEncoder then suffixes with the level (e.g.
            # "..._1.0"), so match by prefix rather than an exact name.
            prefix = f"{feature}_q{q}"
            matches = [c for c in out_cols if c == prefix or c.startswith(prefix + "_")]
            if not matches:
                raise ValueError(
                    f"Numeric feature '{feature}' discloses a quantile (q={q}) "
                    f"but the preprocessor produced no output column prefixed "
                    f"'{prefix}' (got {out_cols}). Use "
                    "AggregateTargetBalanceCalculator, which dichotomizes "
                    "quantile-disclosed features automatically."
                )
            for col in matches:
                # A disclosed quantile (q, value) means P(raw <= value) = q, so
                # by definition (approximately, for discrete columns with ties
                # at `value`) a (1 - q) fraction of the population falls above
                # it, i.e. is "1" in the indicator.
                is_above = col.rsplit("_", 1)[-1] == "1.0"
                rate = 1.0 - q if is_above else q
                idx = feature_index[col]
                means[idx] = rate
                stds[idx] = math.sqrt(max(rate * (1.0 - rate), 0.0))
                constrained[idx] = is_above
                disclosed[idx] = True

    for feature in aggregate_target.headers.categoric:
        if onehot is None:
            raise ValueError(
                "Aggregate categoric features require a CategoricOneHotEncoder "
                "in the preprocessor chain."
            )
        rates = aggregate_target.categoric[feature]
        out_cols = preprocessor.get_feature_names_out(feature)
        encoder = onehot.onehot_encoder
        feature_pos = list(encoder.feature_names_in_).index(feature)
        categories = list(encoder.categories_[feature_pos])
        drop_idx = None
        if encoder.drop is not None:
            drop_idx = encoder.drop_idx_[feature_pos]

        kept_categories = [
            cat for i, cat in enumerate(categories) if drop_idx is None or i != drop_idx
        ]
        if len(kept_categories) != len(out_cols):
            raise RuntimeError(
                f"Mismatch mapping categoric feature '{feature}' to one-hot columns."
            )

        level_rates = [_rate_lookup(rates, category) for category in kept_categories]
        for p, out_col in zip(level_rates, out_cols):
            if p is None:
                # Not disclosed: no constraint, and the placeholder values
                # (pool mean/std) are never matched against.
                continue
            idx = feature_index[out_col]
            means[idx] = p
            stds[idx] = math.sqrt(max(p * (1.0 - p), 0.0))
            constrained[idx] = disclosed[idx] = True

        if drop_idx is None and all(p is not None for p in level_rates):
            # All levels disclosed: the first is implied by the others.
            constrained[feature_index[out_cols[0]]] = False

    mean_t = torch.tensor(means, dtype=torch.float32, device=device).reshape(1, -1)
    std_t = torch.tensor(stds, dtype=torch.float32, device=device).reshape(1, -1)
    return mean_t, std_t, constrained, disclosed


def _quantile_label(q: float) -> str:
    named = {0.0: "min", 0.5: "median", 1.0: "max"}
    if q in named:
        return named[q]
    pct = 100 * q
    if pct != int(pct):
        return f"{q:g} quantile"
    pct = int(pct)
    if 10 <= pct % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(pct % 10, "th")
    return f"{pct}{suffix} percentile"


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    # Smallest value whose cumulative weight reaches q (the weighted inverse CDF).
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    idx = min(int(np.searchsorted(cumulative, q - 1e-12, side="left")), len(values) - 1)
    return float(values[order][idx])


def aggregate_target_constraints(
    matching_data: MatchingData,
    quantiles_as: str = "proportion",
    weights: Optional[str] = None,
    limits_as_proportion: bool = False,
) -> pd.DataFrame:
    """
    List every constraint the AggregateTarget of ``matching_data`` discloses,
    next to the value the (current) pool achieves for it.

    Returns a DataFrame with columns ``feature``, ``constraint``, ``target``
    and ``pool``. One row per disclosed statistic: a numeric feature's
    ``mean`` and ``std``, a disclosed quantile, and each disclosed categoric
    level as ``rate of <level>``. Categoric levels the target does not
    disclose are unconstrained and therefore omitted.

    :param quantiles_as: How a disclosed quantile ``(q, value)`` is listed.
        ``"proportion"`` (default) fixes the cutpoint: the row is
        ``P(x <= value)`` with target ``q`` (a disclosed ``min`` / ``max`` is
        the 0th / 100th quantile and is labelled as such). ``"value"`` fixes
        the quantile instead: the row is e.g. ``75th percentile`` with target
        ``value`` and the pool's own 75th percentile.
    :param weights: Name of a pool column holding patient weights (e.g. from a
        Weighter). The pool's statistics are then weighted, so a reweighted
        pool can be compared to the target without selecting a subset. The
        std is the weighted population std.
    :param limits_as_proportion: List a disclosed ``min`` / ``max`` as
        ``P(x <= limit)`` even when ``quantiles_as="value"``. For a weighted
        pool this is the meaningful form: every patient keeps a positive
        weight, so the extreme *value* never moves, only the weight beyond it.
    """
    if quantiles_as not in ("proportion", "value"):
        raise ValueError(
            f"quantiles_as must be 'proportion' or 'value'; got {quantiles_as!r}."
        )
    if not matching_data.has_aggregate_target:
        raise ValueError("matching_data must have an AggregateTarget.")
    target = matching_data.aggregate_target
    pool = matching_data.get_population(matching_data.pool_name)
    if weights is None:
        w = np.ones(len(pool))
    elif weights in pool.columns:
        w = pool[weights].to_numpy(dtype=float)
    else:
        raise ValueError(f"Weight column {weights!r} not found in the pool.")

    rows = []

    def add(feature, constraint, target_value, pool_value):
        rows.append(
            {
                "feature": feature,
                "constraint": constraint,
                "target": float(target_value),
                "pool": float(pool_value),
            }
        )

    for feature, stats in target.numeric.items():
        raw = pool[feature].to_numpy(dtype=float)
        if "mean" in stats:
            add(feature, "mean", stats["mean"], np.average(raw, weights=w))
        if "std" in stats:
            if weights is None:
                std = raw.std(ddof=1)
            else:
                std = math.sqrt(
                    np.average((raw - np.average(raw, weights=w)) ** 2, weights=w)
                )
            add(feature, "std", stats["std"], std)
        for q, value in stats.get("quantile", []):
            is_limit = q in (0.0, 1.0)
            if quantiles_as == "value" and not (limits_as_proportion and is_limit):
                pool_value = (
                    np.quantile(raw, q)
                    if weights is None
                    else _weighted_quantile(raw, w, q)
                )
                add(feature, _quantile_label(q), value, pool_value)
                continue
            limit = {0.0: " (min)", 1.0: " (max)"}.get(q, "")
            add(
                feature,
                f"P(x <= {value:g}){limit}",
                q,
                np.average(raw <= value, weights=w),
            )

    for feature, rates in target.categoric.items():
        for level, rate in rates.items():
            mask = pool[feature] == level
            if not mask.any():
                # Tolerate int/float/str mismatches between a published table and the pool.
                mask = pool[feature].astype(str) == str(level)
            add(
                feature,
                f"rate of {level}",
                rate,
                np.average(mask.to_numpy(), weights=w),
            )

    return pd.DataFrame(rows, columns=["feature", "constraint", "target", "pool"])
