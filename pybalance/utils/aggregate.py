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
        isinstance(step, (CategoricOneHotEncoder, FloatEncoder, FixedNumericBinsEncoder))
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

    Returns ``(target_mean, target_std)`` with shape ``(1, n_features)``.
    Missing numeric stds fall back to the corresponding pool std when provided;
    otherwise 0. A categoric level not present in a feature's disclosed rates
    is left unconstrained, falling back to the pool's own mean/std for that
    one-hot column when provided (otherwise 0); it is *not* assumed to be 0%.
    Disclosed categoric rates use ``sqrt(p * (1 - p))`` for std.
    """
    if not preprocessor.is_fitted:
        raise RuntimeError("Preprocessor must be fitted before mapping aggregates.")

    _assert_preprocessor_supports_aggregates(preprocessor)
    onehot = _find_onehot_encoder(preprocessor)

    output_features = preprocessor.output_headers["all"]
    means = np.zeros(len(output_features), dtype=np.float32)
    stds = np.zeros(len(output_features), dtype=np.float32)
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

    for feature in aggregate_target.headers.numeric:
        stats = aggregate_target.numeric[feature]
        has_mean = "mean" in stats
        quantiles = stats.get("quantile", [])
        out_cols = preprocessor.get_feature_names_out(feature)
        expected_n = int(has_mean) + len(quantiles)
        if len(out_cols) != expected_n:
            raise ValueError(
                f"Expected numeric feature '{feature}' to map to {expected_n} "
                f"output column(s) (one per disclosed 'mean' and one per "
                f"disclosed quantile/median), got {out_cols}."
            )
        if has_mean:
            idx = feature_index[feature]
            means[idx] = stats["mean"]
            if "std" in stats:
                stds[idx] = stats["std"]
            elif pool_std_vec is not None:
                stds[idx] = pool_std_vec[idx]
            else:
                stds[idx] = 0.0
        for q, value in quantiles:
            # FixedNumericBinsEncoder names its output "{feature}_q{q}", which
            # CategoricOneHotEncoder then suffixes further (e.g. "..._1.0"), so
            # match by prefix rather than relying on get_feature_names_out()'s
            # column order or an exact name.
            prefix = f"{feature}_q{q}"
            matches = [
                c for c in out_cols if c == prefix or c.startswith(prefix + "_")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Numeric feature '{feature}' discloses a quantile (q={q}) "
                    f"but the preprocessor did not produce exactly one output "
                    f"column prefixed '{prefix}' (got {out_cols}). Use "
                    "AggregateTargetBalanceCalculator, which dichotomizes "
                    "quantile-disclosed features automatically."
                )
            idx = feature_index[matches[0]]
            # A disclosed quantile (q, value) means P(raw <= value) = q, so by
            # definition (approximately, for discrete columns with ties at
            # `value`) a (1 - q) fraction of the population falls above it.
            rate = 1.0 - q
            means[idx] = rate
            stds[idx] = math.sqrt(max(rate * (1.0 - rate), 0.0))

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

        for category, out_col in zip(kept_categories, out_cols):
            idx = feature_index[out_col]
            p = _rate_lookup(rates, category)
            if p is None:
                # Category not disclosed: leave it unconstrained (zero loss
                # regardless of composition) by falling back to the pool's
                # own rate for this one-hot column, the same way a numeric
                # feature without a disclosed std falls back to pool_std.
                means[idx] = pool_mean_vec[idx] if pool_mean_vec is not None else 0.0
                stds[idx] = pool_std_vec[idx] if pool_std_vec is not None else 0.0
                continue
            means[idx] = p
            stds[idx] = math.sqrt(max(p * (1.0 - p), 0.0))

    mean_t = torch.tensor(means, dtype=torch.float32, device=device).reshape(1, -1)
    std_t = torch.tensor(stds, dtype=torch.float32, device=device).reshape(1, -1)
    return mean_t, std_t


def aggregate_target_constraints(matching_data: MatchingData) -> pd.DataFrame:
    """
    List every constraint the AggregateTarget of ``matching_data`` discloses,
    next to the value the (current) pool achieves for it.

    Returns a DataFrame with columns ``feature``, ``constraint``, ``target``
    and ``pool``. One row per disclosed statistic: a numeric feature's
    ``mean`` and ``std``, a disclosed quantile ``(q, value)`` as
    ``P(x <= value)`` (target ``q``), and each disclosed categoric level as
    ``rate of <level>``. Categoric levels the target does not disclose are
    unconstrained and therefore omitted.
    """
    if not matching_data.has_aggregate_target:
        raise ValueError("matching_data must have an AggregateTarget.")
    target = matching_data.aggregate_target
    pool = matching_data.get_population(matching_data.pool_name)

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
        raw = pool[feature].astype(float)
        if "mean" in stats:
            add(feature, "mean", stats["mean"], raw.mean())
        if "std" in stats:
            add(feature, "std", stats["std"], raw.std())
        for q, value in stats.get("quantile", []):
            add(feature, f"P(x <= {value:g})", q, (raw <= value).mean())

    for feature, rates in target.categoric.items():
        for level, rate in rates.items():
            mask = pool[feature] == level
            if not mask.any():
                # Tolerate int/float/str mismatches between a published table and the pool.
                mask = pool[feature].astype(str) == str(level)
            add(feature, f"rate of {level}", rate, mask.mean())

    return pd.DataFrame(rows, columns=["feature", "constraint", "target", "pool"])
