from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import math

import numpy as np
import torch

from pybalance.utils.matching_data import AggregateTarget
from pybalance.utils.preprocess import (
    BaseMatchingPreprocessor,
    CategoricOneHotEncoder,
    ChainPreprocessor,
    CrossTermsPreprocessor,
    DecisionTreeEncoder,
    FloatEncoder,
    NumericBinsEncoder,
    StandardMatchingPreprocessor,
)


def _rate_lookup(rates: Dict[Any, float], category: Any) -> float:
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
    return 0.0


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

    if all(isinstance(step, (CategoricOneHotEncoder, FloatEncoder)) for step in steps):
        return

    raise ValueError(
        "Aggregate targets require a StandardMatchingPreprocessor "
        f"(or equivalent). Got: {preprocessor.__class__.__name__}."
    )


def compute_aggregate_feature_moments(
    aggregate_target: AggregateTarget,
    preprocessor: BaseMatchingPreprocessor,
    pool_std: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Map an AggregateTarget into the fitted preprocessor's output feature space.

    Returns ``(target_mean, target_std)`` with shape ``(1, n_features)``.
    Missing numeric stds fall back to the corresponding pool std when provided;
    otherwise 0. Categoric one-hot stds use ``sqrt(p * (1 - p))``.
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

    for feature in aggregate_target.headers.numeric:
        out_cols = preprocessor.get_feature_names_out(feature)
        if len(out_cols) != 1:
            raise ValueError(
                f"Expected numeric feature '{feature}' to map to one output column, "
                f"got {out_cols}."
            )
        out_col = out_cols[0]
        idx = feature_index[out_col]
        stats = aggregate_target.numeric[feature]
        means[idx] = stats["mean"]
        if "std" in stats:
            stds[idx] = stats["std"]
        elif pool_std_vec is not None:
            stds[idx] = pool_std_vec[idx]
        else:
            stds[idx] = 0.0

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
            means[idx] = p
            stds[idx] = math.sqrt(max(p * (1.0 - p), 0.0))

    mean_t = torch.tensor(means, dtype=torch.float32, device=device).reshape(1, -1)
    std_t = torch.tensor(stds, dtype=torch.float32, device=device).reshape(1, -1)
    return mean_t, std_t
