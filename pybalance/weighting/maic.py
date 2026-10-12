from __future__ import annotations

from typing import Dict

from pybalance.utils import MatchingData
from pybalance.weighting.entropy import EntropyBalanceWeighter


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
        MatchingData returned by transform().
    :param verbose: Whether to log fitting diagnostics.
    :param limit_penalty: See ``EntropyBalanceWeighter``.
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
        """Return the weighter's configuration parameters as a dict."""
        params = super().get_params()
        del params["match_variance"]
        return params
